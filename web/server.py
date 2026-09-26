"""Browser front end: a FastAPI server that keeps the model loaded, and one plain HTML page (web/index.html).

  python -m web.server [--host 127.0.0.1] [--port 8000] [--indexes DIR] [--uploads DIR]
  then open http://127.0.0.1:8000

The model loads once at startup; each index is loaded on first use and kept (a cli.QuerySession per index), and a
re-saved index is picked up on the next query, as in `cli.py query --interactive`.

API (JSON):
  GET  /api/indexes          indexes under --indexes, with source, document/chunk counts and chunking mode
  GET  /api/indexes/{name}   one index; also loads it, so the first search does not pay for loading
  POST /api/index            build or update an index; streams NDJSON events (log, progress, done / error):
                               {"path": "D:\\repo", "chunking": "windows"|"ast"}  index a folder on this machine
                               {"index": "<name>"}                               re-index an index's source folder
                               multipart: file=<zip>, name, chunking             unpack under --uploads, index it
  POST /api/query            {"index", "query", "top_k", "code_only"} -> ranked documents, timings

Results are grouped by document: each document has a list of hits (today one: the best-matching chunk of the
current version, "version": null). Retrieval across versions can add hits per document without changing the
shape. One indexing job runs at a time (it uses every core); a second request gets 409.
"""
from __future__ import annotations

import argparse
import json
import queue
import re
import shutil
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

import cli
from index.storage import MANIFEST

HERE = Path(__file__).parent
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
MAX_TOP_K = 100


class QueryRequest(BaseModel):
    index: str
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=MAX_TOP_K)
    code_only: bool = False


def _safe_name(name: str) -> str:
    """A folder name for uploads and indexes: letters, digits, '.', '_' and '-' only."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip(".-")[:100]
    if not cleaned or not _NAME.match(cleaned):
        raise HTTPException(400, f"unusable name {name!r}")
    return cleaned


def extract_zip(data: Path, target: Path) -> int:
    """Unpack a zip into target (replacing it), refusing entries that would land outside it. Returns file count.

    A zip holding a single top-level folder is unpacked from inside that folder, so repo.zip containing repo/...
    gives target/<files>, not target/repo/<files>.
    """
    with zipfile.ZipFile(data) as z:
        members = [m for m in z.infolist() if not m.is_dir()]
        names = [m.filename.replace("\\", "/") for m in members]
        tops = {n.split("/", 1)[0] for n in names}
        strip = len(tops) == 1 and all("/" in n for n in names)
        tmp = Path(tempfile.mkdtemp(prefix=".upload-", dir=target.parent))
        try:
            for m, name in zip(members, names):
                rel = name.split("/", 1)[1] if strip else name
                dest = (tmp / rel).resolve()
                if tmp.resolve() not in dest.parents:
                    raise HTTPException(400, f"zip entry {m.filename!r} would be written outside the upload folder")
                dest.parent.mkdir(parents=True, exist_ok=True)
                with z.open(m) as src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out)
            if target.exists():
                shutil.rmtree(target)
            tmp.replace(target)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
    return len(members)


class Server:
    def __init__(self, indexes_dir: Path, uploads_dir: Path, encoder_factory=None):
        self.indexes_dir = Path(indexes_dir).resolve()
        self.uploads_dir = Path(uploads_dir).resolve()
        self._encoder_factory = encoder_factory or cli._make_encoder
        self._encoder = None
        self.model_load_seconds = None
        self._sessions: dict[str, cli.QuerySession] = {}
        self._sessions_lock = threading.Lock()
        self._indexing = threading.Lock()
        self._info_cache: dict[str, tuple[tuple[int, int], dict]] = {}

    @property
    def encoder(self):
        if self._encoder is None:
            t0 = time.perf_counter()
            self._encoder = self._encoder_factory()
            self.model_load_seconds = round(time.perf_counter() - t0, 1)
        return self._encoder

    # --- indexes -----------------------------------------------------------------------------------------------
    def index_dir(self, name: str) -> Path:
        if not _NAME.match(name) or not (self.indexes_dir / name / MANIFEST).is_file():
            raise HTTPException(404, f"no index named {name!r} in {self.indexes_dir}")
        return self.indexes_dir / name

    def info(self, name: str) -> dict:
        path = self.indexes_dir / name / MANIFEST
        st = path.stat()
        stamp = (st.st_size, st.st_mtime_ns)
        cached = self._info_cache.get(name)
        if cached is None or cached[0] != stamp:
            m = json.loads(path.read_text(encoding="utf-8"))
            source = m.get("source", {})
            info = {
                "name": name,
                "source_kind": source.get("kind"),
                "source": source.get("root") or source.get("dataset") or source.get("kind"),
                "documents": m.get("n_documents", len(m.get("documents", {}))),
                "chunks": m.get("n_chunks"),
                "chunking": m.get("chunking", {}).get("name"),
                "header": bool(m.get("chunking", {}).get("header")),
                "saved_at": m.get("saved_at"),
                "reindexable": source.get("kind") == "directory",
                "versions": None,  # retrieval across versions: not implemented yet
            }
            cached = self._info_cache[name] = (stamp, info)
        return cached[1]

    def list_indexes(self) -> list[dict]:
        if not self.indexes_dir.is_dir():
            return []
        return [self.info(p.name) for p in sorted(self.indexes_dir.iterdir())
                if _NAME.match(p.name) and (p / MANIFEST).is_file()]

    def session(self, name: str) -> tuple[cli.QuerySession, float | None]:
        """The index's loaded session, and how long loading took if it was loaded just now."""
        index_dir = self.index_dir(name)
        with self._sessions_lock:
            s = self._sessions.get(name)
            if s is not None:
                return s, None
            t0 = time.perf_counter()
            try:
                s = cli.QuerySession(index_dir, self.encoder, kind_penalty=cli.DEFAULT_KIND_PENALTY)
            except cli.IndexMismatchError as e:
                raise HTTPException(409, str(e)) from e
            s.reader.preload()  # apps: the dataset the snippets come from, so the first search is not slow
            self._sessions[name] = s
            return s, round((time.perf_counter() - t0) * 1000, 1)

    # --- search ------------------------------------------------------------------------------------------------
    def query(self, req: QueryRequest) -> dict:
        s, load_ms = self.session(req.index)
        s.refresh()
        t0 = time.perf_counter()
        rows = s.search(req.query, req.top_k, code_only=req.code_only)
        query_ms = round((time.perf_counter() - t0) * 1000, 1)
        results = [{
            "rank": r["rank"], "doc_id": r["doc_id"], "source_path": r["source_path"], "language": r["language"],
            "kind": r["kind"], "score": r["score"], "similarity": r["similarity"],
            "hits": [{"version": None, "start_line": r["chunks"][0]["start_line"],
                      "end_line": r["chunks"][0]["end_line"], "score": r["chunks"][0]["score"],
                      "snippet": r["snippet"]}],
        } for r in rows]
        return {"index": self.info(req.index), "query_ms": query_ms, "load_ms": load_ms,
                "top_k": req.top_k, "code_only": req.code_only, "results": results}

    # --- indexing ----------------------------------------------------------------------------------------------
    def stream_index(self, root: Path, out: Path, chunking: str, header: bool, before=None):
        """NDJSON events for one indexing job. Raises 409 at once if another job is running."""
        if not self._indexing.acquire(blocking=False):
            raise HTTPException(409, "another indexing job is running; wait for it to finish")
        events: queue.Queue = queue.Queue()

        def put(event: str, **data):
            events.put(json.dumps({"event": event, **data}) + "\n")

        def job():
            try:
                if before is not None:
                    before(put)
                last = [0.0]

                def progress(done, total):
                    now = time.perf_counter()
                    if now - last[0] >= 0.5 or done == total:
                        last[0] = now
                        put("progress", done=done, total=total)

                result = cli.index_folder(root, out, self.encoder, chunking, header=header, progress=progress,
                                          note=lambda message: put("log", message=message))
                put("done", result=result, info=self.info(out.name))
            except (cli.IndexingError, HTTPException, OSError, zipfile.BadZipFile) as e:
                put("error", message=getattr(e, "detail", None) or str(e))
            except Exception as e:  # reported to the page rather than silently ending the stream
                put("error", message=f"{type(e).__name__}: {e}")
            finally:
                events.put(None)
                self._indexing.release()

        threading.Thread(target=job, daemon=True).start()

        def body():
            while (line := events.get()) is not None:
                yield line

        return StreamingResponse(body(), media_type="application/x-ndjson")

    def out_for(self, root: Path) -> Path:
        return self.indexes_dir / cli.default_index_dir(root).name


def create_app(server: Server) -> FastAPI:
    app = FastAPI(title="Code retrieval", docs_url=None, redoc_url=None)

    @app.get("/")
    def page():
        return FileResponse(HERE / "index.html", media_type="text/html")

    @app.get("/api/indexes")
    def indexes():
        return {"indexes": server.list_indexes(), "indexing": server._indexing.locked(),
                "model_load_seconds": server.model_load_seconds}

    @app.get("/api/indexes/{name}")
    def index_info(name: str):
        _, load_ms = server.session(name)
        return {"index": server.info(name), "load_ms": load_ms}

    @app.post("/api/query")
    def query(req: QueryRequest):
        return server.query(req)

    @app.post("/api/index")
    async def index(request: Request):
        if request.headers.get("content-type", "").startswith("multipart/form-data"):
            form = await request.form()
            upload = form.get("file")
            if upload is None or not hasattr(upload, "read"):
                raise HTTPException(400, "multipart request needs a zip in the 'file' field")
            name = _safe_name(str(form.get("name") or Path(upload.filename or "upload").stem))
            chunking = str(form.get("chunking") or "windows")
            server.uploads_dir.mkdir(parents=True, exist_ok=True)
            saved = server.uploads_dir / f".{name}.zip"
            with open(saved, "wb") as f:
                while chunk := await upload.read(1 << 20):
                    f.write(chunk)
            root = server.uploads_dir / name

            def unpack(put):
                try:
                    n = extract_zip(saved, root)
                finally:
                    saved.unlink(missing_ok=True)
                put("log", message=f"unpacked {n:,} files from {upload.filename} into {root}")

            _check_chunking(chunking)
            return server.stream_index(root, server.out_for(root), chunking, True, before=unpack)

        body = await request.json()
        if body.get("index"):  # re-index: the index's own source folder and chunking settings
            name = body["index"]
            index_dir = server.index_dir(name)
            m = json.loads((index_dir / MANIFEST).read_text(encoding="utf-8"))
            if m.get("source", {}).get("kind") != "directory":
                raise HTTPException(400, f"{name} is not a folder index (the apps index is prebuilt: "
                                         "python -m index.build_apps)")
            chunking = m.get("chunking", {})
            return server.stream_index(Path(m["source"]["root"]), index_dir, chunking.get("name", "windows"),
                                       bool(chunking.get("header")))
        if not body.get("path"):
            raise HTTPException(400, "give a folder 'path', an 'index' to re-index, or upload a zip")
        root = Path(body["path"]).expanduser().resolve()
        if not root.is_dir():
            raise HTTPException(400, f"{root} is not a folder on the server")
        chunking = body.get("chunking") or "windows"
        _check_chunking(chunking)
        return server.stream_index(root, server.out_for(root), chunking, True)

    return app


def _check_chunking(name: str) -> None:
    if name not in ("windows", "ast"):
        raise HTTPException(400, f"chunking must be 'windows' or 'ast', not {name!r}")


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(prog="python -m web.server", description="Browser front end for code retrieval.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--indexes", type=Path, default=Path("indexes"), help="folder holding the indexes")
    p.add_argument("--uploads", type=Path, default=Path("uploads"), help="where uploaded zips are unpacked")
    args = p.parse_args(argv)
    cli.use_offline_hub_if_cached(needs_apps_dataset=(args.indexes / "apps" / MANIFEST).is_file())

    import uvicorn
    server = Server(args.indexes, args.uploads)
    cli.log("loading the model...")
    _ = server.encoder
    cli.log(f"model loaded in {server.model_load_seconds:.1f} s; open http://{args.host}:{args.port}")
    uvicorn.run(create_app(server), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
