"""Build (or incrementally update) the prebuilt AppsRetrieval corpus index, committed under indexes/apps.

Run from the project root:  python -m index.build_apps [out_dir]   (default indexes/apps)
Embeds all 8,765 solutions whole (no chunking) with the shared ONNX encoder: ~40 min on a 12-thread CPU.
If out_dir already holds a compatible index, only new or changed documents are embedded.
"""
import functools
import sys
import time
from pathlib import Path

import loaders.apps
from embedding.onnx_encoder import OnnxEncoder
from index import build_index, chunking_config, load_index, save_index, update_index
from index.storage import MANIFEST

print = functools.partial(print, flush=True)  # show progress live when output goes to a log file

OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("indexes/apps")
LOG_EVERY_SECONDS = 60


def _progress():
    t0 = last = time.perf_counter()

    def report(done: int, total: int) -> None:
        nonlocal last
        now = time.perf_counter()
        if now - last >= LOG_EVERY_SECONDS or done == total:
            last = now
            elapsed = now - t0
            print(f"    {done:,}/{total:,} chunks ({done / total:.0%})  elapsed {elapsed / 60:.1f} min  "
                  f"eta <= {elapsed / done * (total - done) / 60:.1f} min")

    return report


if __name__ == "__main__":
    start = time.perf_counter()
    encoder = OnnxEncoder()
    source = {"kind": "apps", "dataset": loaders.apps.DATASET, "revision": loaders.apps.REVISION}
    docs = list(loaders.apps.load_apps())
    print(f"{len(docs):,} documents -> {OUT_DIR}")

    if (OUT_DIR / MANIFEST).exists():
        index, stats = update_index(load_index(OUT_DIR, encoder=encoder), docs, encoder, source, _progress())
        print(f"updated: {stats}")
    else:
        index = build_index(docs, encoder, chunking_config(loaders.apps.CHUNKING, encoder), source, _progress())
    save_index(index, OUT_DIR)
    print(f"saved {len(index.chunks):,} chunks x {index.dim} dims in {(time.perf_counter() - start) / 60:.1f} min")
