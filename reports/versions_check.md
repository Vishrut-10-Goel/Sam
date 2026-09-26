# Retrieval across versions: psf/requests, six releases

**What it does:** one index over several commits of a repository. A file identical across versions is stored once,
and a changed file reuses the vectors of its unchanged chunks. Search returns one result per file, not one per
version, and shows which versions hold the matched lines, which have them at a different line, and which differ.

**Measured:** indexing six versions took **1,014 s against 2,042 s for six separate builds (50%)**. It embedded
**788 chunks against 1,871 (42%)**, stored 1,001 (46.5% of what separate indexes would store is shared), and
produced **bit-identical vectors** to a separate build of the same commit. Without grouping, the top 10 for a query
held only **2–4 distinct files** (the same file in five versions); grouped, the top 5 are five files.

There is no ground truth for this task, so nothing here is scored: the report records what the system does.

## Setup

- Repository: [psf/requests](https://github.com/psf/requests). The demo clone was shallow (one commit); its full
  history was fetched (`git fetch --unshallow --tags`) without touching the working tree.
- Versions: `v2.32.5` (2025-08-18), `v2.33.0`, `v2.33.1`, `v2.34.0` (type hints added across most modules: 1,296
  lines inserted), `v2.34.2`, and `611c616` (HEAD, 2026-09-21). All use the `src/requests/` layout.
- Chunking as for folder indexes: 1024-token line windows with the path/class/function header.
- Built with `cli.py index-versions D:\prism-demo\requests --commits v2.32.5 v2.33.0 v2.33.1 v2.34.0 v2.34.2 611c6162
  --out indexes/requests-versions`, then each version separately with `cli.py index` on a checkout of it. One build at
  a time, on mains power (logged). Script and raw log: [`versions/measure_builds.sh`](versions/measure_builds.sh),
  [`versions/build_log.txt`](versions/build_log.txt).

## How it works

- **Unit stored: a file revision** (path + content hash, id `path@hash12`), listing every indexed commit that holds
  exactly that file. The index is an ordinary index (same storage, retriever and fingerprint checks); only its
  documents are revisions, and its source records the commits.
- **Reuse, two levels.** A file whose content hash was already indexed is not chunked or embedded again; its revision
  just gains the new commit (the same content-hash test as incremental updates). A changed file is chunked, and each
  chunk whose embedded text (header + lines) was already embedded copies that vector: window chunks above a file's
  first edit are unchanged. The text hashes are kept per revision, so adding commits later reuses them too.
- **Commits are read from a temporary git worktree**, checked out one after another and removed at the end: the
  repository's own working tree, including uncommitted edits, is never touched. Chunks are made under the plain path,
  so each embeds exactly what a folder index of that commit would.
- **Snippets come from git objects** (`git cat-file --filters`, the same line-ending conversion as a checkout), so a
  result from an old version shows that version's lines and is never stale.
- **Ranking across versions.** Every revision is scored; revisions are grouped by path the way chunks are grouped
  into files, and a file ranks by its best revision (equal scores go to the newest version). Selecting one version
  ranks each file by the revision that version holds and leaves out files it lacks. Each result lists every revision
  of the file with its commits and score, and whether it contains the matched lines verbatim (a text search, so it
  does not depend on where chunk boundaries fall in each revision), and at which line.

## Indexing: versioned vs separate builds

| Version | Files | Unchanged since an indexed version | New or changed | Chunks of changed files | Embedded | Copied (identical chunk) | Seconds | Separate build (s) | Separate build chunks |
|---|---|---|---|---|---|---|---|---|---|
| v2.32.5 | 118 | 0 | 118 | 305 | 305 | 0 | 344.2 | 328 | 306 |
| v2.33.0 | 118 | 85 | 33 | 183 | 101 | 82 | 149.4 | 329 | 307 |
| v2.33.1 | 118 | 111 | 7 | 91 | 60 | 31 | 84.8 | 334 | 307 |
| v2.34.0 | 119 | 74 | 45 | 209 | 176 | 33 | 225.8 | 348 | 318 |
| v2.34.2 | 119 | 113 | 6 | 92 | 53 | 39 | 75.6 | 349 | 319 |
| 611c616 | 120 | 97 | 23 | 121 | 93 | 28 | 123.0 | 354 | 320 |
| **Total** | | | **232 revisions** | **1,001 stored** | **788** | **213** | **1,014 wall** | **2,042** | **1,877** |

- **Time: 50%** of six separate builds (1,014 s vs 2,042 s, wall clock, model load included once vs six times).
  Time falls less than embedding work (42%) because each version still pays a checkout, loading all files and
  chunking the changed ones, and the first version is a full build.
- **Chunks: 788 embedded vs 1,871** that separate indexes would hold for the same files; 1,001 stored (46.5% shared).
  Each separate build also indexed the worktree's `.git` pointer file (one chunk each; 1,877 − 6 = 1,871); the
  versioned indexer skips it.
- **Same vectors:** for every chunk of HEAD and of `v2.32.5`, the versioned index's vector equals the separate
  build's for the same file and lines (maximum absolute difference 0.0; 319 and 305 chunks;
  [`versions/equivalence_check.txt`](versions/equivalence_check.txt)). Reuse changes nothing about what is searched.
- **Adding a version later** embeds only its new chunks: re-running with more commits, or "Add HEAD" on the web page,
  keeps the versions already indexed.

## Search: the near-identical-versions problem

Raw output: [`versions/example_queries.txt`](versions/example_queries.txt) (and `.json`), from
[`versions/example_queries.py`](versions/example_queries.py).

**Ranked directly, revisions flood the top 10:**

| Query | Distinct files in the top 10 revisions |
|---|---|
| how are redirects handled | 2 (`sessions.py` ×5, `models.py` ×5) |
| how do I configure retries | 4 |
| where is HTTP basic authentication implemented | 4 (`auth.py` ×4) |
| parse the parameters of a Content-Type header | 2 (`utils.py` ×5, `models.py` ×5) |
| how are proxy settings read from the environment | 4 (`utils.py` ×5) |
| where does it raise an exception for 4xx and 5xx responses | 4 |

**Grouped, one result per file** (the CLI and the web page), e.g. "how are redirects handled":

```
1. src/requests/sessions.py:100-190  score 0.7031
     matched in: 611c616
     same lines also in: v2.34.2 (at line 100); v2.34.0 (at line 100)
     lines differ in: v2.33.0, v2.33.1 (score 0.6953); v2.32.5 (score 0.6953)
2. src/requests/models.py:682-781  score 0.6706
     matched in: v2.33.0, v2.33.1
     same lines also in: v2.32.5 (at line 680)
     lines differ in: 611c616 (score 0.6394); v2.34.2 (score 0.6311); v2.34.0 (score 0.6210)
```

Five different files fill the top 5 for every query, and nothing is hidden: each result names the versions holding
the same lines (with the line when it moved: `v2.32.5 (at line 680)`) and those that differ, with their own scores.

**One version at a time** changes the answer where the code changed. "how are proxy settings read from the
environment" in `v2.33.0` returns `utils.py` (`resolve_proxies`, score 0.6981) first; in `v2.34.0` it returns
`sessions.py` (`merge_environment_settings`, now type-annotated, 0.7125), with `utils.py`'s `get_environ_proxies` second.

**Latency:** search across all six versions takes 33–56 ms once the index is loaded. Loading it in the web server
also reads all 232 revisions from git (1.4 s, once), since grouping compares each result's revisions; without that,
the first queries took 700 and 320 ms.

## What to know when reading results

- **"Lines differ" is strict.** It means the matched chunk (~90 lines) does not appear verbatim in that version; one
  changed line anywhere in the window counts. `v2.34.0` added type hints to most functions, so for many files every
  version before it "differs" even where the logic did not change. The score next to each version shows how close it is.
- **The best version is not always the newest.** Across all versions a file ranks by whichever version scores
  highest, which can be an old one (e.g. `models.py` for the redirects query matches best in `v2.33.x`). Pick a version
  to see a specific one.
- **Files are grouped by path.** A renamed or moved file is two files. None of the six versions moved files.
- **Not measured for quality:** there are no pre-registered answers across versions, so this is a description, not
  a score.
