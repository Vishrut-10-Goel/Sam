# AST chunking vs line windows: psf/requests, 10 fresh questions

**Result: a tie on every retrieval measure. AST chunking stays opt-in (`cli.py index --chunking ast`); the default
stays `windows`.** AST gives much tighter line ranges (median 30 vs 83 lines) and indexes 11% faster, but on these
questions it does not retrieve better: it wins two questions and loses two.

Repository: [psf/requests](https://github.com/psf/requests) at `611c616`, indexed whole (source, tests, docs), 120 files.

## Method

Same as the [Scrapy check](scrapy_retrieval_check.md), with fresh questions:

- Ten plain-language questions a developer new to requests would ask. The Scrapy questions were not reused (they
  were used to choose the kind penalty), and the questions avoid the six topics of an earlier demo survey on this
  repository (redirects, timeouts, basic auth, cookies, retries, response encoding), whose results had been seen.
- The correct answer for each (file + line span of the implementing code, from reading the source) was written down
  **before any retrieval ran on these questions**: [`requests_ground_truth.json`](requests_ground_truth.json),
  committed in `685a333` at 18:22:32; the indexes were built from 18:22:45 and queried after that.
- Both indexes use the current defaults apart from chunking (context header on); queries use the CLI defaults
  (test/docs penalty 0.05, top 10). One run, nothing tuned.
- **Strict top-1:** the top hit is a correct file and its best chunk overlaps the pre-registered span. **File in
  top 3 / top 10**, and **MRR (file)**: by the rank of the first correct file.
- Scripts and raw output: [`requests_chunking/`](requests_chunking/) (`run_eval.py`, `results.json` with the top 10
  per question, `index_log.txt`, `eval_log.txt`).

## Summary

| | windows (default) | ast |
|---|---|---|
| Strict top-1 | **7/10** | **7/10** |
| Right file first | 8/10 | 8/10 |
| Right file in top 3 | 10/10 | 10/10 |
| Right file in top 10 | 10/10 | 10/10 |
| MRR (file) | 0.900 | 0.900 |
| Chunks | 319 | 555 (+74%) |
| Median chunk length | 83 lines | 30 lines |
| Build time (whole repo, wall clock) | 371 s | 330 s (−11%) |
| Query latency (model loaded) | 25–33 ms | 22–28 ms |

Build times were measured one after the other on the same laptop in the same session (its power state was not
recorded, and it moved from battery to mains power at some point that afternoon), so compare them with each other
rather than with other timings in this repository. The first AST build attempt failed to start
(a syntax error in an unrelated edit made while the windows build ran); it was re-run with nothing else running,
as `index_log.txt` records.

## Per question

| # | Question | Pre-registered answer | windows: top hit (file rank / strict) | ast: top hit (file rank / strict) |
|---|---|---|---|---|
| 1 | where are proxy settings read from the environment | `utils.py` 810-939, `sessions.py` 831-868 | `utils.py:813-916` (1 / ✅) | `utils.py:873-910` (1 / ✅) |
| 2 | how is the server's SSL certificate verified and which CA bundle is used | `adapters.py` 307-363, `certs.py` | `tests/certs/.../server.csr` (2 / ❌) | `adapters.py:307-356` (1 / ✅) |
| 3 | how is a dictionary of form fields turned into the request body | `models.py` 150-180, 576-652 | `models.py:109-214` (1 / ✅) | `utils.py:439-473` `parse_dict_header` (2 / ❌) |
| 4 | how are files uploaded as multipart form data | `models.py` 182-251, 576-652 | `models.py:202-306` (1 / ✅) | `models.py:182-232` (1 / ✅) |
| 5 | where does it raise an exception for 4xx and 5xx responses | `models.py` 1144-1171 | `models.py:811-907` (1 / ❌) | `models.py:837-856` `__bool__` (1 / ❌) |
| 6 | how do I download a large file in pieces instead of all at once | `models.py` 906-977 | `models.py:894-993` (1 / ✅) | `models.py:906-913` (1 / ✅) |
| 7 | can it read login credentials from a .netrc file | `utils.py` 231-280, `sessions.py` 511-555 | `utils.py:196-300` (1 / ✅) | `utils.py:231-282` (1 / ✅) |
| 8 | how is the Link header of a response parsed | `utils.py` 965-999, `models.py` 1126-1142 | `models.py:1078-1172` (1 / ✅) | `models.py:1126-1172` (1 / ✅) |
| 9 | where are query string parameters added to the URL | `models.py` 483-563, 150-180 | `models.py:109-214` (1 / ✅) | `utils.py:1040-1069` `prepend_scheme_if_needed` (2 / ❌) |
| 10 | how does a session decide which transport adapter handles a URL | `sessions.py` 870-897 | `tests/test_requests.py:1645-1730` (2 / ❌) | `sessions.py:870-898` (1 / ✅) |

Paths are under `src/requests/` unless shown otherwise.

## Reading

- **Where AST wins (Q2, Q10), it is precision.** With ~100-line windows, `adapters.py`'s `cert_verify` and
  `sessions.py`'s `get_adapter` share their vector with neighbouring code, and a test fixture (a certificate file) or
  a test that exercises adapters edges ahead even after the test penalty. As single functions they win clearly.
- **Where AST loses (Q3, Q9), a small function with the question's words wins.** `parse_dict_header` ("dictionary")
  and `prepend_scheme_if_needed` ("URL") are short, lexically close functions; the right file is second both times,
  but on a chunk next to the pre-registered span (`models.py` 357-406 and 108-145), so neither counts. A window
  that holds the whole `RequestEncodingMixin` carries enough context to win these.
- **Q5 fails both ways, near the answer:** both return `Response.__bool__` / `ok` ("True if status_code is less than
  400"), which sits next to `raise_for_status` in meaning and in the file.
- With ten questions a two-for-two swap is noise. The measures that do differ are mechanical: AST cites 30-line spans
  instead of 83-line ones, embeds 9% fewer tokens (see `experiments/ast_chunk_stats.py`), builds 11% faster, and
  stores 74% more vectors.

## Decision

The rule set before the run: make AST the default only if it wins. It ties, so **`windows` stays the default and
`--chunking ast` stays opt-in**, now measured rather than unmeasured. Its tighter ranges matter less for display than
they did, since result snippets now open on the lines that match the query (`retrieval/snippets.py`,
`focus_offset`). Revisit with more questions (these ten are now spent too) or a larger repository, where precision
might count for more.
