# Real-codebase retrieval check: Scrapy

Repository: [scrapy/scrapy](https://github.com/scrapy/scrapy) at `e4eda778f9`, indexed whole (source, tests, docs) with `cli.py index`: **654 files, 1,980 chunks, 42.8 min** on a 12-thread CPU.

Index log: `654 files; skipped {'minified': 2, 'binary': 30, 'empty': 19, 'not_utf8': 3}`

## Method

- Ten questions of the kind a developer new to the codebase would ask, phrased in plain language.
- The correct answer for each (file + line span of the implementing code) was written down from reading the source **before any retrieval was run**, together with 'related' answers (docs, tests, settings) that would be acceptable but weaker.
- **Strict grade:** the top hit is a pre-registered correct file and its best chunk overlaps the pre-registered span. My judgement is given alongside when it differs.
- Queries were run in one `--interactive`-style session (model loaded once), top 10 kept.
- Raw data: [`scrapy_ground_truth.json`](scrapy_ground_truth.json) (the pre-registered answers) and
  [`scrapy_results.json`](scrapy_results.json) (top 10 per query, with snippets).

## Summary

| | Result |
|---|---|
| Top-1 correct (strict) | **5 / 10** |
| Top-1 useful (my judgement: also counts the Q2 docs answer) | 6 / 10 |
| Correct file in top 3 | 7 / 10 |
| Correct file in top 10 | 9 / 10 |
| Top-1 hit is a doc or test file | 5 / 10 |
| Query latency (model loaded) | 68-150 ms |

**Verdict:** usable but not good. When the question uses the code's own vocabulary (retry, redirect, duplicate, robots.txt, encoding) the right file comes first. When it does not, prose wins: documentation and test files outrank the implementation, because natural-language questions are closer to prose than to code. One question (Q10) fails outright on a vocabulary gap.

## What goes wrong

1. **Docs and tests outrank code.** 5 of the 10 top hits are `.rst` docs or tests. Docs are prose, like the questions; tests repeat the implementation's vocabulary many times. Nothing in retrieval knows that 'where is X' means 'the implementation of X'.
2. **Chunks are large and full.** Median chunk is 88 lines / 1012 tokens (90th percentile 1022 tokens): almost every window is packed to the 1024-token limit, so a chunk typically mixes imports, several functions and a class. One embedding has to represent all of it, and the cited line range (~90 lines) is coarse. It also makes indexing slow (43 min for 654 files).
3. **Vocabulary gap.** 'too many links deep' never matches 'depth' (Q10); the code's own identifiers (`DEPTH_LIMIT`, `DepthMiddleware`) and its file path carry the concept, but the model only sees the chunk text, not the path.
4. **Scores are compressed.** Top-10 scores mostly sit between 0.62 and 0.75, so small, irrelevant differences (a docs chunk that happens to say 'concurrency' five times) decide the order.

## Queries

### Q1. "where is the retry logic for failed requests"

- **Pre-registered answer:** `scrapy/downloadermiddlewares/retry.py` 35-203
- **Strict grade:** ✅ correct   **My judgement:** correct
- **Correct file rank:** 1   **Latency:** 150 ms

| # | Result | Lines | Score |
|---|---|---|---|
| 1 | `scrapy/downloadermiddlewares/retry.py` | 1-99 | 0.742 |
| 2 | `docs/topics/downloader-middleware.rst` | 1067-1188 | 0.705 |
| 3 | `tests/test_downloadermiddleware_retry.py` | 152-256 | 0.691 |

Top hit is `retry.py` lines 1-99: the module docstring and `get_retry_request()`, the retry decision itself. Exactly the answer. #2 (middleware docs) and #3 (retry tests) are useful context.

<details><summary>Top hit, first lines</summary>

```
"""
An extension to retry failed requests that are potentially caused by temporary
problems such as a connection timeout or HTTP 500 error.

You can change the behaviour of this middleware by modifying the scraping settings:
RETRY_TIMES - how many times to retry a failed page
RETRY_HTTP_CODES - which HTTP response codes to retry
"""

from __future__ import annotations

from logging import Logger, getLevelName, getLogger
```
</details>

<details><summary>Ranks 4-10</summary>

4. `docs/topics/stats.rst` 643-743 (0.665)
5. `tests/test_crawl.py` 939-1026 (0.656)
6. `tests/test_pipeline_media.py` 353-430 (0.649)
7. `scrapy/exceptions.py` 102-200 (0.641)
8. `tests/test_request_attribute_binding.py` 101-200 (0.638)
9. `docs/topics/request-response.rst` 959-1080 (0.637)
10. `tests/test_command_parse.py` 99-217 (0.631)
</details>

### Q2. "how are download timeouts configured"

- **Pre-registered answer:** `scrapy/downloadermiddlewares/downloadtimeout.py` 1-60, `scrapy/settings/default_settings.py` 315-315
- **Strict grade:** ❌ wrong   **My judgement:** acceptable (docs)
- **Correct file rank:** 4   **Latency:** 118 ms

| # | Result | Lines | Score |
|---|---|---|---|
| 1 | `docs/topics/settings.rst` | 1161-1275 | 0.736 |
| 2 | `scrapy/core/downloader/__init__.py` | 162-246 | 0.712 |
| 3 | `tests/test_downloadermiddleware_downloadtimeout.py` | 1-52 | 0.699 |

Top hit is the `settings.rst` section that defines `DOWNLOAD_TIMEOUT` (and says it can be overridden per request). For a *how is it configured* question that is a genuinely useful answer, but I pre-registered docs as 'related', not 'correct', so strictly it is a miss. The implementation (`downloadtimeout.py`) is #4. #2 (downloader slot jitter code) is irrelevant.

<details><summary>Top hit, first lines</summary>

```
        }

.. note::

    For other downloader slots default settings values will be used:

    -   :setting:`DOWNLOAD_DELAY`: ``delay``
    -   :setting:`CONCURRENT_REQUESTS_PER_DOMAIN`: ``concurrency``
    -   :setting:`DOWNLOAD_DELAY_JITTER`: ``jitter``

Requests are assigned to a slot based on their URL domain. To assign a request
to a specific slot instead, set the name of the slot as the ``download_slot``
```
</details>

<details><summary>Ranks 4-10</summary>

4. `scrapy/downloadermiddlewares/downloadtimeout.py` 1-44 (0.688)
5. `docs/topics/request-response.rst` 866-969 (0.668)
6. `tests/utils/bases/download_handlers_http.py` 825-900 (0.652)
7. `tests/test_core_downloader.py` 80-153 (0.649)
8. `scrapy/templates/project/module/settings.py.tmpl` 79-88 (0.639)
9. `tests/test_crawl.py` 94-167 (0.639)
10. `scrapy/settings/default_settings.py` 323-390 (0.638)
</details>

### Q3. "where do we figure out the character encoding of the response body and decode it to text"

- **Pre-registered answer:** `scrapy/http/response/text.py` 100-160
- **Strict grade:** ✅ correct   **My judgement:** correct
- **Correct file rank:** 1   **Latency:** 121 ms

| # | Result | Lines | Score |
|---|---|---|---|
| 1 | `scrapy/http/response/text.py` | 1-116 | 0.735 |
| 2 | `docs/topics/request-response.rst` | 1434-1527 | 0.706 |
| 3 | `tests/test_http_response_text.py` | 149-223 | 0.686 |

`text.py` lines 1-116: `TextResponse.encoding`, `_declared_encoding()` (BOM, headers, declared) and the start of the decode path. Right file and right region; `_body_inferred_encoding()` sits just after the chunk (line 126), in the next window.

<details><summary>Top hit, first lines</summary>

```
"""
This module implements the TextResponse class which adds encoding handling and
discovering (through HTTP headers) to base Response class.

See documentation in docs/topics/request-response.rst
"""

from __future__ import annotations

import json
from contextlib import suppress
from typing import TYPE_CHECKING, Any, cast
```
</details>

<details><summary>Ranks 4-10</summary>

4. `scrapy/downloadermiddlewares/httpcompression.py` 94-181 (0.676)
5. `docs/topics/shell.rst` 254-344 (0.649)
6. `docs/topics/extraction.rst` 98-201 (0.635)
7. `scrapy/spiders/sitemap.py` 159-176 (0.635)
8. `tests/test_selector.py` 75-171 (0.623)
9. `tests/test_robotstxt_interface.py` 73-142 (0.617)
10. `docs/topics/dynamic-content.rst` 175-273 (0.614)
</details>

### Q4. "how are redirects followed, and where is the maximum number of redirects enforced"

- **Pre-registered answer:** `scrapy/downloadermiddlewares/redirect.py` 93-200
- **Strict grade:** ✅ correct   **My judgement:** correct
- **Correct file rank:** 1   **Latency:** 147 ms

| # | Result | Lines | Score |
|---|---|---|---|
| 1 | `scrapy/downloadermiddlewares/redirect.py` | 84-171 | 0.675 |
| 2 | `docs/topics/downloader-middleware.rst` | 865-967 | 0.655 |
| 3 | `tests/utils/bases/redirect.py` | 1-97 | 0.650 |

`redirect.py` lines 84-171 contains `_redirect()`, which checks `redirect_times` against `REDIRECT_MAX_TIMES`. Exactly the answer.

<details><summary>Top hit, first lines</summary>

```
                f"handle_referer() method."
            )
        logger.warning(
            f"{redirect_cls} found no {referer_cls} instance to handle "
            f"Referer header handling, so the Referer header will be removed "
            f"on redirects. To set a Referer header on redirects, enable "
            f"{referer_cls} (or a subclass), or {replacement}.",
        )

    def _redirect(
        self, redirected: Request, request: Request, reason: str | int
    ) -> Request:
```
</details>

<details><summary>Ranks 4-10</summary>

4. `tests/test_downloadermiddleware_cookies.py` 514-625 (0.636)
5. `tests/test_downloadermiddleware_redirect.py` 310-396 (0.628)
6. `tests/utils/bases/http_response.py` 260-343 (0.626)
7. `scrapy/spidermiddlewares/referer.py` 90-186 (0.624)
8. `tests/test_http_response_text.py` 287-375 (0.623)
9. `tests/test_spidermiddleware_referer.py` 320-421 (0.615)
10. `tests/utils/bases/download_handlers_http.py` 1689-1780 (0.607)
</details>

### Q5. "where are cookies stored and attached to outgoing requests"

- **Pre-registered answer:** `scrapy/downloadermiddlewares/cookies.py` 41-140, `scrapy/http/cookies.py` 1-400
- **Strict grade:** ❌ wrong   **My judgement:** wrong (right answer at #3-#4)
- **Correct file rank:** 3   **Latency:** 85 ms

| # | Result | Lines | Score |
|---|---|---|---|
| 1 | `docs/topics/cookies.rst` | 118-135 | 0.656 |
| 2 | `tests/test_downloadermiddleware_cookies.py` | 381-453 | 0.649 |
| 3 | `scrapy/http/cookies.py` | 86-199 | 0.644 |

Top hit is a `cookies.rst` fragment about `COOKIES_DEBUG` log output, not how cookies are stored or attached. #2 is a test. The real answers are #3 (`http/cookies.py`, the `CookieJar` wrapper) and #4 (`CookiesMiddleware`). A developer scanning the top result would be misled.

<details><summary>Top hit, first lines</summary>

```
header) and all cookies received in responses (i.e. the ``Set-Cookie``
header)::

    2011-04-06 14:35:10-0300 [scrapy.core.engine] INFO: Spider opened
    2011-04-06 14:35:10-0300 [scrapy.downloadermiddlewares.cookies] DEBUG: Sending cookies to: <GET http://www.diningcity.com/netherlands/index.html>
            Cookie: clientlanguage_nl=en_EN
    2011-04-06 14:35:14-0300 [scrapy.downloadermiddlewares.cookies] DEBUG: Received cookies from: <200 http://www.diningcity.com/netherlands/index.html>
            Set-Cookie: JSESSIONID=B~FA4DC0C496C8762AE4F1A620EAB34F38; Path=/
            Set-Cookie: ip_isocode=US
            Set-Cookie: clientlanguage_nl=en_EN; Expires=Thu, 07-Apr-2011 21:21:34 GMT; Path=/
    2011-04-06 14:49:50-0300 [scrapy.core.engine] DEBUG: Crawled (200) <GET http://www.diningcity.com/netherlands/index.html> (referer: None)
    [...]
```
</details>

<details><summary>Ranks 4-10</summary>

4. `scrapy/downloadermiddlewares/cookies.py` 1-117 (0.632)
5. `scrapy/extensions/httpcache.py` 324-405 (0.625)
6. `docs/topics/jobs.rst` 113-198 (0.609)
7. `docs/topics/feed-exports.rst` 96-196 (0.608)
8. `scrapy/spidermiddlewares/referer.py` 90-186 (0.597)
9. `docs/news/1.x.rst` 69-155 (0.594)
10. `docs/topics/request-response.rst` 266-385 (0.592)
</details>

### Q6. "what limits how many requests run in parallel against the same website"

- **Pre-registered answer:** `scrapy/core/downloader/__init__.py` 46-270
- **Strict grade:** ❌ wrong   **My judgement:** wrong (code at #4)
- **Correct file rank:** 4   **Latency:** 110 ms

| # | Result | Lines | Score |
|---|---|---|---|
| 1 | `docs/topics/optimize.rst` | 156-256 | 0.716 |
| 2 | `docs/topics/practices.rst` | 470-558 | 0.677 |
| 3 | `docs/topics/stats.rst` | 546-652 | 0.668 |

Top 3 are all prose docs (performance advice, practices, stats) that talk about concurrency in general. The code that enforces per-domain concurrency (downloader `Slot` / `_get_slot`) is #4, and the chunk found is `_get_slot` rather than the `Slot` class. The query has no identifier-like words, so prose wins.

<details><summary>Top hit, first lines</summary>

```
    the terms of service may state a rate.

-   Crawl when the website is idle, in its own timezone, so that the capacity
    you take is capacity nobody else wanted.

-   Raise concurrency gradually and watch the website respond, changing it on
    the running crawl from the :ref:`telnet console <telnet-concurrency>`.
    :stat:`downloader/response_status_count/{status_code}` counts for 429, 503
    or the ban page of the website, growing :stat:`retry/count`, or a
    :ref:`download latency <download-latency>` that climbs as you push harder,
    all mean you have gone past the limit.

```
</details>

<details><summary>Ranks 4-10</summary>

4. `scrapy/core/downloader/__init__.py` 237-331 (0.668)
5. `tests/benchmarks/test_crawl.py` 187-275 (0.668)
6. `docs/topics/autothrottle.rst` 1-97 (0.647)
7. `docs/topics/security.rst` 94-205 (0.639)
8. `tests/benchmarks/__init__.py` 1-70 (0.634)
9. `scrapy/core/engine.py` 429-537 (0.630)
10. `tests/test_utils_datatypes.py` 376-401 (0.629)
</details>

### Q7. "where are duplicate requests filtered out"

- **Pre-registered answer:** `scrapy/dupefilters.py` 46-200
- **Strict grade:** ✅ correct   **My judgement:** correct
- **Correct file rank:** 1   **Latency:** 107 ms

| # | Result | Lines | Score |
|---|---|---|---|
| 1 | `scrapy/dupefilters.py` | 1-117 | 0.749 |
| 2 | `tests/test_dupefilters.py` | 97-196 | 0.710 |
| 3 | `docs/topics/request-response.rst` | 372-491 | 0.674 |

`dupefilters.py` lines 1-117: `BaseDupeFilter.request_seen` and the `RFPDupeFilter` class. Correct (the RFP `request_seen` body is in the next window at line 140).

<details><summary>Top hit, first lines</summary>

```
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING
from warnings import warn

from scrapy.exceptions import ScrapyDeprecationWarning
from scrapy.utils.conf import _job_dir
from scrapy.utils.deprecate import method_is_overridden
from scrapy.utils.request import (
    RequestFingerprinter,
```
</details>

<details><summary>Ranks 4-10</summary>

4. `scrapy/spidermiddlewares/metacopy.py` 1-65 (0.654)
5. `sep/sep-018.rst` 74-182 (0.653)
6. `tests/spiders.py` 380-486 (0.643)
7. `sep/sep-014.rst` 361-483 (0.636)
8. `scrapy/core/scheduler.py` 332-424 (0.629)
9. `docs/topics/settings.rst` 1354-1478 (0.627)
10. `tests/test_downloadermiddleware_offsite.py` 154-237 (0.624)
</details>

### Q8. "how does the crawler check robots.txt before fetching a page"

- **Pre-registered answer:** `scrapy/downloadermiddlewares/robotstxt.py` 36-200, `scrapy/robotstxt.py` 1-200
- **Strict grade:** ✅ correct   **My judgement:** correct, but not the best answer
- **Correct file rank:** 1   **Latency:** 68 ms

| # | Result | Lines | Score |
|---|---|---|---|
| 1 | `scrapy/robotstxt.py` | 93-144 | 0.753 |
| 2 | `tests/test_downloadermiddleware_robotstxt.py` | 145-223 | 0.739 |
| 3 | `scrapy/downloadermiddlewares/robotstxt.py` | 1-95 | 0.727 |

`scrapy/robotstxt.py` lines 93-144: the parser wrappers' `allowed()` / `can_fetch()`. Pre-registered as correct, but the 'before fetching a page' part is the middleware's `process_request`, which is #3.

<details><summary>Top hit, first lines</summary>

```
        user_agent = to_unicode(user_agent)
        url = to_unicode(url)
        return self.rp.can_fetch(user_agent, url)

    def crawl_delay(self, user_agent: str | bytes) -> float | None:
        delay = self.rp.crawl_delay(to_unicode(user_agent))
        return None if delay is None else float(delay)


class RerpRobotParser(RobotParser):
    def __init__(self, robotstxt_body: bytes, spider: Spider | None):
        from robotexclusionrulesparser import RobotExclusionRulesParser  # noqa: PLC0415
```
</details>

<details><summary>Ranks 4-10</summary>

4. `tests/test_robotstxt_interface.py` 131-204 (0.721)
5. `docs/topics/downloader-middleware.rst` 1166-1279 (0.718)
6. `tests/test_downloadermiddleware_offsite.py` 230-312 (0.701)
7. `tests/CrawlerProcess/reactorless.py` 1-3 (0.688)
8. `docs/topics/practices.rst` 470-558 (0.686)
9. `docs/faq.rst` 88-202 (0.683)
10. `tests/AsyncCrawlerProcess/reactorless_telnetconsole_enabled.py` 1-21 (0.683)
</details>

### Q9. "where are gzip-compressed responses decompressed"

- **Pre-registered answer:** `scrapy/downloadermiddlewares/httpcompression.py` 41-200, `scrapy/utils/_compression.py` 1-200, `scrapy/utils/gz.py` 1-100
- **Strict grade:** ❌ wrong   **My judgement:** wrong (right answer at #2)
- **Correct file rank:** 2   **Latency:** 80 ms

| # | Result | Lines | Score |
|---|---|---|---|
| 1 | `tests/test_downloadermiddleware_httpcompression.py` | 385-472 | 0.726 |
| 2 | `scrapy/utils/_compression.py` | 97-138 | 0.708 |
| 3 | `tests/benchmarks/test_httpcompression.py` | 108-176 | 0.704 |

Top hit is a *test* of the compression middleware. #2 is the real decompression loop (`utils/_compression.py`) and #4 the middleware. Tests share all the vocabulary of the code they test.

<details><summary>Top hit, first lines</summary>

```
        response = self._getresponse("gzip")
        response.headers["Content-Type"] = "application/octet-stream"
        assert response.request
        request = response.request

        newresponse = self.mw.process_response(request, response)
        assert newresponse is not response
        assert newresponse.body.startswith(b"<!DOCTYPE")
        assert "Content-Encoding" not in newresponse.headers
        self.assertStatsEqual("httpcompression/response_count", 1)
        self.assertStatsEqual("httpcompression/response_bytes", 74837)

```
</details>

<details><summary>Ranks 4-10</summary>

4. `scrapy/downloadermiddlewares/httpcompression.py` 1-103 (0.691)
5. `scrapy/utils/gz.py` 1-16 (0.675)
6. `tests/test_downloadermiddleware.py` 1-112 (0.635)
7. `docs/topics/shell.rst` 254-344 (0.626)
8. `tests/test_utils_compression.py` 1-63 (0.623)
9. `docs/topics/request-response.rst` 1239-1350 (0.618)
10. `scrapy/pipelines/files.py` 265-347 (0.600)
</details>

### Q10. "how do I stop the crawler from going too many links deep"

- **Pre-registered answer:** `scrapy/spidermiddlewares/depth.py` 30-120
- **Strict grade:** ❌ wrong   **My judgement:** wrong (complete miss)
- **Correct file rank:** not in top 10   **Latency:** 76 ms

| # | Result | Lines | Score |
|---|---|---|---|
| 1 | `docs/topics/practices.rst` | 470-558 | 0.721 |
| 2 | `docs/topics/optimize.rst` | 334-389 | 0.721 |
| 3 | `tests/test_spidermiddleware_urllength.py` | 1-63 | 0.709 |

Nothing relevant in the top 10: operational docs about crawl processes. `spidermiddlewares/depth.py` ranks **126th**; the `DEPTH_LIMIT` docs are not in the top 10 either. Rephrased as 'how do I limit the crawl depth', `depth.py` comes 3rd, so this is a vocabulary gap: 'too many links deep' never meets the word 'depth'.

<details><summary>Top hit, first lines</summary>

```
:ref:`AutoThrottle <topics-autothrottle>` also throttles each crawler
separately. When crawling simultaneously, divide those values by the number of
crawlers to keep the combined load on your hardware and on target websites
unchanged.

Because of this, running the same spider several times in the same process
multiplies those limits instead of increasing crawling capacity. To crawl
faster, raise :setting:`CONCURRENT_REQUESTS` on a single crawler.

.. seealso:: :ref:`run-from-script`.

.. skip: end
```
</details>

<details><summary>Ranks 4-10</summary>

4. `tests/CrawlerProcess/reactorless.py` 1-3 (0.694)
5. `docs/topics/settings.rst` 2420-2530 (0.685)
6. `scrapy/core/downloader/handlers/http11.py` 97-186 (0.685)
7. `scrapy/crawler.py` 740-835 (0.681)
8. `tests/CrawlerRunner/reactorless.py` 1-3 (0.678)
9. `docs/topics/benchmarking.rst` 50-86 (0.675)
10. `tests/test_request_attribute_binding.py` 1-114 (0.675)
</details>
