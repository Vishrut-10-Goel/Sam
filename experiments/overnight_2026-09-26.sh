#!/usr/bin/env bash
# Overnight 2026-09-26, run after experiments/encode_apps_queries.py, one step at a time (no memory contention).
# 1. demo indexes with the current defaults (timed)  2. full test suite  3. AST chunking mechanical checks.
# Run from the repository root: bash experiments/overnight_2026-09-26.sh > experiments/overnight_2026-09-26.log 2>&1
set -u
PY=venv/Scripts/python.exe
DEMO=/d/prism-demo
stamp() { date +%H:%M:%S; }

echo "== $(stamp) 1. demo indexes (defaults: windows chunking, headers on)"
for repo in scrapy requests express; do
  start=$(date +%s)
  $PY cli.py index "$DEMO/$repo" --rebuild 2>&1 | grep -vE "HF_TOKEN|embedded [0-9,]+/"
  echo "INDEX_TIME $repo $(( $(date +%s) - start )) s"
done

echo "== $(stamp) 2. test suite"
for t in test_loaders_chunking test_index test_retrieval test_eval test_cli test_file_hash test_ast_chunks; do
  echo -n "$t: "; $PY -m tests.$t 2>&1 | grep -E "FAIL|passed|Error" | tr '\n' ' '; echo
done

echo "== $(stamp) 3a. AST chunking statistics vs windows (no embedding)"
$PY experiments/ast_chunk_stats.py "$DEMO/scrapy" "$DEMO/requests" "$DEMO/express" 2>&1 | grep -v HF_TOKEN

echo "== $(stamp) 3b. end-to-end: cli.py index --chunking ast (requests), then a query"
start=$(date +%s)
$PY cli.py index "$DEMO/requests" --chunking ast --out indexes/requests-ast --rebuild 2>&1 | grep -vE "HF_TOKEN|embedded [0-9,]+/"
echo "INDEX_TIME requests-ast $(( $(date +%s) - start )) s"
$PY cli.py query "how are redirects handled" --index indexes/requests-ast --top-k 3 2>&1 | grep -vE "HF_TOKEN" | grep -E "^ *[0-9]+\. "
echo "== $(stamp) done"
