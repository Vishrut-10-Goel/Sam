#!/usr/bin/env bash
# Versioned index of N commits vs N separate builds, on psf/requests (reports/versions_check.md).
# Run from the repository root:  bash reports/versions/measure_builds.sh > reports/versions/build_log.txt 2>&1
# One build at a time; the power state (Windows battery status: 1 = on battery, 2 = on mains) is logged each time.
set -u
PY=venv/Scripts/python.exe
REPO=/d/prism-demo/requests
COMMITS="v2.32.5 v2.33.0 v2.33.1 v2.34.0 v2.34.2 611c6162"
power() { powershell.exe -NoProfile -Command "(Get-CimInstance Win32_Battery).BatteryStatus" 2>/dev/null | tr -d '\r'; }
stamp() { date +%H:%M:%S; }

echo "== $(stamp) versioned: one index of all six commits (power $(power))"
start=$(date +%s)
$PY cli.py index-versions "$REPO" --commits $COMMITS --out indexes/requests-versions 2>&1 | grep -vE "HF_TOKEN|embedded [0-9,]+/"
echo "WALL versioned $(( $(date +%s) - start )) s"

for c in $COMMITS; do
  wt=$(mktemp -d)/wt
  git -C "$REPO" worktree add --quiet --detach "$wt" "$c"
  echo "== $(stamp) separate build of $c (power $(power))"
  start=$(date +%s)
  $PY cli.py index "$wt" --out "indexes/sep-$c" --rebuild 2>&1 | grep -vE "HF_TOKEN|embedded [0-9,]+/"
  echo "WALL separate $c $(( $(date +%s) - start )) s"
  git -C "$REPO" worktree remove --force "$wt"
done
git -C "$REPO" worktree prune
echo "== $(stamp) done (power $(power))"
