#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn
#
# tools/run_tool_tests.sh -- every tool test (tools/test_*.py, tools/tests/test_*.py) in one command, on every core.
#
#   bash tools/run_tool_tests.sh [file-or-glob ...]
#
# Parallel by file: one interpreter per test file, MOJOLEARN_TEST_WORKERS of them at a time (default: the cores this
# machine has, nproc / hw.ncpu). A file with test functions or TestCase classes (tools/python_test_inventory.py) runs
# under pytest (the repo's pixi `test` environment declares it); a module-style check (a test_*.py with no test
# function) runs as a script. tools/test_duration_report.py is a report tool, not a test, and is left out.
# Prints pass / fail / error counts and every failing test name, writes one log per file under OUT, and exits 1 when
# anything failed (2 when no Python with pytest is found). Wired into nothing: run it by hand.
#
#   MOJOLEARN_TEST_PY       the interpreter (default: .pixi/envs/test/bin/python of this checkout or of the main
#                           checkout this worktree belongs to, else python3)
#   MOJOLEARN_TEST_WORKERS  parallel files (default: the core count)
#   MOJOLEARN_TOOL_TESTS_OUT  log dir (default ~/mojolearn-evidence/tool-tests/<UTC time>; never /private/tmp)
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT" || exit 2

has_pytest() { [ -x "$1" ] && "$1" -c 'import pytest' > /dev/null 2>&1; }
PY=${MOJOLEARN_TEST_PY:-}
if [ -z "$PY" ]; then
    main=$(cd "$(git rev-parse --git-common-dir 2> /dev/null || echo .git)/.." 2> /dev/null && pwd)
    for c in "$ROOT/.pixi/envs/test/bin/python" "${main:-$ROOT}/.pixi/envs/test/bin/python" "$(command -v python3)"; do
        if has_pytest "$c"; then PY=$c; break; fi
    done
fi
if [ -z "$PY" ] || ! has_pytest "$PY"; then
    echo "run_tool_tests: no Python with pytest (pixi install -e test, or set MOJOLEARN_TEST_PY)" >&2
    exit 2
fi
CORES=$(nproc 2> /dev/null || sysctl -n hw.ncpu 2> /dev/null || echo 1)
WORKERS=${MOJOLEARN_TEST_WORKERS:-$CORES}
OUT=${MOJOLEARN_TOOL_TESTS_OUT:-$HOME/mojolearn-evidence/tool-tests/$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$OUT/logs" || exit 2

if [ $# -gt 0 ]; then FILES=("$@"); else FILES=(tools/test_*.py tools/tests/test_*.py); fi
# kind<TAB>path per file: pytest (test functions / TestCase classes) or script (a module-style check)
"$PY" - "${FILES[@]}" > "$OUT/files.tsv" << 'EOF'
import os, sys
sys.path.insert(0, "tools")
from python_test_inventory import inventory
want = [os.path.normpath(p) for p in sys.argv[1:] if os.path.basename(p) != "test_duration_report.py"]
kinds = {}
for d in sorted({os.path.dirname(p) or "." for p in want}):
    s = inventory(d)
    for k, key in (("pytest", "pytest"), ("script", "gates")):
        for p in s[key]:
            kinds[os.path.normpath(str(p))] = k
for p in want:
    print("%s\t%s" % (kinds.get(p, "pytest"), p))
EOF
NFILES=$(wc -l < "$OUT/files.tsv" | tr -d ' ')
echo "run_tool_tests: $NFILES files, $WORKERS workers of $CORES cores, $PY, logs $OUT"

export PY OUT
one() {   # kind path: one file, its log, and one "rc kind path seconds" line in rc.txt
    local kind=$1 path=$2 log rc t0
    log="$OUT/logs/$(echo "$path" | tr '/' '.').log"
    t0=$(date +%s)
    if [ "$kind" = pytest ]; then
        "$PY" -m pytest -q -p no:cacheprovider -p no:xdist -rfE "$path" > "$log" 2>&1 < /dev/null; rc=$?
    else
        "$PY" "$path" > "$log" 2>&1 < /dev/null; rc=$?
    fi
    printf '%s\t%s\t%s\t%s\n' "$rc" "$kind" "$path" "$(( $(date +%s) - t0 ))" >> "$OUT/rc.txt"
}
export -f one
: > "$OUT/rc.txt"
start=$(date +%s)
tr '\t' '\n' < "$OUT/files.tsv" | xargs -n 2 -P "$WORKERS" bash -c 'one "$0" "$1"'
secs=$(( $(date +%s) - start ))

"$PY" - "$OUT" "$secs" << 'EOF'
import os, re, sys
out, secs = sys.argv[1], sys.argv[2]
counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
bad_files, names, times = [], [], []
for line in open(os.path.join(out, "rc.txt")):
    rc, kind, path, took = line.rstrip("\n").split("\t")
    times.append((float(took), path))
    log = os.path.join(out, "logs", path.replace("/", ".") + ".log")
    text = open(log, errors="replace").read() if os.path.exists(log) else ""
    if kind == "pytest":
        tail = (text.strip().splitlines() or [""])[-1]
        for n, w in re.findall(r"(\d+) (passed|failed|errors?|skipped)", tail):
            counts["errors" if w.startswith("error") else w] += int(n)
        hit = re.findall(r"^(?:FAILED|ERROR) (\S+)", text, re.M)
        names += hit
        if rc not in ("0", "5") and not hit:          # 5: nothing collected; any other rc without a name
            names.append("%s (rc %s, see %s)" % (path, rc, log))
        if rc not in ("0", "5"):
            bad_files.append(path)
    else:
        counts["passed" if rc == "0" else "failed"] += 1
        if rc != "0":
            bad_files.append(path)
            names.append("%s (script rc %s, see %s)" % (path, rc, log))
print("TOOL-TESTS passed=%(passed)d failed=%(failed)d errors=%(errors)d skipped=%(skipped)d" % counts
      + " files_failing=%d seconds=%s logs=%s" % (len(bad_files), secs, out))
for n in sorted(set(names)):
    print("  FAILED " + n)
print("slowest files (seconds):")
for took, path in sorted(times, reverse=True)[:10]:
    print("  %6.0f  %s" % (took, path))
with open(os.path.join(out, "SUMMARY.txt"), "w") as fh:
    fh.write("passed=%(passed)d failed=%(failed)d errors=%(errors)d skipped=%(skipped)d\n" % counts)
    fh.writelines("FAILED %s\n" % n for n in sorted(set(names)))
sys.exit(1 if bad_files else 0)
EOF
