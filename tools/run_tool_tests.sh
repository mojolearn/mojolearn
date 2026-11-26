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
# A file in SPLIT (below) is split by test id into one shard per worker, the shards first in the queue, so the
# longest files no longer set the wall time alone. The log and rc row of a shard are named <path>#<k>. The ten
# slowest units are printed after the counts.
# Prints pass / fail / error counts and every failing test name, writes one log per file under OUT, and exits 1 when
# anything failed (2 when no Python with pytest is found). Wired into nothing: run it by hand.
#
#   MOJOLEARN_TEST_PY       the interpreter (default: .pixi/envs/test/bin/python of this checkout or of the main
#                           checkout this worktree belongs to, else python3)
#   MOJOLEARN_TEST_WORKERS  parallel files (default: the core count)
#   MOJOLEARN_TOOL_TESTS_OUT  log dir (default ~/mojolearn-evidence/tool-tests/<UTC time>; never /private/tmp)
#
# Every test starts from a scrubbed environment (lane fx-box-tooltests, 2026-11-20): once the three settings above are
# read, every exported MOJOLEARN_*, CPU_CHECK_*, CPU_CHECKS_*, CPC_* and NVC_* variable is unset, because a run
# started inside a box job inherited that job's MOJOLEARN_RESULTS_DIR, MOJOLEARN_PERSIST_TAG, MOJOLEARN_BOX_PERSIST,
# MOJOLEARN_PERSIST_RUNNER (tools/box_persist_run.sh) and CPU_CHECK_SHA / CPU_CHECK_SUITE (tools/cpu_central.sh
# check), and the tools under test took them for their own (test_cpu_central: the job script read the enclosing
# job's results dir and sha instead of printing usage). Each unit also starts with SIGINT and SIGQUIT at their
# defaults, as from a terminal or a GitHub step: a run started in the background by a non-interactive shell (`cmd &`,
# tools/ops/cpu_box_checks_job.sh) inherits both ignored, and neither bash nor Python can then take Ctrl-C.
# OUT/ENV.txt lists the scrubbed names and every skipped test with its reason; the summary prints them as the
# run's environment notes.
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

SCRUBBED=()
while IFS= read -r v; do
    case "$v" in
        MOJOLEARN_*|CPU_CHECK_*|CPU_CHECKS_*|CPC_*|NVC_*) SCRUBBED+=("$v"); unset "$v" ;;
    esac
done < <(compgen -e)
printf 'scrubbed\t%s\n' "${SCRUBBED[*]:-none}" > "$OUT/ENV.txt"
# A test that runs a board or verify path without naming its own summary directory would write a run summary into
# docs/measurements/latest/auto/ of THIS tree (tools/retain_run_summary.py falls back to the repository; 2026-11-26: a
# full run rewrote the four bench-board-*.json there). Every unit writes it under this run's output instead; the tests
# that check the summary set their own directory.
export MOJOLEARN_SUMMARY_REPO="$OUT/run-summary-repo"
mkdir -p "$MOJOLEARN_SUMMARY_REPO/.git" "$MOJOLEARN_SUMMARY_REPO/tools" || exit 2  # the shape retain_run_summary.py accepts

if [ $# -gt 0 ]; then FILES=("$@"); else FILES=(tools/test_*.py tools/tests/test_*.py); fi
# kind<TAB>path per file: pytest (test functions / TestCase classes) or script (a module-style check)
WORKERS=$WORKERS OUT=$OUT "$PY" - "${FILES[@]}" > "$OUT/files.tsv" << 'EOF'
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
# Files split by test id across the workers. Each re-derives the whole lane map (about 5 min of pure-Python Mojo
# parsing) in every test that edits the tree or patches an edge, by design, so as one process they ran 51 and 29
# minutes and set the wall time of the whole run (2026-11-17 run: 3264 s at 4 workers, lane_select 3073 s of it).
SPLIT = {"tools/test_lane_select.py", "tools/test_lane_select_rebuild.py"}
workers = int(os.environ["WORKERS"])
os.makedirs(os.path.join(os.environ["OUT"], "shards"), exist_ok=True)
rows = []
for p in want:
    kind = kinds.get(p, "pytest")
    if p in SPLIT and kind == "pytest" and workers > 1:
        import subprocess
        got = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--collect-only", p],
                             capture_output=True, text=True)
        ids = [l for l in got.stdout.splitlines() if l.startswith(p + "::")]
        if got.returncode == 0 and ids:
            for k in range(min(workers, len(ids))):
                name = "%s#%d" % (p, k)
                with open(os.path.join(os.environ["OUT"], "shards", name.replace("/", ".") + ".ids"), "w") as fh:
                    fh.write("\n".join(ids[k::workers]) + "\n")
                rows.insert(0, ("shard", name))
            continue
    rows.append((kind, p))
for kind, p in rows:
    print("%s\t%s" % (kind, p))
EOF
NFILES=$(wc -l < "$OUT/files.tsv" | tr -d ' ')
echo "run_tool_tests: $NFILES units, $WORKERS workers of $CORES cores, $PY, logs $OUT"
echo "run_tool_tests: environment scrubbed of ${#SCRUBBED[@]} job variables: ${SCRUBBED[*]:-none}"

export PY OUT
dfl() {   # run "$@" with SIGINT and SIGQUIT at their defaults (an ignored disposition survives exec; bash cannot undo it)
    "$PY" -c 'import os, signal, sys
for s in (signal.SIGINT, signal.SIGQUIT):
    signal.signal(s, signal.SIG_DFL)
os.execvp(sys.argv[1], sys.argv[1:])' "$@"
}
one() {   # kind path: one file, its log, and one "rc kind path seconds" line in rc.txt
    local kind=$1 path=$2 log rc t0
    log="$OUT/logs/$(echo "$path" | tr '/' '.').log"
    t0=$(date +%s)
    if [ "$kind" = pytest ]; then
        dfl "$PY" -m pytest -q -p no:cacheprovider -p no:xdist -rfEs "$path" > "$log" 2>&1 < /dev/null; rc=$?
    elif [ "$kind" = shard ]; then   # this shard's test ids, one per line (ids may hold spaces)
        dfl "$PY" -c 'import sys, pytest; sys.exit(pytest.main(["-q", "-p", "no:cacheprovider", "-p", "no:xdist", "-rfEs",
            *open(sys.argv[1]).read().splitlines()]))' "$OUT/shards/$(echo "$path" | tr '/' '.').ids" \
            > "$log" 2>&1 < /dev/null; rc=$?
    else
        dfl "$PY" "$path" > "$log" 2>&1 < /dev/null; rc=$?
    fi
    printf '%s\t%s\t%s\t%s\n' "$rc" "$kind" "$path" "$(( $(date +%s) - t0 ))" >> "$OUT/rc.txt"
}
export -f dfl one
: > "$OUT/rc.txt"
start=$(date +%s)
tr '\t' '\n' < "$OUT/files.tsv" | xargs -n 2 -P "$WORKERS" bash -c 'one "$0" "$1"'
secs=$(( $(date +%s) - start ))

"$PY" - "$OUT" "$secs" << 'EOF'
import os, re, sys
out, secs = sys.argv[1], sys.argv[2]
counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
bad_files, names, times, skips = [], [], [], []
for line in open(os.path.join(out, "rc.txt")):
    rc, kind, path, took = line.rstrip("\n").split("\t")
    times.append((float(took), path))
    log = os.path.join(out, "logs", path.replace("/", ".") + ".log")
    text = open(log, errors="replace").read() if os.path.exists(log) else ""
    if kind in ("pytest", "shard"):
        tail = (text.strip().splitlines() or [""])[-1]
        for n, w in re.findall(r"(\d+) (passed|failed|errors?|skipped)", tail):
            counts["errors" if w.startswith("error") else w] += int(n)
        hit = re.findall(r"^(?:FAILED|ERROR) (\S+)", text, re.M)
        names += hit
        skips += re.findall(r"^SKIPPED \[\d+\] (.+)$", text, re.M)
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
# the environment notes: what was scrubbed and every skip with its reason (a skip is never counted as a pass)
with open(os.path.join(out, "ENV.txt"), "a") as fh:
    fh.writelines("skipped\t%s\n" % s for s in skips)
env_lines = open(os.path.join(out, "ENV.txt")).read().splitlines()
print("environment notes (%s):" % os.path.join(out, "ENV.txt"))
for line in env_lines:
    print("  " + line.replace("\t", ": ", 1))
print("slowest units (seconds):")
for took, path in sorted(times, reverse=True)[:10]:
    print("  %6.0f  %s" % (took, path))
with open(os.path.join(out, "SUMMARY.txt"), "w") as fh:
    fh.write("passed=%(passed)d failed=%(failed)d errors=%(errors)d skipped=%(skipped)d\n" % counts)
    fh.writelines("FAILED %s\n" % n for n in sorted(set(names)))
sys.exit(1 if bad_files else 0)
EOF
