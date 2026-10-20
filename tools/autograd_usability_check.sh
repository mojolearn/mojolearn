#!/usr/bin/env bash
# Focused feature qualification, not a performance measurement.
# Complete logs, source bytes and native artifacts stay in the supplied run
# directory. On rented machines use /root/results-* so box_archive retains it.
set -euo pipefail
action=${1:?build or test}
target=${2:?cpu, metal, cuda or hip}
output=${3:?absolute evidence directory}
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
mkdir -p "$output"
git rev-parse HEAD > "$output/source-head.txt"
git diff --binary HEAD > "$output/source-changes.patch"
case "$action" in
build)
    arch=${4:?accelerator architecture}
    mkdir -p "$output/bindings" "$output/tmp"
    export TMPDIR="$output/tmp" MOJOLEARN_COMPILE_JOBS=1
    # Patch-synced build trees keep new source files untracked.
    git ls-files --cached --others --exclude-standard '*.mojo' 'python/mojolearn/*.py' 'python/mojolearn/tests/test_*' \
        'tools/autograd_usability_check.sh' > "$output/source-files.txt"
    tar -czf "$output/source.tar.gz" -T "$output/source-files.txt"
    set +e
    pixi run mojo build -j 1 --emit shared-lib --target-cpu x86-64-v3 \
        --target-accelerator "$arch" -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . -I bindings \
        bindings/entry/entry__mojolearn_x_sequence.mojo \
        -o "$output/bindings/_mojolearn_x_sequence.so" > "$output/build.log" 2>&1
    rc=$?
    set -e
    echo "$rc" > "$output/build.rc"
    echo "BUILD $target rc=$rc"
    exit "$rc"
    ;;
test)
    test_python=${MOJOLEARN_TEST_PYTHON:?set the test Python executable}
    mkdir -p "$output/tmp"
    export TMPDIR="$output/tmp"
    export PYTHONPATH="$root/python" MOJOLEARN_NUMERIC_MODE=identical
    case "$target" in
        cpu) export MOJOLEARN_VENDOR=cpu ;;
        cuda|hip) export MOJOLEARN_VENDOR="$target" ;;
        metal) unset MOJOLEARN_VENDOR ;;
        *) echo "unknown target $target" >&2; exit 2 ;;
    esac
    "$test_python" - "$target" > "$output/binding.json" <<'PY'
import json
import sys
from mojolearn import _backend
b = _backend.binding("_mojolearn_x_sequence")
info = {"vendor": b.x_sequence_vendor(), "numeric_mode": b.x_sequence_numeric_mode(),
        "binding": getattr(b, "__file__", None), "binding_type": type(b).__name__}
print(json.dumps(info, indent=2))
assert info["vendor"] == sys.argv[1], info
assert info["numeric_mode"] == "identical", info
PY
    set +e
    "$test_python" -m pytest -q -s -p no:cacheprovider \
        python/mojolearn/tests/test_autograd_usability.py \
        python/mojolearn/tests/test_autograd_model.py \
        python/mojolearn/tests/test_ops_higher.py \
        python/mojolearn/tests/test_ops_optimizer.py \
        python/mojolearn/tests/test_ops_program.py \
        python/mojolearn/tests/test_ops_rankn.py \
        python/mojolearn/tests/test_ops_session.py \
        python/mojolearn/tests/test_autograd_public.py \
        python/mojolearn/tests/test_autograd_save_resume.py \
        --junitxml="$output/tests.xml" > "$output/tests.log" 2>&1
    rc=$?
    set -e
    echo "$rc" > "$output/tests.rc"
    "$test_python" - "$output" "$rc" <<'PY'
import collections
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
out = Path(sys.argv[1])
report = {"exit_code": int(sys.argv[2]), "counts": {}, "nonpassing": [], "digests": {}}
if (out / "tests.xml").exists():
    root = ET.parse(out / "tests.xml").getroot()
    for suite in root.iter("testsuite"):
        for k in ("tests", "failures", "errors", "skipped"):
            report["counts"][k] = report["counts"].get(k, 0) + int(suite.get(k, 0))
    for case in root.iter("testcase"):
        for child in case:
            if child.tag in ("failure", "error", "skipped"):
                report["nonpassing"].append({"test": case.get("name"), "status": child.tag,
                                              "message": child.get("message", "")})
digests = collections.defaultdict(set)
for match in re.finditer(r"([A-Z_0-9]+_DIGEST) ([a-zA-Z_0-9/.-]+) ([0-9a-f]+)", (out / "tests.log").read_text()):
    digests[match[1] + " " + match[2]].add(match[3])
report["digests"] = {k: sorted(v) for k, v in sorted(digests.items())}
(out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
print("TEST", report["exit_code"], report["counts"], "digests", len(digests))
PY
    exit "$rc"
    ;;
*) echo "unknown action $action" >&2; exit 2 ;;
esac
