#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# The builders refuse, before any compile, a tree whose kernel directory or RUN
# module imports Python (tools/python_free_dirs.sh, an import grep).
#
# Builds a scratch copy of the gate (the script, its doors list and
# bindings/merged_build.sh with the two files it sources), plants
# `from python import PythonObject` in a scratch kernel file, and checks that
# tools/python_free_dirs.sh, bindings/merged_build.sh and
# tools/development_build.py all refuse. Nothing is compiled: `mojo` and `pixi`
# on PATH and MOJOLEARN_MERGED_PYTHON are stubs that leave a marker, and the
# test fails if the compiler stub ran. Runs in a second or two.
#
# Usage: bash tools/tests/test_python_free_gate.sh
set -u -o pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
scratch=$(mktemp -d "${TMPDIR:-/tmp}/python-free-gate.XXXXXX")
trap 'rm -rf "$scratch"' EXIT
fails=0
pass() { echo "ok   $1"; }
fail() { echo "FAIL $1" >&2; fails=$((fails + 1)); }

stubs=$scratch/stubs
mkdir -p "$stubs"
for tool in mojo pixi; do
    printf '#!/bin/sh\necho %s >> "%s/compiler.marker"\nexit 99\n' "$tool" "$scratch" > "$stubs/$tool"
done
printf '#!/bin/sh\necho "$*" >> "%s/plan.marker"\nexit 7\n' "$scratch" > "$stubs/plan-python"
chmod +x "$stubs"/*
export PATH="$stubs:$PATH"

tree=$scratch/tree
mkdir -p "$tree/tools" "$tree/bindings" "$tree/core"
cp "$repo/tools/python_free_dirs.sh" "$repo/tools/python_free_doors.txt" "$tree/tools/"
cp "$repo/bindings/merged_build.sh" "$repo/bindings/build_defines.sh" "$repo/bindings/compile_jobs.sh" "$tree/bindings/"
# A clean kernel, a listed entry table, a clean RUN module, and a binding that
# imports Python (allowed: bindings/ holds UNPACK/PACK).
printf 'def kernel(x: Int) -> Int:\n    return x + 1\n' > "$tree/core/clean_kernel.mojo"
printf 'def entry(x: Int) -> Int:\n    return x\n' > "$tree/core/demo_entry_table.mojo"
printf 'def run(x: Int) -> Int:\n    return x\n' > "$tree/bindings/demo_run.mojo"
printf 'from python import PythonObject\n\ndef unpack(o: PythonObject):\n    pass\n' > "$tree/bindings/demo_unpack.mojo"
git -C "$tree" init -q
git -C "$tree" add -A
git -C "$tree" -c user.name=test -c user.email=test@localhost commit -q -m clean

plant='from python import PythonObject'

# expect_refusal <label> <expected text>  (reads the gate in $tree)
merged() { (cd "$tree" && env -u MOJOLEARN_EXTRA_DEFINES MOJOLEARN_MERGED_PYTHON="$stubs/plan-python" \
    MOJOLEARN_NUMERIC_MODE=identical sh bindings/merged_build.sh host) > "$scratch/merged.log" 2>&1; }
devbuild() {
    python3 - "$repo/tools" "$tree" "$scratch" > "$scratch/dev.log" 2>&1 <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import development_build as d
d.REPO = Path(sys.argv[2])
scratch = Path(sys.argv[3])
sys.exit(d.main(["build", "--layout", "development", "--cache", str(scratch / "cache"),
                 "--evidence", str(scratch / "evidence")]))
PY
}

# 1. The clean tree passes, and merged_build.sh goes on to its plan (the stub).
bash "$tree/tools/python_free_dirs.sh" > "$scratch/clean.log" 2>&1 \
    && pass "clean tree passes" || { fail "clean tree passes"; tail -n 5 "$scratch/clean.log" >&2; }
merged
[ -s "$scratch/plan.marker" ] && pass "clean tree: merged_build.sh passes the gate and reaches its plan" \
    || { fail "clean tree: merged_build.sh reaches its plan"; tail -n 5 "$scratch/merged.log" >&2; }
rm -f "$scratch/plan.marker"

# 2. A scratch kernel file that imports Python (untracked: the compiler reads it).
printf '%s\n\ndef k(o: PythonObject):\n    pass\n' "$plant" > "$tree/core/scratch_kernel.mojo"
bash "$tree/tools/python_free_dirs.sh" > "$scratch/dirty.log" 2>&1
rc=$?
[ $rc = 1 ] && grep -q 'core/scratch_kernel.mojo' "$scratch/dirty.log" \
    && pass "python_free_dirs.sh refuses the planted kernel (rc 1)" \
    || { fail "python_free_dirs.sh refuses the planted kernel (rc $rc)"; tail -n 5 "$scratch/dirty.log" >&2; }
merged
rc=$?
[ $rc != 0 ] && grep -q 'REFUSED before any compile' "$scratch/merged.log" \
    && grep -q 'core/scratch_kernel.mojo' "$scratch/merged.log" && [ ! -e "$scratch/plan.marker" ] \
    && pass "merged_build.sh refuses before its plan (rc $rc)" \
    || { fail "merged_build.sh refuses before its plan (rc $rc)"; tail -n 8 "$scratch/merged.log" >&2; }
devbuild
rc=$?
[ $rc != 0 ] && grep -q 'REFUSED before any compile' "$scratch/dev.log" \
    && grep -q 'core/scratch_kernel.mojo' "$scratch/dev.log" \
    && pass "development_build.py refuses before any compile (rc $rc)" \
    || { fail "development_build.py refuses before any compile (rc $rc)"; tail -n 8 "$scratch/dev.log" >&2; }

# 3. The committed plant, judged as the pre-push hook does (a commit argument).
git -C "$tree" add -A
git -C "$tree" -c user.name=test -c user.email=test@localhost commit -q -m plant
(cd "$tree" && bash tools/python_free_dirs.sh HEAD) > "$scratch/commit.log" 2>&1
rc=$?
[ $rc = 1 ] && grep -q 'core/scratch_kernel.mojo' "$scratch/commit.log" \
    && pass "a commit with the plant is refused (rc 1)" || fail "a commit with the plant is refused (rc $rc)"
git -C "$tree" rm -q core/scratch_kernel.mojo
git -C "$tree" -c user.name=test -c user.email=test@localhost commit -q -m unplant

# 4. A RUN module in bindings/ that imports Python (the *_run.mojo pattern).
printf '%s\n' "$plant" >> "$tree/bindings/demo_run.mojo"
bash "$tree/tools/python_free_dirs.sh" > "$scratch/run.log" 2>&1
rc=$?
[ $rc = 1 ] && grep -q 'bindings/demo_run.mojo' "$scratch/run.log" \
    && pass "a *_run.mojo in bindings/ that imports Python is refused" || fail "*_run.mojo refused (rc $rc)"
git -C "$tree" checkout -q -- bindings/demo_run.mojo

# 5. A listed door that imports Python, and a listed door that does not exist.
printf 'bindings/demo_door.mojo\nbindings/gone.mojo\n' >> "$tree/tools/python_free_doors.txt"
printf '%s\n' "$plant" > "$tree/bindings/demo_door.mojo"
bash "$tree/tools/python_free_dirs.sh" > "$scratch/door.log" 2>&1
rc=$?
[ $rc = 1 ] && grep -q 'bindings/demo_door.mojo' "$scratch/door.log" && grep -q 'bindings/gone.mojo' "$scratch/door.log" \
    && pass "a listed door that imports Python, and a missing listed door, are refused" || fail "listed doors (rc $rc)"
rm -f "$tree/bindings/demo_door.mojo"
git -C "$tree" checkout -q -- tools/python_free_doors.txt

# 6. Outside git (a `git archive` source tree on a box): the plain-grep fallback.
rm -rf "$tree/.git"
bash "$tree/tools/python_free_dirs.sh" > "$scratch/nogit-clean.log" 2>&1 \
    && pass "no git: clean tree passes" || { fail "no git: clean tree passes"; tail -n 5 "$scratch/nogit-clean.log" >&2; }
printf '%s\n' "$plant" > "$tree/core/scratch_kernel.mojo"
merged
rc=$?
[ $rc != 0 ] && grep -q 'core/scratch_kernel.mojo' "$scratch/merged.log" && [ ! -e "$scratch/plan.marker" ] \
    && pass "no git: merged_build.sh refuses the planted kernel" || fail "no git: merged_build.sh refuses (rc $rc)"
printf '%s\n' "$plant" >> "$tree/bindings/demo_run.mojo"
rm -f "$tree/core/scratch_kernel.mojo"
bash "$tree/tools/python_free_dirs.sh" > "$scratch/nogit-run.log" 2>&1
rc=$?
[ $rc = 1 ] && grep -q 'bindings/demo_run.mojo' "$scratch/nogit-run.log" \
    && pass "no git: a *_run.mojo that imports Python is refused" || fail "no git: *_run.mojo refused (rc $rc)"

if [ -e "$scratch/compiler.marker" ]; then
    fail "a compiler stub ran: $(tr '\n' ' ' < "$scratch/compiler.marker")"
else
    pass "no compiler ran"
fi
[ $fails = 0 ] && echo "test_python_free_gate: all passed" || { echo "test_python_free_gate: $fails failed" >&2; exit 1; }
