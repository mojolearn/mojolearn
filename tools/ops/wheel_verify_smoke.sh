#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# tools/ops/wheel_verify_smoke.sh WHEEL_DIR OUT
#
# DOES VERIFY STILL RUN, AND DO THE PYTHON BOUNDARIES STILL WORK, from a BUILT Linux CPU wheel. Runs after every
# integration build on GitHub's free CPU runners (.github/workflows/verify-smoke.yml). It is NOT the identity
# verification: it compares nothing across vendors, records nothing, writes no reference and never replaces the
# release's one verification (AGENTS.md "Verify once, before the release"). The verify command is protected
# (AGENTS.md): this is how a wave learns it broke verify before the release does.
#
# WHEEL_DIR holds the packed wheels of one measurement build (<work dir>/<commit12>/linux/final/ of
# `tools/release.py candidate`): the core wheel (mojolearn_core-*.whl) is installed, exactly as built, into a fresh
# venv with NumPy (the `verify` extra; the release columns install it beside the wheel the same way). Every step
# runs from an empty directory with `python -I`, so no source tree is on sys.path. Steps, each logged under OUT
# with its exit code on one line of OUT/SUMMARY.txt; a failed step never stops the later ones:
#   install    fresh venv, the core wheel + numpy
#   imports    (a) `import mojolearn` and every public module tools/public_api_surface.py lists in the installed
#              package, one process per module
#   selftest   (b) `python -m mojolearn verify --self-test` on the CPU column
#   lanes      (c) every CPU lane of verify's own harness (mojolearn._identity_break, which `verify --all` calls per
#              cell), base fixture, one fit, one process per lane. No `verify` flag runs every lane without
#              comparing against the shipped reference table (--all/--quick/--lanes compare with it, --coverage fits
#              nothing, --smoke covers only NOT APPLICABLE lanes, --cross-check refuses on a CPU-only install), so
#              the harness runs directly; --no-batch --no-rlpair stamp each JSON partial_column, which
#              _verify_reference.admit() refuses: nothing here can become a record.
#   test-deps  pytest (pinned like tools/cpu_wheel_checks.sh), scipy, scikit-learn, added AFTER (a)-(c)
#   frames     (e) the Python frames census (tools/ops/python_frames_census.py): the Python that runs inside every
#              public call, with verify's own fixtures, one process per public class and per CPU lane;
#              OUT/python-frames/census.tsv and the totals in SUMMARY.txt. Reports; gates nothing (always rc=0)
#   tests      (d) the pytest files of SOURCE_ROOT/python/mojolearn/tests against the installed runtime
#              (tools/run_python_tests.py INSTALLED_BOOTSTRAP), one process per file (one per test for a file
#              longer than an even share of the pool); GPU and optional-dependency tests skip with their reason
#              and a file of skips only is listed as `env` (tests/ENV.txt), never as passed
# Parallel plan: tools/ops/wheel_verify_smoke.py (one worker per core; units longest first by the measured seconds
# of tools/ops/wheel_verify_smoke_plan/<pool>.tsv or MEMORY_FROM; each unit starts when its measured peak RSS fits).
# Crashes: a unit or step that dies by a signal is rerun under `python -X faulthandler` and gdb (when on PATH);
# traces in OUT/crashes/<unit>.txt, the top frame per crash in SUMMARY.txt.
# Environment: PYTHON (default python3), SOURCE_ROOT (the checkout whose tests run; default this script's tree),
# SOURCE_COMMIT (the wheel's commit, for the harness's provenance), VENV (default a fresh mktemp dir),
# MEMORY_FROM (an earlier OUT: plan each pool from its results.tsv instead of the committed plan), STEPS (default all).
# Exit 0 only when every step passed.
set -u
[ $# -eq 2 ] || { echo "usage: $0 WHEEL_DIR OUT" >&2; exit 2; }
WHEEL_DIR=$(cd "$1" && pwd) || exit 2
mkdir -p "$2" && OUT=$(cd "$2" && pwd) || exit 2
HERE=$(cd "$(dirname "$0")" && pwd)
SOURCE_ROOT=${SOURCE_ROOT:-$(cd "$HERE/../.." && pwd)}
PY=${PYTHON:-python3}
VENV=${VENV:-$(mktemp -d)/venv}
STEPS=${STEPS:-install,imports,selftest,lanes,frames,test-deps,tests}
HELPER="$PY $HERE/wheel_verify_smoke.py"
: > "$OUT/SUMMARY.txt"

wheels=("$WHEEL_DIR"/mojolearn_core-*.whl)
[ -e "${wheels[0]}" ] || wheels=("$WHEEL_DIR"/mojolearn-*.whl)
{
  echo "# wheel verify smoke: does verify run and do the Python boundaries work (compares and records nothing)"
  echo "# started $(date -u +%FT%TZ) host=$(uname -srm) cores=$(nproc 2>/dev/null || sysctl -n hw.ncpu) python=$($PY --version 2>&1)"
  echo "# source_root=$SOURCE_ROOT source_commit=${SOURCE_COMMIT:-unset}"
  for w in "${wheels[@]}"; do [ -e "$w" ] && echo "# wheel $(basename "$w") sha256=$(sha256sum "$w" 2>/dev/null | cut -c1-64 || shasum -a 256 "$w" | cut -c1-64)"; done
} >> "$OUT/SUMMARY.txt"
if [ ! -e "${wheels[0]}" ] || [ "${#wheels[@]}" -ne 1 ]; then
  echo "install rc=1 expected exactly one mojolearn_core-*.whl (or one mojolearn-*.whl) in $WHEEL_DIR, found ${#wheels[@]}" >> "$OUT/SUMMARY.txt"
fi
WHEEL=${wheels[0]}
VPY="$VENV/bin/python"

want() { case ",$STEPS," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
mem() { [ -n "${MEMORY_FROM:-}" ] && [ -f "$MEMORY_FROM/$1/results.tsv" ] && echo "--memory-from $MEMORY_FROM/$1/results.tsv"; }

if want install && [ -e "$WHEEL" ]; then
  $HELPER step --out "$OUT" --name install -- bash -c \
    "'$PY' -m venv --clear '$VENV' && '$VPY' -m pip install --disable-pip-version-check --no-input -q --no-cache-dir '$WHEEL' numpy && '$VPY' -m pip list --disable-pip-version-check --format=freeze"
fi
want imports && $HELPER imports --python "$VPY" --out "$OUT" $(mem imports)
want selftest && $HELPER step --out "$OUT" --name selftest --timeout 900 -- "$VPY" -I -m mojolearn verify --self-test
want lanes && $HELPER lanes --python "$VPY" --out "$OUT" --commit "${SOURCE_COMMIT:-}" $(mem lanes)
want frames && $HELPER frames --python "$VPY" --out "$OUT" $(mem python-frames)
want test-deps && $HELPER step --out "$OUT" --name test-deps -- "$VPY" -m pip install --disable-pip-version-check \
  --no-input -q --no-cache-dir pytest==8.3.5 scipy scikit-learn
want tests && $HELPER tests --python "$VPY" --out "$OUT" --tests-root "$SOURCE_ROOT/python/mojolearn/tests" $(mem tests)

# the verdict: every step named in STEPS must have a line, and every line rc=0
failed=""
IFS=, read -ra named <<< "$STEPS"
for s in "${named[@]}"; do
  line=$(grep -m1 -E "^$s rc=" "$OUT/SUMMARY.txt")
  [ -n "$line" ] || { echo "$s rc=1 did not run (an earlier step left nothing to run it with)" >> "$OUT/SUMMARY.txt"; failed="$failed $s"; continue; }
  case "$line" in "$s rc=0 "*) ;; *) failed="$failed $s" ;; esac
done
if [ -z "$failed" ]; then
  echo "overall rc=0 every step passed" >> "$OUT/SUMMARY.txt"; exit 0
fi
echo "overall rc=1 failed:$failed" >> "$OUT/SUMMARY.txt"
exit 1
