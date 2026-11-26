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
#   selftest   (b) `python -m mojolearn verify --tamper-test` on the CPU column
#   lanes      (c) every CPU lane of verify's own harness (mojolearn._identity_break, which `verify --all` calls per
#              cell), base fixture, one fit, one process per lane. No `verify` flag runs every lane without
#              comparing against the shipped reference table (--all/--quick/--lanes compare with it, --coverage fits
#              nothing, --smoke covers only NOT APPLICABLE lanes, --cross-check refuses on a CPU-only install), so
#              the harness runs directly; --no-batch --no-rlpair stamp each JSON partial_column, which
#              _verify_reference.admit() refuses: nothing here can become a record.
#   test-deps  pytest (pinned like tools/cpu_wheel_checks.sh), scipy, scikit-learn, and the boundary/adapter test
#              dependencies SMOKE_EXTRAS (pinned: CPU torch with onnx and onnxscript, the `mojolearn-core[torch]`
#              extra; pandas), added AFTER (a)-(c) so those still run on the core wheel + numpy alone. The
#              extras are DOWNLOADED in the background from the start of `install` (extras-download.log, its own
#              SUMMARY line, not a verdict step), in parallel with the install, imports, selftest, lanes and
#              frames steps, so test-deps only installs local files. MAX (`mojolearn-core[max]`) is not added: its
#              cp312 manylinux_2_34 wheel would install here, but it pulls max-core and the Mojo compiler, the CPU
#              core wheel refuses the framework adapters by design and no test imports max
#              (verify-fixtures-env, 2026-11-16).
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
# MEMORY_FROM (an earlier OUT: plan each pool from its results.tsv instead of the committed plan), STEPS (default all),
# CENSUS_STANDS (non-empty: the frames step is skipped with that line; set by the dispatcher's census skip rule).
# GPU MODE (SMOKE_SIDE=gpu; Andrew 2026-11-25: the smoke runs on the GPU side too, cheap and fast, after every build;
# tools/ops/gpu_smoke_job.sh runs it on the L40S): WHEEL_DIR holds the core wheel and exactly one GPU plugin wheel
# (mojolearn_nvidia-*.whl or mojolearn_amd-*.whl); `install` installs both and fails unless mojolearn.vendor() is a
# GPU vendor; `lanes` runs every lane on BOTH sides in its process (a refused GPU side fails the lane as a refused CPU
# side does; a side a lane declares it does not have reads n/a, never a failure; lanes/SIDES.tsv); every pool's
# workers are the core allocation capped by the GPU's free memory per worker (SMOKE_GPU_MB_PER_WORKER, default
# wheel_verify_smoke.py GPU_MEM_PER_WORKER_MB). The header adds the GPU name, driver and memory and the parallel plan.
# Default STEPS leave out `frames` (it reports and gates nothing, and the CPU smoke of the same wheel takes it).
# Same SUMMARY.txt format.
# Exit 0 only when every step passed.
set -u
[ $# -eq 2 ] || { echo "usage: $0 WHEEL_DIR OUT" >&2; exit 2; }
WHEEL_DIR=$(cd "$1" && pwd) || exit 2
mkdir -p "$2" && OUT=$(cd "$2" && pwd) || exit 2
HERE=$(cd "$(dirname "$0")" && pwd)
SOURCE_ROOT=${SOURCE_ROOT:-$(cd "$HERE/../.." && pwd)}
PY=${PYTHON:-python3}
VENV=${VENV:-$(mktemp -d)/venv}
SIDE=${SMOKE_SIDE:-cpu}
case "$SIDE" in cpu|gpu) ;; *) echo "SMOKE_SIDE is cpu or gpu, not $SIDE" >&2; exit 2 ;; esac
if [ "$SIDE" = gpu ]; then
  STEPS=${STEPS:-install,imports,selftest,lanes,test-deps,tests}
  GPU_FLAG=--gpu
else
  STEPS=${STEPS:-install,imports,selftest,lanes,frames,test-deps,tests}
  GPU_FLAG=
fi
#: the pandas and torch boundary tests and the torch adapter tests (python/mojolearn/tests: 20 files importorskip
#: pandas, 14 torch, 2 onnx); torch is the CPU build of the pin packaging/adapters/pins.json names (2.13)
SMOKE_EXTRAS=${SMOKE_EXTRAS:-"torch==2.13.0+cpu onnx==1.18.0 onnxscript==0.5.6 pandas==2.2.3"}
TORCH_CPU_INDEX=https://download.pytorch.org/whl/cpu
EXTRAS_DIR="$(dirname "$VENV")/extras-wheels"
HELPER="$PY $HERE/wheel_verify_smoke.py"
: > "$OUT/SUMMARY.txt"

wheels=("$WHEEL_DIR"/mojolearn_core-*.whl)
[ -e "${wheels[0]}" ] || wheels=("$WHEEL_DIR"/mojolearn-*.whl)
plugins=()
if [ "$SIDE" = gpu ]; then
  for w in "$WHEEL_DIR"/mojolearn_nvidia-*.whl "$WHEEL_DIR"/mojolearn_amd-*.whl; do [ -e "$w" ] && plugins+=("$w"); done
fi
{
  echo "# wheel verify smoke: does verify run and do the Python boundaries work (compares and records nothing)"
  echo "# started $(date -u +%FT%TZ) host=$(uname -srm) cores=$(nproc 2>/dev/null || sysctl -n hw.ncpu) python=$($PY --version 2>&1)"
  echo "# source_root=$SOURCE_ROOT source_commit=${SOURCE_COMMIT:-unset} side=$SIDE"
  for w in "${wheels[@]}" ${plugins[@]+"${plugins[@]}"}; do [ -e "$w" ] && echo "# wheel $(basename "$w") sha256=$(sha256sum "$w" 2>/dev/null | cut -c1-64 || shasum -a 256 "$w" | cut -c1-64)"; done
  # shellcheck disable=SC2086
  $HELPER plan $GPU_FLAG
} >> "$OUT/SUMMARY.txt"
# The tests step runs SOURCE_ROOT's pytest files against the wheel, so the tree must be the wheel's commit
# (lane gs-fix: the L40S smoke of 3cd641b37e ran a tree holding five test files and changed tests of the fixed15
# lanes, which failed against that wheel as product reds). A git tree whose HEAD or tests differ is refused.
if [ -n "${SOURCE_COMMIT:-}" ] && git -C "$SOURCE_ROOT" rev-parse --git-dir >/dev/null 2>&1; then
  src_head=$(git -C "$SOURCE_ROOT" rev-parse HEAD 2>/dev/null)
  case "$src_head" in
    "$SOURCE_COMMIT"*) ;;
    *) echo "source rc=1 SOURCE_ROOT $SOURCE_ROOT is at $src_head, not the wheel's commit $SOURCE_COMMIT" >> "$OUT/SUMMARY.txt"; exit 1 ;;
  esac
  src_dirty=$(git -C "$SOURCE_ROOT" status --porcelain -- python/mojolearn/tests tools/identity_lanes 2>/dev/null | head -n 5)
  if [ -n "$src_dirty" ]; then
    echo "source rc=1 SOURCE_ROOT $SOURCE_ROOT has test files that differ from $SOURCE_COMMIT: $(echo "$src_dirty" | tr '\n' ' ')" >> "$OUT/SUMMARY.txt"
    exit 1
  fi
fi
if [ ! -e "${wheels[0]}" ] || [ "${#wheels[@]}" -ne 1 ]; then
  echo "install rc=1 expected exactly one mojolearn_core-*.whl (or one mojolearn-*.whl) in $WHEEL_DIR, found ${#wheels[@]}" >> "$OUT/SUMMARY.txt"
elif [ "$SIDE" = gpu ] && [ "${#plugins[@]}" -ne 1 ]; then
  echo "install rc=1 GPU mode expects exactly one GPU plugin wheel (mojolearn_nvidia-*.whl or mojolearn_amd-*.whl) in $WHEEL_DIR, found ${#plugins[@]}" >> "$OUT/SUMMARY.txt"
fi
WHEEL=${wheels[0]}
INSTALL_SET="'$WHEEL'"
GPU_CHECK=true
if [ "$SIDE" = gpu ]; then
  [ "${#plugins[@]}" -eq 1 ] && INSTALL_SET="'$WHEEL' '${plugins[0]}'" || WHEEL=/nonexistent
  # the GPU plugin must have loaded: a GPU smoke on a CPU-only answer would run every lane's CPU side only
  GPU_CHECK="'$VENV/bin/python' -I -c \"import mojolearn as ml; v = ml.vendor(); print('vendor', v); raise SystemExit(0 if v in ('cuda', 'hip', 'metal') else 3)\""
fi
VPY="$VENV/bin/python"

want() { case ",$STEPS," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
mem() { [ -n "${MEMORY_FROM:-}" ] && [ -f "$MEMORY_FROM/$1/results.tsv" ] && echo "--memory-from $MEMORY_FROM/$1/results.tsv"; }

extras_pid=""
if want test-deps && [ -n "$SMOKE_EXTRAS" ]; then
  # network-bound and one process: it overlaps the core steps, which keep every core busy
  # shellcheck disable=SC2086
  $HELPER step --out "$OUT" --name extras-download -- "$PY" -m pip download --disable-pip-version-check --no-input \
    -q --dest "$EXTRAS_DIR" --extra-index-url "$TORCH_CPU_INDEX" $SMOKE_EXTRAS > /dev/null &
  extras_pid=$!
fi
if want install && [ -e "$WHEEL" ]; then
  $HELPER step --out "$OUT" --name install -- bash -c \
    "'$PY' -m venv --clear '$VENV' && '$VPY' -m pip install --disable-pip-version-check --no-input -q --no-cache-dir $INSTALL_SET numpy && '$VPY' -m pip list --disable-pip-version-check --format=freeze && $GPU_CHECK"
fi
# shellcheck disable=SC2046,SC2086
want imports && $HELPER imports --python "$VPY" --out "$OUT" $GPU_FLAG $(mem imports)
want selftest && $HELPER step --out "$OUT" --name selftest --timeout 900 -- "$VPY" -I -m mojolearn verify --tamper-test
# shellcheck disable=SC2046,SC2086
want lanes && $HELPER lanes --python "$VPY" --out "$OUT" --commit "${SOURCE_COMMIT:-}" $GPU_FLAG $(mem lanes)
# CENSUS_STANDS (tools/ops/verify_smoke_dispatch.py's skip rule): nothing the census reads changed since the last
# census of this branch, so the step is skipped with one line naming the wheel whose census stands
if want frames && [ -n "${CENSUS_STANDS:-}" ]; then
  printf 'frames rc=0 skipped: %s\n' "$(printf '%s' "$CENSUS_STANDS" | tr -d '\n' | cut -c1-400)" >> "$OUT/SUMMARY.txt"
elif want frames; then
  $HELPER frames --python "$VPY" --out "$OUT" $(mem python-frames)
fi
[ -n "$extras_pid" ] && wait "$extras_pid"
# shellcheck disable=SC2086
want test-deps && $HELPER step --out "$OUT" --name test-deps -- "$VPY" -m pip install --disable-pip-version-check \
  --no-input -q --no-cache-dir --find-links "$EXTRAS_DIR" --extra-index-url "$TORCH_CPU_INDEX" \
  pytest==8.3.5 scipy scikit-learn $SMOKE_EXTRAS
# shellcheck disable=SC2046,SC2086
want tests && $HELPER tests --python "$VPY" --out "$OUT" --tests-root "$SOURCE_ROOT/python/mojolearn/tests" $GPU_FLAG $(mem tests)

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
