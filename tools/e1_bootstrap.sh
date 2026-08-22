#!/usr/bin/env bash
# E1 remote bootstrap: run ON the GPU box (MI300X droplet, RunPod pod, or
# the Mac reference side). Executes E1_RUNBOOK.md Phases 0-3 and leaves
# every artifact under bench/results/e1/<stamp>-<host>/.
#
# usage:  bash tools/e1_bootstrap.sh
# The repo checkout this script lives in IS the source; rsync it to the
# box first (rsync carries the exact commit; no GitHub access needed):
#   rsync -a --exclude .pixi --exclude bench/external/.gbm-bench \
#       ~/CascadeProjects/mojolearn/ <box>:~/mojolearn/
#
# THE MODE FLIP: this script flips GLOBAL_NUMERIC_MODE to IDENTICAL for
# the session and reverts it at the end (trap). It is never committed.
# On AMD the flip is mandatory -- the FAST build is a compile error at
# the 64-wide wavefront asserts, by design.
#
# Mac reference side, after this script: the tracked .so files were
# rebuilt IDENTICAL for the driver; restore the shipped FAST binaries:
#   git checkout -- python/mojolearn/_mojolearn_trees.so \
#       python/mojolearn/_mojolearn_rf.so python/mojolearn/_mojolearn_estimators.so
#   bash bindings/build_gbdt.sh   # gbdt .so is untracked; rebuild FAST
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
STAMP="$(date +%Y-%m-%d_%H%M%S)-$(hostname -s)"
OUT="$REPO/bench/results/e1/$STAMP"
mkdir -p "$OUT"
LOG="$OUT/bootstrap.log"
exec > >(tee "$LOG") 2>&1

step() { echo; echo "=== $* === $(date +%T)"; }

step "provenance"
git rev-parse HEAD | tee "$OUT/commit.txt"
uname -a | tee "$OUT/uname.txt"
bash bench/external/record_environment.sh > "$OUT/environment.txt" 2>&1 || true

step "pixi"
if ! command -v pixi >/dev/null; then
  curl -fsSL https://pixi.sh/install.sh | bash
  export PATH="$HOME/.pixi/bin:$PATH"
fi
pixi install

step "mode flip -> IDENTICAL (session-local, reverted on exit)"
NUMERICS=mojo_only/numerics.mojo
grep -q "^comptime GLOBAL_NUMERIC_MODE = NUMERIC_FAST$" "$NUMERICS" || {
  echo "numerics.mojo not in the expected FAST state; refusing"; exit 2; }
sed -i.e1bak 's/^comptime GLOBAL_NUMERIC_MODE = NUMERIC_FAST$/comptime GLOBAL_NUMERIC_MODE = NUMERIC_IDENTICAL/' "$NUMERICS"
revert() { mv -f "$NUMERICS.e1bak" "$NUMERICS" 2>/dev/null || true; }
trap revert EXIT

step "phase 0: smoke (hardware matrix, column detection)"
pixi run check-hardware-matrix || echo "PHASE0-FINDING: hardware matrix (see log)"

step "phase 1: vendor characterization (row 10 precondition)"
pixi run check-ieee-arith || echo "PHASE1-FINDING: ieee-arith (see log)"

step "phase 2: gates under IDENTICAL"
for gate in check-depthwise check-lossguide-policy check-random-strength; do
  echo "--- $gate"
  pixi run "$gate" || echo "PHASE2-FINDING: $gate FAILED"
done
echo "--- extratrees suite"
bash extratrees/tools/check.sh || echo "PHASE2-FINDING: extratrees suite (see log)"

step "phase 3: build IDENTICAL .so + traced fits"
# ALL FIVE bindings: the python package's __init__ imports cluster ->
# _mojolearn.so and friends, so an rsync'd foreign-platform .so anywhere
# in the package breaks every import ("invalid ELF header", run 2's
# finding). Remove the foreign binaries LOUDLY first, and skip each
# build's own smoke gate -- during a from-scratch five-binding build
# every gate imports siblings that do not exist yet (run 3's finding).
# The traced driver below is the real gate: it launches kernels through
# every lib. Run inside the gbmbench env so python has numpy.
rm -f python/mojolearn/_mojolearn*.so
export MOJOLEARN_SKIP_BUILD_GATE=1
for b in build.sh build_estimators.sh build_gbdt.sh build_rf.sh build_trees.sh; do
  echo "--- bindings/$b"
  pixi run -e gbmbench bash "bindings/$b" \
    || echo "PHASE3-FINDING: bindings/$b failed (see log)"
done
unset MOJOLEARN_SKIP_BUILD_GATE
PYTHONPATH="$REPO/python" pixi run -e gbmbench python3 tools/e1_traced_fit.py "$OUT" \
  || PYTHONPATH="$REPO/python" python3 tools/e1_traced_fit.py "$OUT" \
  || echo "PHASE3-FINDING: traced driver failed (see log)"

step "done"
echo "artifacts in $OUT"
echo "next: fetch this directory beside the other machine's and run"
echo "  python3 tools/identity_trace_diff.py <mac>/<fit>.card <gpu>/<fit>.card"
