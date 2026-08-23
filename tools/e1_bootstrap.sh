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
# THE MODE: IDENTICAL, by BUILD DEFINE (-D MOJOLEARN_NUMERIC_IDENTICAL=1 via
# tools/with_identical_mode.sh; bindings into python/mojolearn/identical/;
# drivers select it with MOJOLEARN_NUMERIC_MODE=identical). No file in the
# tree is edited, nothing is reverted, the shipped FAST binaries are never
# touched. On AMD a FAST build is a compile error at the 64-wide wavefront
# asserts, by design, so IDENTICAL is the only mode that runs there.
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

step "mode: IDENTICAL by build define (no source flip since 2026-08-23)"
# Every gate below runs through tools/with_identical_mode.sh, which injects
# -D MOJOLEARN_NUMERIC_IDENTICAL=1 into the mojo command (or exports
# MOJOLEARN_MOJO_DEFINES for scripts that call mojo themselves); the
# bindings build into python/mojolearn/identical/ and the Python drivers
# select that set with MOJOLEARN_NUMERIC_MODE=identical. Nothing in the
# tree is edited, so there is nothing to revert and no shipped binary to
# restore afterwards. On AMD a FAST build is still a compile error at the
# 64-wide wavefront asserts, by design.
IDENT="$REPO/tools/with_identical_mode.sh"
export MOJOLEARN_NUMERIC_MODE=identical

step "phase 0: smoke (hardware matrix, column detection)"
"$IDENT" pixi run check-hardware-matrix || echo "PHASE0-FINDING: hardware matrix (see log)"

step "phase 1: vendor characterization (row 10 precondition)"
"$IDENT" pixi run check-ieee-arith || echo "PHASE1-FINDING: ieee-arith (see log)"
# row 12's certificate line: the printed device hash must be the SAME
# NUMBER on every vendor column (Apple measured 8705486125800438413)
"$IDENT" pixi run check-portable-translog || echo "PHASE1-FINDING: portable-translog (see log)"
"$IDENT" pixi run check-portable-sqrtcos || echo "PHASE1-FINDING: portable-sqrtcos (see log)"

step "phase 2: gates under IDENTICAL"
for gate in check-depthwise check-lossguide-policy check-random-strength; do
  echo "--- $gate"
  "$IDENT" pixi run "$gate" || echo "PHASE2-FINDING: $gate FAILED"
done
echo "--- extratrees suite"
"$IDENT" bash extratrees/tools/check.sh || echo "PHASE2-FINDING: extratrees suite (see log)"

step "phase 3: build IDENTICAL .so + traced fits"
# ALL FIVE bindings: the python package's __init__ imports cluster ->
# _mojolearn.so and friends, so an rsync'd foreign-platform .so anywhere
# in the package breaks every import ("invalid ELF header", run 2's
# finding). Remove the foreign binaries LOUDLY first, and skip each
# build's own smoke gate -- during a from-scratch five-binding build
# every gate imports siblings that do not exist yet (run 3's finding).
# The traced driver below is the real gate: it launches kernels through
# every lib. Run inside the gbmbench env so python has numpy.
# the identical set lands in python/mojolearn/identical/ (MOJOLEARN_NUMERIC_MODE
# is exported above and the build scripts read it); a foreign-platform .so
# left there from an rsync would break every import, so clear it first
rm -f python/mojolearn/identical/_mojolearn*.so
for b in build.sh build_estimators.sh build_gbdt.sh build_rf.sh build_trees.sh; do
  echo "--- bindings/$b (identical)"
  pixi run -e gbmbench bash "bindings/$b" \
    || echo "PHASE3-FINDING: bindings/$b failed (see log)"
done
# the FAST set must exist too: the package imports every binding and the
# selector only swaps the five it loads; on a fresh box build FAST as well
# (it cannot build on AMD -- the wavefront asserts -- so a missing FAST set
# is tolerated: the selector never touches it under identical)
if [ ! -f python/mojolearn/_mojolearn.so ]; then
  for b in build.sh build_estimators.sh build_gbdt.sh build_rf.sh build_trees.sh; do
    MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_SKIP_BUILD_GATE=1 pixi run -e gbmbench bash "bindings/$b" >/dev/null 2>&1 \
      || echo "note: FAST bindings/$b did not build here (expected on AMD)"
  done
fi
PYTHONPATH="$REPO/python" pixi run -e gbmbench python3 tools/e1_traced_fit.py "$OUT" \
  || PYTHONPATH="$REPO/python" python3 tools/e1_traced_fit.py "$OUT" \
  || echo "PHASE3-FINDING: traced driver failed (see log)"

step "phase 4: E2 sub-feature matrix (every loss/bootstrap/score/estimator/searcher/bins/cat/NaN/criterion)"
# one subprocess per cell, so a device fault in one configuration leaves
# the other cards intact; e2_cells.json is rewritten after every cell.
# `tools/e2_matrix_diff.py <mac>:APPLE <gpu>:VENDOR` is the verdict table.
PYTHONPATH="$REPO/python" pixi run -e gbmbench python3 tools/e2_matrix_fit.py "$OUT" \
  || PYTHONPATH="$REPO/python" python3 tools/e2_matrix_fit.py "$OUT" \
  || echo "PHASE4-FINDING: E2 matrix driver failed (see log)"
# the Python-unreachable training paths (depthwise/lossguide growth,
# MultiClassOneVsAll, ...) get their cards from Mojo probes, one fit per
# file, when that script is present
if [ -x tools/e2_mojo_cards.sh ]; then
  bash tools/e2_mojo_cards.sh "$OUT" || echo "PHASE4-FINDING: e2_mojo_cards failed (see log)"
fi

step "phase 5: the unsupervised cards (k-means, k-NN, DBSCAN) -- IDENTICAL"
# tools/e1_unsupervised.sh is the unsupervised lane's leg (rows 19-26); it
# re-enters itself through the injector and writes bench/results/e1u/<stamp>;
# a copy of that directory lands beside this run's artifacts
sh tools/e1_unsupervised.sh "$OUT/e1u" || echo "PHASE5-FINDING: unsupervised leg (see log)"

step "phase 6: the linear-algebra identity gates (GEMM, column stats, Jacobi/PCA, OLS) -- both modes"
# rows 27-32; verdicts, not numbers (both modes, ~22 device runs). The
# decomposition bindings did not BUILD on AMD before 4ecf43c; the gate
# running at all here is part of the result.
pixi run check-linalg-identity || echo "PHASE6-FINDING: linalg identity (see log)"
pixi run check-unsupervised-identity || echo "PHASE6-FINDING: unsupervised identity (see log)"

step "done"
echo "artifacts in $OUT"
echo "next: fetch this directory beside the other machine's and run"
echo "  python3 tools/e2_matrix_diff.py <mac>:APPLE <gpu>:VENDOR --write"
echo "  python3 tools/identity_trace_diff.py <mac>/<cell>.card <gpu>/<cell>.card"
