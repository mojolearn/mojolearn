#!/usr/bin/env bash
# Run NVIDIA's gbm-bench with the mojolearn arms added.
#
# usage:
#   bench/external/run_gbm_bench.sh <dataset> <ntrees> [algorithms]
# examples:
#   pixi run -e gbmbench bash bench/external/run_gbm_bench.sh year 100 gbdt
#   pixi run -e gbmbench bash bench/external/run_gbm_bench.sh covtype 100 forest
#
# [algorithms] shorthands:
#   gbdt    mojolearn-gbdt-gpu,cat-cpu           (the symmetric-trees pairing)
#   forest  mojolearn-et-gpu,skl-et-cpu,lgbm-et-cpu,lgbm-rf-cpu
#   (anything else is passed through as a comma-separated list)
#
# All arms of a shorthand run interleaved inside ONE process, which is what
# makes the comparison survive this box's thermal drift. Repeat the whole
# invocation for spread. Publication rules: bench/external/README.md.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="${GBM_BENCH_HOME:-$REPO_ROOT/bench/external/.gbm-bench}"
DATA="${GBM_BENCH_DATA:-$REPO_ROOT/bench/external/.gbm-datasets}"

DATASET="${1:?dataset required (year, covtype, higgs, fraud, epsilon, airline, bosch)}"
NTREES="${2:?ntrees required}"
case "${3:-}" in
  gbdt)   ALGOS="mojolearn-gbdt-gpu,cat-cpu" ;;
  forest) ALGOS="mojolearn-et-gpu,skl-et-cpu,lgbm-et-cpu,lgbm-rf-cpu" ;;
  "")     ALGOS="mojolearn-gbdt-gpu,cat-cpu" ;;
  *)      ALGOS="$3" ;;
esac

STAMP="$(date +%Y-%m-%d_%H%M%S)"
OUT="$REPO_ROOT/bench/results/gbm_bench_${DATASET}_${STAMP}.json"

if [ ! -d "$WORK" ]; then
  echo "==> cloning gbm-bench into $WORK"
  git clone --depth 1 https://github.com/NVIDIA/gbm-bench.git "$WORK"
fi

# Environment preconditions, checked BEFORE anything is downloaded (these
# datasets are gigabytes; failing on an import after fetching one is the
# expensive ordering).
PY_BIN="${GBM_BENCH_PYTHON:-python3}"
export PYTHONPATH="$REPO_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
if ! "$PY_BIN" -c "import pandas, sklearn, lightgbm, catboost, mojolearn" 2>/dev/null; then
  cat >&2 <<'PRECHECK'
run_gbm_bench.sh: the interpreter cannot import pandas, sklearn, lightgbm,
catboost and mojolearn together.

This script needs the gbmbench environment AND this repository's python/ on
the path. The invocation that works here:

  pixi run -e gbmbench bash bench/external/run_gbm_bench.sh <dataset> <ntrees> [algos]

PYTHONPATH is set for you and already points at the repository's python/.
PRECHECK
  exit 2
fi

echo "==> pinning the harness commit for the record"
GBM_SHA="$(git -C "$WORK" rev-parse HEAD)"

echo "==> patching (idempotent)"
"$PY_BIN" "$REPO_ROOT/bench/external/patch_gbm_bench.py" "$WORK"

mkdir -p "$DATA" "$REPO_ROOT/bench/results"

echo "==> recording the box"
"$REPO_ROOT/bench/external/record_environment.sh" > "${OUT%.json}.env.txt" 2>&1 || true
echo "gbm_bench_commit=$GBM_SHA" >> "${OUT%.json}.env.txt"

# Thread count: gbm-bench's -cpus reaches every arm identically. On a
# bare-metal box leave GBM_BENCH_CPUS unset and the upstream default (the
# machine's count) applies.
CPUS_ARG=()
if [ -n "${GBM_BENCH_CPUS:-}" ]; then
  echo "==> pinning every arm to $GBM_BENCH_CPUS threads"
  CPUS_ARG=(-cpus "$GBM_BENCH_CPUS")
fi

echo "==> running: dataset=$DATASET ntrees=$NTREES algorithms=$ALGOS"
cd "$WORK"
"$PY_BIN" runme.py \
  -root "$DATA" \
  -dataset "$DATASET" \
  -algorithm "$ALGOS" \
  -ntrees "$NTREES" \
  "${CPUS_ARG[@]}" \
  -output "$OUT" \
  -verbose

echo
echo "==> wrote $OUT"
echo "    and the box record beside it at ${OUT%.json}.env.txt"
