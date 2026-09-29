# bench/lowbit_quality/jobs/env.sh -- sourced by every job of lane/lowbit-quality.
# Runs ON THE POD (nvc1), in the lane's tree, never on the laptop.
#   the model and the text   staged from the R2 dataset store by
#                            `tools/dataset_store.sh stage` (pins checked on the box)
#   the interpreter          a venv over the pod's own torch, with tokenizers,
#                            safetensors and transformers from pip
#   the host binding         bindings/build_linalg_host.sh, built on the pod into
#                            $WORK/host: what `mojolearn.lowbit.pack` runs through
set -eu
TREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
WORK=${LOWBIT_QUALITY_WORK:-/root/lowbit-quality-work}
PY=${LOWBIT_QUALITY_PY:-/root/lowbit-quality-venv/bin/python}
MODEL=${LOWBIT_QUALITY_MODEL:-/root/models/SmolLM2-360M}
CORPUS=${LOWBIT_QUALITY_CORPUS:-/root/mojolearn-wt/lowbit-quality/training/corpus/enwik8/input.txt}
COMMIT=$(cat "$TREE/.lowbit_quality_commit" 2>/dev/null || git -C "$TREE" rev-parse HEAD 2>/dev/null || echo unknown)
export MOJOLEARN_HOST_DIR="$WORK/host"
export PYTHONPATH="$TREE/python"
export PYTHONUNBUFFERED=1
[ -f "$MODEL/model.safetensors" ] || { echo "REFUSED: $MODEL is not staged (tools/dataset_store.sh stage ... models/SmolLM2-360M)"; exit 3; }
[ -f "$CORPUS" ] || { echo "REFUSED: $CORPUS is not staged (tools/dataset_store.sh stage ... corpus/enwik8/input.txt)"; exit 3; }
cd "$TREE"
echo "job=${NVQ_JOB_ID:-none} lane=${NVQ_LANE:-none} gpus=${CUDA_VISIBLE_DEVICES:-unset} commit=$COMMIT started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
