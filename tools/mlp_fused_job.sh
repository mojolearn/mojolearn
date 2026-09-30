#!/bin/bash
# PR #7 (fused small-MLP step, one wait fewer per matmul) on one NVIDIA or AMD box.
set -uo pipefail
cd "$(dirname "$0")/.."
V=${1:-nvidia}; O=/root/mlp-fused; mkdir -p $O
if [ $V = nvidia ]; then BK=cuda; AR=sm_89; else BK=hip; AR=gfx942; fi
export PATH=/root/.pixi/bin:/opt/rocm/bin:$PATH MOJOLEARN_TARGET_COLUMN=$V MOJOLEARN_GPU_ARCHS=$AR \
  MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_COMPILE_JOBS=2 MOJOLEARN_BENCH_INSTALLED=1 PYTHONUNBUFFERED=1
unset PYTHONPATH
st() { echo "{\"phase\":\"$1\",\"utc\":\"$(date -u +%FT%TZ)\"}" > $O/status.json; }
rc() { echo "$(date -u +%T) $1 rc=$2" >> $O/rc.txt; }
git rev-parse HEAD > $O/head.txt
st pixi; pixi install > $O/pixi.log 2>&1
base=$(pixi run python3 -c 'import sys;print(sys.executable)' | tail -1)
[ -x $O/venv/bin/python ] || $base -m venv $O/venv
PY=$O/venv/bin/python
$PY -m pip -q install mojolearn==0.8.31 numpy==2.5.2 scipy==1.18.0 pytest > $O/pip.log 2>&1
$PY -m pip -q install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu >> $O/pip.log 2>&1
SITE=$($PY -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')/mojolearn
T=$SITE/$BK/$AR/identical; N=$O/native; mkdir -p $N
build() {  # module tag
  st build-$2; rm -f python/mojolearn/identical/_mojolearn_$1.so python/mojolearn/_mojolearn_$1.so
  bash bindings/build_$1.sh > $O/build-$2.log 2>&1; rc build-$2 $?
  so=$(ls -t python/mojolearn/identical/_mojolearn_$1.so python/mojolearn/_mojolearn_$1.so 2>/dev/null | head -1)
  [ -n "$so" ] && cp $so $N/$2.so && sha256sum $N/$2.so >> $O/bindings.sha256
}
# main's linalg first (the matmul A/B), then the PR's
git stash list >/dev/null
MB=$(git merge-base HEAD origin/main 2>/dev/null || git rev-parse HEAD~1)
git show $MB:gemm/host_entry.mojo > /tmp/host_entry.main.mojo
cp gemm/host_entry.mojo /tmp/host_entry.pr.mojo
cp /tmp/host_entry.main.mojo gemm/host_entry.mojo; build linalg linalg-main
cp /tmp/host_entry.pr.mojo gemm/host_entry.mojo; build linalg linalg-pr
build training training-pr
cp -r python/mojolearn/*.py $SITE/; find $SITE -name __pycache__ -exec rm -rf {} +
cp $N/training-pr.so $T/_mojolearn_training.so
for arm in main pr; do cp $N/linalg-$arm.so $T/_mojolearn_linalg.so; st matmul-$arm
  timeout 900 $PY tools/matmul_digest.py > $O/matmul-$arm.json 2>$O/matmul-$arm.err; rc matmul-$arm $?; done
st mlp-check-256; timeout 1800 $PY tools/mlp_step_check.py --json $O/mlp-check-256.json > $O/mlp-check-256.log 2>&1; rc mlp-check-256 $?
st mlp-check-32; timeout 1800 $PY tools/mlp_step_check.py --rows 32 --json $O/mlp-check-32.json > $O/mlp-check-32.log 2>&1; rc mlp-check-32 $?
st pytest; (cd python && timeout 1800 $PY -m pytest -q mojolearn/tests/test_small_mlp_surface.py mojolearn/tests/test_small_mlp_numerical_edges.py) > $O/pytest.log 2>&1; rc pytest $?
export MOJOLEARN_REPO_COMMIT=$(cat $O/head.txt)
for f in 1 0; do st race-mlp-fused$f
  MOJOLEARN_MLP_FUSED=$f timeout 1800 $PY tools/bench_board_neural.py race --lane mlp-train-step --shape full --arms ours --rounds 5 \
    --out $O/race-mlp-fused$f --work $O/work --ours-python $PY > $O/race-mlp-fused$f.log 2>&1; rc race-mlp-fused$f $?; done
st done; echo done > $O/done
