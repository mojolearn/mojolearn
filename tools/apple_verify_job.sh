#!/bin/bash
# Apple (Metal) proof of everything merged 2026-09-30/10-01, on the published mojolearn 0.8.32 wheel (no build):
# digests to compare with the NVIDIA and AMD evidence on main. Its own venv; never the board's.
set -uo pipefail
cd "$(dirname "$0")/.."
O=$HOME/apple-verify-0832; mkdir -p $O; rm -f $O/rc.txt
export MOJOLEARN_NUMERIC_MODE=identical PYTHONUNBUFFERED=1; unset PYTHONPATH
rc() { echo "$(date -u +%T) $1 rc=$2" >> $O/rc.txt; }
sysctl -n machdep.cpu.brand_string > $O/cpu.txt
BASE=$HOME/mojolearn/.pixi/envs/default/bin/python3.13
[ -x $O/venv/bin/python ] || $BASE -m venv $O/venv
PY=$O/venv/bin/python
$PY -m pip -q install mojolearn==0.8.32 numpy==2.5.2 scipy==1.18.0 pytest > $O/pip.log 2>&1; rc pip $?
$PY -m pip install -q torch==2.13.0 >> $O/pip.log 2>&1
(cd /tmp && $PY -c "import mojolearn;print(mojolearn.__version__)") > $O/version.txt 2>&1
cd /tmp
R=$OLDPWD
# Mamba-3 output and all gradients, both shapes (NVIDIA = AMD = 26abf7d3 / 6dcc4669 md5 of the JSON)
timeout 1800 $PY $R/tools/strides_digest.py 2 512 384 > $O/digest-board.json 2>$O/digest-board.err; rc digest-board $?
timeout 3600 $PY $R/tools/strides_digest.py 8 512 768 > $O/digest-default.json 2>$O/digest-default.err; rc digest-default $?
timeout 1800 $PY $R/tools/matmul_digest.py > $O/matmul.json 2>$O/matmul.err; rc matmul $?
timeout 1800 $PY $R/tools/mlp_step_check.py > $O/mlp-check-256.log 2>&1; rc mlp-check-256 $?
timeout 1800 $PY $R/tools/mlp_step_check.py --rows 32 > $O/mlp-check-32.log 2>&1; rc mlp-check-32 $?
for k in adamw adam sgd; do timeout 1800 $PY $R/tools/optimizer_resident_check.py --kind $k > $O/opt-check-$k.log 2>&1; rc opt-check-$k $?; done
mkdir -p $O/classical
for c in "lu 1024" "sgd-reg 20000" "sgd-clf 20000" "lars 200000" "ivf 40000"; do set -- $c
  timeout 3600 $PY $R/tools/classical_pass_ab.py case $1 $2 --out $O/classical/$1-new-$2.json > $O/classical/$1-new-$2.log 2>&1; rc classical-$1 $?; done
[ -f $HOME/taxi-board-10000.npy ] && for r in 256 1000 10000; do timeout 1800 $PY $R/tools/kernel_pca_trial_check.py --data $HOME/taxi-board-10000.npy --rows $r --out $O/kpca-rows-$r.json > $O/kpca-rows-$r.log 2>&1; rc kpca-$r $?; done
echo done > $O/done
