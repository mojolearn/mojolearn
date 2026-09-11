#!/bin/sh
# usage: ab.sh <dataset> <outdir> <lanes> <arms> <rounds> [probe modes]   (PROBE_BIN env)
# Lane linear-cluster-istella. Same shape as the linear-cluster-speed lane's
# ab.sh: MOJOLEARN_CTD_BASE_PY is the BEFORE tree (origin/main's build), the
# working tree is the AFTER, and both run in one interleaved race.
ds=$1; out=$2; lanes=$3; arms=$4; rounds=$5; probe=${6:-}
PROBE_BIN=${PROBE_BIN:-/root/lane/stage_probe}
cd /root/mojolearn || exit 9
export PATH="$HOME/.pixi/bin:$PATH" MOJOLEARN_NUMERIC_MODE=identical GBM_BENCH_DATA=/root/datasets/gbm-bench MOJOLEARN_CTD_BASE_PY=/root/lane/python_base
# DBSCAN eps and min_samples, one value per dataset for every arm; the rule and
# the picker's output are in this directory's README.
export MOJOLEARN_CTD_DBSCAN_TAXI="${MOJOLEARN_CTD_DBSCAN_TAXI:-}"
export MOJOLEARN_CTD_DBSCAN_ISTELLA="${MOJOLEARN_CTD_DBSCAN_ISTELLA:-}"
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS
mkdir -p "$out"
for lane in $(echo $lanes | tr ',' ' '); do
  t0=$(date +%s)
  timeout -k 30 3000 python3 tools/classical_two_datasets.py race --lane $lane --dataset $ds --data /root/ctd-data \
     --out "$out" --work /root/ctd-work --root /root/mojolearn --rounds $rounds --arms $arms \
     --ours-python python3 --theirs-python python3 > "$out/race-$lane-$ds.log" 2>&1
  echo "race-$lane-$ds rc=$? s=$(( $(date +%s) - t0 ))" >> "$out/status.txt"
  pkill -9 -f 'classical_two_datasets.py worker' > /dev/null 2>&1
  python3 tools/classical_two_datasets.py summary --out "$out" > "$out/summary.log" 2>&1
done
if [ -n "$probe" ]; then
  shape=$(python3 -c "import json;r=json.load(open('/root/ctd-data/big-$ds.json'));s=r['arrays']['X']['shape'];print(s[0],s[1])")
  for m in $(echo $probe | tr ',' ' '); do
    $PROBE_BIN $m /root/lane/bins/$ds $shape 3 > "$out/probe-$m-$ds.log" 2>&1
    echo "probe-$m-$ds rc=$?" >> "$out/status.txt"
  done
fi
echo done >> "$out/status.txt"
