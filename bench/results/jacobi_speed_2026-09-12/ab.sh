#!/bin/sh
# usage: ab.sh <dataset> <outdir> <lanes> <arms> <rounds>
# Lane jacobi-speed (DEVIATION 2680). Same shape as the linear-cluster-istella
# lane's ab.sh: MOJOLEARN_CTD_BASE_PY is the BEFORE tree (origin/main's build,
# /root/lane/python_base), /root/mojolearn/python is the AFTER, and both run in
# ONE interleaved race so the two arms see the same box heat.
ds=$1; out=$2; lanes=$3; arms=$4; rounds=$5
cd /root/mojolearn || exit 9
export PATH="$HOME/.pixi/bin:$PATH" MOJOLEARN_NUMERIC_MODE=identical
export GBM_BENCH_DATA=/root/datasets/gbm-bench
export MOJOLEARN_CTD_BASE_PY=/root/lane/python_base
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS
mkdir -p "$out"
for lane in $(echo $lanes | tr ',' ' '); do
  t0=$(date +%s)
  timeout -k 30 1800 python3 tools/classical_two_datasets.py race \
     --lane $lane --dataset $ds --data /root/ctd-data \
     --out "$out" --work /root/ctd-work --root /root/mojolearn \
     --rounds $rounds --arms $arms \
     --ours-python python3 --theirs-python python3 > "$out/race-$lane-$ds.log" 2>&1
  echo "race-$lane-$ds rc=$? s=$(( $(date +%s) - t0 ))" >> "$out/status.txt"
  pkill -9 -f 'classical_two_datasets.py worker' > /dev/null 2>&1
done
python3 tools/classical_two_datasets.py summary --out "$out" > "$out/summary.log" 2>&1
echo done >> "$out/status.txt"
