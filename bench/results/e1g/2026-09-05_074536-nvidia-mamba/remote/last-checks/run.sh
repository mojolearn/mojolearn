#!/usr/bin/env bash
set -uo pipefail
cd /root/mojolearn
export PATH=/root/.pixi/bin:$PATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_CPU_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=/root/mojolearn/python
OUT=/root/gemm_leg_out/last-checks
mkdir -p "$OUT"
PY=/root/mojolearn/.nvidia-comparator-venv/bin/python
cores=$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0))[:2])))')
taskset -pc "$cores" $$
run() { local name=$1 cap=$2 status; shift 2; timeout -k 5 "$cap" "$@" > "$OUT/$name.log" 2>&1; status=$?; printf '%s\t%s\n' "$name" "$status" | tee -a "$OUT/results.tsv"; return "$status"; }
cp /root/nvidia-last-checks.sh "$OUT/run.sh"
cp tools/mamba_external_compare.py "$OUT/adapter.py"
for family in mamba1 mamba2; do
 run "$family" 60 "$PY" tools/mamba_external_compare.py --family "$family" --native "/root/gemm_leg_out/mamba-cert/$family/native" --out "$OUT/$family" --samples 7
done
run mamba3 90 env PYTHONPATH=/root/mamba3-deps:/root/mamba3-source:/root/mojolearn/python "$PY" tools/mamba_external_compare.py --family mamba3 --native /root/gemm_leg_out/mamba-cert/mamba3/native --out "$OUT/mamba3" --samples 7
