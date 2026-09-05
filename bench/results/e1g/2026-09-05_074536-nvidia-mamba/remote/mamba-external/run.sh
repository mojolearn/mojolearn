#!/usr/bin/env bash
set -uo pipefail
cd /root/mojolearn
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MOJOLEARN_CPU_THREADS=2 MOJOLEARN_NUMERIC_MODE=identical
export PYTHONPATH=/root/mojolearn/python
OUT=/root/gemm_leg_out/mamba-external
mkdir -p "$OUT"
PY=/root/mojolearn/.nvidia-comparator-venv/bin/python
cores=$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0))[:2])))')
taskset -pc "$cores" $$
run() {
 local name=$1 cap=$2 status
 shift 2
 timeout -k 10 "$cap" "$@" > "$OUT/$name.log" 2>&1
 status=$?
 printf '%s\t%s\n' "$name" "$status" | tee -a "$OUT/results.tsv"
 return "$status"
}
: > "$OUT/results.tsv"
printf 'native_commit=1d1f8fdd37022dad1c9e5752311296580e7762dd\nadapter_commit=d5415d25\n' > "$OUT/source.txt"
cp tools/mamba_external_compare.py "$OUT/adapter.py"
cp /root/nvidia-external-supplement.sh "$OUT/run.sh"
run causal-conv-wheel 120 "$PY" -m pip install --no-deps --only-binary=:all: 'https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/causal_conv1d-1.5.0.post8%2Bcu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl'
for family in mamba1 mamba2; do
 run "$family" 240 "$PY" tools/mamba_external_compare.py --family "$family" --native "/root/gemm_leg_out/mamba-cert/$family/native" --out "$OUT/$family" --samples 7
done
mkdir -p /root/mamba3-source
if run mamba3-source 120 curl -fL --max-time 110 https://codeload.github.com/state-spaces/mamba/tar.gz/e9594ce1c732d97440f0332fdc43170a2294dbfa -o /root/mamba3-source.tar.gz; then
 tar -xzf /root/mamba3-source.tar.gz --strip-components=1 -C /root/mamba3-source
 run mamba3-triton 120 "$PY" -m pip install --target /root/mamba3-deps --no-deps --only-binary=:all: triton==3.5.1
 run mamba3 300 env PYTHONPATH=/root/mamba3-deps:/root/mamba3-source:/root/mojolearn/python "$PY" tools/mamba_external_compare.py --family mamba3 --native /root/gemm_leg_out/mamba-cert/mamba3/native --out "$OUT/mamba3" --samples 7
fi
printf 'Supplement complete; inspect individual statuses\n'
