#!/usr/bin/env bash
set -uo pipefail
cd /root/mojolearn
export PATH=/root/.pixi/bin:$PATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MOJOLEARN_CPU_THREADS=2 MOJOLEARN_COMMIT=1d1f8fdd37022dad1c9e5752311296580e7762dd
export PYTHONPATH=/root/mojolearn/python
OUT=/root/gemm_leg_out/final-supplement
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
cp /root/nvidia-final-supplement.sh "$OUT/run.sh"
cp tools/nvidia_public_compare.py bench/knn_smallk_selection_main.mojo neighbors/checks/select_smallk_identical_candidate.mojo "$OUT/"
run smallk-specialized 180 env MOJOLEARN_SMALLK_LARGE=1 pixi run mojo run -j 2 -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . bench/knn_smallk_selection_main.mojo
for rows in 32 128 1000; do
 run "torch-knn-$rows" 120 "$PY" tools/nvidia_public_compare.py --lane knn --knn-external torch --index 100000 --queries "$rows" --rounds 7 --out "$OUT/torch-knn-$rows"
done
run transformer-api-unbuffered 120 env MOJOLEARN_NUMERIC_MODE=identical PYTHONUNBUFFERED=1 pixi run -e skgpu python -u python/mojolearn/tests/test_transformer_surface.py
printf 'Final supplement completed\n'
