#!/usr/bin/env bash
set -uo pipefail
cd /root/mojolearn
export PATH=/root/.pixi/bin:$PATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MOJOLEARN_CPU_THREADS=2 MOJOLEARN_COMMIT=1d1f8fdd37022dad1c9e5752311296580e7762dd
export PYTHONPATH=/root/mojolearn/python
OUT=/root/gemm_leg_out/priority-supplement
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
cp /root/nvidia-priority-supplement.sh "$OUT/run.sh"
cp bench/knn_index_layout_main.mojo bench/knn_smallk_selection_main.mojo neighbors/checks/select_smallk_identical_candidate.mojo "$OUT/"
run smallk 180 env MOJOLEARN_SMALLK_LARGE=1 pixi run mojo run -j 2 -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . bench/knn_smallk_selection_main.mojo
for rows in 32 128; do
 run "random-distance-$rows" 180 env MOJOLEARN_KNN_LAYOUT_LARGE=1 MOJOLEARN_KNN_LAYOUT_ROWS="$rows" pixi run mojo run -j 2 -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . bench/knn_index_layout_main.mojo
done
run rapids-retry 360 "$PY" -m pip install --disable-pip-version-check --no-input --only-binary=:all: cupy-cuda12x cuml-cu12 --constraint /root/gemm_leg_out/campaign/vendor-constraints.txt --extra-index-url https://pypi.nvidia.com
for rows in 32 128 1000; do
 run "public-knn-$rows" 180 "$PY" tools/nvidia_public_compare.py --lane knn --index 100000 --queries "$rows" --rounds 7 --out "$OUT/public-knn-$rows"
done
run public-umap 180 "$PY" tools/nvidia_public_compare.py --lane umap --rounds 7 --out "$OUT/public-umap"
printf 'Priority supplement completed\n'
