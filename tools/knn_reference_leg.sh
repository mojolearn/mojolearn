#!/usr/bin/env bash
# The k-NN reference grid ON THE BOX: ours IDENTICAL (kernel-matrix defaults),
# then cuML brute-force NearestNeighbors, same fixture and shapes. Parent owns
# the pod, the lease, the fetch and the teardown.
#
#   MOJOLEARN_KNN_REF_OUT=<fresh dir> bash tools/knn_reference_leg.sh
#
# Optional: MOJOLEARN_KNN_REF_INDEX_LIST="100000 400000",
# MOJOLEARN_KNN_REF_QUERY_LIST="32 128 1000 4000", MOJOLEARN_KNN_REF_K_LIST="10 15",
# MOJOLEARN_KNN_REF_ROUNDS=7, MOJOLEARN_KNN_REF_SKIP_CUML=1, MOJOLEARN_KNN_REF_PY=<python>.
set -uo pipefail
cd "${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
OUT=${MOJOLEARN_KNN_REF_OUT:?set a fresh evidence directory}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
[ ! -e "$OUT/status.tsv" ] || { echo 'refusing to overwrite evidence'; exit 2; }
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
INDEX_LIST=${MOJOLEARN_KNN_REF_INDEX_LIST:-"100000 400000"}
QUERY_LIST=${MOJOLEARN_KNN_REF_QUERY_LIST:-"32 128 1000 4000"}
K_LIST=${MOJOLEARN_KNN_REF_K_LIST:-"10 15"}
ROUNDS=${MOJOLEARN_KNN_REF_ROUNDS:-7}
rc=0
run() {
    local name=$1 code=0
    shift
    "$@" > "$OUT/$name.log" 2>&1 || code=$?
    printf '%s\t%s\n' "$name" "$code" >> "$OUT/status.tsv"
    if (( code != 0 )); then rc=1; fi
    return "$code"
}
: > "$OUT/status.tsv"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$OUT/nvidia.csv" 2>&1
uname -a > "$OUT/uname.txt"
pixi run mojo --version > "$OUT/mojo_version.txt" 2>&1
mkdir -p "$OUT/dump"
if run build-ours pixi run mojo build -j "${MOJOLEARN_COMPILE_JOBS:-4}" -I . \
        -D MOJOLEARN_NUMERIC_IDENTICAL=1 bench/knn_reference_price_main.mojo -o "$OUT/ours-identical"; then
    for n in $INDEX_LIST; do
        for q in $QUERY_LIST; do
            for k in $K_LIST; do
                run "ours-$n-$q-$k" env MOJOLEARN_KNN_REF_INDEX="$n" MOJOLEARN_KNN_REF_QUERIES="$q" \
                    MOJOLEARN_KNN_REF_K="$k" MOJOLEARN_KNN_REF_ROUNDS="$ROUNDS" \
                    MOJOLEARN_KNN_REF_DUMP="$OUT/dump/ours-$n-$q-$k.u32" "$OUT/ours-identical"
            done
        done
    done
fi
rm -f "$OUT/ours-identical"
if [ "${MOJOLEARN_KNN_REF_SKIP_CUML:-0}" != 1 ]; then
    PY=${MOJOLEARN_KNN_REF_PY:-}
    if [ -z "$PY" ]; then
        run cuml-venv python3 -m venv --system-site-packages "$OUT/../cuml-venv"
        PY="$OUT/../cuml-venv/bin/python"
        run cuml-wheels "$PY" -m pip install --disable-pip-version-check --no-input --only-binary=:all: \
            numpy==2.4.6 cupy-cuda12x==14.2.0 cuml-cu12==26.8.0 --extra-index-url https://pypi.nvidia.com
    fi
    run cuml-freeze "$PY" -m pip freeze
    # shellcheck disable=SC2086
    run cuml-grid "$PY" tools/knn_cuml_reference.py --index $INDEX_LIST --queries $QUERY_LIST \
        --k $K_LIST --rounds "$ROUNDS" --ours-dump "$OUT/dump" --out "$OUT/cuml-reference.json"
fi
rm -rf "$OUT/dump"
printf 'leg_exit=%s\n' "$rc" > "$OUT/completion.txt"
exit "$rc"
