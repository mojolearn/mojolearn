#!/bin/bash
# On-pod payload for lane/knn-selector (2026-09-09). Phases, each writing
# $OUT/<phase>.done with its exit code so the desk can poll one file:
#
#   bash tools/knn_selector_pod_run.sh bootstrap          pixi + mojo version
#   bash tools/knn_selector_pod_run.sh profile <tag>      per-launch-class timers
#                                                         (-D MOJOLEARN_KNN_PHASE_TIMERS=1)
#                                                         at the eight many-query shapes
#   bash tools/knn_selector_pod_run.sh ref <tag>          the 16-shape reference
#                                                         table, ours alone, 7 rounds,
#                                                         plus the 400k no-index-tile
#                                                         fingerprint gate
#   bash tools/knn_selector_pod_run.sh gates              four-arm dispatch cells,
#                                                         adversarial cells, identity
#                                                         check, knn_main, the kNN card,
#                                                         the UMAP stage identity logs
#   bash tools/knn_selector_pod_run.sh umap1m             UMAP phase split at 1M rows
#
# `MOJOLEARN_KNN_DEFINES` carries extra `-D` flags into every IDENTICAL build
# of the profile/ref phases (an A/B arm); `<tag>` names the log set.
# Nothing here runs on the Mac; `RUN OWED` in docs/lanes/HANDOFF_knn_selector.md
# lists the Apple commands.
set -u
ROOT=${MOJOLEARN_KNN_ROOT:-/root/mojolearn}
OUT=${MOJOLEARN_KNN_OUT:-/root/knn_out}
mkdir -p "$OUT"
cd "$ROOT" || exit 9
export PATH="$HOME/.pixi/bin:$PATH"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export MOJOLEARN_COMPILE_JOBS=${MOJOLEARN_COMPILE_JOBS:-4}
# MAX's bundled CUDA 13 assembler needs driver 580. Older-driver pods use
# their installed assembler at BOTH build and runtime; a runtime-only
# override cannot repair a binary already containing CUDA 13 images.
driver_major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
if [[ "$driver_major" =~ ^[0-9]+$ ]] && (( driver_major < 580 )) && [[ -x /usr/local/cuda/bin/ptxas ]]; then
    export MODULAR_NVPTX_COMPILER_PATH=${MODULAR_NVPTX_COMPILER_PATH:-/usr/local/cuda/bin/ptxas}
fi
phase=${1:?phase}
tag=${2:-default}
rc=0
run() {
    local name=$1 code=0
    shift
    "$@" > "$OUT/$name.log" 2>&1 || code=$?
    printf '%s\t%s\n' "$name" "$code" >> "$OUT/status.tsv"
    if (( code != 0 )); then rc=1; fi
    return "$code"
}
finish() { echo "$rc" > "$OUT/$phase.done"; exit "$rc"; }
rm -f "$OUT/$phase.done"
IDENT="-D MOJOLEARN_NUMERIC_IDENTICAL=1"
EXTRA=${MOJOLEARN_KNN_DEFINES:-}
build() {
    # build <name> <source> [extra -D ...]
    local name=$1 src=$2
    shift 2
    # shellcheck disable=SC2086
    run "build-$name" pixi run mojo build -j "$MOJOLEARN_COMPILE_JOBS" -I . $IDENT $EXTRA "$@" "$src" -o "$OUT/bin/$name"
}
case "$phase" in
bootstrap)
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$OUT/nvidia.csv" 2>&1
    uname -a > "$OUT/uname.txt"
    cat commit.txt > "$OUT/commit.txt" 2>/dev/null
    command -v nsys > "$OUT/nsys.txt" 2>&1 || echo "no nsys" > "$OUT/nsys.txt"
    if [ ! -x "$HOME/.pixi/bin/pixi" ] && ! command -v pixi >/dev/null 2>&1; then
        run pixi-install sh -c 'curl -fsSL --max-time 300 https://pixi.sh/install.sh | sh'
    fi
    run pixi-env pixi install --locked
    run mojo-version pixi run mojo --version
    cat "$OUT/mojo-version.log"
    finish ;;
profile)
    mkdir -p "$OUT/bin"
    build "knn-prof-$tag" bench/knn_reference_price_main.mojo -D MOJOLEARN_KNN_PHASE_TIMERS=1 || finish
    for n in ${MOJOLEARN_KNN_REF_INDEX_LIST:-100000 400000}; do
        for q in ${MOJOLEARN_KNN_REF_QUERY_LIST:-1000 4000}; do
            for k in ${MOJOLEARN_KNN_REF_K_LIST:-10 15}; do
                run "prof-$tag-$n-$q-$k" env MOJOLEARN_KNN_REF_INDEX="$n" MOJOLEARN_KNN_REF_QUERIES="$q" \
                    MOJOLEARN_KNN_REF_K="$k" MOJOLEARN_KNN_REF_ROUNDS=3 "$OUT/bin/knn-prof-$tag"
            done
        done
    done
    grep -h "KNN_PHASE_TIMERS\|KNN_REF_RESULT\|KNN_REF_HEADER" "$OUT"/prof-"$tag"-*.log > "$OUT/prof-$tag-summary.txt"
    finish ;;
ref)
    mkdir -p "$OUT/bin"
    build "knn-ref-$tag" bench/knn_reference_price_main.mojo || finish
    ROUNDS=${MOJOLEARN_KNN_REF_ROUNDS:-7}
    for n in ${MOJOLEARN_KNN_REF_INDEX_LIST:-100000 400000}; do
        for q in ${MOJOLEARN_KNN_REF_QUERY_LIST:-32 128 1000 4000}; do
            for k in ${MOJOLEARN_KNN_REF_K_LIST:-10 15}; do
                run "ref-$tag-$n-$q-$k" env MOJOLEARN_KNN_REF_INDEX="$n" MOJOLEARN_KNN_REF_QUERIES="$q" \
                    MOJOLEARN_KNN_REF_K="$k" MOJOLEARN_KNN_REF_ROUNDS="$ROUNDS" "$OUT/bin/knn-ref-$tag"
            done
        done
    done
    if [ "${MOJOLEARN_KNN_REF_NOTILE:-1}" = 1 ]; then
        build "knn-ref-$tag-notile" bench/knn_reference_price_main.mojo -D MOJOLEARN_KNN_IDENTICAL_NO_INDEX_TILE=1 || finish
        for k in 10 15; do
            run "notile-$tag-400000-1000-$k" env MOJOLEARN_KNN_REF_INDEX=400000 MOJOLEARN_KNN_REF_QUERIES=1000 \
                MOJOLEARN_KNN_REF_K="$k" MOJOLEARN_KNN_REF_ROUNDS=1 "$OUT/bin/knn-ref-$tag-notile"
        done
    fi
    grep -h "KNN_REF_HEADER\|KNN_REF_RESULT\|KNN_REF_FINGERPRINT" "$OUT"/ref-"$tag"-*.log "$OUT"/notile-"$tag"-*.log > "$OUT/ref-$tag-summary.txt" 2>/dev/null
    finish ;;
gates)
    mkdir -p "$OUT/bin" "$OUT/gates"
    run knn-distance-fma-boundary pixi run mojo run $IDENT $EXTRA -I . neighbors/checks/knn_distance_fma_boundary_check.mojo || finish
    run knn-selector-long-rows pixi run mojo run $IDENT $EXTRA -I . neighbors/checks/knn_selector_long_rows_check.mojo || finish
    run knn-distance-components pixi run mojo run $IDENT $EXTRA -I . bench/knn_index_layout_main.mojo || finish
    # four-arm dispatch check + six-arm price at 100k (the check binaries print the _CELL lines)
    run layout-price env MOJOLEARN_LAYOUT_PRICE_OUT="$OUT/layout" bash tools/knn_layout_dispatch_price.sh
    for arm in baseline selector transpose both; do
        grep _CELL "$OUT/layout/check-$arm.log" | sort | sha256sum | sed "s/-\$/cells-$arm/" >> "$OUT/gates/CELL_SHA256SUMS"
        grep -c _CELL "$OUT/layout/check-$arm.log" >> "$OUT/gates/CELL_COUNTS"
        if [[ $(grep -c _CELL "$OUT/layout/check-$arm.log") != 143628 ]] || \
           [[ $(grep _CELL "$OUT/layout/check-$arm.log" | sort | sha256sum | cut -d' ' -f1) != 49c0f02513c2722db1855acd01d6d02941ccd29f4fb1b27d59a533db6b7350ce ]]; then
            echo "dispatch cell identity failed: $arm" >> "$OUT/gates/assertions.log"
            rc=1
        fi
    done
    # adversarial, both and baseline arms (tools/continued_cert_checks.sh's recipe)
    for arm in baseline both; do
        read -r -a extra_flags <<< "$EXTRA"
        flags=(-D MOJOLEARN_NUMERIC_IDENTICAL=1 "${extra_flags[@]}")
        case "$arm" in
            both) flags+=(-D MOJOLEARN_EXPERIMENTAL_SMALLK_IDENTICAL=1 -D MOJOLEARN_EXPERIMENTAL_KNN_TRANSPOSE_IDENTICAL=1);;
            *) flags+=(-D MOJOLEARN_KNN_IDENTICAL_LEGACY_SELECT=1 -D MOJOLEARN_KNN_IDENTICAL_LEGACY_LAYOUT=1);;
        esac
        if run "adv-build-$arm" pixi run mojo build -j "$MOJOLEARN_COMPILE_JOBS" -I . "${flags[@]}" bench/knn_layout_adversarial_check.mojo -o "$OUT/bin/adv-$arm"; then
            run "adv-$arm" "$OUT/bin/adv-$arm"
            grep _CELL "$OUT/adv-$arm.log" | sort | sha256sum | sed "s/-\$/adv-$arm/" >> "$OUT/gates/CELL_SHA256SUMS"
            if [[ $(grep -c _CELL "$OUT/adv-$arm.log") != 5440 ]] || \
               [[ $(grep _CELL "$OUT/adv-$arm.log" | sort | sha256sum | cut -d' ' -f1) != 27a1e2771acd23cb67fcac8fe59d160ae1a3e95cc0ce781058f2052234f564d2 ]]; then
                echo "adversarial cell identity failed: $arm" >> "$OUT/gates/assertions.log"
                rc=1
            fi
        fi
    done
    # identity check, knn_main, the card
    run knn-identity pixi run mojo run $IDENT $EXTRA -I . neighbors/checks/knn_identity_check.mojo
    run knn-main pixi run mojo run $IDENT $EXTRA -I . neighbors/knn_main.mojo
    run knn-card env MOJOLEARN_IDENTITY_TRACE="$OUT/gates/knn.card" MOJOLEARN_UNSUP_ARM=knn \
        pixi run mojo run $IDENT $EXTRA -I . bench/unsupervised_trace_main.mojo
    grep -E "^input\.|^output\.|^query_tile" "$OUT/knn-card.log" > "$OUT/gates/knn.hashes" || true
    # the UMAP stage identity logs and the 20k launch-width fingerprint (the self-kNN feeds them)
    run umap-identity pixi run mojo run $IDENT $EXTRA -I . umap/checks/identity_check.mojo
    run umap-identity-broader pixi run mojo run $IDENT $EXTRA -I . umap/checks/identity_broader_check.mojo
    build umap-phase bench/umap_phase_price_main.mojo
    run umap-gate-20k env MOJOLEARN_UMAP_ROWS=20000 "$OUT/bin/umap-phase"
    grep -h "embedding_fnv1a64" "$OUT/umap-gate-20k.log" | sed 's/.*embedding_fnv1a64/embedding_fnv1a64/' > "$OUT/gates/umap-20k-fingerprint.txt"
    if ! grep -q 'embedding_fnv1a64 12938647291752780014' "$OUT/gates/umap-20k-fingerprint.txt"; then
        echo "UMAP 20k fingerprint mismatch" >> "$OUT/gates/assertions.log"
        rc=1
    fi
    finish ;;
umap1m)
    mkdir -p "$OUT/bin"
    [ -x "$OUT/bin/umap-phase" ] || build umap-phase bench/umap_phase_price_main.mojo
    run umap-1m env MOJOLEARN_UMAP_ROWS=1000000 MOJOLEARN_UMAP_ROUNDS=1 timeout 1500 "$OUT/bin/umap-phase"
    grep -h "UMAP_PHASE" "$OUT/umap-1m.log" > "$OUT/umap-1m-summary.txt"
    finish ;;
*)
    echo "unknown phase $phase"; rc=2; finish ;;
esac
