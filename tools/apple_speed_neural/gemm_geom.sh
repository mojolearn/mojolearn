#!/bin/sh
# lane/neural-apple (2026-09-28): the twelve T3-shard GEMM calls
# (bench/gemm_excp_ab_main.mojo, the shipped entry points) under several
# Apple matrix-plan tile geometries (-D MOJOLEARN_APPLE_MMA_*; scheduling
# only), each build's hash and median per call and operand kind, on one Mac.
# Every geometry must print the same hashes.
#   GEOMS  space separated "label:defines" with defines comma separated
#   GEOM_KINDS / GEOM_CALLS / GEOM_ROUNDS  the harness's filters
set -u
OUT=${GEOM_OUT:-$HOME/mojolearn-evidence/neural-apple-speed/geom-$(git rev-parse --short HEAD)-$(date -u +%H%M%S)}
mkdir -p "$OUT"
echo "GEOM OUT $OUT commit $(git rev-parse --short HEAD) host $(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
rc=0
for g in ${GEOMS:-default: fm2fn2:MOJOLEARN_APPLE_MMA_FM=2,MOJOLEARN_APPLE_MMA_FN=2 sg1fm4:MOJOLEARN_APPLE_MMA_SGM=1,MOJOLEARN_APPLE_MMA_SGN=1 sg4x2:MOJOLEARN_APPLE_MMA_SGM=4,MOJOLEARN_APPLE_MMA_SGN=2 fm2fn4:MOJOLEARN_APPLE_MMA_FM=2}; do
    label=${g%%:*}; defs=${g#*:}
    dflags=""
    for d in $(echo "$defs" | tr ',' ' '); do dflags="$dflags -D $d"; done
    # shellcheck disable=SC2086
    pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 $dflags -I . bench/gemm_excp_ab_main.mojo \
        -o "$OUT/ab-$label" > "$OUT/build-$label.log" 2>&1 || { echo "GEOM $label BUILD FAILED"; tail -5 "$OUT/build-$label.log"; rc=1; continue; }
    MOJOLEARN_EXCP_AB_KINDS=${GEOM_KINDS:-ordinary,mixed,sparse} MOJOLEARN_EXCP_AB_CALLS=${GEOM_CALLS:-} \
        MOJOLEARN_EXCP_AB_ROUNDS=${GEOM_ROUNDS:-3} "$OUT/ab-$label" > "$OUT/run-$label.log" 2>&1 || rc=1
    sed "s/^/GEOM $label /" "$OUT/run-$label.log" | grep -E "EXCP_AB |DISPATCH|rror"
done
exit $rc
