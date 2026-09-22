#!/usr/bin/env sh
# THE EXTRATREES AND FUSED-kNN ARMS OF THE CROSS-BLOCK MUTEX DEFECT, ON gfx942.
#
# Both subsystems
# carry the same claim_device_mutex protocol as the random forest and NEITHER HAS EVER
# BEEN MEASURED. This body measures them. It does not touch the forest; a peer session
# owns that.
#
# READ THIS BEFORE READING ANY NUMBER BELOW IT.
# The source establishes, with no box at all, that the two sites have REACHABILITY GATES the forest does not:
#
#   * ExtraTrees claims from bpn = ceildiv(k, TPB) blocks per node, k the sampled column
#     count and TPB 512 on a 64-lane wavefront. At k <= TPB there is ONE claimant per
#     mutex and the hole cannot fire. So on gfx942 ExtraTrees needs MORE THAN 512 SAMPLED
#     COLUMNS. The forest needs eleven. The column count is therefore an ARM here and not
#     a nuisance parameter, and the sub-threshold cell is a control that shares its binary
#     with the cell above it.
#
#   * The fused kNN mutex is UNREACHABLE in every shipped build: fused_l2_knn.mojo pins
#     grid_x = 1 under PIN_DETERMINISM and bindings/build.sh refuses any mode but
#     IDENTICAL for the binding that carries it. This body reaches it through
#     fused_l2_knn_launch directly, which is the only path that can.
#
# WHAT WOULD MAKE THIS LEG CONCLUSIVE. The STOCK arm moving where the repaired arm does
# not, at a column count above the threshold, with the sub-threshold cell of the SAME
# STOCK BINARY stable. That is a second live instance of a shipped defect.
#
# WHAT A NULL MEANS. At the forest's measured 4.3% to 5.3%, 300 fits expect 13 to 16
# moves and P(0) is about 2e-6. So 0/300 excludes the forest's rate. It does NOT clear
# the protocol: it bounds the rate at roughly 1% (the 95% rule-of-three bound, 3/300).
# Say the bound. A null without a detectable rate is not a clearance.
set -u
cd /root/mojolearn || exit 9
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 MOJOLEARN_CPU_THREADS=1
mkdir -p /root/gemm_leg_out
exec > /root/gemm_leg_out/et_knn_mutex.log 2>&1

TOKEN=ETKNN-gfx942
ARCH="${MOJOLEARN_GPU_ARCHS:-gfx942}"
ARMD=/root/armdir
mkdir -p "$ARMD"
JSON=/root/gemm_leg_out/et_knn.json
PUT='@PUTURL@'

echo "$TOKEN start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "$TOKEN commit $(cat /root/mojolearn/commit.txt 2>/dev/null)"
echo "$TOKEN arch $ARCH column ${MOJOLEARN_TARGET_COLUMN:-unset}"
rocminfo 2>/dev/null | grep -m1 -o 'gfx[0-9a-f]*'
pixi run mojo --version
sha256sum core/device_mutex.mojo \
          extratrees/impl/decisiontree/batched_levelalgo/split.mojo \
          neighbors/impl/detail/fused_l2_knn.mojo \
          neighbors/checks/fused_mutex_repeat_main.mojo \
          tools/mutex_et_knn/et_ab.py

upload() { [ -s "$JSON" ] || return 1
    curl -fsS --max-time 180 -X PUT --upload-file "$JSON" "$PUT" > /dev/null 2>&1; }
( while :; do sleep 60; upload && echo "$TOKEN partial_uploaded $(wc -c < "$JSON")"; done ) &
UPLOADER=$!
echo "$TOKEN uploader_pid=$UPLOADER"

sect() {  # <expect> <a> <b> <what>
    echo "$TOKEN section $4  expect $1"
    nice -n 19 python3 tools/compare_binary_sections.py --expect "$1" \
        --label-a "$2" --label-b "$3" "$ARMD/$2" "$ARMD/$3"
    echo "$TOKEN section $4 rc=$?"
}

# =====================================================================
# PART 1. THE FUSED kNN MERGE. Small mains, so this part is cheap and it
# runs first: if the section comparator cannot report DIFFER on this
# target, nothing below it is readable either.
# =====================================================================
echo "$TOKEN ==== PART 1 fused kNN ===="
kbuild() {  # <label> then literal defines
    _l=$1; shift
    nice -n 19 pixi run mojo build -j 1 --target-accelerator "$ARCH" -I . \
        -D MOJOLEARN_NUMERIC_IDENTICAL=1 "$@" \
        neighbors/checks/fused_mutex_repeat_main.mojo -o "$ARMD/knn_$_l" 2>&1 | tail -4
    echo "$TOKEN knnbuild $_l size=$(stat -c%s "$ARMD/knn_$_l" 2>/dev/null)"
}
kbuild fixed
kbuild fixed2
kbuild stock -D MOJOLEARN_RF_MUTEX_CLAIM_STOCK=1
kbuild sab   -D MOJOLEARN_MUTEX_SECTION_SABOTAGE=1

sect same   knn_fixed knn_fixed2 "NEGATIVE CONTROL two builds of one source"
sect differ knn_fixed knn_sab    "POSITIVE CONTROL the section sabotage"
sect differ knn_fixed knn_stock  "THE ARMS repaired against stock"
nice -n 19 python3 tools/compare_binary_sections.py --expect differ \
    "$ARMD/knn_fixed" "$ARMD/knn_stock" > /dev/null 2>&1
KGATE=$?
if [ "$KGATE" != 0 ]; then
    echo "$TOKEN KNN REFUSED: the two arms are the SAME PROGRAM on $ARCH."
else
    for a in stock fixed; do
        echo "$TOKEN ---- knn arm $a, computed grid ----"
        MOJOLEARN_KNN_MUTEX_REPEATS=300 nice -n 19 "$ARMD/knn_$a"
        echo "$TOKEN knn_${a}_rc=$?"
    done
    for a in stock fixed; do
        echo "$TOKEN ---- knn arm $a, forced grid_x=8 ----"
        MOJOLEARN_KNN_MUTEX_REPEATS=200 MOJOLEARN_KNN_MUTEX_GX=8 nice -n 19 "$ARMD/knn_$a"
        echo "$TOKEN knn_${a}_gx8_rc=$?"
    done
fi

# =====================================================================
# PART 2. EXTRATREES. The binding, three builds, then the cells.
# =====================================================================
echo "$TOKEN ==== PART 2 ExtraTrees ===="
SO=/root/mojolearn/python/mojolearn/identical/_mojolearn_trees.so
tbuild() {  # <label> <defines as ONE string, consumed by sh word splitting>
    _l=$1
    echo "$TOKEN treesbuild $_l defines='$2' start $(date -u +%H:%M:%SZ)"
    rm -f "$SO"
    MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_TARGET_COLUMN=amd \
    MOJOLEARN_GPU_ARCHS="$ARCH" MOJOLEARN_COMPILE_JOBS=4 \
    MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_EXTRA_DEFINES="$2" \
        nice -n 19 sh bindings/build_trees.sh > "/root/gemm_leg_out/treesbuild_$_l.log" 2>&1
    _rc=$?
    echo "$TOKEN treesbuild $_l rc=$_rc $(date -u +%H:%M:%SZ)"
    if [ "$_rc" != 0 ] || [ ! -f "$SO" ]; then
        tail -12 "/root/gemm_leg_out/treesbuild_$_l.log"
        return 1
    fi
    cp "$SO" "$ARMD/trees_$_l.so"
    echo "$TOKEN treesbuild $_l sha256=$(sha256sum "$ARMD/trees_$_l.so" | cut -c1-16)"
    return 0
}

# The define reaches the compiler through MOJOLEARN_EXTRA_DEFINES, which
# bindings/build_trees.sh:132 expands UNQUOTED under sh, so it DOES word-split
# there. It is checked anyway, by section, below: this project has already had a
# leg in which both arms were one program because a zsh loop did not split.
tbuild fixed  ""            || echo "$TOKEN ET FIXED BUILD FAILED"
tbuild fixed2 ""            || echo "$TOKEN ET FIXED2 BUILD FAILED"
tbuild stock  "-D MOJOLEARN_RF_MUTEX_CLAIM_STOCK=1" || echo "$TOKEN ET STOCK BUILD FAILED"

sect same   trees_fixed.so trees_fixed2.so "ET NEGATIVE CONTROL"
sect differ trees_fixed.so trees_stock.so  "ET THE ARMS"
nice -n 19 python3 tools/compare_binary_sections.py --expect differ \
    "$ARMD/trees_fixed.so" "$ARMD/trees_stock.so" > /dev/null 2>&1
TGATE=$?
echo "$TOKEN et_arms_gate_rc=$TGATE"

etrun() {   # <arm> <cols> <repeats> <seconds>
    cp "$ARMD/trees_$1.so" "$SO" || return 1
    echo "$TOKEN ---- ET cell arm=$1 cols=$2 repeats=$3 $(date -u +%H:%M:%SZ) ----"
    PYTHONPATH=/root/mojolearn/python:/root/mojolearn \
    MOJOLEARN_NUMERIC_MODE=identical \
    ET_JSON="$JSON" ET_ARM="$1" ET_COLS="$2" ET_REPEATS="$3" ET_SECS="$4" \
    ET_ROWS=2000 ET_TREES=16 ET_DEPTH=8 ET_TPB=512 \
    ET_SO_SHA="$(sha256sum "$ARMD/trees_$1.so" | cut -d' ' -f1)" \
        nice -n 19 pixi run python tools/mutex_et_knn/et_ab.py
    echo "$TOKEN et_cell_${1}_${2}_rc=$?"
    upload && echo "$TOKEN uploaded after $1/$2"
}

if [ "$TGATE" = 0 ]; then
    # The order is deliberate. The cell that has to MOVE runs FIRST, so a leg that
    # runs out of lease still carries its own control (the same binary below the
    # threshold) rather than only the arm that is expected to be quiet.
    # SIZED FROM A MEASUREMENT, NOT A GUESS. One fit of this configuration took
    # about 2 s on an M4 (3 fits, 7.5 s wall including interpreter start), so 600
    # fits fit in a 700 s cell with room. At the forest's 4.3% that arm expects 26
    # moves and P(0) is about 5e-12; a 0/600 bounds the rate at 0.5% by the 95%
    # rule of three. COMPUTE THIS BEFORE READING THE RESULT, which is the whole
    # point of writing it in the body rather than in the write-up.
    etrun stock 2048 600 700   # bpn = 4 on gfx942. THE ARM.
    etrun stock  256 300 250   # bpn = 1, SAME BINARY, cannot contend. THE CONTROL.
    etrun fixed 2048 600 700   # bpn = 4, repaired.
    etrun fixed  256 300 250
else
    echo "$TOKEN ET SKIPPED: the arms did not pass the section gate"
fi

kill "$UPLOADER" 2>/dev/null
upload && echo "$TOKEN final_uploaded $(wc -c < "$JSON")"
echo "$TOKEN ---- summary ----"
grep -hE "^ARM |^CONTROL |^POSITIVE_CONTROL |^CELL |grid_x " /root/gemm_leg_out/et_knn_mutex.log 2>/dev/null | tail -40
echo "$TOKEN finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
