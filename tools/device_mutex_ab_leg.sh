#!/usr/bin/env sh
# SPDX-License-Identifier: Apache-2.0
# THE STOCK ARM OF THE DEVICE MUTEX CHECK, ON gfx942. Top owed item of the
# 2026-09-16 random forest mutex reconciliation.
#
# WHY THIS LEG EXISTS. The 2026-09-16 gfx942 run of the primitive check passed the
# REPAIRED arm only, and the repair it was built with emitted no instructions at all, so
# what that run really showed is that its contention pattern does not reach the window.
# That is a NULL. A pass of a repaired arm with no stock arm beside it says nothing about
# whether the stock protocol loses updates.
#
# WHAT WOULD MAKE THIS LEG CONCLUSIVE. The STOCK arm losing at least one increment where
# the FENCE arm loses none. That converts a replicated effect into a mechanism, away from
# random forests, at the primitive.
#
# WHAT A NULL HERE MEANS. If the stock arm also loses nothing, this check still has not
# reached the window and the result is a NULL, not a clearance for the stock protocol.
# Report it as a null. Do not write it up as evidence that the protocol is sound.
set -u
cd /root/mojolearn || exit 9
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 MOJOLEARN_CPU_THREADS=1
export MOJOLEARN_COMPILE_JOBS=1 MOJOLEARN_BUILD_JOBS=1
mkdir -p /root/gemm_leg_out
exec > /root/gemm_leg_out/device_mutex_ab.log 2>&1

TOKEN=MUTEXAB-gfx942
echo "$TOKEN start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "$TOKEN commit $(cat /root/mojolearn/commit.txt 2>/dev/null)"
printf '%s arch %s\n' "$TOKEN" "${MOJOLEARN_GPU_ARCHS:-unset}"
rocminfo 2>/dev/null | grep -m1 -o 'gfx[0-9a-f]*'
pixi run mojo --version
sha256sum core/device_mutex.mojo core/device_mutex_check.mojo tools/compare_binary_sections.py

echo "$TOKEN ---- model check (a MODEL, not a device result) ----"
nice -n 19 python3 tools/check_mutex_handoff_model.py; echo "$TOKEN model_rc=$?"
echo "$TOKEN ---- model check negative control, MUST exit 1 ----"
nice -n 19 python3 tools/check_mutex_handoff_model.py --protocol stock; echo "$TOKEN model_stock_rc=$? (1 is correct)"

ARCH="${MOJOLEARN_GPU_ARCHS:-gfx942}"
build() {   # <label> [defines...]
    _lab=$1; shift
    echo "$TOKEN build $_lab defines='$*'"
    nice -n 19 pixi run mojo build -j 1 --target-accelerator "$ARCH" \
        -I . "$@" core/device_mutex_check.mojo -o "/tmp/dmc_$_lab" 2>&1 | tail -5
    _rc=$?
    echo "$TOKEN build $_lab rc=$_rc size=$(stat -c%s "/tmp/dmc_$_lab" 2>/dev/null)"
    return $_rc
}

build fixed || { echo "$TOKEN FIXED BUILD FAILED, leg uninterpretable"; exit 1; }
build stock -D MOJOLEARN_RF_MUTEX_CLAIM_STOCK=1 || { echo "$TOKEN STOCK BUILD FAILED, no control"; exit 1; }
build sab -D MOJOLEARN_MUTEX_SECTION_SABOTAGE=1 || echo "$TOKEN sabotage build failed, comparator has no positive control"

echo "$TOKEN ---- SECTION GATE, before trusting either arm ----"
echo "$TOKEN whole-file digests (the guard that CANNOT fail; ignore them)"
sha256sum /tmp/dmc_fixed /tmp/dmc_stock /tmp/dmc_sab 2>/dev/null
echo "$TOKEN POSITIVE CONTROL fixed vs sabotage, must DIFFER"
nice -n 19 python3 tools/compare_binary_sections.py --expect differ \
    --label-a fixed --label-b sabotage /tmp/dmc_fixed /tmp/dmc_sab
echo "$TOKEN control_rc=$?"
echo "$TOKEN THE ARMS fixed(fence) vs stock(no fence), must DIFFER"
nice -n 19 python3 tools/compare_binary_sections.py --expect differ \
    --label-a fixed_fence --label-b stock /tmp/dmc_fixed /tmp/dmc_stock
GATE=$?
echo "$TOKEN arms_rc=$GATE"
if [ "$GATE" != 0 ]; then
    echo "$TOKEN REFUSED: the two arms are the SAME PROGRAM on this target. Any"
    echo "$TOKEN contrast below would be one program compared with itself. STOPPING."
    exit 1
fi

echo "$TOKEN ---- THE STOCK ARM. This is what the leg is for. ----"
echo "$TOKEN a nonzero rc here with LOST-UPDATE ARMS is the RESULT, not a failure"
nice -n 19 /tmp/dmc_stock; echo "$TOKEN STOCK_RC=$?"

echo "$TOKEN ---- the fence arm ----"
nice -n 19 /tmp/dmc_fixed; echo "$TOKEN FIXED_RC=$?"

echo "$TOKEN ---- repeat both, so a single run is not the whole story ----"
for i in 1 2 3; do
    nice -n 19 /tmp/dmc_stock > "/tmp/stock_$i.out" 2>&1; echo "$TOKEN rep$i STOCK_RC=$?"
    grep -E "lost|LOST|PASS" "/tmp/stock_$i.out" | tail -4
    nice -n 19 /tmp/dmc_fixed > "/tmp/fixed_$i.out" 2>&1; echo "$TOKEN rep$i FIXED_RC=$?"
    grep -E "lost|LOST|PASS" "/tmp/fixed_$i.out" | tail -4
done
cp /tmp/stock_*.out /tmp/fixed_*.out /root/gemm_leg_out/ 2>/dev/null

echo "$TOKEN ---- ISA evidence: NOT YET WORKING, read this before believing a number ----"
# A `grep -c` over a command that produced nothing returns 0, and a 0 here is
# INDISTINGUISHABLE from "the instruction is absent". llvm-objdump with an amdgcn
# triple does NOT disassemble the GPU code object embedded in the HOST ELF, so the
# first version of this block printed 0 for BOTH arms and meant nothing by it. The
# authoritative emission evidence is the SECTION GATE above: .rodata carries the
# embedded code object, and it MOVED between the arms.
for a in fixed stock; do
    dis=$(llvm-objdump -d --triple=amdgcn-amd-amdhsa "/tmp/dmc_$a" 2>/dev/null | wc -l)
    if [ "$dis" -lt 10 ]; then
        echo "$TOKEN $a buffer_inv=UNAVAILABLE (objdump produced $dis lines; not a zero count)"
    else
        echo "$TOKEN $a buffer_inv_occurrences=$(llvm-objdump -d \
            --triple=amdgcn-amd-amdhsa "/tmp/dmc_$a" 2>/dev/null | grep -c buffer_inv)"
    fi
done

echo "$TOKEN finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
