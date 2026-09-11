#!/bin/sh
# tools/hotaisle_smoke_body.sh. The body smoke for tools/hotaisle_leg.sh (its
# third runner test). Runs from /root/mojolearn after `pixi install`, like any
# MOJOLEARN_GEMM_LEG_EXTRA body. Writes /root/gemm_leg_out/smoke.txt.
#
#   MOJOLEARN_HOTAISLE_LANE=hotaisle-runner MOJOLEARN_GEMM_LEG_EXTRA=tools/hotaisle_smoke_body.sh \
#   bash tools/hotaisle_leg.sh amd --rent --minutes 20 --skip-gates
set -u
O=/root/gemm_leg_out/smoke.txt
{
    echo "== devices"
    ls -l /dev/kfd /dev/dri 2>&1 | head -8
    [ -e /dev/kfd ] && echo KFD_PRESENT || echo KFD_ABSENT
    rocm-smi --showproductname 2>&1 | head -20
    echo "rocminfo_gfx=$(rocminfo 2>/dev/null | grep -Eo 'gfx[0-9a-f]+' | sort -u | tr '\n' ' ')"
    echo "MOJOLEARN_GPU_ARCHS=${MOJOLEARN_GPU_ARCHS:-<unset>} MOJOLEARN_TARGET_COLUMN=${MOJOLEARN_TARGET_COLUMN:-<unset>}"
    echo "== python"
    echo "python3: $(python3 --version 2>&1)"
    echo "pixi python: $(pixi run python --version 2>&1 | tail -1)"
    echo "== disk"
    df -h / /root 2>&1
    echo "== cpu"
    echo "nproc=$(nproc)"
    lscpu 2>/dev/null | grep -E '^(Model name|CPU\(s\)|Thread|Core|Socket)'
    echo "== reachability (HEAD)"
    for u in https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet \
             http://library.istella.it/dataset/istella-s-letor.tar.gz; do
        echo "-- $u"
        curl -sSIL --max-time 30 "$u" 2>&1 | grep -iE '^(HTTP/|content-length|location|curl:)' | tr -d '\r'
        echo "curl_exit=$?"
    done
} > "$O" 2>&1

# The base binding, IDENTICAL, for this box's arch (MOJOLEARN_GPU_ARCHS is exported by the runner).
t0=$(date +%s)
tools/with_identical_mode.sh sh bindings/build.sh > /root/gemm_leg_out/build_base.log 2>&1
rc=$?
{
    echo "== build (tools/with_identical_mode.sh sh bindings/build.sh)"
    echo "build_exit=$rc build_seconds=$(( $(date +%s) - t0 ))"
    ls -l python/mojolearn/_mojolearn*.so 2>&1
} >> "$O"
cat "$O"
exit "$rc"
