#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
#
# tools/ols_illconditioned_leg.sh -- the ols-illconditioned lane's box body
# (DEVIATIONS 2620, 2621). A MOJOLEARN_GEMM_LEG_EXTRA body: runs ON THE BOX
# from /root/mojolearn after `pixi install` and writes everything under
# /root/gemm_leg_out/ols-illconditioned, which every runner fetches home.
#
#   AMD, Hot Aisle:
#     MOJOLEARN_HOTAISLE_LANE=ols-illconditioned \
#     MOJOLEARN_GEMM_LEG_EXTRA=tools/ols_illconditioned_leg.sh \
#     MOJOLEARN_GEMM_LEG_OUT=$HOME/mojolearn-evidence/ols-illconditioned/leg1-amd \
#     bash tools/hotaisle_leg.sh amd --rent --minutes 60 --skip-gates
#   NVIDIA, RunPod (the gemm payload's card diff is not this leg's evidence):
#     MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key \
#     MOJOLEARN_GEMM_LEG_LOCAL_CARD=$HOME/mojolearn-evidence/classical-runpod-amd/apple_12a22776.card \
#     MOJOLEARN_GEMM_LEG_EXTRA=tools/ols_illconditioned_leg.sh \
#     MOJOLEARN_GEMM_LEG_OUT=$HOME/mojolearn-evidence/ols-illconditioned/leg2-nvidia \
#     sh tools/gemm_remote_leg.sh nvidia --payload gemm --rent --minutes 60
#
# THREE STREAMS FROM t=0, THEN THE FITS
#   data      Istella-S and taxi fetched (sha256 against the classical lane's
#             pins), decoded by tools/speed_gbdt_arm.py, prepped by the probe,
#             then `probe ref` (scikit-learn, spectra, float32 emulations)
#   fixed     the IDENTICAL base and estimators bindings of THIS commit, then
#             glm/ols_main.mojo under IDENTICAL (every OLS check)
#   sabotage  a copy of the tree with its own pixi env: the three new checks
#             under sabotage f (no equilibration), g (absolute 1e-10) and
#             f+g, each applied to lstsq.mojo by sed with the diff kept; then
#             the estimators binding built WITH f+g, which is the pre-fix
#             arithmetic (the BEFORE binding)
#   fits      `probe ours` with the after and the before bindings, `probe
#             cuml` on NVIDIA, the summary, and glm/ols_main.mojo under FAST
#             if time is left
# status.tsv has one line per step: name, exit code, seconds, UTC time.
set -u
ROOT=/root/mojolearn
OUT=${MOJOLEARN_OLSIC_OUT:-/root/gemm_leg_out/ols-illconditioned}
DATA=/root/olsic-data
SAB=/root/olsic-sab
VENV=/root/olsic-venv
PROBE="$ROOT/tools/ols_illconditioned_probe.py"
LSQ=glm/impl/linalg/detail/lstsq.mojo
BODY_START=$(date +%s)
BODY_SECONDS=${MOJOLEARN_OLSIC_BODY_SECONDS:-2700}
JOBS=${MOJOLEARN_COMPILE_JOBS:-$(nproc)}
TAXI_WAIT=${MOJOLEARN_OLSIC_TAXI_WAIT:-900}
GBM_BENCH_DATA=${GBM_BENCH_DATA:-/root/datasets/gbm-bench}
export GBM_BENCH_DATA
export PATH="$HOME/.pixi/bin:$PATH"
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS
export MOJOLEARN_NUMERIC_MODE=identical
TAXI_SHA_2024_01=c4d59da7bbc8abaeeeb1727947ee93d9891a71acb42854bd80db1571b2030510
TAXI_SHA_2024_02=c76c43c18c6c6664080dd920baab4928988d5786a6b65980792ca7cd796f9f20
ISTELLA_TGZ_SHA=41b21116a3650cc043dbe16f02ee39f4467f9405b37fdbcc9a6a05e230a38981
UA="Mozilla/5.0 (X11; Linux x86_64) mojolearn-bench"
SED_F='s/^    if (bits >> 31) != UInt32(0) or field == 255:$/    if True:/'
SED_G='s/^    return Float32(Float64(n) \* OLS_PINV_EPS32 \* Float64(max_abs_eig))$/    return OLS_NONZERO_THRESH/'

mkdir -p "$OUT/probe" "$DATA"
cd "$ROOT" || exit 9
touch "$OUT/status.tsv"
VENDOR=$(sed -n 's/^vendor=//p' /root/gemm_leg_out/leg.txt 2>/dev/null | head -1)
if [ -z "$VENDOR" ]; then
    if [ -e /dev/kfd ] && command -v rocminfo > /dev/null 2>&1; then VENDOR=amd
    elif command -v nvidia-smi > /dev/null 2>&1; then VENDOR=nvidia
    else VENDOR=unknown; fi
fi

record() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$(date -u +%H:%M:%S)" >> "$OUT/status.tsv"; }
run() {
    _name=$1
    shift
    _t0=$(date +%s)
    "$@" > "$OUT/$_name.log" 2>&1
    _rc=$?
    record "$_name" "$_rc" "$(( $(date +%s) - _t0 ))"
    return $_rc
}
left() { echo $(( BODY_START + BODY_SECONDS - $(date +%s) )); }
wait_for() {  # <file>: true once it exists, false at the deadline
    while [ ! -f "$1" ]; do
        [ "$(left)" -gt 30 ] || return 1
        sleep 10
    done
    return 0
}

{
    echo "lane=ols-illconditioned (DEVIATIONS 2620, 2621)"
    echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ) body_seconds=$BODY_SECONDS jobs=$JOBS"
    echo "vendor=$VENDOR gpu_archs=${MOJOLEARN_GPU_ARCHS:-} target_column=${MOJOLEARN_TARGET_COLUMN:-}"
    echo "commit=$(sed -n 's/^commit=//p' /root/gemm_leg_out/leg.txt 2>/dev/null | head -1)"
    uname -a
    nproc
    lscpu 2>/dev/null | grep -E 'Model name|^CPU\(s\)'
    free -g 2>/dev/null | head -2
    command -v rocm-smi > /dev/null 2>&1 && rocm-smi --showproductname 2>&1 | head -12
    command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>&1
    sha256sum "$LSQ" glm/checks/ols_check.mojo "$PROBE"
} > "$OUT/env.txt" 2>&1
record env 0 0

# ============================================================================
# data stream
# ============================================================================
(
    mkdir -p "$GBM_BENCH_DATA/istella" "$GBM_BENCH_DATA/taxi"
    (
        _tgz="$GBM_BENCH_DATA/istella/istella-s-letor.tar.gz"
        _i=0
        while [ "$_i" -lt 4 ] && [ "$(sha256sum "$_tgz" 2>/dev/null | cut -c1-64)" != "$ISTELLA_TGZ_SHA" ]; do
            _i=$((_i + 1))
            [ "$_i" -ge 3 ] && rm -f "$_tgz"
            timeout -k 10 1200 curl -fsSL -C - --retry 3 --retry-delay 5 -A "$UA" -o "$_tgz" \
                http://library.istella.it/dataset/istella-s-letor.tar.gz >> "$OUT/istella_curl.log" 2>&1
            echo "attempt=$_i curl_exit=$? size=$(stat -c %s "$_tgz" 2>/dev/null || echo 0) $(date -u +%T)" >> "$OUT/istella_curl.log"
        done
        if [ "$(sha256sum "$_tgz" 2>/dev/null | cut -c1-64)" = "$ISTELLA_TGZ_SHA" ]; then
            run istella-untar tar -xzf "$_tgz" -C "$GBM_BENCH_DATA/istella"
        else
            record istella-fetch SHA_MISMATCH 0
        fi
        : > "$OUT/fetch_istella.done"
    ) &
    (
        for m in 2024-01 2024-02; do
            _f="$GBM_BENCH_DATA/taxi/yellow_tripdata_$m.parquet"
            [ -f "$_f" ] || { timeout -k 10 600 curl -fsSL --retry 3 -A "$UA" -o "$_f.part" \
                "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_$m.parquet" \
                > "$OUT/taxi_curl_$m.txt" 2>&1 && mv "$_f.part" "$_f"; }
            rm -f "$_f.part"
        done
        # A CDN that blocks the box (DigitalOcean): wait up to TAXI_WAIT
        # seconds for the Mac to scp both months into $GBM_BENCH_DATA/taxi
        # (sha256 checked, so a half-written upload is not taken).
        _w0=$(date +%s)
        while :; do
            _ok=1
            [ "$(sha256sum "$GBM_BENCH_DATA/taxi/yellow_tripdata_2024-01.parquet" 2>/dev/null | cut -c1-64)" = "$TAXI_SHA_2024_01" ] || _ok=0
            [ "$(sha256sum "$GBM_BENCH_DATA/taxi/yellow_tripdata_2024-02.parquet" 2>/dev/null | cut -c1-64)" = "$TAXI_SHA_2024_02" ] || _ok=0
            [ "$_ok" = 1 ] && break
            : > "$OUT/taxi_upload_wanted"
            [ $(( $(date +%s) - _w0 )) -ge "$TAXI_WAIT" ] && break
            sleep 15
        done
        record taxi-fetch "$( [ "$_ok" = 1 ] && echo 0 || echo SHA_MISMATCH)" "$(( $(date +%s) - _w0 ))"
        : > "$OUT/fetch_taxi.done"
    ) &
    (
        if [ "$VENDOR" = nvidia ]; then
            PY=python3
            run pip-base timeout -k 30 900 python3 -m pip install --no-input --disable-pip-version-check \
                scikit-learn pyarrow threadpoolctl
        else
            PY="$VENV/bin/python"
            if ! run venv python3 -m venv "$VENV"; then
                rm -rf "$VENV"
                run uv-get sh -c 'curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL=/root/.local/uv sh'
                run venv-uv /root/.local/uv/uv venv --seed --python 3.12 "$VENV"
            fi
            run wheels timeout -k 30 900 "$VENV/bin/pip" install --disable-pip-version-check --no-input \
                --only-binary=:all: numpy scipy scikit-learn threadpoolctl pyarrow
        fi
        echo "$PY" > "$OUT/theirs_python.txt"
        "$PY" -c 'import numpy, scipy, sklearn; print("numpy", numpy.__version__, "scipy", scipy.__version__, "sklearn", sklearn.__version__)' > "$OUT/versions.txt" 2>&1
        : > "$OUT/venv.done"
        if [ "$VENDOR" = nvidia ]; then
            run pip-cuml timeout -k 30 1500 python3 -m pip install --no-input --disable-pip-version-check \
                --extra-index-url=https://pypi.nvidia.com cuml-cu12==26.8.0
            python3 -c 'import cuml, cupy; print("cuml", cuml.__version__, "cupy", cupy.__version__)' >> "$OUT/versions.txt" 2>&1
            : > "$OUT/cuml.ready"
        fi
    ) &
    wait_for "$OUT/venv.done"
    PY=$(cat "$OUT/theirs_python.txt")
    wait_for "$OUT/fetch_istella.done"
    wait_for "$OUT/fetch_taxi.done"
    ( run decode-istella timeout -k 30 1500 "$PY" tools/speed_gbdt_arm.py --download istella
      run prep-istella timeout -k 30 900 "$PY" "$PROBE" prep --data "$DATA" --datasets istella ) &
    ( run decode-taxi timeout -k 30 1200 "$PY" tools/speed_gbdt_arm.py --download taxi
      run prep-taxi timeout -k 30 900 "$PY" "$PROBE" prep --data "$DATA" --datasets taxi ) &
    wait
    : > "$OUT/prep.done"
    run ref timeout -k 30 1500 "$PY" "$PROBE" ref --data "$DATA" --out "$OUT/probe"
    : > "$OUT/data.done"
) &

# ============================================================================
# fixed stream: this commit's bindings and every OLS check under IDENTICAL
# ============================================================================
(
    for b in build.sh build_estimators.sh; do
        run "build-after-${b%.sh}" env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 \
            MOJOLEARN_COMPILE_JOBS="$JOBS" timeout -k 30 1500 sh "bindings/$b"
    done
    sha256sum python/mojolearn/identical/*.so > "$OUT/bindings-after.txt" 2>&1
    run checks-identical timeout -k 30 1500 pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . glm/ols_main.mojo
    : > "$OUT/fixed.done"
) &

# ============================================================================
# sabotage stream: a copy of the tree, sabotages applied there only
# ============================================================================
(
    rm -rf "$SAB"
    mkdir -p "$SAB"
    tar -C "$ROOT" --exclude=./.pixi -cf - . | tar -C "$SAB" -xf -
    cd "$SAB" || exit 9
    run sab-pixi-install timeout -k 30 1500 pixi install
    cp "$LSQ" "$OUT/lstsq.fixed.mojo"
    cat > glm/olsic_sabotage_main.mojo <<'EOF'
from glm.checks.ols_check import (
    check_ols_mixed_scale_design_matches_float64_oracle,
    check_ols_rank_deficient_design_drops_the_noise_direction,
    check_ols_rank_guard_is_scale_invariant,
)


def main() raises:
    var failed = 0
    try:
        check_ols_rank_guard_is_scale_invariant()
        print("SABOTAGE-PASS check_ols_rank_guard_is_scale_invariant")
    except e:
        print("SABOTAGE-FAIL check_ols_rank_guard_is_scale_invariant: " + String(e))
        failed += 1
    try:
        check_ols_mixed_scale_design_matches_float64_oracle()
        print("SABOTAGE-PASS check_ols_mixed_scale_design_matches_float64_oracle")
    except e:
        print("SABOTAGE-FAIL check_ols_mixed_scale_design_matches_float64_oracle: " + String(e))
        failed += 1
    try:
        check_ols_rank_deficient_design_drops_the_noise_direction()
        print("SABOTAGE-PASS check_ols_rank_deficient_design_drops_the_noise_direction")
    except e:
        print("SABOTAGE-FAIL check_ols_rank_deficient_design_drops_the_noise_direction: " + String(e))
        failed += 1
    print("SABOTAGE-SUMMARY failed=" + String(failed) + " of 3")
EOF
    sab() {  # <tag> <sed expression>...: apply, keep the diff, run the three checks
        _tag=$1
        shift
        cp "$OUT/lstsq.fixed.mojo" "$LSQ"
        _want=0
        for _e in "$@"; do
            sed -i "$_e" "$LSQ"
            _want=$((_want + 1))
        done
        diff "$OUT/lstsq.fixed.mojo" "$LSQ" > "$OUT/sabotage-$_tag.diff"
        _got=$(grep -c '^>' "$OUT/sabotage-$_tag.diff")
        if [ "$_got" != "$_want" ]; then
            record "sabotage-$_tag" "SED_MATCHED_${_got}_OF_${_want}" 0
            return 1
        fi
        run "sabotage-$_tag" timeout -k 30 1500 pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . glm/olsic_sabotage_main.mojo
    }
    sab none
    sab f "$SED_F"
    sab g "$SED_G"
    if sab fg "$SED_F" "$SED_G"; then :; fi
    # The BEFORE binding: f and g together are the pre-fix arithmetic (a
    # multiply by 1.0 and the absolute 1e-10), still applied to this copy.
    if [ "$(grep -c '^>' "$OUT/sabotage-fg.diff" 2>/dev/null)" = 2 ]; then
        run build-before-estimators env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 \
            MOJOLEARN_COMPILE_JOBS="$JOBS" timeout -k 30 1500 sh bindings/build_estimators.sh
    fi
    cp "$OUT/lstsq.fixed.mojo" "$LSQ"
    : > "$OUT/sabotage.done"
) &

# ============================================================================
# fits
# ============================================================================
wait_for "$OUT/fixed.done"
wait_for "$OUT/sabotage.done"
wait_for "$OUT/prep.done"
if [ -f python/mojolearn/identical/_mojolearn_estimators.so ]; then
    run ours-after env PYTHONPATH="$ROOT/python" timeout -k 30 1200 \
        pixi run python3 "$PROBE" ours --data "$DATA" --out "$OUT/probe" --tag after
fi
if [ -f "$SAB/python/mojolearn/identical/_mojolearn_estimators.so" ]; then
    for so in python/mojolearn/identical/*.so; do
        case "$so" in *_mojolearn_estimators*) ;; *) cp "$so" "$SAB/python/mojolearn/identical/" ;; esac
    done
    sha256sum "$SAB"/python/mojolearn/identical/*.so > "$OUT/bindings-before.txt" 2>&1
    ( cd "$SAB" && run ours-before env PYTHONPATH="$SAB/python" timeout -k 30 1200 \
        pixi run python3 "$PROBE" ours --data "$DATA" --out "$OUT/probe" --tag before )
fi
if [ "$VENDOR" = nvidia ] && wait_for "$OUT/cuml.ready"; then
    run cuml timeout -k 30 900 python3 "$PROBE" cuml --data "$DATA" --out "$OUT/probe"
fi
wait_for "$OUT/data.done"
PY=$(cat "$OUT/theirs_python.txt" 2>/dev/null || echo python3)
"$PY" "$PROBE" summary --out "$OUT/probe" > "$OUT/summary.txt" 2>&1
record summary $? 0
if [ "$(left)" -gt 700 ]; then
    run checks-fast timeout -k 30 "$(( $(left) - 120 ))" env MOJOLEARN_NUMERIC_MODE=fast \
        pixi run mojo run -I . glm/ols_main.mojo
fi
record body-done 0 "$(( $(date +%s) - BODY_START ))"
exit 0
