#!/bin/sh
# tools/gemm_ksplit_classical_amd_leg.sh -- the classical GEMM callers under
# the ksplit default (DEVIATION 2595; AMD row 110 since 190fb7a4) against the
# old plan, on the Hot Aisle AMD MI300X. ENGINEERING_RULES.md section 9 owes
# this A/B for a shared kernel's flip on every lane it reaches
# (docs/lanes/BRIEF_gemm_long_k_2026-09-11.md section 12).
#
# A tools/hotaisle_leg.sh BODY. It runs ON THE BOX inside
# rocm/dev-ubuntu-22.04:6.4.1-complete, from /root/mojolearn after `pixi
# install`, with MOJOLEARN_TARGET_COLUMN=amd and MOJOLEARN_GPU_ARCHS exported by
# the runner. Knobs arrive through MOJOLEARN_HOTAISLE_EXTRA_ENV, which allows
# no spaces, so lists are commas. Everything it keeps is under
# /root/gemm_leg_out/ksplit-classical-amd, which the runner fetches to
# <MOJOLEARN_GEMM_LEG_OUT>/remote/ksplit-classical-amd. Binaries, datasets
# and blocks stay outside that tree.
#
# Two legs under the 60-minute cap (brief 12.6), the callers the rule can
# take first:
#
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/gemm_ksplit_classical_amd_leg.sh \
#   MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date -u +%Y-%m-%d_%H%M%S)-amd-mi300x-hotaisle-gemm-ksplit-classical-kde-svc \
#   MOJOLEARN_HOTAISLE_EXTRA_ENV=MOJOLEARN_CLASSICAL_AB_LANES=kde,svc \
#   bash tools/hotaisle_leg.sh amd --rent --minutes 60 --skip-gates
#
#   (leg 2: MOJOLEARN_CLASSICAL_AB_LANES=gp,ols,pca and the -gp-ols-pca suffix)
#
# WHY A SECOND BODY, NOT tools/gemm_ksplit_classical_leg.sh. That file is the
# H100 leg, and RunPod passes a body no environment, so its defaults ARE that
# leg (gp, ols and pca on the trees loader). This leg adds the kde and svc
# callers on the classical lane's own blocks (tools/classical_two_datasets.py
# prep, which needs the SVM binding and a prep step), orders the callers by
# priority, ends inside the Hot Aisle runner's work bound, and pre-fetches
# Istella-S with a resumable curl. Folding that into the H100 body behind
# vendor branches would change the file the H100 leg ships while its run is
# still owed. The Python driver, tools/gemm_ksplit_classical_ab.py, is shared:
# this body calls it with --source ctd, the H100 body with its defaults.
#
# WHAT RUNS, each step with its own exit code in status.tsv. A red step is a
# finding, never an abort: later steps still run (set -u, not set -e).
#
#   fetch-*        at t=0, in the background: the Istella-S tarball (curl, resumable,
#                  sha256-checked, untarred) and the two taxi months (sha256-checked)
#   numpy, pyarrow the pixi python's prerequisites; a uv Python 3.12 venv decodes
#                  and preps instead when pyarrow will not import there
#   download-*,    in the background, per dataset once its fetch is done: the trees
#   prep-*         harness's decode (tools/speed_gbdt_arm.py --download) and the
#                  classical blocks (tools/classical_two_datasets.py prep) for the
#                  lanes that have one; then <dataset>.ready
#   build-price    bench/gemm_step_price_main.mojo, IDENTICAL + trial, label mode
#                  only: plans.tsv, the DEFAULT line, and dispatch.txt at the end
#   build-binding-*  the IDENTICAL bindings the lanes import (build.sh; build_svm.sh
#                  for svc; build_estimators.sh for kde, ols, pca; build_gp.sh for
#                  gp), each with -D MOJOLEARN_GEMM_ARM_TRIAL=1 through
#                  MOJOLEARN_BUILD_EXTRA_DEFINES
#   smoke          tools/gemm_ksplit_classical_ab.py smoke --lanes <lanes>
#   wait-data      until every dataset is ready, or until only
#                  MOJOLEARN_CLASSICAL_AB_TIMING_FLOOR seconds are left, so no
#                  decode runs beside a timed process unless the lease forces it
#                  (then each such log carries FSPEED-NOTE context=bg=data_work_running)
#   <lane>.<dataset>.<arm>.<block>
#                  one timed process each, blocks in ABBA order (tuned128 then
#                  default, then default then tuned128). Group FIRST
#                  (default kde,svc) runs before every other lane; inside a group
#                  taxi runs before Istella-S. An item that cannot start with
#                  MOJOLEARN_CLASSICAL_AB_MIN_ITEM seconds left before the tail is
#                  SKIPPED_DEADLINE; one whose dataset failed is SKIPPED_NODATA.
#                  Either leaves that caller UNMEASURED, never guessed.
#   dispatch       every FSPEED-GEMM caller shape through the label mode, shipped
#                  and tuned128, on this box's build (the AMD row, not Apple's)
#   verdicts       tools/gemm_ksplit_classical_ab.py verdict: flip_verdict per
#                  caller, then `caller=<lane> verdict=HOLDS|REGRESSES|UNMEASURED|IDENTITY-BREAK`
#
# KNOBS (defaults in parentheses):
#   MOJOLEARN_CLASSICAL_AB_LANES (kde,svc,gp,ols,pca)   the callers
#   MOJOLEARN_CLASSICAL_AB_FIRST (kde,svc)              the priority group
#   MOJOLEARN_CLASSICAL_AB_DATASETS (taxi,istella)
#   MOJOLEARN_CLASSICAL_AB_ROUNDS (3)          timed rounds per block, plus one warm-up
#   MOJOLEARN_CLASSICAL_AB_BLOCKS (2)          blocks per arm; 2 measures the noise band
#   MOJOLEARN_CLASSICAL_AB_GP_TRAIN (4000), _GP_TEST (1000), _PREP (standardize)   gp only
#   MOJOLEARN_CLASSICAL_AB_DEADLINE (1200)     seconds cap per timed process
#   MOJOLEARN_CLASSICAL_AB_BODY_SECONDS (3300) body bound; the runner's work bound
#                                              minus 150 s wins when smaller
#   MOJOLEARN_CLASSICAL_AB_TIMING_FLOOR (1500) seconds left at which timing starts
#                                              even while a dataset still decodes
#   MOJOLEARN_CLASSICAL_AB_MIN_ITEM (90), _TAIL (150)   per-item floor; seconds kept
#                                              for dispatch and verdicts
#   MOJOLEARN_CLASSICAL_AB_SMOKE_ROWS (empty)  N: prep --max-rows N into <data>-smoke;
#                                              a smoke is never a timing row
#   MOJOLEARN_CLASSICAL_AB_DATA (/root/ctd-data), _BIN (/root/ksplit-ab-bin),
#   MOJOLEARN_CLASSICAL_AB_VENV (/root/ksplit-ab-venv), _OUT (/root/gemm_leg_out/ksplit-classical-amd)
#   MOJOLEARN_COMPILE_JOBS (nproc)
#
# POSIX sh only: the box's /bin/sh is dash.
set -u
HOME=${HOME:-/root}
export HOME
ROOT=${MOJOLEARN_GEMM_STEP_ROOT:-/root/mojolearn}
OUT=${MOJOLEARN_CLASSICAL_AB_OUT:-/root/gemm_leg_out/ksplit-classical-amd}
LANES=${MOJOLEARN_CLASSICAL_AB_LANES:-kde,svc,gp,ols,pca}
FIRST=${MOJOLEARN_CLASSICAL_AB_FIRST:-kde,svc}
DATASETS=${MOJOLEARN_CLASSICAL_AB_DATASETS:-taxi,istella}
ROUNDS=${MOJOLEARN_CLASSICAL_AB_ROUNDS:-3}
BLOCKS=${MOJOLEARN_CLASSICAL_AB_BLOCKS:-2}
GP_TRAIN=${MOJOLEARN_CLASSICAL_AB_GP_TRAIN:-4000}
GP_TEST=${MOJOLEARN_CLASSICAL_AB_GP_TEST:-1000}
PREP=${MOJOLEARN_CLASSICAL_AB_PREP:-standardize}
DEADLINE=${MOJOLEARN_CLASSICAL_AB_DEADLINE:-1200}
BODY_SECONDS=${MOJOLEARN_CLASSICAL_AB_BODY_SECONDS:-3300}
TIMING_FLOOR=${MOJOLEARN_CLASSICAL_AB_TIMING_FLOOR:-1500}
MIN_ITEM=${MOJOLEARN_CLASSICAL_AB_MIN_ITEM:-90}
TAIL=${MOJOLEARN_CLASSICAL_AB_TAIL:-150}
SMOKE_ROWS=${MOJOLEARN_CLASSICAL_AB_SMOKE_ROWS:-}
DATA=${MOJOLEARN_CLASSICAL_AB_DATA:-/root/ctd-data}
BIN=${MOJOLEARN_CLASSICAL_AB_BIN:-/root/ksplit-ab-bin}
VENV=${MOJOLEARN_CLASSICAL_AB_VENV:-/root/ksplit-ab-venv}
JOBS=${MOJOLEARN_COMPILE_JOBS:-}
[ -n "$JOBS" ] || JOBS=$(nproc 2>/dev/null || echo 8)
GBM_BENCH_DATA=${GBM_BENCH_DATA:-/root/datasets/gbm-bench}
export GBM_BENCH_DATA
# The Mac's copies of the two TLC months and the Istella-S tarball
# (tools/classical_two_datasets_leg.sh, 2026-09-11).
TAXI_SHA_2024_01=c4d59da7bbc8abaeeeb1727947ee93d9891a71acb42854bd80db1571b2030510
TAXI_SHA_2024_02=c76c43c18c6c6664080dd920baab4928988d5786a6b65980792ca7cd796f9f20
ISTELLA_TGZ_SHA=41b21116a3650cc043dbe16f02ee39f4467f9405b37fdbcc9a6a05e230a38981
UA="Mozilla/5.0 (X11; Linux x86_64) mojolearn-bench"
BODY_START=$(date +%s)

mkdir -p "$OUT" || exit 9
cd "$ROOT" || exit 9
export PATH="$HOME/.pixi/bin:$PATH"
export MOJOLEARN_NUMERIC_MODE=identical
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
unset MOJOLEARN_GEMM_ARM MOJOLEARN_GEMM_ARM_SABOTAGE MOJOLEARN_CLASSICAL_AB_CONTEXT

gate_fail() {  # <message>: nothing run
    echo "$1; nothing run" >> "$OUT/gate.txt"
    exit 9
}
for _list in "$LANES" "$FIRST"; do
    case "$_list" in
        *[!a-z,]*) gate_fail "MOJOLEARN_CLASSICAL_AB_LANES / _FIRST=$_list: letters and commas only" ;;
    esac
    for _l in $(echo "$_list" | tr ',' ' '); do
        case "$_l" in
            kde|svc|gp|ols|pca) ;;
            *) gate_fail "lane $_l is not kde, svc, gp, ols or pca" ;;
        esac
    done
done
[ -n "$LANES" ] || gate_fail "MOJOLEARN_CLASSICAL_AB_LANES is empty"
for _d in $(echo "$DATASETS" | tr ',' ' '); do
    case "$_d" in
        taxi|istella) ;;
        *) gate_fail "dataset $_d is not taxi or istella" ;;
    esac
done
case "$ROUNDS$BLOCKS$GP_TRAIN$GP_TEST$DEADLINE$BODY_SECONDS$TIMING_FLOOR$MIN_ITEM$TAIL$JOBS$SMOKE_ROWS" in
    *[!0-9]*) gate_fail "numeric knobs must be digits" ;;
esac
case "$PREP" in
    standardize|raw) ;;
    *) gate_fail "MOJOLEARN_CLASSICAL_AB_PREP=$PREP is not standardize or raw" ;;
esac
SMOKE_ARG=""
if [ -n "$SMOKE_ROWS" ]; then
    DATA="$DATA-smoke"
    SMOKE_ARG="--max-rows $SMOKE_ROWS"
fi
mkdir -p "$BIN" "$DATA" || gate_fail "cannot create $BIN or $DATA"

wants() {  # <lane>...: true when any is in LANES
    for _w in "$@"; do
        case ",$LANES," in *,"$_w",*) return 0 ;; esac
    done
    return 1
}
# The lanes with a classical block (prep writes theirs), and the two groups
# in LANES order.
CTD_LANES=""
FIRST_GROUP=""
REST_GROUP=""
for _l in $(echo "$LANES" | tr ',' ' '); do
    case "$_l" in ols|pca|kde|svc) CTD_LANES="${CTD_LANES:+$CTD_LANES,}$_l" ;; esac
    case ",$FIRST," in
        *,"$_l",*) FIRST_GROUP="${FIRST_GROUP:+$FIRST_GROUP,}$_l" ;;
        *) REST_GROUP="${REST_GROUP:+$REST_GROUP,}$_l" ;;
    esac
done

# ---- the vendor and the arch (one mojo build is one GPU arch) ----------------
case "${MOJOLEARN_TARGET_COLUMN:-}" in
    amd|nvidia) VENDOR=$MOJOLEARN_TARGET_COLUMN ;;
    *) if [ -e /dev/kfd ] && { command -v rocminfo || command -v rocm-smi; } > /dev/null 2>&1; then
           VENDOR=amd
       elif command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi -L > /dev/null 2>&1; then
           VENDOR=nvidia
       else
           VENDOR=unknown
       fi ;;
esac
[ "$VENDOR" = unknown ] && gate_fail "vendor=unknown: no MOJOLEARN_TARGET_COLUMN, no /dev/kfd with rocminfo, no working nvidia-smi"
export MOJOLEARN_TARGET_COLUMN="$VENDOR"
GPU_NAME=""
if [ "$VENDOR" = amd ]; then
    if [ -z "${MOJOLEARN_GPU_ARCHS:-}" ]; then
        # The agent Name: field only (a bare gfx grep also hits a stray "gfx9").
        MOJOLEARN_GPU_ARCHS=$(rocminfo 2>/dev/null | awk '$1 == "Name:" && $2 ~ /^gfx[0-9a-f]+$/ {print $2; exit}')
    fi
    GPU_NAME=$(rocminfo 2>/dev/null | awk '$1 == "Name:" && $2 ~ /^gfx[0-9a-f]+$/ {g = 1; next} g && /Marketing Name:/ {sub(/^[^:]*: */, ""); print; exit}')
else
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    if [ -z "${MOJOLEARN_GPU_ARCHS:-}" ]; then
        _cap=$(nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
        case "$_cap" in
            9.0) MOJOLEARN_GPU_ARCHS=sm_90a ;;
            [0-9].[0-9]) MOJOLEARN_GPU_ARCHS="sm_$(echo "$_cap" | tr -d .)" ;;
            *) MOJOLEARN_GPU_ARCHS="" ;;
        esac
    fi
fi
[ -n "${MOJOLEARN_GPU_ARCHS:-}" ] || gate_fail "vendor=$VENDOR gpu_archs=MISSING: one mojo build is one GPU arch; set MOJOLEARN_GPU_ARCHS"
export MOJOLEARN_GPU_ARCHS
GPU_NAME=$(printf '%s' "${GPU_NAME:-${VENDOR}_$MOJOLEARN_GPU_ARCHS}" | tr -s ' \t' '__')

gpu_snapshot() {  # <file>
    if [ "$VENDOR" = amd ]; then
        { rocm-smi --showproductname --showdriverversion --showuse --showmemuse --showtemp 2>&1 \
            || echo "rocm-smi did not answer"; } > "$1"
    else
        nvidia-smi --query-gpu=name,driver_version,uuid,clocks.sm,temperature.gpu,memory.used --format=csv > "$1" 2>&1
    fi
}

# ---- one deadline, inside the runner's work bound ------------------------------
# tools/hotaisle_leg.sh runs the body under the container's PID 1
# `timeout -k 30 <work seconds> sh /root/gemm_leg.sh`, and the remote body
# writes started= in /root/gemm_leg_out/leg.txt first. End 150 s inside that.
_work=""
for _c in /proc/1/cmdline /proc/[0-9]*/cmdline; do
    [ -r "$_c" ] || continue
    _work=$(tr '\0' ' ' < "$_c" 2>/dev/null | sed -n 's|^timeout -k [0-9]* \([0-9][0-9]*\) sh /root/gemm_leg.sh.*|\1|p')
    [ -n "$_work" ] && break
done
_started=$(sed -n 's/^started=//p' /root/gemm_leg_out/leg.txt 2>/dev/null | head -1)
_s0=""
[ -n "$_started" ] && _s0=$(date -d "$_started" +%s 2>/dev/null)
if [ -n "$_work" ] && [ -n "$_s0" ]; then
    _bound=$(( _s0 + _work - 150 - BODY_START ))
    [ "$_bound" -lt "$BODY_SECONDS" ] && BODY_SECONDS=$_bound
fi
END=$(( BODY_START + BODY_SECONDS ))

rc=0
run() {
    # run <name> <command...>: log to $OUT/<name>.log, record the exit code.
    _name=$1
    shift
    _t0=$(date +%s)
    "$@" > "$OUT/$_name.log" 2>&1
    _code=$?
    printf '%s\t%s\t%ss\n' "$_name" "$_code" "$(( $(date +%s) - _t0 ))" >> "$OUT/status.tsv"
    [ "$_code" -eq 0 ] || rc=1
    return "$_code"
}
record() {  # <name> <status> <detail>
    printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "$OUT/status.tsv"
}

{
    echo "brief=docs/lanes/BRIEF_gemm_long_k_2026-09-11.md section 12"
    echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "root=$ROOT"
    echo "lanes=$LANES first_group=${FIRST_GROUP:-none} rest_group=${REST_GROUP:-none} ctd_lanes=${CTD_LANES:-none} datasets=$DATASETS"
    echo "rounds=$ROUNDS blocks=$BLOCKS order=ABBA gp_train=$GP_TRAIN gp_test=$GP_TEST gp_prep=$PREP deadline=$DEADLINE"
    echo "body_seconds=$BODY_SECONDS runner_work_seconds=${_work:-unread} runner_started=${_started:-unread} timing_floor=$TIMING_FLOOR min_item=$MIN_ITEM tail=$TAIL"
    echo "vendor=$VENDOR gpu=$GPU_NAME gpu_archs=$MOJOLEARN_GPU_ARCHS column=$MOJOLEARN_TARGET_COLUMN jobs=$JOBS nproc=$(nproc 2>/dev/null)"
    echo "data=$DATA smoke_rows=${SMOKE_ROWS:-none} data_root=$GBM_BENCH_DATA bin=$BIN"
    [ -f /root/gemm_leg_out/leg.txt ] && grep -E '^(commit|vendor|provider|size|image|runtime)=' /root/gemm_leg_out/leg.txt
} > "$OUT/gate.txt"
: > "$OUT/status.tsv"
gpu_snapshot "$OUT/gpu_before.txt"

# ---- the raw fetches, at t=0 (curl only, no python) ------------------------------
fetch_istella() {
    mkdir -p "$GBM_BENCH_DATA/istella"
    _t0=$(date +%s)
    _tgz="$GBM_BENCH_DATA/istella/istella-s-letor.tar.gz"
    # Resumed attempts, a clean restart from the third (the MI325X got HTTP
    # 504 on a resume, tools/trees_hotaisle_body.sh).
    _i=0
    while [ "$_i" -lt 4 ] && [ "$(sha256sum "$_tgz" 2>/dev/null | cut -c1-64)" != "$ISTELLA_TGZ_SHA" ]; do
        _i=$((_i + 1))
        [ "$_i" -ge 3 ] && rm -f "$_tgz"
        timeout -k 10 1500 curl -fsSL -C - --retry 3 --retry-delay 5 -A "$UA" -o "$_tgz" \
            http://library.istella.it/dataset/istella-s-letor.tar.gz >> "$OUT/istella_curl.log" 2>&1
        echo "attempt=$_i curl_exit=$? size=$(stat -c %s "$_tgz" 2>/dev/null || echo 0) $(date -u +%T)" >> "$OUT/istella_curl.log"
    done
    _got=$(sha256sum "$_tgz" 2>/dev/null | cut -c1-64)
    # A wrong file is deleted, so the decode step fetches again or reports the miss.
    [ "$_got" = "$ISTELLA_TGZ_SHA" ] || rm -f "$_tgz"
    record fetch-istella "$( [ "$_got" = "$ISTELLA_TGZ_SHA" ] && echo 0 || echo 1)" "$(( $(date +%s) - _t0 ))s sha256=${_got:-none}"
    if [ -f "$_tgz" ]; then
        _t0=$(date +%s)
        tar -xzf "$_tgz" -C "$GBM_BENCH_DATA/istella" > "$OUT/istella_untar.log" 2>&1
        record untar-istella "$?" "$(( $(date +%s) - _t0 ))s"
    fi
    : > "$OUT/fetch_istella.done"
}
fetch_taxi() {
    mkdir -p "$GBM_BENCH_DATA/taxi"
    _t0=$(date +%s)
    _ok=1
    for _m in 2024-01 2024-02; do
        _f="$GBM_BENCH_DATA/taxi/yellow_tripdata_$_m.parquet"
        [ -f "$_f" ] || { timeout -k 10 600 curl -fsSL --retry 3 -A "$UA" -o "$_f.part" \
            "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_$_m.parquet" \
            >> "$OUT/taxi_curl.log" 2>&1 && mv "$_f.part" "$_f"; }
        rm -f "$_f.part"
        case $_m in 2024-01) _want=$TAXI_SHA_2024_01 ;; *) _want=$TAXI_SHA_2024_02 ;; esac
        if [ "$(sha256sum "$_f" 2>/dev/null | cut -c1-64)" != "$_want" ]; then
            _ok=0
            rm -f "$_f"
        fi
    done
    record fetch-taxi "$( [ "$_ok" = 1 ] && echo 0 || echo 1)" "$(( $(date +%s) - _t0 ))s"
    : > "$OUT/fetch_taxi.done"
}
for _d in $(echo "$DATASETS" | tr ',' ' '); do
    case "$_d" in
        istella) fetch_istella > "$OUT/fetch-istella.console" 2>&1 & ;;
        *) fetch_taxi > "$OUT/fetch-taxi.console" 2>&1 & ;;
    esac
done

# ---- python prerequisites --------------------------------------------------------
PY="pixi run python"
pixi run mojo --version > "$OUT/mojo_version.txt" 2>&1
if ! $PY -c 'import numpy' > "$OUT/numpy.log" 2>&1; then
    $PY -m pip install numpy >> "$OUT/numpy.log" 2>&1
fi
if ! $PY -c 'import pyarrow' > "$OUT/pyarrow.log" 2>&1; then
    $PY -m pip install pyarrow >> "$OUT/pyarrow.log" 2>&1
fi
# The decode and the prep need numpy and pyarrow; the timed processes need
# numpy and the IDENTICAL package, so they always run on the pixi python.
DPY=$PY
if ! $PY -c 'import numpy, pyarrow' >> "$OUT/pyarrow.log" 2>&1; then
    if [ ! -x "$VENV/bin/python" ]; then
        run uv-get sh -c 'curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL=/root/.local/uv sh'
        run venv /root/.local/uv/uv venv --seed --python 3.12 "$VENV"
    fi
    run venv-wheels timeout -k 30 900 "$VENV/bin/pip" install --disable-pip-version-check --no-input \
        --only-binary=:all: numpy pyarrow
    DPY="$VENV/bin/python"
fi
echo "timed_python=$PY decode_python=$DPY" >> "$OUT/gate.txt"

# ---- the decodes and the classical blocks, in the background ---------------------
data_chain() {  # <dataset>: writes <dataset>.ready, or <dataset>.failed
    while [ ! -f "$OUT/fetch_$1.done" ]; do sleep 5; done
    # shellcheck disable=SC2086
    if ! run "download-$1" timeout -k 30 2700 $DPY tools/speed_gbdt_arm.py --download "$1"; then
        : > "$OUT/$1.failed"
        return 1
    fi
    if [ -n "$CTD_LANES" ]; then
        # shellcheck disable=SC2086
        if ! run "prep-$1" timeout -k 30 1500 $DPY tools/classical_two_datasets.py prep \
                --data "$DATA" --lanes "$CTD_LANES" --datasets "$1" $SMOKE_ARG; then
            : > "$OUT/$1.failed"
            return 1
        fi
    fi
    : > "$OUT/$1.ready"
}
(
    for _d in $(echo "$DATASETS" | tr ',' ' '); do
        data_chain "$_d" &
    done
    wait
    : > "$OUT/data.done"
) > "$OUT/data.console" 2>&1 &

# ---- the label binary ------------------------------------------------------------
IDENT="-D MOJOLEARN_NUMERIC_IDENTICAL=1"
TRIAL="-D MOJOLEARN_GEMM_ARM_TRIAL=1"
rm -f "$BIN/step-price"
# shellcheck disable=SC2086
run build-price pixi run mojo build -j "$JOBS" -I . $IDENT $TRIAL \
    bench/gemm_step_price_main.mojo -o "$BIN/step-price"
: > "$OUT/plans.tsv"
if [ -x "$BIN/step-price" ]; then
    for arm in shipped tuned128; do
        env MOJOLEARN_GEMM_ARM="$arm" MOJOLEARN_GEMM_STEP_LABEL_ONLY=1 \
            timeout 120 "$BIN/step-price" < /dev/null > "$OUT/label-$arm.txt" 2>&1
        plan_label=$(sed -n 's/^PLANLABEL arm=[^ ]* label=//p' "$OUT/label-$arm.txt" | head -1)
        printf '%s\t%s\n' "$arm" "${plan_label:-unlabeled}" >> "$OUT/plans.tsv"
    done
    grep -h '^DEFAULT' "$OUT/label-shipped.txt" >> "$OUT/gate.txt" 2>/dev/null
fi
awk -F '\t' '{ print "plan " $1 "=" $2 }' "$OUT/plans.tsv" >> "$OUT/gate.txt"

# ---- the bindings, with the trial hook -------------------------------------------
SOS="python/mojolearn/identical/_mojolearn.so python/mojolearn/identical/_mojolearn_svm.so python/mojolearn/identical/_mojolearn_estimators.so python/mojolearn/identical/_mojolearn_gp.so"
# shellcheck disable=SC2086
rm -f $SOS
build_binding() {  # <name> <script>
    run "build-binding-$1" env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 \
        MOJOLEARN_COMPILE_JOBS="$JOBS" MOJOLEARN_BUILD_EXTRA_DEFINES="$TRIAL" \
        timeout -k 30 1500 sh "bindings/$2"
}
build_binding base build.sh
wants svc && build_binding svm build_svm.sh
wants kde ols pca && build_binding estimators build_estimators.sh
wants gp && build_binding gp build_gp.sh
: > "$OUT/bindings.sha256"
for so in $SOS; do
    [ -f "$so" ] || continue
    _n=unread
    command -v strings > /dev/null 2>&1 && _n=$(strings "$so" 2>/dev/null | grep -c "$MOJOLEARN_GPU_ARCHS")
    printf '%s  %s  %s_strings=%s\n' "$(sha256sum "$so" | cut -c1-64)" "$so" "$MOJOLEARN_GPU_ARCHS" "$_n" >> "$OUT/bindings.sha256"
done
# shellcheck disable=SC2086
run smoke timeout -k 30 600 $PY tools/gemm_ksplit_classical_ab.py smoke --lanes "$LANES"

# ---- wait for the data, unless the lease forces timing beside it -----------------
_w0=$(date +%s)
while [ ! -f "$OUT/data.done" ] && [ $(( END - $(date +%s) )) -gt "$TIMING_FLOOR" ]; do
    sleep 10
done
record wait-data "$( [ -f "$OUT/data.done" ] && echo 0 || echo FLOOR)" "$(( $(date +%s) - _w0 ))s"

# ---- the timed A/B ---------------------------------------------------------------
dataset_ok() {  # <dataset>: 0 when ready; waits while the item floor allows
    while [ ! -f "$OUT/$1.ready" ] && [ ! -f "$OUT/$1.failed" ] \
            && [ $(( END - TAIL - MIN_ITEM - $(date +%s) )) -gt 0 ]; do
        sleep 10
    done
    [ -f "$OUT/$1.ready" ]
}
timed() {  # <lane> <dataset> <arm name> <block>
    _item="$1.$2.$3.$4"
    _left=$(( END - TAIL - $(date +%s) ))
    if [ "$_left" -lt "$MIN_ITEM" ]; then
        record "$_item" SKIPPED_DEADLINE "0s"
        return 0
    fi
    _cap=$DEADLINE
    [ "$_left" -lt "$_cap" ] && _cap=$_left
    _env_arm=""
    _plan_arm=shipped
    if [ "$3" = tuned128 ]; then
        _env_arm=tuned128
        _plan_arm=tuned128
    fi
    _label=$(awk -F '\t' -v a="$_plan_arm" '$1 == a { print $2; exit }' "$OUT/plans.tsv" 2>/dev/null)
    _bg=quiet
    [ -f "$OUT/data.done" ] || _bg=data_work_running
    # shellcheck disable=SC2086
    run "$_item" env MOJOLEARN_GEMM_ARM="$_env_arm" MOJOLEARN_GEMM_PLAN_LABEL="${_label:-unlabeled}" \
        MOJOLEARN_CLASSICAL_AB_CONTEXT="bg=$_bg" \
        timeout -k 30 "$_cap" $PY tools/gemm_ksplit_classical_ab.py time \
        --lane "$1" --dataset "$2" --arm-name "$3" --block "$4" --rounds "$ROUNDS" \
        --source ctd --data "$DATA" --gp-train "$GP_TRAIN" --gp-test "$GP_TEST" --prep "$PREP" \
        --device "$GPU_NAME"
}

gpu_snapshot "$OUT/gpu_before_timing.txt"
for group in "$FIRST_GROUP" "$REST_GROUP"; do
    [ -n "$group" ] || continue
    for ds in $(echo "$DATASETS" | tr ',' ' '); do
        ds_ready=0
        dataset_ok "$ds" && ds_ready=1
        for lane in $(echo "$group" | tr ',' ' '); do
            b=1
            while [ "$b" -le "$BLOCKS" ]; do
                if [ $((b % 2)) -eq 1 ]; then
                    order="tuned128 default"
                else
                    order="default tuned128"
                fi
                for arm in $order; do
                    if [ "$ds_ready" = 1 ]; then
                        timed "$lane" "$ds" "$arm" "$b"
                    else
                        record "$lane.$ds.$arm.$b" SKIPPED_NODATA "0s"
                    fi
                done
                b=$((b + 1))
            done
        done
    done
done
gpu_snapshot "$OUT/gpu_after.txt"
[ -f "$OUT/data.done" ] || echo "data_chain=still_running_at_the_end_of_timing" >> "$OUT/gate.txt"

# ---- which plan each caller GEMM ran (the Mojo dispatch on this box) -------------
grep -h '^FSPEED-GEMM ' "$OUT"/*.log 2>/dev/null \
    | sed -n 's/.* caller=\([^ ]*\) op=\([A-Z]*\) m=\([0-9]*\) n=\([0-9]*\) k=\([0-9]*\).*/\1 \2 \3 \4 \5/p' \
    | sort -u > "$OUT/gemm_shapes.txt"
: > "$OUT/dispatch.txt"
if [ -x "$BIN/step-price" ]; then
    while read -r caller op m n k; do
        for arm in shipped tuned128; do
            env MOJOLEARN_GEMM_ARM="$arm" MOJOLEARN_GEMM_STEP_LABEL_ONLY=1 \
                MOJOLEARN_GEMM_STEP_LABEL_CALLER="$caller.$op" MOJOLEARN_GEMM_STEP_LABEL_M="$m" \
                MOJOLEARN_GEMM_STEP_LABEL_N="$n" MOJOLEARN_GEMM_STEP_LABEL_K="$k" \
                timeout 120 "$BIN/step-price" < /dev/null 2>&1 | grep '^DISPATCH' >> "$OUT/dispatch.txt"
        done
    done < "$OUT/gemm_shapes.txt"
    record dispatch 0 "$(wc -l < "$OUT/dispatch.txt" | tr -d ' ') lines"
else
    record dispatch 9 "no label binary"
    rc=1
fi

# ---- the verdict -----------------------------------------------------------------
# shellcheck disable=SC2086
run verdicts $PY tools/gemm_ksplit_classical_ab.py verdict --out "$OUT" --lanes "$LANES"
echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ) seconds=$(( $(date +%s) - BODY_START ))" >> "$OUT/gate.txt"
exit "$rc"
