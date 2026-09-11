#!/bin/sh
# tools/classical_hotaisle_leg.sh -- DEVIATION 2572, the classical opponent
# rows (DEVIATION 2570) on a Hot Aisle AMD MI300X VM (ENGINEERING_RULES.md
# sections 9 and 10; MI300X rows are a new tuple). A tools/hotaisle_leg.sh
# body: runs ON THE BOX from /root/mojolearn after pixi install and the gates.
#
# One lease runs, in this order, under ONE deadline (MOJOLEARN_CTD_BODY_SECONDS
# from the body's start; a race that cannot fit is SKIPPED_DEADLINE in
# status.tsv, and a second lease runs only what was skipped):
#   1. setup  env record, curl fetches at t=0, the opponents' venv (torch ROCm,
#             scikit-learn), decodes, and every IDENTICAL binding the six
#             lanes need
#   2. smoke  prep + races of all six lanes on both datasets at 20,000 rows,
#             1 round (never a timing row; a harness bug costs minutes)
#   3. prep   the full blocks of all six lanes
#   4. leg 1  races kmeans, pca, ols, knn on taxi then Istella-S
#   5. leg 2  races kde, svc on taxi then Istella-S
# MOJOLEARN_CTD_HOTAISLE_STAGE=leg2 skips the smoke and leg 1 (a second lease).
#
# Arms on AMD: ours IDENTICAL; scikit-learn on every CPU core (CPU); torch ROCm
# (GPU) for kmeans, pca, ols (lstsq and eigh) and knn. cuML has no ROCm path.
#
# Env values reach the box through MOJOLEARN_HOTAISLE_EXTRA_ENV, which allows
# no spaces: lists are commas. POSIX sh only.
set -u
L=/root/mojolearn/tools/classical_two_datasets_leg.sh
STAGE=${MOJOLEARN_CTD_HOTAISLE_STAGE:-all}
MOJOLEARN_CTD_BODY_START=$(date +%s)
MOJOLEARN_CTD_BODY_SECONDS=${MOJOLEARN_CTD_BODY_SECONDS:-3000}
MOJOLEARN_CTD_OUT=${MOJOLEARN_CTD_OUT:-/root/gemm_leg_out/classical-hotaisle}
MOJOLEARN_COMPILE_JOBS=${MOJOLEARN_COMPILE_JOBS:-$(nproc)}
export MOJOLEARN_CTD_BODY_START MOJOLEARN_CTD_BODY_SECONDS MOJOLEARN_CTD_OUT MOJOLEARN_COMPILE_JOBS
ALL=kmeans,pca,ols,knn,kde,svc
mkdir -p "$MOJOLEARN_CTD_OUT"
echo "stage=$STAGE body_seconds=$MOJOLEARN_CTD_BODY_SECONDS start=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$MOJOLEARN_CTD_OUT/hotaisle_body.txt"
step() {  # <label> <env words>...: one call of the leg file, console kept
    _label=$1
    shift
    echo "step=$_label begin=$(date -u +%H:%M:%S)" >> "$MOJOLEARN_CTD_OUT/hotaisle_body.txt"
    env "$@" sh "$L" > "$MOJOLEARN_CTD_OUT/console-$_label.log" 2>&1
    echo "step=$_label rc=$? end=$(date -u +%H:%M:%S)" >> "$MOJOLEARN_CTD_OUT/hotaisle_body.txt"
}

step setup MOJOLEARN_CTD_LANES=$ALL MOJOLEARN_CTD_PHASES=setup
if [ "$STAGE" = all ]; then
    step smoke MOJOLEARN_CTD_LANES=$ALL MOJOLEARN_CTD_PHASES=prep,races MOJOLEARN_CTD_SMOKE_ROWS=20000 \
        MOJOLEARN_CTD_MIN_RACE=60
    step prep MOJOLEARN_CTD_LANES=$ALL MOJOLEARN_CTD_PHASES=prep
    step leg1 MOJOLEARN_CTD_LANES=kmeans,pca,ols,knn MOJOLEARN_CTD_PHASES=races
else
    step prep MOJOLEARN_CTD_LANES=kde,svc MOJOLEARN_CTD_PHASES=prep
fi
step leg2 MOJOLEARN_CTD_LANES=kde,svc MOJOLEARN_CTD_PHASES=races
echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$MOJOLEARN_CTD_OUT/hotaisle_body.txt"
exit 0
