#!/bin/bash
# On-box body of lane/gbdt-train-speed (2026-09-17, DEVIATION 3040-3059): one
# NVIDIA pod rented through tools/trees_leg.sh, driven over ssh phase by phase.
# The pattern is tools/gbdt_resident_body.sh's.
#
#   bash tools/gbdt_train_body.sh setup        builds /root/mojolearn (shipped at main), copies it to
#                                              /root/mojolearn-before (the BEFORE arm, never rebuilt)
#                                              and to the CPU-only copies; installs catboost, xgboost
#   bash tools/gbdt_train_body.sh build-after  rebuilds the AFTER gbdt binding (GPU and host family)
#                                              after a `push`, then the sabotage arm in /root/mojolearn-sab
#   bash tools/gbdt_train_body.sh identity     identity_break columns (before/after/sabotage cuda,
#                                              before/after cpu) over LANES, five fixtures, two repeats
#   bash tools/gbdt_train_body.sh identity-diff
#
# Layout on the box: /root/mojolearn (AFTER), /root/mojolearn-sab, /root/mojolearn-before,
# /root/mojolearn-cpu and /root/mojolearn-before-cpu, /root/leg_out (every log and JSON).
set -u
R=/root/mojolearn
B=/root/mojolearn-before
OUT=/root/leg_out
mkdir -p "$OUT"
export PATH="$HOME/.pixi/bin:$PATH"
export MOJOLEARN_NUMERIC_MODE=identical
export MOJOLEARN_COMPILE_JOBS=${MOJOLEARN_COMPILE_JOBS:-16}
export MOJOLEARN_BUILD_JOBS=${MOJOLEARN_BUILD_JOBS:-16}
export GBM_BENCH_DATA=/root/datasets/gbm-bench
export PYTHONUNBUFFERED=1
LANES="${LANES:-gbdt-symmetric,gbdt-depthwise,gbdt-lossguide,gbdt-rmse,gbdt-ordered-rmse,gbdt-feature-freq,gbdt-multiclass,gbdt-onevsall,gbdt-parametric-losses,gbdt-lossguide-newtoncosine,gbdt-pointwise-l2-bayesian-eval,gbdt-exact-mae,gbdt-categorical-ctr,gbdt-categorical-ctr-tables,gbdt-tensor-ctr-tables,gbdt-nan-modes,gbdt-adapter-clf,gbdt-adapter-reg,gbdt-query-rmse,gbdt-pair-logit,gbdt-yeti-rank,gbdt-adapter-score-weighted}"
FIXTURES="${FIXTURES:-base,ties,odd,dupes,wide}"
SAB_DEFINE="${SAB_DEFINE:--D MOJOLEARN_GBDT_YETI_SABOTAGE=1}"

say() { printf '[%s body] %s\n' "$(date +%T)" "$*"; }
step() {
    _name=$1; shift
    say "$_name: $*"
    ( "$@" ) > "$OUT/$_name.log" 2>&1
    _rc=$?
    say "$_name rc=$_rc"
    [ "$_rc" -eq 0 ] || echo "$_name rc=$_rc" >> "$OUT/failed.txt"
    return $_rc
}

build_gpu() {
    # $1 tree, $2 label, $3 extra gbdt defines (may be empty)
    _t=$1; _l=$2; _d=${3:-}
    cd "$_t" || return 1
    if [ -z "$_d" ] && [ "${ONLY_GBDT:-0}" != "1" ]; then
        step "${_l}_build_base" bash bindings/build.sh
        step "${_l}_build_metrics" bash bindings/build_metrics.sh
    else
        rm -f python/mojolearn/identical/_mojolearn_gbdt.so
    fi
    MOJOLEARN_EXTRA_DEFINES="$_d" step "${_l}_build_gbdt" bash bindings/build_gbdt.sh
    sha256sum python/mojolearn/identical/*.so > "$OUT/${_l}_so_sha256.txt" 2>&1
}

build_host() {
    _t=$1; _l=$2; shift 2
    cd "$_t" || return 1
    for fam in "$@"; do
        step "${_l}_host_$fam" sh bindings/build_host_family.sh "$fam"
    done
    sha256sum python/mojolearn/host/*.so > "$OUT/${_l}_host_so_sha256.txt" 2>&1
}

phase_setup() {
    cd "$R"
    [ -x "$HOME/.pixi/bin/pixi" ] || curl -fsSL https://pixi.sh/install.sh | sh > "$OUT/pixi_get.log" 2>&1
    step pixi_install pixi install
    pixi run mojo --version > "$OUT/mojo_version.txt" 2>&1
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > "$OUT/gpu.txt"
    lscpu | grep "Model name" >> "$OUT/gpu.txt"; nproc >> "$OUT/gpu.txt"
    command -v rsync > /dev/null || (apt-get update > /dev/null 2>&1; apt-get install -y rsync > "$OUT/apt_rsync.log" 2>&1)
    build_gpu "$R" main ""
    build_host "$R" main core forest rf trees gbdt metrics
    # BEFORE is this build, frozen: the shipped commit is the lane's base
    rsync -a --exclude .pixi "$R/" "$B/"
    ln -sfn "$R/.pixi" "$B/.pixi"
    cp "$R/SHIPPED_COMMIT.txt" "$B/COMMIT"
    rsync -a --exclude .pixi --exclude 'python/mojolearn/identical' "$B/" "$B-cpu/"
    ln -sfn "$R/.pixi" "$B-cpu/.pixi"
    (cd "$R" && PYTHONPATH=python pixi run python3 -c "import mojolearn; m = mojolearn.GradientBoosting(); print('main import', mojolearn.vendor(), m.numeric_mode_used(), m.vendor_used())") > "$OUT/main_import.log" 2>&1
    cat "$OUT/main_import.log"
    cd "$R"
    (pixi run python3 -m pip install --quiet catboost xgboost scikit-learn || (pixi run python3 -m ensurepip && pixi run python3 -m pip install --quiet catboost xgboost scikit-learn)) > "$OUT/pip_opponents.log" 2>&1
    pixi run python3 -c "import catboost, xgboost, sklearn; print('catboost', catboost.__version__, 'xgboost', xgboost.__version__, 'sklearn', sklearn.__version__)" >> "$OUT/pip_opponents.log" 2>&1
    tail -1 "$OUT/pip_opponents.log"
    : > "$OUT/setup.done"
}

phase_build_after() {
    rm -f "$OUT/build_after.done"
    cd "$R"
    ONLY_GBDT=1 build_gpu "$R" after ""
    build_host "$R" after gbdt
    rsync -a --exclude .pixi "$R/" "$R-sab/"
    ln -sfn "$R/.pixi" "$R-sab/.pixi"
    build_gpu "$R-sab" sab "$SAB_DEFINE"
    rsync -a --exclude .pixi --exclude 'python/mojolearn/identical' "$R/" "$R-cpu/"
    ln -sfn "$R/.pixi" "$R-cpu/.pixi"
    ls -l --time-style=full-iso "$R/python/mojolearn/identical/_mojolearn_gbdt.so" "$R-sab/python/mojolearn/identical/_mojolearn_gbdt.so" \
        "$B/python/mojolearn/identical/_mojolearn_gbdt.so" > "$OUT/after_mtimes.txt" 2>&1
    find "$R/gbdt" "$R/bindings/_mojolearn_gbdt.mojo" -name '*.mojo' -newer "$R/python/mojolearn/identical/_mojolearn_gbdt.so" >> "$OUT/after_mtimes.txt" 2>&1
    cat "$OUT/after_mtimes.txt"
    : > "$OUT/build_after.done"
}

identity_column() {
    # $1 tree, $2 label, $3 commit, then env words
    _t=$1; _l=$2; _c=$3; shift 3
    cd "$_t" || return 1
    say "identity $_l in $_t"
    env MOJOLEARN_COMMIT="$_c" PYTHONPATH=python "$@" pixi run python3 tools/identity_break.py \
        --lanes "$LANES" --fixtures "$FIXTURES" --repeats 2 --vendor "$_l" \
        --json "$OUT/identity/$_l.json" > "$OUT/identity/$_l.log" 2>&1
    _rc=$?
    say "identity $_l rc=$_rc $(grep -c '^# DONE' "$OUT/identity/$_l.log") DONE cells $(grep -c '^REFUSED' "$OUT/identity/$_l.log") REFUSED"
    return $_rc
}

identity_diffs() {
    cd "$R"
    for pair in "before-cuda after-cuda" "after-cuda sabotage-cuda" "before-cpu after-cpu" "after-cuda after-cpu" "before-cuda before-cpu"; do
        set -- $pair
        [ -f "$OUT/identity/$1.json" ] && [ -f "$OUT/identity/$2.json" ] || continue
        PYTHONPATH=python pixi run python3 tools/identity_break.py --diff "$OUT/identity/$1.json" "$OUT/identity/$2.json" \
            > "$OUT/identity/diff.$1.vs.$2.txt" 2>&1
        say "diff $1 vs $2: $(grep '^summary' "$OUT/identity/diff.$1.vs.$2.txt" | tr '\n' ' ')"
    done
}

phase_identity() {
    rm -f "$OUT/identity.done"
    mkdir -p "$OUT/identity"
    AC=$(cat "$R/SHIPPED_COMMIT.txt")
    BC=$(cat "$B/COMMIT")
    (
        [ "${SKIP_BEFORE:-0}" = "1" ] || identity_column "$B" before-cuda "$BC" MOJOLEARN_IDENTITY_HOST_INFER=0
        identity_column "$R" after-cuda "$AC" MOJOLEARN_IDENTITY_HOST_INFER=0
        identity_column "$R-sab" sabotage-cuda "$AC" MOJOLEARN_IDENTITY_HOST_INFER=0
    ) &
    (
        [ "${SKIP_BEFORE:-0}" = "1" ] || identity_column "$B-cpu" before-cpu "$BC" MOJOLEARN_IDENTITY_HOST_INFER=1
        identity_column "$R-cpu" after-cpu "$AC" MOJOLEARN_IDENTITY_HOST_INFER=1
    ) &
    wait
    identity_diffs
    : > "$OUT/identity.done"
}

case "${1:-}" in
    setup) phase_setup ;;
    build-after) phase_build_after ;;
    identity) phase_identity ;;
    identity-diff) identity_diffs ;;
    *) sed -n '2,16p' "$0"; exit 2 ;;
esac
