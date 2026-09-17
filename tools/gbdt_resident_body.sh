#!/bin/bash
# On-box body of lane/gbdt-resident-predict (2026-09-17, DEVIATION 2980): one
# NVIDIA pod rented through tools/trees_leg.sh, driven over ssh phase by phase.
# The pattern is tools/infer_speed_trees_body.sh's.
#
#   bash tools/gbdt_resident_body.sh setup       builds AFTER (/root/mojolearn), the CUDA
#                                                sabotage arm (/root/mojolearn-sab, the gbdt
#                                                binding under -D MOJOLEARN_GBDT_RESIDENT_SABOTAGE=1),
#                                                BEFORE (/root/mojolearn-before, main at the lane's
#                                                base commit, shipped separately with a COMMIT file)
#                                                and the CPU-only copies; installs catboost
#   bash tools/gbdt_resident_body.sh build-after rebuilds only the AFTER gbdt binding and the
#                                                sabotage arm (after a `push`)
#   bash tools/gbdt_resident_body.sh identity    five identity_break columns over every gbdt-*
#                                                lane, five fixtures, two repeats, plus the diffs
#   bash tools/gbdt_resident_body.sh identity-diff   the diffs over the JSONs on disk
#   bash tools/gbdt_resident_body.sh speed       prepare the models, then the interleaved
#                                                resident / per-call timings
#   bash tools/gbdt_resident_body.sh catboost    CatBoost GPU and CPU predict on the same rows
#   bash tools/gbdt_resident_body.sh summarize   the speed tables
#
# Layout on the box: /root/mojolearn (AFTER), /root/mojolearn-sab, /root/mojolearn-before,
# /root/mojolearn-cpu and /root/mojolearn-before-cpu (no GPU binding directory, so the
# package imports as a CPU-only install), /root/ab (models and rows), /root/leg_out
# (every log and JSON, pulled home after each phase).
set -u
R=/root/mojolearn
B=/root/mojolearn-before
OUT=/root/leg_out
AB=/root/ab
mkdir -p "$OUT" "$AB"
export PATH="$HOME/.pixi/bin:$PATH"
export MOJOLEARN_NUMERIC_MODE=identical
export MOJOLEARN_COMPILE_JOBS=${MOJOLEARN_COMPILE_JOBS:-16}
export MOJOLEARN_BUILD_JOBS=${MOJOLEARN_BUILD_JOBS:-16}
export GBM_BENCH_DATA=/root/datasets/gbm-bench
export PYTHONUNBUFFERED=1
LANES="${LANES:-gbdt-symmetric,gbdt-depthwise,gbdt-lossguide,gbdt-rmse,gbdt-ordered-rmse,gbdt-feature-freq,gbdt-multiclass,gbdt-onevsall,gbdt-parametric-losses,gbdt-lossguide-newtoncosine,gbdt-pointwise-l2-bayesian-eval,gbdt-exact-mae,gbdt-categorical-ctr,gbdt-categorical-ctr-tables,gbdt-tensor-ctr-tables,gbdt-nan-modes,gbdt-adapter-clf,gbdt-adapter-reg,gbdt-query-rmse,gbdt-pair-logit,gbdt-yeti-rank,gbdt-adapter-score-weighted}"
FIXTURES="${FIXTURES:-base,ties,odd,dupes,wide}"
SAB_DEFINE="-D MOJOLEARN_GBDT_RESIDENT_SABOTAGE=1"

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
    if [ -z "$_d" ]; then
        step "${_l}_build_base" bash bindings/build.sh
        step "${_l}_build_metrics" bash bindings/build_metrics.sh
    else
        rm -f python/mojolearn/identical/_mojolearn_gbdt.so
    fi
    MOJOLEARN_EXTRA_DEFINES="$_d" step "${_l}_build_gbdt" bash bindings/build_gbdt.sh
    sha256sum python/mojolearn/identical/*.so > "$OUT/${_l}_so_sha256.txt" 2>&1
}

build_host() {
    _t=$1; _l=$2
    cd "$_t" || return 1
    for fam in core forest rf trees gbdt metrics; do
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
    build_gpu "$R" after ""
    build_host "$R" after
    # the CUDA sabotage arm: the AFTER tree with the gbdt binding rebuilt
    rsync -a --exclude .pixi "$R/" "$R-sab/"
    ln -sfn "$R/.pixi" "$R-sab/.pixi"
    build_gpu "$R-sab" sab "$SAB_DEFINE"
    # BEFORE: main at the base commit, shipped separately with its COMMIT file
    if [ -d "$B" ]; then
        ln -sfn "$R/.pixi" "$B/.pixi"
        build_gpu "$B" before ""
        build_host "$B" before
    else
        say "no $B: the BEFORE tree was not shipped"
    fi
    # CPU-only copies: no GPU binding directory, the package imports as cpu
    for pair in "$R:$R-cpu" "$B:$B-cpu"; do
        src=${pair%%:*}; dst=${pair##*:}
        [ -d "$src" ] || continue
        rsync -a --exclude .pixi --exclude 'python/mojolearn/identical' "$src/" "$dst/"
        ln -sfn "$R/.pixi" "$dst/.pixi"
        (cd "$dst" && PYTHONPATH=python pixi run python3 -c "import mojolearn; print('$dst vendor', mojolearn.vendor())") > "$OUT/cpu_probe_$(basename "$dst").log" 2>&1
    done
    (cd "$R" && PYTHONPATH=python pixi run python3 -c "import mojolearn; m = mojolearn.GradientBoosting(); b = m._bind('_mojolearn_gbdt'); print('after import', mojolearn.vendor(), m.numeric_mode_used(), m.vendor_used(), 'resident door', hasattr(b, 'gbdt_resident_prepare'))") > "$OUT/after_import.log" 2>&1
    (cd "$B" && PYTHONPATH=python pixi run python3 -c "import mojolearn; m = mojolearn.GradientBoosting(); b = m._bind('_mojolearn_gbdt'); print('before import', mojolearn.vendor(), m.numeric_mode_used(), m.vendor_used(), 'resident door', hasattr(b, 'gbdt_resident_prepare'))") > "$OUT/before_import.log" 2>&1
    cat "$OUT"/cpu_probe_*.log "$OUT/after_import.log" "$OUT/before_import.log"
    cd "$R"
    (pixi run python3 -m pip install --quiet catboost || (pixi run python3 -m ensurepip && pixi run python3 -m pip install --quiet catboost)) > "$OUT/pip_catboost.log" 2>&1
    pixi run python3 -c "import catboost; print('catboost', catboost.__version__)" >> "$OUT/pip_catboost.log" 2>&1
    tail -1 "$OUT/pip_catboost.log"
    : > "$OUT/setup.done"
}

phase_build_after() {
    cd "$R"
    MOJOLEARN_EXTRA_DEFINES="" step after_build_gbdt bash bindings/build_gbdt.sh
    sha256sum python/mojolearn/identical/*.so > "$OUT/after_so_sha256.txt" 2>&1
    rsync -a --exclude .pixi --exclude 'python/mojolearn/identical' --exclude 'python/mojolearn/host' "$R/" "$R-sab/"
    rsync -a --exclude .pixi --exclude 'python/mojolearn/identical' --exclude 'python/mojolearn/host' "$R/" "$R-cpu/"
    build_gpu "$R-sab" sab "$SAB_DEFINE"
    (cd "$R" && PYTHONPATH=python pixi run python3 -c "import mojolearn; m = mojolearn.GradientBoosting(); b = m._bind('_mojolearn_gbdt'); print('after import', mojolearn.vendor(), m.numeric_mode_used(), m.vendor_used(), 'resident door', hasattr(b, 'gbdt_resident_prepare'))") > "$OUT/after_import.log" 2>&1
    cat "$OUT/after_import.log"
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
    mkdir -p "$OUT/identity"
    AC=$(cat "$R/SHIPPED_COMMIT.txt")
    BC=$(cat "$B/COMMIT")
    # the CUDA group and the CPU group side by side: identity is not a
    # timing, and the box has 16 cores beside the one GPU
    (
        identity_column "$B" before-cuda "$BC" MOJOLEARN_IDENTITY_HOST_INFER=0
        identity_column "$R" after-cuda "$AC" MOJOLEARN_IDENTITY_HOST_INFER=0
        identity_column "$R-sab" sabotage-cuda "$AC" MOJOLEARN_IDENTITY_HOST_INFER=0
    ) &
    (
        identity_column "$B-cpu" before-cpu "$BC" MOJOLEARN_IDENTITY_HOST_INFER=1
        identity_column "$R-cpu" after-cpu "$AC" MOJOLEARN_IDENTITY_HOST_INFER=1
    ) &
    wait
    identity_diffs
    : > "$OUT/identity.done"
}

phase_identity_diff() {
    identity_diffs
    : > "$OUT/identity.done"
}

AB_ROUNDS="${AB_ROUNDS:-5}"
SPEED_DIR="${SPEED_DIR:-speed}"
DATASETS="${DATASETS:-taxi,taxireg,higgs,covtype}"

time_one() {
    # $1 model key, $2 path, $3 calls, $4 order, $5 index
    _m=$1; _p=$2; _c=$3; _o=$4; _i=$5
    cd "$R" || return 1
    _x=$(PYTHONPATH=python pixi run python3 -c "import json; m = json.load(open('$AB/manifest.json'))['models']['$_m']; print(m['x'])")
    _y=$(PYTHONPATH=python pixi run python3 -c "import json; m = json.load(open('$AB/manifest.json'))['models']['$_m']; print(m['y'])")
    PYTHONPATH=python pixi run python3 bench/speed/gbdt_resident_ab.py time \
        --model "$AB/$_m.npz" --x "$_x" --y "$_y" --path "$_p" --rounds "$AB_ROUNDS" --calls "$_c" --order "$_o" \
        --label "after" --json "$OUT/$SPEED_DIR/$_m.$_p.c$_c.$_o.$_i.json" 2>&1 | tail -1
}

model_keys() {
    cd "$R" && PYTHONPATH=python pixi run python3 -c "import json; print(' '.join(sorted(json.load(open('$AB/manifest.json'))['models'])))"
}

phase_speed() {
    mkdir -p "$OUT/$SPEED_DIR"
    cd "$R"
    if [ ! -f "$AB/manifest.json" ]; then
        step prepare env PYTHONPATH=python pixi run python3 bench/speed/gbdt_resident_ab.py prepare \
            --out "$AB" --rows "${AB_ROWS:-1000000}" --train-rows "${AB_TRAIN_ROWS:-1000000}" \
            --gbdt-iterations "${AB_GBDT_ITERS:-100,1000}" --datasets "$DATASETS" --catboost
        cp "$AB/manifest.json" "$OUT/$SPEED_DIR/manifest.json"
    fi
    for key in $(model_keys); do
        case "$key" in *rmse*) paths="predict" ;; *) paths="predict proba" ;; esac
        for p in $paths; do
            for i in 1 2; do
                say "$(time_one "$key" "$p" 1 C "$i")"
            done
            say "$(time_one "$key" "$p" 8 C 1)"
        done
        # the F-order input: what the call costs without the per-call transpose
        say "$(time_one "$key" predict 1 F 1)"
    done
    : > "$OUT/$SPEED_DIR.done"
}

phase_catboost() {
    mkdir -p "$OUT/$SPEED_DIR"
    cd "$R"
    for key in $(model_keys); do
        _cb=$(PYTHONPATH=python pixi run python3 -c "import json; m = json.load(open('$AB/manifest.json'))['models']['$key']; print(m.get('catboost', {}).get('path', ''))")
        [ -n "$_cb" ] || { say "no catboost model for $key"; continue; }
        _x=$(PYTHONPATH=python pixi run python3 -c "import json; m = json.load(open('$AB/manifest.json'))['models']['$key']; print(m['x'])")
        _y=$(PYTHONPATH=python pixi run python3 -c "import json; m = json.load(open('$AB/manifest.json'))['models']['$key']; print(m['y'])")
        case "$key" in *rmse*) paths="predict" ;; *) paths="predict proba" ;; esac
        for p in $paths; do
            for c in 1 8; do
                PYTHONPATH=python pixi run python3 bench/speed/gbdt_resident_ab.py catboost \
                    --model "$_cb" --x "$_x" --y "$_y" --path "$p" --rounds "$AB_ROUNDS" --calls "$c" \
                    --devices "${CB_DEVICES:-gpu,cpu,cpu1}" --label catboost \
                    --json "$OUT/$SPEED_DIR/catboost-$key.$p.c$c.json" > "$OUT/$SPEED_DIR/catboost-$key.$p.c$c.log" 2>&1
                say "catboost $key $p c$c: $(grep '^catboost' "$OUT/$SPEED_DIR/catboost-$key.$p.c$c.log" | tr '\n' ' ')"
            done
        done
    done
    : > "$OUT/catboost.done"
}

phase_summarize() {
    cd "$R"
    PYTHONPATH=python pixi run python3 bench/speed/gbdt_resident_ab.py summarize "$OUT/$SPEED_DIR"/*.json \
        --out "$OUT/$SPEED_DIR/summary.json" > "$OUT/$SPEED_DIR/summary.txt" 2>&1
    tail -3 "$OUT/$SPEED_DIR/summary.txt"
}

case "${1:-}" in
    setup) phase_setup ;;
    build-after) phase_build_after ;;
    identity) phase_identity ;;
    identity-diff) phase_identity_diff ;;
    speed) phase_speed ;;
    catboost) phase_catboost ;;
    summarize) phase_summarize ;;
    *) sed -n '2,24p' "$0"; exit 2 ;;
esac
