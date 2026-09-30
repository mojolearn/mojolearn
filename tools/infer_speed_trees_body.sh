#!/bin/bash
# On-box body of lane/infer-speed-trees (2026-09-17): one NVIDIA pod rented
# through tools/trees_leg.sh, driven over ssh phase by phase.
#
#   bash tools/infer_speed_trees_body.sh setup      builds AFTER (/root/mojolearn), BEFORE
#                                                  (/root/mojolearn-before), the sabotage
#                                                  arms and the CPU-only copies
#   bash tools/infer_speed_trees_body.sh identity   six identity_break columns + diffs
#   bash tools/infer_speed_trees_body.sh identity-cuda   the three CUDA columns only
#                                                  (the CPU columns already on disk)
#   bash tools/infer_speed_trees_body.sh identity-diff   the six diffs over the JSONs on disk
#   bash tools/infer_speed_trees_body.sh identity-sw     the two score-weighted lanes as a
#                                                  separate CUDA part (setup built no
#                                                  metrics GPU binding, so the columns
#                                                  above REFUSE them); builds it, runs
#                                                  the part per CUDA tree, diffs
#   bash tools/infer_speed_trees_body.sh speed      prepare the models, then the
#                                                  alternating BEFORE/AFTER timings
#   SPECS="kind:model:x:path ..." AB_ROUNDS=15 SPEED_DIR=speed-rerun \
#     bash tools/infer_speed_trees_body.sh speed-rerun   the same, more rounds, own dir
#
# Layout on the box:
#   /root/mojolearn          AFTER source (the shipped lane HEAD), GPU bindings in
#                            python/mojolearn/identical, host families in
#                            python/mojolearn/host, the sabotage host set in
#                            python/mojolearn/host-sabotage
#   /root/mojolearn-sab      AFTER source with rf, trees and gbdt GPU bindings built
#                            under -D MOJOLEARN_FOREST_HOST_SABOTAGE=1 (the CUDA
#                            sabotage arm)
#   /root/mojolearn-before   main at the lane's base commit, same builds, no sabotage
#   /root/mojolearn-cpu, /root/mojolearn-before-cpu
#                            the same two trees with no GPU binding directory, so the
#                            package imports as a CPU-only install (the CPU columns)
#   /root/leg_out            every log and JSON, pulled home after each phase
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
LANES="rf-clf,rf-reg,et-clf,et-reg,gbdt-symmetric,gbdt-depthwise,gbdt-lossguide,gbdt-rmse,gbdt-ordered-rmse,gbdt-feature-freq,rf-clf-entropy-log2-noboot,rf-clf-balanced-parallel,rf-reg-poisson,rf-reg-gamma-ig,et-clf-entropy-bestfirst,et-reg-bootstrap-parallel,gbdt-multiclass,gbdt-onevsall,gbdt-parametric-losses,gbdt-lossguide-newtoncosine,gbdt-pointwise-l2-bayesian-eval,gbdt-exact-mae,gbdt-categorical-ctr,gbdt-categorical-ctr-tables,gbdt-tensor-ctr-tables,gbdt-nan-modes,gbdt-adapter-clf,gbdt-adapter-reg,gbdt-query-rmse,gbdt-pair-logit,gbdt-yeti-rank,gbdt-adapter-score-weighted,rf-score-weighted"
FIXTURES="base,ties,odd,dupes,wide"
# NO LONGER SKIPPED (DEVIATION 3010, 2026-09-18). These two hung the CUDA
# column at `rf-clf-balanced-parallel/base repeat=1/batch` (90 minutes in
# futex_wait, 0% CPU, 0% GPU) on the UNMODIFIED main tree, and it was never
# the batch protocol: it was the resident forest destroying its DeviceContext
# with the buffer frees its release had enqueued still in flight, which left
# the MAX runtime allocator's lock held for the whole PROCESS. The `model`
# part's reloaded estimator is what died before the batch part asked for its
# first buffer. With the drain, both lanes run the full protocol on one GPU:
# repeats 2, batch stable, 0 moved.
SKIP_CUDA="${SKIP_CUDA:-}"

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

build_tree() {
    # $1 tree, $2 label, $3 extra GPU defines (may be empty)
    _t=$1; _l=$2; _d=${3:-}
    cd "$_t" || return 1
    if [ -n "$_d" ]; then
        rm -f python/mojolearn/identical/_mojolearn_rf.so python/mojolearn/identical/_mojolearn_trees.so python/mojolearn/identical/_mojolearn_gbdt.so
    else
        step "${_l}_build_base" bash bindings/build.sh
    fi
    MOJOLEARN_EXTRA_DEFINES="$_d" step "${_l}_build_gbdt" bash bindings/build_gbdt.sh
    MOJOLEARN_EXTRA_DEFINES="$_d" step "${_l}_build_rf" bash bindings/build_rf.sh
    MOJOLEARN_EXTRA_DEFINES="$_d" step "${_l}_build_trees" bash bindings/build_trees.sh
    if [ -z "$_d" ]; then
        step "${_l}_build_metrics" bash bindings/build_metrics.sh
        for fam in core forest rf trees gbdt metrics; do
            step "${_l}_host_$fam" sh bindings/build_host_family.sh "$fam"
        done
    fi
    sha256sum python/mojolearn/identical/*.so python/mojolearn/host/*.so > "$OUT/${_l}_so_sha256.txt" 2>&1
}

phase_setup() {
    cd "$R"
    [ -x "$HOME/.pixi/bin/pixi" ] || curl -fsSL https://pixi.sh/install.sh | sh > "$OUT/pixi_get.log" 2>&1
    step pixi_install pixi install
    pixi run mojo --version > "$OUT/mojo_version.txt" 2>&1
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > "$OUT/gpu.txt"
    lscpu | grep "Model name" >> "$OUT/gpu.txt"; nproc >> "$OUT/gpu.txt"
    build_tree "$R" after ""
    # the sabotage host set: the four families under the define, the two
    # helpers copied as built (their read-back is not the control's)
    cd "$R"
    mkdir -p python/mojolearn/host-sabotage
    cp python/mojolearn/host/_mojolearn_core_host.so python/mojolearn/host/_mojolearn_metrics_host.so python/mojolearn/host-sabotage/
    for fam in forest rf trees gbdt; do
        MOJOLEARN_HOST_OUTDIR=python/mojolearn/host-sabotage MOJOLEARN_FOREST_HOST_OUTDIR=python/mojolearn/host-sabotage \
            MOJOLEARN_BUILD_EXTRA_DEFINES="-D MOJOLEARN_FOREST_HOST_SABOTAGE=1" \
            step "sab_host_$fam" sh bindings/build_host_family.sh "$fam"
    done
    sha256sum python/mojolearn/host-sabotage/*.so > "$OUT/sab_host_so_sha256.txt"
    # the CUDA sabotage arm: the AFTER tree with the three GPU bindings rebuilt
    rsync -a --exclude .pixi --exclude 'python/mojolearn/host-sabotage' "$R/" "$R-sab/"
    ln -sfn "$R/.pixi" "$R-sab/.pixi"
    build_tree "$R-sab" sab "-D MOJOLEARN_FOREST_HOST_SABOTAGE=1"
    # BEFORE: main at the base commit, shipped separately with its COMMIT file
    ln -sfn "$R/.pixi" "$B/.pixi"
    build_tree "$B" before ""
    # CPU-only copies: no GPU binding directory, the package imports as cpu
    for pair in "$R:$R-cpu" "$B:$B-cpu"; do
        src=${pair%%:*}; dst=${pair##*:}
        rsync -a --exclude .pixi --exclude 'python/mojolearn/identical' "$src/" "$dst/"
        ln -sfn "$R/.pixi" "$dst/.pixi"
        (cd "$dst" && PYTHONPATH=python pixi run python3 -c "import mojolearn; print('$dst vendor', mojolearn.vendor())") > "$OUT/cpu_probe_$(basename "$dst").log" 2>&1
    done
    (cd "$R" && PYTHONPATH=python pixi run python3 -c "import mojolearn; m = mojolearn.GradientBoosting(); print('after import', mojolearn.vendor(), m.numeric_mode_used(), m.vendor_used())") > "$OUT/after_import.log" 2>&1
    (cd "$B" && PYTHONPATH=python pixi run python3 -c "import mojolearn; m = mojolearn.GradientBoosting(); print('before import', mojolearn.vendor(), m.numeric_mode_used(), m.vendor_used())") > "$OUT/before_import.log" 2>&1
    cat "$OUT"/cpu_probe_*.log "$OUT/after_import.log" "$OUT/before_import.log"
    : > "$OUT/setup.done"
}

identity_column() {
    # $1 tree, $2 label, $3 commit, $4 extra env (string of VAR=val words)
    # A label ending in -cuda passes --skip "$SKIP_CUDA".
    _t=$1; _l=$2; _c=$3; shift 3
    cd "$_t" || return 1
    _skip=""
    case "$_l" in *-cuda) _skip="$SKIP_CUDA" ;; esac
    say "identity $_l in $_t skip=[$_skip]"
    env MOJOLEARN_COMMIT="$_c" PYTHONPATH=python "$@" pixi run python3 tools/identity_break.py \
        --lanes "$LANES" --fixtures "$FIXTURES" --repeats 2 --vendor "$_l" --skip "$_skip" \
        --json "$OUT/identity/$_l.json" > "$OUT/identity/$_l.log" 2>&1
    _rc=$?
    say "identity $_l rc=$_rc $(grep -c '^# DONE' "$OUT/identity/$_l.log") DONE cells"
    return $_rc
}

identity_cuda_columns() {
    AC=$(cat "$R/SHIPPED_COMMIT.txt")
    BC=$(cat "$B/COMMIT")
    identity_column "$B" before-cuda "$BC" MOJOLEARN_IDENTITY_HOST_INFER=0
    identity_column "$R" after-cuda "$AC" MOJOLEARN_IDENTITY_HOST_INFER=0
    identity_column "$R-sab" sabotage-cuda "$AC" MOJOLEARN_IDENTITY_HOST_INFER=0
}

identity_diffs() {
    cd "$R"
    for pair in "before-cuda after-cuda" "after-cuda sabotage-cuda" "before-cpu after-cpu" "after-cpu sabotage-cpu" "after-cuda after-cpu" "before-cuda before-cpu"; do
        set -- $pair
        PYTHONPATH=python pixi run python3 tools/identity_break.py --diff "$OUT/identity/$1.json" "$OUT/identity/$2.json" \
            > "$OUT/identity/diff.$1.vs.$2.txt" 2>&1
        say "diff $1 vs $2: $(tail -1 "$OUT/identity/diff.$1.vs.$2.txt")"
    done
}

phase_identity() {
    mkdir -p "$OUT/identity"
    AC=$(cat "$R/SHIPPED_COMMIT.txt")
    BC=$(cat "$B/COMMIT")
    # the CUDA group and the CPU group run side by side: identity is not a
    # timing, and the box has 16 cores beside the one GPU
    ( identity_cuda_columns ) &
    (
        identity_column "$B-cpu" before-cpu "$BC" MOJOLEARN_IDENTITY_HOST_INFER=1
        identity_column "$R-cpu" after-cpu "$AC" MOJOLEARN_IDENTITY_HOST_INFER=1
        identity_column "$R-cpu" sabotage-cpu "$AC" MOJOLEARN_IDENTITY_HOST_INFER=1 \
            MOJOLEARN_HOST_DIR="$R-cpu/python/mojolearn/host-sabotage" MOJOLEARN_HOST_ALLOW_SABOTAGE=1 \
            MOJOLEARN_FOREST_HOST_ALLOW_SABOTAGE=1
    ) &
    wait
    identity_diffs
    : > "$OUT/identity.done"
}

phase_identity_cuda() {
    mkdir -p "$OUT/identity"
    identity_cuda_columns
    : > "$OUT/identity-cuda.done"
}

phase_identity_diff() {
    identity_diffs
    : > "$OUT/identity.done"
}

SW_LANES="gbdt-adapter-score-weighted,rf-score-weighted"

identity_sw_part() {
    # $1 tree, $2 label, $3 commit
    _t=$1; _l=$2; _c=$3
    cd "$_t" || return 1
    [ -f python/mojolearn/identical/_mojolearn_metrics.so ] || step "${_l}_build_metrics" sh bindings/build_metrics.sh
    say "identity $_l-sw in $_t"
    env MOJOLEARN_COMMIT="$_c" PYTHONPATH=python MOJOLEARN_IDENTITY_HOST_INFER=0 pixi run python3 tools/identity_break.py \
        --lanes "$SW_LANES" --fixtures "$FIXTURES" --repeats 2 --vendor "$_l" \
        --json "$OUT/identity/$_l-sw.json" > "$OUT/identity/$_l-sw.log" 2>&1
    say "identity $_l-sw rc=$? $(grep -c '^# DONE' "$OUT/identity/$_l-sw.log") DONE cells $(grep -c '^REFUSED' "$OUT/identity/$_l-sw.log") REFUSED"
}

phase_identity_sw() {
    mkdir -p "$OUT/identity"
    AC=$(cat "$R/SHIPPED_COMMIT.txt")
    BC=$(cat "$B/COMMIT")
    identity_sw_part "$B" before-cuda "$BC"
    identity_sw_part "$R" after-cuda "$AC"
    identity_sw_part "$R-sab" sabotage-cuda "$AC"
    sha256sum "$R"/python/mojolearn/identical/_mojolearn_metrics.so "$B"/python/mojolearn/identical/_mojolearn_metrics.so \
        "$R-sab"/python/mojolearn/identical/_mojolearn_metrics.so > "$OUT/metrics_so_sha256.txt" 2>&1
    cd "$R"
    for pair in "before-cuda-sw after-cuda-sw" "after-cuda-sw sabotage-cuda-sw" "after-cuda-sw after-cpu" "before-cuda-sw before-cpu"; do
        set -- $pair
        PYTHONPATH=python pixi run python3 tools/identity_break.py --diff "$OUT/identity/$1.json" "$OUT/identity/$2.json" \
            > "$OUT/identity/diff.$1.vs.$2.txt" 2>&1
        say "diff $1 vs $2: $(tail -1 "$OUT/identity/diff.$1.vs.$2.txt")"
    done
    : > "$OUT/identity-sw.done"
}

SPEED_DIR="${SPEED_DIR:-speed}"
AB_ROUNDS="${AB_ROUNDS:-5}"

time_one() {
    # $1 tree, $2 label, $3 kind, $4 model, $5 x, $6 path, $7 index
    _t=$1; _l=$2; _k=$3; _m=$4; _x=$5; _p=$6; _i=$7
    cd "$_t" || return 1
    PYTHONPATH=python pixi run python3 bench/speed/infer_speed_trees_ab.py time \
        --model "$_m" --x "$_x" --kind "$_k" --path "$_p" --rounds "$AB_ROUNDS" --label "$_l" \
        --json "$OUT/$SPEED_DIR/$(basename "$_m" .npz).$_p.$_l.$_i.json" 2>&1 | tail -1
}

phase_speed_rerun() {
    # The cells whose five-round process failed the 1.10 spread gate, run
    # again with AB_ROUNDS rounds into their own directory (SPEED_DIR), so
    # the summary of that directory stands on its own processes.
    mkdir -p "$OUT/$SPEED_DIR"
    for spec in $SPECS; do
        IFS=: read -r kind model x path <<EOF
$spec
EOF
        [ -f "$AB/$model" ] || { say "no $AB/$model"; continue; }
        for i in 1 2; do
            say "$(time_one "$B" before "$kind" "$AB/$model" "$AB/$x" "$path" "$i")"
            say "$(time_one "$R" after "$kind" "$AB/$model" "$AB/$x" "$path" "$i")"
        done
    done
    cd "$R"
    PYTHONPATH=python pixi run python3 bench/speed/infer_speed_trees_ab.py summarize "$OUT/$SPEED_DIR/*.*.*.*.json" \
        --out "$OUT/$SPEED_DIR/summary.json" > "$OUT/$SPEED_DIR/summary.txt" 2>&1
    : > "$OUT/$SPEED_DIR.done"
}

phase_speed() {
    mkdir -p "$OUT/speed"
    cd "$R"
    if [ ! -f "$AB/manifest.json" ]; then
        step prepare env PYTHONPATH=python pixi run python3 bench/speed/infer_speed_trees_ab.py prepare \
            --out "$AB" --rows "${AB_ROWS:-1000000}" --train-rows "${AB_TRAIN_ROWS:-1000000}" \
            --trees 100 --depth 16 --gbdt-iterations "${AB_GBDT_ITERS:-100,1000}"
        cp "$AB/manifest.json" "$OUT/speed/manifest.json"
    fi
    # alternating processes: before, after, before, after, per path
    for spec in "rf-reg:rf-reg-100x16.npz:x_taxireg.npy:gpu-predict" \
                "rf-reg:rf-reg-100x16.npz:x_taxireg.npy:host-predict" \
                "et-reg:et-reg-100x16.npz:x_taxireg.npy:gpu-predict" \
                "et-reg:et-reg-100x16.npz:x_taxireg.npy:host-predict" \
                "gbdt:gbdt-logloss-1000.npz:x_taxi.npy:host-predict" \
                "gbdt:gbdt-logloss-1000.npz:x_taxi.npy:host-proba" \
                "gbdt:gbdt-logloss-1000.npz:x_taxi.npy:gpu-predict" \
                "gbdt:gbdt-logloss-1000.npz:x_taxi.npy:gpu-proba" \
                "gbdt:gbdt-logloss-1000.npz:x_taxi.npy:gbdt-parse" \
                "gbdt:gbdt-logloss-100.npz:x_taxi.npy:host-predict" \
                "gbdt:gbdt-logloss-100.npz:x_taxi.npy:gpu-proba" \
                "gbdt:gbdt-logloss-100.npz:x_taxi.npy:gbdt-parse"; do
        IFS=: read -r kind model x path <<EOF
$spec
EOF
        [ -f "$AB/$model" ] || { say "no $AB/$model"; continue; }
        for i in 1 2; do
            say "$(time_one "$B" before "$kind" "$AB/$model" "$AB/$x" "$path" "$i")"
            say "$(time_one "$R" after "$kind" "$AB/$model" "$AB/$x" "$path" "$i")"
        done
    done
    cd "$R"
    PYTHONPATH=python pixi run python3 bench/speed/infer_speed_trees_ab.py summarize "$OUT/speed/*.*.*.*.json" \
        --out "$OUT/speed/summary.json" > "$OUT/speed/summary.txt" 2>&1
    tail -5 "$OUT/speed/summary.txt"
    : > "$OUT/speed.done"
}

case "${1:-}" in
    setup) phase_setup ;;
    identity) phase_identity ;;
    identity-cuda) phase_identity_cuda ;;
    identity-diff) phase_identity_diff ;;
    identity-sw) phase_identity_sw ;;
    speed) phase_speed ;;
    speed-rerun) phase_speed_rerun ;;
    *) sed -n '2,12p' "$0"; exit 2 ;;
esac
