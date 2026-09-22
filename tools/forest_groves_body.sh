#!/bin/bash
# On-box body of lane/forest-groves-cpu-and-speed (2026-09-17): one NVIDIA pod
# rented through tools/trees_leg.sh, driven over ssh phase by phase.
#
#   bash tools/forest_groves_body.sh setup      pixi, AFTER (/root/mojolearn) GPU rf/trees/
#                                              metrics bindings + host families, the two
#                                              forest sabotage host sets, BEFORE
#                                              (/root/mojolearn-before, main at the base
#                                              commit) and the CPU-only copies
#   bash tools/forest_groves_body.sh variants   the kernel/glue candidates: AFTER copies
#                                              with rf+trees rebuilt under one define each
#                                              (pinned, packed, shared, profile, and the
#                                              combination named in COMBO)
#   bash tools/forest_groves_body.sh identity   identity_break: before/after cuda, before/
#                                              after/sabotage cpu (host_model route), diffs
#   bash tools/forest_groves_body.sh groves     tools/forest_groves_identity.py: GPU groves
#                                              vs host groves on every rf and et lane, the
#                                              production host set and both sabotage sets,
#                                              then the HIGGS and Covtype sized models
#   bash tools/forest_groves_body.sh speed      tools/forest_groves_speed.py prepare, then
#                                              alternating processes over every arm tree
#   bash tools/forest_groves_body.sh fil        cuML FIL on the system python (installs
#                                              cuml-cu12 and treelite on first use)
#
# Layout on the box: /root/mojolearn (AFTER), /root/mojolearn-before (BEFORE),
# /root/mojolearn-cpu and -before-cpu (no GPU binding dir), /root/mojolearn-v-<name>
# (variants), /root/leg_out (every log and JSON, pulled home after each phase),
# /root/ab (saved models and fixed prediction rows).
set -u
# The four roots are overridable so a second lane can share a pod whose
# /root/mojolearn, /root/leg_out and /root/ab belong to another lane.
R=${FG_R:-/root/mojolearn}
B=${FG_B:-/root/mojolearn-before}
OUT=${FG_OUT:-/root/leg_out}
AB=${FG_AB:-/root/ab}
mkdir -p "$OUT" "$AB"
export PATH="$HOME/.pixi/bin:$PATH"
export MOJOLEARN_NUMERIC_MODE=identical
export MOJOLEARN_COMPILE_JOBS=${MOJOLEARN_COMPILE_JOBS:-16}
export MOJOLEARN_BUILD_JOBS=${MOJOLEARN_BUILD_JOBS:-16}
export GBM_BENCH_DATA=/root/datasets/gbm-bench
export PYTHONUNBUFFERED=1
LANES="rf-clf,rf-reg,et-clf,et-reg,rf-clf-entropy-log2-noboot,rf-clf-balanced-parallel,rf-reg-poisson,rf-reg-gamma-ig,et-clf-entropy-bestfirst,et-reg-bootstrap-parallel,rf-score-weighted"
FIXTURES="base,ties,odd,dupes,wide"
# The two parallel lanes hung the one-GPU box on main (lane/infer-speed-trees,
# 2026-09-17). FIXED by DEVIATION 3010 on 2026-09-18 and no longer skipped:
# the cause was the resident forest's teardown, not the batch protocol.
SKIP_CUDA="${SKIP_CUDA:-}"
VARIANTS="${VARIANTS:-pinned:-D MOJOLEARN_FOREST_PINNED_STAGE=1|packed:-D MOJOLEARN_FOREST_PACKED_NODES=1|shared:-D MOJOLEARN_FOREST_SHARED_ROWS=1|profile:-D MOJOLEARN_FOREST_PROFILE=1}"

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
    # $1 tree, $2 label, $3 extra GPU defines (may be empty); rf and trees only
    _t=$1; _l=$2; _d=${3:-}
    cd "$_t" || return 1
    rm -f python/mojolearn/identical/_mojolearn_rf.so python/mojolearn/identical/_mojolearn_trees.so
    MOJOLEARN_EXTRA_DEFINES="$_d" step "${_l}_build_rf" bash bindings/build_rf.sh
    MOJOLEARN_EXTRA_DEFINES="$_d" step "${_l}_build_trees" bash bindings/build_trees.sh
    sha256sum python/mojolearn/identical/*.so > "$OUT/${_l}_so_sha256.txt" 2>&1
}

build_full() {
    # $1 tree, $2 label: base + metrics GPU bindings, rf/trees, host families
    _t=$1; _l=$2
    cd "$_t" || return 1
    step "${_l}_build_base" bash bindings/build.sh
    step "${_l}_build_metrics" bash bindings/build_metrics.sh
    build_gpu "$_t" "$_l" ""
    for fam in core forest rf trees metrics; do
        step "${_l}_host_$fam" sh bindings/build_host_family.sh "$fam"
    done
    sha256sum python/mojolearn/identical/*.so python/mojolearn/host/*.so > "$OUT/${_l}_so_sha256.txt" 2>&1
}

clone_tree() {
    # $1 src, $2 dst: source copy without the GPU binding dir, .pixi linked
    rsync -a --exclude .pixi --exclude 'python/mojolearn/identical' --exclude 'python/mojolearn/host-*' "$1/" "$2/"
    ln -sfn "$R/.pixi" "$2/.pixi"
}

phase_setup() {
    cd "$R"
    [ -x "$HOME/.pixi/bin/pixi" ] || curl -fsSL https://pixi.sh/install.sh | sh > "$OUT/pixi_get.log" 2>&1
    step pixi_install pixi install
    pixi run mojo --version > "$OUT/mojo_version.txt" 2>&1
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > "$OUT/gpu.txt"
    lscpu | grep "Model name" >> "$OUT/gpu.txt"; nproc >> "$OUT/gpu.txt"
    ls -la /root/datasets/gbm-bench/*/ >> "$OUT/gpu.txt" 2>&1
    build_full "$R" after
    # the sabotage host sets: the forest family under each control, the
    # other families copied as built
    cd "$R"
    for pair in "host-sabotage:-D MOJOLEARN_FOREST_HOST_SABOTAGE=1" "host-groves-sabotage:-D MOJOLEARN_FOREST_GROVES_SABOTAGE=1"; do
        dir=${pair%%:*}; def=${pair#*:}
        mkdir -p "python/mojolearn/$dir"
        for fam in core rf trees metrics; do
            cp "python/mojolearn/host/_mojolearn_${fam}_host.so" "python/mojolearn/$dir/"
        done
        MOJOLEARN_FOREST_HOST_OUTDIR="python/mojolearn/$dir" MOJOLEARN_BUILD_EXTRA_DEFINES="$def" \
            step "sab_${dir}_forest" sh bindings/build_host_family.sh forest
        sha256sum "python/mojolearn/$dir"/*.so > "$OUT/${dir}_so_sha256.txt"
    done
    # BEFORE: main at the base commit, shipped separately with its COMMIT file
    ln -sfn "$R/.pixi" "$B/.pixi"
    build_full "$B" before
    # CPU-only copies: no GPU binding directory, the package imports as cpu
    for pair in "$R:$R-cpu" "$B:$B-cpu"; do
        src=${pair%%:*}; dst=${pair##*:}
        clone_tree "$src" "$dst"
        [ -d "$src/python/mojolearn/host-sabotage" ] && cp -r "$src/python/mojolearn/host-sabotage" "$src/python/mojolearn/host-groves-sabotage" "$dst/python/mojolearn/"
        (cd "$dst" && PYTHONPATH=python pixi run python3 -c "import mojolearn; print('$dst vendor', mojolearn.vendor())") > "$OUT/cpu_probe_$(basename "$dst").log" 2>&1
    done
    (cd "$R" && PYTHONPATH=python pixi run python3 -c "import mojolearn; print('after import', mojolearn.vendor())") > "$OUT/after_import.log" 2>&1
    (cd "$B" && PYTHONPATH=python pixi run python3 -c "import mojolearn; print('before import', mojolearn.vendor())") > "$OUT/before_import.log" 2>&1
    cat "$OUT"/cpu_probe_*.log "$OUT/after_import.log" "$OUT/before_import.log"
    : > "$OUT/setup.done"
}

phase_setup_lite() {
    # AFTER only, no BEFORE tree and no sabotage host sets: for a lane whose
    # default build IS the before arm and whose candidates are variants.
    cd "$R"
    pixi run mojo --version > "$OUT/mojo_version.txt" 2>&1
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > "$OUT/gpu.txt"
    lscpu | grep "Model name" >> "$OUT/gpu.txt"; nproc >> "$OUT/gpu.txt"
    build_full "$R" after
    (cd "$R" && PYTHONPATH=python pixi run python3 -c "import mojolearn; print('after import', mojolearn.vendor())") > "$OUT/after_import.log" 2>&1
    cat "$OUT/after_import.log"
    : > "$OUT/setup_lite.done"
}

phase_variants() {
    # VARIANTS="name:defines|name:defines"; each an AFTER copy with rf+trees rebuilt
    echo "$VARIANTS" | tr '|' '\n' | while IFS=: read -r name defs; do
        [ -n "$name" ] || continue
        dst="$R-v-$name"
        clone_tree "$R" "$dst"
        mkdir -p "$dst/python/mojolearn/identical"
        cp "$R"/python/mojolearn/identical/*.so "$dst/python/mojolearn/identical/"
        ln -sfn "$R/python/mojolearn/host" "$dst/python/mojolearn/host"
        build_gpu "$dst" "v-$name" "$defs"
    done
    : > "$OUT/variants.done"
}

identity_column() {
    # $1 tree, $2 label, $3 commit, then extra env words
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

phase_identity() {
    mkdir -p "$OUT/identity"
    AC=$(cat "$R/SHIPPED_COMMIT.txt")
    BC=$(cat "$B/COMMIT")
    ( identity_column "$B" before-cuda "$BC" MOJOLEARN_IDENTITY_HOST_INFER=0
      identity_column "$R" after-cuda "$AC" MOJOLEARN_IDENTITY_HOST_INFER=0 ) &
    ( identity_column "$B-cpu" before-cpu "$BC" MOJOLEARN_IDENTITY_HOST_INFER=1
      identity_column "$R-cpu" after-cpu "$AC" MOJOLEARN_IDENTITY_HOST_INFER=1
      identity_column "$R-cpu" sabotage-cpu "$AC" MOJOLEARN_IDENTITY_HOST_INFER=1 \
          MOJOLEARN_HOST_DIR="$R-cpu/python/mojolearn/host-sabotage" MOJOLEARN_HOST_ALLOW_SABOTAGE=1 \
          MOJOLEARN_FOREST_HOST_ALLOW_SABOTAGE=1
      identity_column "$R-cpu" groves-sabotage-cpu "$AC" MOJOLEARN_IDENTITY_HOST_INFER=1 \
          MOJOLEARN_HOST_DIR="$R-cpu/python/mojolearn/host-groves-sabotage" MOJOLEARN_HOST_ALLOW_SABOTAGE=1 \
          MOJOLEARN_FOREST_HOST_ALLOW_SABOTAGE=1 ) &
    wait
    cd "$R"
    for pair in "before-cuda after-cuda" "before-cpu after-cpu" "after-cuda after-cpu" "after-cpu sabotage-cpu" "after-cpu groves-sabotage-cpu" "before-cuda before-cpu"; do
        set -- $pair
        PYTHONPATH=python pixi run python3 tools/identity_break.py --diff "$OUT/identity/$1.json" "$OUT/identity/$2.json" \
            > "$OUT/identity/diff.$1.vs.$2.txt" 2>&1
        say "diff $1 vs $2: $(tail -1 "$OUT/identity/diff.$1.vs.$2.txt")"
    done
    : > "$OUT/identity.done"
}

phase_groves() {
    mkdir -p "$OUT/groves"
    cd "$R"
    # the production host set, then the two sabotage sets, on every rf and et
    # lane's five fixtures; the GPU groves engine is the reference column
    for pair in "production:$R/python/mojolearn/host" "sabotage:$R/python/mojolearn/host-sabotage" "groves-sabotage:$R/python/mojolearn/host-groves-sabotage"; do
        label=${pair%%:*}; dir=${pair#*:}
        step "groves_$label" env PYTHONPATH=python MOJOLEARN_HOST_DIR="$dir" MOJOLEARN_HOST_ALLOW_SABOTAGE=1 \
            MOJOLEARN_FOREST_HOST_ALLOW_SABOTAGE=1 pixi run python3 tools/forest_groves_identity.py lanes \
            --fixtures "$FIXTURES" --json "$OUT/groves/$label.json"
        tail -3 "$OUT/groves_$label.log"
    done
    # HIGGS and Covtype sized models, saved once under $AB by the speed
    # prepare (or here when speed has not run), production set only
    step groves_large env PYTHONPATH=python pixi run python3 tools/forest_groves_identity.py large \
        --models-dir "$AB" --json "$OUT/groves/large.json"
    tail -3 "$OUT/groves_large.log"
    : > "$OUT/groves.done"
}

SPEED_DIR="${SPEED_DIR:-speed}"
AB_ROUNDS="${AB_ROUNDS:-7}"

time_one() {
    # $1 tree, $2 label, $3 model, $4 x, $5 path, $6 index
    _t=$1; _l=$2; _m=$3; _x=$4; _p=$5; _i=$6
    cd "$_t" || return 1
    _stem="$OUT/$SPEED_DIR/$(basename "$_m" .npz).$_p.$_l.$_i"
    # the host walks (16 threads) are single-call context, not the A/B
    _calls=8; case "$_p" in groves) ;; *) _calls=1 ;; esac
    PYTHONPATH=python pixi run python3 tools/forest_groves_speed.py time \
        --model "$_m" --x "$_x" --path "$_p" --rounds "$AB_ROUNDS" --calls "$_calls" --label "$_l" \
        --json "$_stem.json" --save-prediction "$OUT/$SPEED_DIR" > "$_stem.log" 2>&1
    tail -1 "$_stem.log"
}

phase_speed() {
    mkdir -p "$OUT/$SPEED_DIR"
    cd "$R"
    if [ ! -f "$AB/manifest.json" ]; then
        step prepare env PYTHONPATH=python pixi run python3 tools/forest_groves_speed.py prepare --out "$AB" --models "${PREPARE_MODELS:-}"
        cp "$AB/manifest.json" "$OUT/$SPEED_DIR/manifest.json"
    fi
    # ARMS="label:tree ..." default: before, after and every variant built
    ARMS="${ARMS:-before:$B after:$R $(for d in "$R"-v-*; do [ -d "$d" ] && printf 'v-%s:%s ' "${d##*-v-}" "$d"; done)}"
    MODELS="${MODELS:-rf-higgs-100x16 et-higgs-100x16 rf-covtype-100x16 et-year-100x16 rf-higgs-500x16}"
    for model in $MODELS; do
        x=$(PYTHONPATH=python pixi run python3 -c "import json;print(json.load(open('$AB/manifest.json'))['models']['$model']['x'])")
        [ -f "$AB/$model.npz" ] || { say "no $AB/$model.npz"; continue; }
        for i in 1 2; do
            for arm in $ARMS; do
                label=${arm%%:*}; tree=${arm#*:}
                [ "$label" = v-profile ] && continue
                say "$(time_one "$tree" "$label" "$AB/$model.npz" "$AB/$x" groves "$i")"
            done
        done
        # the host groves door and the sequential GPU-class engine, AFTER only,
        # one process each, single calls; not for the 500-tree model
        case "$model" in *-500x*) continue ;; esac
        say "$(time_one "$R" after "$AB/$model.npz" "$AB/$x" host-groves 1)"
        say "$(time_one "$R" after "$AB/$model.npz" "$AB/$x" sequential 1)"
    done
    # the stage profile, one process per model, AFTER's profile variant
    if [ -d "$R-v-profile" ]; then
        for model in $MODELS; do
            x=$(PYTHONPATH=python pixi run python3 -c "import json;print(json.load(open('$AB/manifest.json'))['models']['$model']['x'])")
            [ -f "$AB/$model.npz" ] || continue
            say "$(time_one "$R-v-profile" v-profile "$AB/$model.npz" "$AB/$x" groves 1)"
        done
    fi
    cd "$R"
    PYTHONPATH=python pixi run python3 tools/forest_groves_speed.py summarize "$OUT/$SPEED_DIR/*.*.*.*.json" \
        --out "$OUT/$SPEED_DIR/summary.json" > "$OUT/$SPEED_DIR/summary.txt" 2>&1
    tail -40 "$OUT/$SPEED_DIR/summary.txt"
    : > "$OUT/$SPEED_DIR.done"
}

phase_fil() {
    mkdir -p "$OUT/fil"
    cd "$R"
    if ! /usr/bin/python3 -c "import cuml, treelite" 2>/dev/null; then
        step fil_pip /usr/bin/python3 -m pip install --quiet "cuml-cu12==26.8.*" treelite
    fi
    /usr/bin/python3 -c "import cuml, treelite; print('cuml', cuml.__version__, 'treelite', treelite.__version__)" > "$OUT/fil/versions.txt" 2>&1
    cat "$OUT/fil/versions.txt"
    MODELS="${MODELS:-rf-higgs-100x16 et-higgs-100x16 rf-covtype-100x16 et-year-100x16 rf-higgs-500x16}"
    for model in $MODELS; do
        [ -f "$AB/$model.npz" ] || continue
        x=$(/usr/bin/python3 -c "import json;print(json.load(open('$AB/manifest.json'))['models']['$model']['x'])")
        step "fil_$model" /usr/bin/python3 tools/forest_groves_fil.py --model "$AB/$model.npz" --x "$AB/$x" \
            --rounds "$AB_ROUNDS" --calls 8 --json "$OUT/fil/$model.json" --ours "$OUT/$SPEED_DIR"
        tail -2 "$OUT/fil_$model.log"
    done
    : > "$OUT/fil.done"
}

phase_final() {
    # After the A/B chose the packed layout as the default: rebuild AFTER's
    # rf and trees bindings from the pushed sources, run the small NVIDIA
    # layout matrix (both layouts, IDENTICAL), rerun the after-cuda identity
    # column and the groves column against the new build, then the AFTER
    # timing processes (and BEFORE where a process is missing).
    cd "$R"
    build_gpu "$R" after2 ""
    mkdir -p "$OUT/layouts"
    for layout in packed_default separate_arrays; do
        defs="-D MOJOLEARN_NUMERIC_IDENTICAL=1"
        [ "$layout" = separate_arrays ] && defs="$defs -D MOJOLEARN_FOREST_SEPARATE_NODES=1"
        stem="$OUT/layouts/$layout"
        # shellcheck disable=SC2086
        step "layout_build_$layout" pixi run mojo build -I . $defs checks/forest_inference_model.mojo -o "$stem"
        step "layout_run_$layout" "$stem"
        tail -2 "$OUT/layout_run_$layout.log"
    done
    mkdir -p "$OUT/identity" "$OUT/groves"
    AC=$(cat "$R/SHIPPED_COMMIT.txt")
    identity_column "$R" after2-cuda "$AC" MOJOLEARN_IDENTITY_HOST_INFER=0
    for pair in "before-cuda after2-cuda" "after2-cuda after-cpu"; do
        set -- $pair
        PYTHONPATH=python pixi run python3 tools/identity_break.py --diff "$OUT/identity/$1.json" "$OUT/identity/$2.json" \
            > "$OUT/identity/diff.$1.vs.$2.txt" 2>&1
        say "diff $1 vs $2: $(grep '^summary' "$OUT/identity/diff.$1.vs.$2.txt" | tr '\n' ' ')"
    done
    step groves2_production env PYTHONPATH=python pixi run python3 tools/forest_groves_identity.py lanes \
        --fixtures "$FIXTURES" --json "$OUT/groves/production2.json"
    tail -1 "$OUT/groves2_production.log"
    step groves2_large env PYTHONPATH=python pixi run python3 tools/forest_groves_identity.py large \
        --models-dir "$AB" --json "$OUT/groves/large2.json"
    tail -1 "$OUT/groves2_large.log"
    MODELS="${MODELS:-rf-higgs-100x16 et-higgs-100x16 rf-covtype-100x16 et-year-100x16 rf-higgs-500x16}"
    for model in $MODELS; do
        x=$(PYTHONPATH=python pixi run python3 -c "import json;print(json.load(open('$AB/manifest.json'))['models']['$model']['x'])")
        for i in 1 2; do
            [ -f "$OUT/$SPEED_DIR/$model.groves.before.$i.json" ] || say "$(time_one "$B" before "$AB/$model.npz" "$AB/$x" groves "$i")"
            say "$(time_one "$R" after2 "$AB/$model.npz" "$AB/$x" groves "$i")"
        done
    done
    cd "$R"
    PYTHONPATH=python pixi run python3 tools/forest_groves_speed.py summarize "$OUT/$SPEED_DIR/*.*.*.*.json" \
        --out "$OUT/$SPEED_DIR/summary.json" > "$OUT/$SPEED_DIR/summary.txt" 2>&1
    : > "$OUT/final.done"
}

case "${1:-}" in
    setup) phase_setup ;;
    setup-lite) phase_setup_lite ;;
    final) phase_final ;;
    variants) phase_variants ;;
    identity) phase_identity ;;
    groves) phase_groves ;;
    speed) phase_speed ;;
    fil) phase_fil ;;
    *) sed -n '2,24p' "$0"; exit 2 ;;
esac
