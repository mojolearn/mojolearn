#!/bin/bash
# On-box body of lane/forest-train-speed (2026-09-17): one NVIDIA pod rented
# through tools/trees_leg.sh, driven over ssh phase by phase. The shape is
# tools/forest_groves_body.sh's.
#
#   bash tools/forest_train_body.sh setup      pixi, the GPU base/metrics/rf/trees bindings
#                                             and the core/forest/rf/trees/metrics host
#                                             families in /root/mojolearn
#   bash tools/forest_train_body.sh profile    the attribution: MOJOLEARN_STAGE_TIMES=1 fits
#                                             and (when nsys is present) an nsys kernel
#                                             summary, rf and et on taxi and istella
#   bash tools/forest_train_body.sh fast       the FAST tier's rf and trees bindings beside them
#   bash tools/forest_train_body.sh variants   /root/mojolearn-v-<name>: a source copy with
#                                             rf+trees rebuilt under one define set each
#                                             (VARIANTS="name:defines|name:defines")
#   bash tools/forest_train_body.sh identity   identity_break columns named in COLUMNS
#                                             ("label:tree label:tree"), then DIFFS
#                                             ("a:b a:c")
#   bash tools/forest_train_body.sh ab         alternating processes over ARMS
#                                             ("label:tree label:tree"), CELLS
#                                             ("lane:dataset:rows ..."), AB_ROUNDS passes
#
# Everything lands in /root/leg_out, pulled home after each phase.
set -u
R=/root/mojolearn
OUT=/root/leg_out
mkdir -p "$OUT"
export PATH="$HOME/.pixi/bin:$PATH"
export MOJOLEARN_NUMERIC_MODE=${MOJOLEARN_NUMERIC_MODE:-identical}
export MOJOLEARN_COMPILE_JOBS=${MOJOLEARN_COMPILE_JOBS:-16}
export MOJOLEARN_BUILD_JOBS=${MOJOLEARN_BUILD_JOBS:-16}
export GBM_BENCH_DATA=/root/datasets/gbm-bench
export PYTHONUNBUFFERED=1
LANES="rf-clf,rf-reg,et-clf,et-reg,rf-clf-entropy-log2-noboot,rf-reg-poisson,rf-reg-gamma-ig,et-clf-entropy-bestfirst,rf-score-weighted"
FIXTURES="base,ties,odd,dupes,wide"
CELLS="${CELLS:-et:taxi:4000000 et:istella:2000000 rf:taxi:4000000 rf:istella:2000000}"

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
    # $1 tree, $2 label, $3 extra defines (may be empty); the families named
    # in VARIANT_FAMILIES (default "rf trees"), each binding removed first so a
    # failed build cannot leave the copied one in place
    _t=$1; _l=$2; _d=${3:-}
    cd "$_t" || return 1
    for fam in ${VARIANT_FAMILIES:-rf trees}; do
        rm -f "python/mojolearn/identical/_mojolearn_$fam.so"
        MOJOLEARN_EXTRA_DEFINES="$_d" step "${_l}_build_$fam" bash "bindings/build_$fam.sh"
    done
    sha256sum python/mojolearn/identical/*.so > "$OUT/${_l}_so_sha256.txt" 2>&1
    # the mtime check the brief asks for: binding against newest forest source
    { ls -l --time-style=full-iso python/mojolearn/identical/_mojolearn_rf.so python/mojolearn/identical/_mojolearn_trees.so
      find ensemble extratrees bindings -name '*.mojo' -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' | sort | tail -3
    } > "$OUT/${_l}_mtimes.txt" 2>&1
    echo "$_d" > "$_t/VARIANT_DEFINES.txt"
}

clone_tree() {
    rsync -a --exclude .pixi --exclude 'python/mojolearn/identical' "$1/" "$2/"
    ln -sfn "$R/.pixi" "$2/.pixi"
}

phase_setup() {
    cd "$R"
    [ -x "$HOME/.pixi/bin/pixi" ] || curl -fsSL https://pixi.sh/install.sh | sh > "$OUT/pixi_get.log" 2>&1
    step pixi_install pixi install
    pixi run mojo --version > "$OUT/mojo_version.txt" 2>&1
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > "$OUT/gpu.txt"
    lscpu | grep "Model name" >> "$OUT/gpu.txt"; nproc >> "$OUT/gpu.txt"
    ls -la /root/datasets/gbm-bench/*/ >> "$OUT/gpu.txt" 2>&1
    step main_build_base bash bindings/build.sh
    step main_build_metrics bash bindings/build_metrics.sh
    build_gpu "$R" main ""
    for fam in core forest rf trees metrics; do
        step "main_host_$fam" sh bindings/build_host_family.sh "$fam"
    done
    (cd "$R" && PYTHONPATH=python pixi run python3 -c "import mojolearn; print('import', mojolearn.vendor())") > "$OUT/import.log" 2>&1
    cat "$OUT/import.log"
    which nsys > "$OUT/nsys_which.txt" 2>&1 || ls /usr/local/cuda*/bin/nsys /opt/nvidia/nsight-systems/*/bin/nsys >> "$OUT/nsys_which.txt" 2>&1
    : > "$OUT/setup.done"
}

find_nsys() {
    for c in nsys /usr/local/cuda/bin/nsys /opt/nvidia/nsight-systems/*/bin/nsys /usr/local/cuda-*/bin/nsys; do
        if command -v "$c" > /dev/null 2>&1; then command -v "$c"; return 0; fi
    done
    return 1
}

phase_profile() {
    # PROFILE_TREE: the checkout to profile (default main's)
    _t="${PROFILE_TREE:-$R}"; _l="${PROFILE_LABEL:-main}"
    export MOJOLEARN_NUMERIC_MODE="${PROFILE_MODE:-identical}"
    mkdir -p "$OUT/profile"
    cd "$_t" || return 1
    for cell in $CELLS; do
        IFS=: read -r lane ds rows <<< "$cell"
        stem="$OUT/profile/$_l.$lane.$ds.$rows"
        # 1. untimed: the honest wall time beside the serialized table
        PYTHONPATH=python pixi run python3 tools/forest_train_ab.py fit --lane "$lane" --dataset "$ds" \
            --rows "$rows" --rounds "${PROFILE_ROUNDS:-3}" ${PROFILE_TREES:+--trees $PROFILE_TREES} --label "$_l" --json "$stem.plain.json" > "$stem.plain.log" 2>&1
        say "plain $cell: $(grep '^FTRAIN ' "$stem.plain.log" | tr '\n' ' ' | cut -c1-300)"
        # 2. the stage table (drains per stage; attribution, never a timing)
        MOJOLEARN_STAGE_TIMES=1 PYTHONPATH=python pixi run python3 tools/forest_train_ab.py fit --lane "$lane" \
            --dataset "$ds" --rows "$rows" --rounds 1 ${PROFILE_TREES:+--trees $PROFILE_TREES} --label "$_l-staged" --json "$stem.staged.json" > "$stem.staged.log" 2>&1
        say "staged $cell done"
        # 3. nsys, when the box has it: per-kernel GPU time, CUDA API time, memcpy
        if NSYS=$(find_nsys); then
            "$NSYS" profile -t cuda,osrt -s none --force-overwrite true -o "$stem.nsys" \
                env PYTHONPATH=python pixi run python3 tools/forest_train_ab.py fit --lane "$lane" --dataset "$ds" \
                --rows "$rows" --rounds 1 ${PROFILE_TREES:+--trees $PROFILE_TREES} --label "$_l-nsys" --json "$stem.nsys.json" > "$stem.nsys.log" 2>&1
            "$NSYS" stats --force-export true -r cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
                --format csv -o "$stem.nsysstats" "$stem.nsys.nsys-rep" > "$stem.nsysstats.log" 2>&1
            say "nsys $cell done"
        fi
    done
    : > "$OUT/profile.$_l.done"
}

phase_fast() {
    # The FAST tier's rf and trees bindings beside the IDENTICAL ones (the
    # FAST build lands in python/mojolearn/, MOJOLEARN_NUMERIC_MODE=fast
    # selects it at import). FAST_TREE names the checkout (default main's).
    _t="${FAST_TREE:-$R}"
    cd "$_t" || return 1
    MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_SKIP_BUILD_GATE=1 step "fast_build_rf_$(basename "$_t")" bash bindings/build_rf.sh
    MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_SKIP_BUILD_GATE=1 step "fast_build_trees_$(basename "$_t")" bash bindings/build_trees.sh
    ls -l --time-style=full-iso python/mojolearn/_mojolearn_rf*.so python/mojolearn/_mojolearn_trees*.so > "$OUT/fast_so_$(basename "$_t").txt" 2>&1
    : > "$OUT/fast.done"
}

phase_variants() {
    # every variant builds in its own copy, in parallel (VARIANT_JOBS each)
    IFS='|' read -r -a _vs <<< "${VARIANTS:-}"
    for v in "${_vs[@]}"; do
        name=${v%%:*}; defs=${v#*:}
        [ -n "$name" ] || continue
        (
            dst="$R-v-$name"
            clone_tree "${VARIANT_SRC:-$R}" "$dst"
            mkdir -p "$dst/python/mojolearn/identical"
            cp -n "$R"/python/mojolearn/identical/*.so "$dst/python/mojolearn/identical/"
            MOJOLEARN_COMPILE_JOBS="${VARIANT_JOBS:-16}" build_gpu "$dst" "v-$name" "$defs"
        ) &
    done
    wait
    : > "$OUT/variants.done"
}

identity_column() {
    # $1 tree, $2 label, then extra env words
    _t=$1; _l=$2; shift 2
    cd "$_t" || return 1
    say "identity $_l in $_t"
    env MOJOLEARN_COMMIT="$(cat "$_t/SHIPPED_COMMIT.txt" 2>/dev/null || echo unknown)" PYTHONPATH=python "$@" \
        pixi run python3 tools/identity_break.py \
        --lanes "$LANES" --fixtures "$FIXTURES" --repeats 2 --vendor "$_l" \
        --json "$OUT/identity/$_l.json" > "$OUT/identity/$_l.log" 2>&1
    _rc=$?
    say "identity $_l rc=$_rc $(grep -c '^# DONE' "$OUT/identity/$_l.log") DONE cells"
    return $_rc
}

phase_identity() {
    mkdir -p "$OUT/identity"
    for col in ${COLUMNS:-}; do
        label=${col%%:*}; tree=${col#*:}
        case "$label" in
            *-cpu) identity_column "$tree" "$label" MOJOLEARN_IDENTITY_HOST_INFER=1 ;;
            *)     identity_column "$tree" "$label" MOJOLEARN_IDENTITY_HOST_INFER=0 ;;
        esac
    done
    cd "$R"
    for pair in ${DIFFS:-}; do
        a=${pair%%:*}; b=${pair#*:}
        PYTHONPATH=python pixi run python3 tools/identity_break.py --diff "$OUT/identity/$a.json" "$OUT/identity/$b.json" \
            > "$OUT/identity/diff.$a.vs.$b.txt" 2>&1
        say "diff $a vs $b: $(grep -i '^summary\|IDENTICAL\|DIVERGENT' "$OUT/identity/diff.$a.vs.$b.txt" | tail -2 | tr '\n' ' ')"
    done
    : > "$OUT/identity.done"
}

phase_ab() {
    _dir="$OUT/${AB_DIR:-ab}"
    mkdir -p "$_dir"
    for cell in $CELLS; do
        IFS=: read -r lane ds rows <<< "$cell"
        for i in $(seq 1 "${AB_ROUNDS:-5}"); do
            for arm in ${ARMS:-}; do
                # label:tree or label:tree:mode (mode defaults to identical)
                IFS=: read -r label tree mode <<< "$arm"
                stem="$_dir/$lane.$ds.$rows.$label.$i"
                ( cd "$tree" && MOJOLEARN_NUMERIC_MODE="${mode:-identical}" PYTHONPATH=python pixi run python3 tools/forest_train_ab.py fit --lane "$lane" \
                    --dataset "$ds" --rows "$rows" --rounds "${AB_FITS:-2}" ${AB_TREES:+--trees $AB_TREES} --label "$label" ${AB_SCORE:+--score} \
                    --json "$stem.json" > "$stem.log" 2>&1 )
                say "$cell $label $i: $(grep '^FTRAIN ' "$stem.log" | sed 's/.*round=/r/' | tr '\n' ' ')"
            done
        done
    done
    cd "$R"
    PYTHONPATH=python pixi run python3 tools/forest_train_ab.py summarize "$_dir/*.json" \
        --before "${AB_BEFORE:-before}" --after "${AB_AFTER:-after}" --out "$_dir/summary.json" > "$_dir/summary.txt" 2>&1
    cat "$_dir/summary.txt"
    : > "$_dir.done"
}

case "${1:-}" in
    setup) phase_setup ;;
    profile) phase_profile ;;
    fast) phase_fast ;;
    variants) phase_variants ;;
    identity) phase_identity ;;
    ab) phase_ab ;;
    *) sed -n '2,22p' "$0"; exit 2 ;;
esac
