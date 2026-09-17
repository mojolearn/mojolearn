#!/bin/sh
# lane/knn-selector-speed, 2026-09-17: the on-pod sequence (RunPod NVIDIA,
# CUDA 13 driver). Run ON THE POD from /root/mojolearn (the branch's source,
# shipped and then pushed by tools/trees_leg.sh). POSIX sh. The shape is
# tools/knn_tiled_body.sh's. Stages:
#
#   sh tools/knn_selector_body.sh setup            pixi, the two kNN blocks from the staged R2 npz,
#                                                  /root/mojolearn-base (a copy of the source AS SHIPPED,
#                                                  which is main), base core + estimators + host sets
#   sh tools/knn_selector_body.sh arm NAME "DEFS"  one core binding of the CURRENT source with DEFS,
#                                                  into /root/gpubins/NAME and the tree /root/t-NAME
#   sh tools/knn_selector_body.sh basearm NAME "DEFS"   the same from /root/mojolearn-base
#   sh tools/knn_selector_body.sh probe TAG ARM [ENV=VALUE ...]   tools/knn_selector_probe.py on /root/t-ARM
#   sh tools/knn_selector_body.sh race TAG "ARM,ARM,..." "KS" "ROWS"  bench/speed/knn_selector_race.py
#   sh tools/knn_selector_body.sh identity TAG ARM [LANES]   cuda identity column of /root/t-ARM
#   sh tools/knn_selector_body.sh cpuidentity TAG base|after  cpu identity column
#   sh tools/knn_selector_body.sh hostafter        the branch's host sets (CPU column, after)
#   sh tools/knn_selector_body.sh check NAME "DEFS" FILE   mojo run one neighbors/checks file
#
# No opponent stage: the cuML and cuVS rows for these blocks on the RTX 4090
# are looked up (bench/OPPONENT_REFERENCE.md rule), never re-measured here.
#
# Every stage writes under /root/kss_out/<stage or TAG>; `pull` the whole
# directory home after each stage. Every arm records the source mtime and
# the binding mtime (so an "after" column can be checked against its source).
set -u
STAGE=${1:?stage}
R=/root/mojolearn
B=/root/mojolearn-base
export PATH="$HOME/.pixi/bin:$PATH"
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1
export GBM_BENCH_DATA=/root/datasets/gbm-bench
export MOJOLEARN_COMPILE_JOBS=16 MOJOLEARN_BUILD_JOBS=16
PIXI="$HOME/.pixi/bin/pixi"
P=$R/.pixi/envs/default/bin/python3
DATA=/root/ctd-data
KNN_LANES="knn,knn-chebyshev,knn-clf,knn-clf-distance,knn-cosine,knn-manhattan,knn-minkowski-p3,knn-rbc,knn-reg,knn-reg-distance,knn-sqeuclidean,radius,radius-chebyshev,radius-manhattan,radius-minkowski-p3"
FIX5="base,ties,odd,dupes,wide"
TOP=/root/kss_out
mkdir -p "$TOP"
note() { echo "$* $(date -u +%H:%M:%S)" | tee -a "$OUT/progress.txt"; }
step() {
    _n=$1; _cap=$2; shift 2
    _t=$(date +%s)
    timeout -k 30 "$_cap" "$@" > "$OUT/logs/$_n.log" 2>&1
    _rc=$?
    printf '%s\t%s\t%s\n' "$_n" "$_rc" "$(( $(date +%s) - _t ))" >> "$OUT/status.tsv"
    note "$_n=$_rc"
    return $_rc
}
build_core() { # tree defines name
    ( cd "$1" && MOJOLEARN_GPU_ARCHS=sm_89 MOJOLEARN_BUILD_EXTRA_DEFINES="$2" bash bindings/build.sh ) || return 1
    mkdir -p "/root/gpubins/$3" && cp "$1/python/mojolearn/identical/_mojolearn.so" "/root/gpubins/$3/_mojolearn.so"
}
build_estimators() {
    ( cd "$1" && MOJOLEARN_GPU_ARCHS=sm_89 MOJOLEARN_BUILD_EXTRA_DEFINES="$2" bash bindings/build_estimators.sh ) || return 1
    mkdir -p "/root/gpubins/$3" && cp "$1/python/mojolearn/identical/_mojolearn_estimators.so" "/root/gpubins/$3/_mojolearn_estimators.so"
}
make_tree() { # name source-tree
    rm -rf "/root/t-$1" && mkdir -p "/root/t-$1"
    ( cd "$2" && tar cf - --exclude=.pixi --exclude='*.so' . ) | ( cd "/root/t-$1" && tar xf - )
    mkdir -p "/root/t-$1/python/mojolearn/identical"
    cp "/root/gpubins/$1/_mojolearn.so" "/root/t-$1/python/mojolearn/identical/"
    cp "/root/gpubins/base/_mojolearn_estimators.so" "/root/t-$1/python/mojolearn/identical/"
}
tree_env() { echo "PYTHONPATH=/root/t-$1/python:/root/t-$1/tools MOJOLEARN_NUMERIC_MODE=identical"; }
stamp_arm() { # name tree defines
    {
        printf 'arm\t%s\ndefines\t%s\nbuilt_utc\t%s\n' "$1" "$3" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'binding_mtime\t%s\n' "$(stat -c %Y "/root/gpubins/$1/_mojolearn.so")"
        printf 'newest_source_mtime\t%s\n' "$(find "$2/neighbors" "$2/checks" "$2/bindings" "$2/core" -name '*.mojo' -printf '%T@\n' | sort -n | tail -1)"
        printf 'source_sha256\t%s\n' "$(cd "$2" && find neighbors checks/kernel_matrix.mojo -name '*.mojo' -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)"
        printf 'binding_sha256\t%s\n' "$(sha256sum "/root/gpubins/$1/_mojolearn.so" | cut -d' ' -f1)"
    } > "/root/gpubins/$1/ARM.txt"
    cp "/root/gpubins/$1/ARM.txt" "$TOP/arm_$1.txt"
}

case "$STAGE" in
_build_core) build_core "$2" "$3" "$4"; exit $? ;;
_build_est) build_estimators "$2" "$3" "$4"; exit $? ;;
setup)
    OUT=$TOP/setup; mkdir -p "$OUT/logs"
    note start shipped="$(cat "$R/SHIPPED_COMMIT.txt")"
    nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader > "$OUT/gpu.txt" 2>&1
    { nproc; grep -m1 'model name' /proc/cpuinfo; uname -a; python3 --version; free -g | head -2; } > "$OUT/box.txt" 2>&1
    # The base tree: the source exactly as shipped (the branch point, main).
    if [ ! -d "$B" ]; then
        mkdir -p "$B" && ( cd "$R" && tar cf - --exclude=.pixi --exclude='*.so' . ) | ( cd "$B" && tar xf - )
    fi
    cd "$R" || exit 9
    [ -x "$HOME/.pixi/bin/pixi" ] || curl -fsSL https://pixi.sh/install.sh | sh > "$OUT/logs/pixi_get.log" 2>&1
    step pixi_install 1500 pixi install
    pixi run mojo --version > "$OUT/mojo_version.txt" 2>&1
    [ -e "$B/.pixi" ] || ln -s "$R/.pixi" "$B/.pixi"
    step prep_blocks 1500 pixi run python3 tools/classical_two_datasets.py prep --data "$DATA" --lanes knn --datasets taxi,istella
    step build_base_core 1800 sh "$0" _build_core "$B" "" base
    step build_base_est 1800 sh "$0" _build_est "$B" "" base
    make_tree base "$B"; stamp_arm base "$B" ""
    ( cd "$B" && MOJOLEARN_HOST_OUTDIR=/root/hostbins/base step host_core_base 1500 sh bindings/build_core_host.sh )
    ( cd "$B" && MOJOLEARN_HOST_OUTDIR=/root/hostbins/base step host_est_base 1500 sh bindings/build_estimators_host.sh )
    ( cd /root/t-base && env $(tree_env base) "$P" -c "import mojolearn as ml; print('base import OK', ml.vendor(), ml.numeric_mode())" ) > "$OUT/imports.txt" 2>&1
    note setup_done
    : > "$OUT/setup.done"
    ;;
arm|basearm)
    NAME=${2:?name}; DEFS=${3-}
    OUT=$TOP/arms; mkdir -p "$OUT/logs"
    SRC=$R; [ "$STAGE" = basearm ] && SRC=$B
    rm -f "$OUT/$NAME.done"
    step "build_$NAME" 1800 sh "$0" _build_core "$SRC" "$DEFS" "$NAME" || { : > "$OUT/$NAME.failed"; exit 1; }
    make_tree "$NAME" "$SRC"; stamp_arm "$NAME" "$SRC" "$DEFS"
    ( cd "/root/t-$NAME" && env $(tree_env "$NAME") "$P" -c "import mojolearn as ml; print('$NAME import OK', ml.vendor(), ml.numeric_mode())" ) >> "$OUT/imports.txt" 2>&1
    : > "$OUT/$NAME.done"
    ;;
hostafter)
    OUT=$TOP/arms; mkdir -p "$OUT/logs"
    cd "$R" || exit 9
    MOJOLEARN_HOST_OUTDIR=/root/hostbins/after step host_core_after 1500 sh bindings/build_core_host.sh
    MOJOLEARN_HOST_OUTDIR=/root/hostbins/after step host_est_after 1500 sh bindings/build_estimators_host.sh
    rm -rf /root/mojolearn-cpu && mkdir -p /root/mojolearn-cpu && ( cd "$R" && tar cf - --exclude=.pixi --exclude='*.so' . ) | ( cd /root/mojolearn-cpu && tar xf - )
    : > "$OUT/hostafter.done"
    ;;
probe)
    TAG=${2:?tag}; ARM=${3:?arm}; shift 3
    OUT=$TOP/$TAG; mkdir -p "$OUT/logs"
    ( cd "/root/t-$ARM" && env $(tree_env "$ARM") "$@" "$P" "$R/tools/knn_selector_probe.py" --arm "$ARM" --json "$OUT/probe_$ARM.json" > "$OUT/probe_$ARM.console" 2>&1 )
    note "probe_$ARM=$? $*"
    ;;
race)
    TAG=${2:?tag}; ARMS=${3:?arms}; KS=${4:-32,64}; ROWS=${5:-4000}
    OUT=$TOP/$TAG; mkdir -p "$OUT/logs"
    "$P" "$R/bench/speed/knn_selector_race.py" --arms "$ARMS" --ks "$KS" --rows "$ROWS" --python "$P" --out "$OUT" --outer "${KSS_OUTER:-5}" > "$OUT/race.console" 2>&1
    note "race=$? arms=$ARMS ks=$KS rows=$ROWS"
    : > "$OUT/race.done"
    ;;
identity)
    TAG=${2:?tag}; ARM=${3:?arm}; LANES=${4:-$KNN_LANES}
    OUT=$TOP/$TAG; mkdir -p "$OUT/logs"
    SHA=$(cat "$R/SHIPPED_COMMIT.txt")
    ( cd "/root/t-$ARM" && env $(tree_env "$ARM") MOJOLEARN_COMMIT="$SHA" \
      "$PIXI" run --manifest-path "$R/pixi.toml" python3 tools/identity_break.py --require-backend cuda --lanes "$LANES" --fixtures "$FIX5" --repeats 2 \
      --json "$OUT/cuda-$ARM.json" > "$OUT/logs/cuda-$ARM.log" 2>&1 )
    note "cuda-$ARM rc=$?"
    ;;
cpuidentity)
    TAG=${2:?tag}; WHICH=${3:?base|after}
    OUT=$TOP/$TAG; mkdir -p "$OUT/logs"
    SHA=$(cat "$R/SHIPPED_COMMIT.txt")
    CPUTREE=/root/mojolearn-cpu
    if [ "$WHICH" = base ]; then
        CPUTREE=/root/mojolearn-cpu-base
        rm -rf "$CPUTREE" && mkdir -p "$CPUTREE" && ( cd "$B" && tar cf - --exclude=.pixi --exclude='*.so' . ) | ( cd "$CPUTREE" && tar xf - )
    fi
    ( cd "$CPUTREE" && env PYTHONPATH="$CPUTREE/python:$CPUTREE/tools" MOJOLEARN_HOST_DIR="/root/hostbins/$WHICH" MOJOLEARN_COMMIT="$SHA" MOJOLEARN_NUMERIC_MODE=identical \
      "$PIXI" run --manifest-path "$R/pixi.toml" python3 tools/identity_break.py --require-backend cpu --lanes "$KNN_LANES" --fixtures "$FIX5" --repeats 2 \
      --json "$OUT/cpu-$WHICH.json" > "$OUT/logs/cpu-$WHICH.log" 2>&1 )
    note "cpu-$WHICH rc=$?"
    ;;
diff)
    TAG=${2:?tag}; NAME=${3:?name}; shift 3
    OUT=$TOP/$TAG; mkdir -p "$OUT/logs"
    ( cd "$R" && PYTHONPATH="$R/python:$R/tools" "$PIXI" run python3 tools/identity_break.py --diff "$@" > "$OUT/diff.$NAME.txt" 2>&1 )
    note "diff $NAME rc=$?"
    ;;
check)
    NAME=${2:?name}; DEFS=${3-}; FILE=${4:?file}
    OUT=$TOP/checks; mkdir -p "$OUT/logs"
    # shellcheck disable=SC2086
    ( cd "$R" && "$PIXI" run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 $DEFS -I . "$FILE" > "$OUT/$NAME.txt" 2>&1 )
    note "check $NAME rc=$?"
    ;;
*)
    echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
