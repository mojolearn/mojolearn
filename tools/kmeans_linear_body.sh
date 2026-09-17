#!/bin/sh
# lane/kmeans-linear-speed, 2026-09-17: the on-pod sequence (RunPod NVIDIA,
# CUDA 13 driver). Run ON THE POD from /root/mojolearn (the branch's source,
# shipped by tools/trees_leg.sh). POSIX sh. Stages:
#
#   sh tools/kmeans_linear_body.sh setup            pixi, the big blocks from the staged R2 npz
#   sh tools/kmeans_linear_body.sh arm NAME "DEFS"  one binding arm (core + estimators) and its tree /root/t-NAME
#   sh tools/kmeans_linear_body.sh host NAME        the host (CPU column) binding set /root/hostbins/NAME from tree /root/t-NAME
#   sh tools/kmeans_linear_body.sh diff LABEL A.json B.json ...   identity_break --diff into /root/kls_out/diff.LABEL.txt
#   sh tools/kmeans_linear_body.sh identity NAME    cuda column of every reached lane on arm NAME
#   sh tools/kmeans_linear_body.sh identity_cpu NAME   cpu column with /root/hostbins/NAME
#
# Every stage writes under /root/kls_out/<stage>[-NAME]; pull the directory home.
set -u
STAGE=${1:?stage}
NAME=${2:-}
R=/root/mojolearn
OUT=/root/kls_out/$STAGE${NAME:+-$NAME}
mkdir -p "$OUT/logs"
export PATH="$HOME/.pixi/bin:$PATH"
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1
export GBM_BENCH_DATA=/root/datasets/gbm-bench
export MOJOLEARN_COMPILE_JOBS=16 MOJOLEARN_BUILD_JOBS=16
P=$R/.pixi/envs/default/bin/python3
DATA=/root/ctd-data
ARCH=${KLS_GPU_ARCH:-sm_89}
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
LANES="kmeans,kmeans-random,kmeans-array,kmeans-weighted,kmeans-sqrt,kmeans-classic-pp,kmeans-cosine,pca,pca-whiten,pca-full-whiten,tsvd,ols,ridge,ols-no-intercept,ols-weighted,ridge-no-intercept"
FIX5="base,ties,odd,dupes,wide"
# The other families whose fits run `kmeans_fit_main_traced` or `kmeans_fit`
# (spectral in the core binding; GaussianMixture's k-means init; the IVF
# coarse quantizer; the parallel k-means and GMM drivers).
LANES2="spectral,spectral-precomputed,gmm,gmm-sample,ivf,ivf-euclidean,ivf-extend,par-kmeans,par-gmm"

case "$STAGE" in
_build_core)
    ( cd "$R" && MOJOLEARN_GPU_ARCHS=$ARCH MOJOLEARN_BUILD_EXTRA_DEFINES="$3" bash bindings/build.sh ) || exit 1
    mkdir -p "/root/gpubins/$2" && cp "$R/python/mojolearn/identical/_mojolearn.so" "/root/gpubins/$2/"
    ;;
_build_est)
    ( cd "$R" && MOJOLEARN_GPU_ARCHS=$ARCH MOJOLEARN_BUILD_EXTRA_DEFINES="$3" bash bindings/build_estimators.sh ) || exit 1
    mkdir -p "/root/gpubins/$2" && cp "$R/python/mojolearn/identical/_mojolearn_estimators.so" "/root/gpubins/$2/"
    ;;
setup)
    note start branch="$(cat "$R/SHIPPED_COMMIT.txt")"
    nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader > "$OUT/gpu.txt" 2>&1
    { nproc; grep -m1 'model name' /proc/cpuinfo; uname -a; free -g | head -2; } > "$OUT/box.txt" 2>&1
    cd "$R" || exit 9
    [ -x "$HOME/.pixi/bin/pixi" ] || curl -fsSL https://pixi.sh/install.sh | sh > "$OUT/logs/pixi_get.log" 2>&1
    step pixi_install 1500 pixi install
    pixi run mojo --version > "$OUT/mojo_version.txt" 2>&1
    step prep_blocks 1500 pixi run python3 tools/classical_two_datasets.py prep --data "$DATA" --lanes kmeans,pca,ols --datasets taxi,istella
    note setup_done
    : > "$OUT/setup.done"
    ;;
arm)
    DEFS=${3:-}
    cd "$R" || exit 9
    step build_core 2400 sh "$0" _build_core "$NAME" "$DEFS"
    step build_est 2400 sh "$0" _build_est "$NAME" "$DEFS"
    rm -rf "/root/t-$NAME" && mkdir -p "/root/t-$NAME"
    ( cd "$R" && tar cf - --exclude=.pixi --exclude='*.so' . ) | ( cd "/root/t-$NAME" && tar xf - )
    mkdir -p "/root/t-$NAME/python/mojolearn/identical"
    cp /root/gpubins/"$NAME"/*.so "/root/t-$NAME/python/mojolearn/identical/"
    ( cd "/root/t-$NAME" && stat -c '%y %n' python/mojolearn/identical/*.so cluster/impl/detail/kmeans.mojo glm/impl/ols.mojo ) > "$OUT/mtimes.txt" 2>&1
    sha256sum /root/gpubins/"$NAME"/*.so > "$OUT/so_sha256.txt"
    echo "$DEFS" > "$OUT/defines.txt"
    ( cd "/root/t-$NAME" && PYTHONPATH="/root/t-$NAME/python" "$P" -c "import mojolearn as ml; print('import OK', ml.vendor(), ml.numeric_mode())" ) > "$OUT/import.txt" 2>&1
    note arm_done
    : > "$OUT/arm.done"
    ;;
arm_extra)
    # The ivf and mixture bindings of arm NAME, into its gpubins and its tree.
    DEFS=${3:-}
    cd "$R" || exit 9
    for fam in ivf mixture; do
        step build_$fam 2400 env MOJOLEARN_GPU_ARCHS=$ARCH MOJOLEARN_BUILD_EXTRA_DEFINES="$DEFS" bash bindings/build_$fam.sh
        cp "$R/python/mojolearn/identical/_mojolearn_$fam.so" "/root/gpubins/$NAME/" && cp "/root/gpubins/$NAME/_mojolearn_$fam.so" "/root/t-$NAME/python/mojolearn/identical/"
    done
    sha256sum /root/gpubins/"$NAME"/*.so > "$OUT/so_sha256.txt"
    : > "$OUT/arm_extra.done"
    ;;
identity2)
    cd "/root/t-$NAME" || exit 9
    MOJOLEARN_COMMIT=$(cat "/root/t-$NAME/SHIPPED_COMMIT.txt"); export MOJOLEARN_COMMIT
    step identity2 5400 env PYTHONPATH="/root/t-$NAME/python:/root/t-$NAME/tools" "$P" tools/identity_break.py \
        --require-backend cuda --lanes "$LANES2" --fixtures "$FIX5" --repeats 2 --json "$OUT/identity.json"
    : > "$OUT/identity2.done"
    ;;
host)
    # NAME's host (CPU column) binding set, built from the source of tree
    # /root/t-NAME (the pixi environment is shared through a symlink).
    T="/root/t-$NAME"
    [ -e "$T/.pixi" ] || ln -s "$R/.pixi" "$T/.pixi"
    cd "$T" || exit 9
    MOJOLEARN_HOST_OUTDIR=/root/hostbins/$NAME step host_core 2400 sh bindings/build_core_host.sh
    MOJOLEARN_HOST_OUTDIR=/root/hostbins/$NAME step host_est 2400 sh bindings/build_estimators_host.sh
    sha256sum /root/hostbins/"$NAME"/*.so > "$OUT/so_sha256.txt" 2>&1
    : > "$OUT/host.done"
    ;;
identity)
    cd "/root/t-$NAME" || exit 9
    MOJOLEARN_COMMIT=$(cat "/root/t-$NAME/SHIPPED_COMMIT.txt"); export MOJOLEARN_COMMIT
    step identity 5400 env PYTHONPATH="/root/t-$NAME/python:/root/t-$NAME/tools" "$P" tools/identity_break.py \
        --require-backend cuda --lanes "$LANES" --fixtures "$FIX5" --repeats 2 --json "$OUT/identity.json"
    : > "$OUT/identity.done"
    ;;
identity_cpu)
    # The cpu column: tree /root/t-NAME's Python with NO GPU binding beside it,
    # host bindings from /root/hostbins/NAME.
    rm -rf "/root/cpu-$NAME" && mkdir -p "/root/cpu-$NAME"
    ( cd "/root/t-$NAME" && tar cf - --exclude=.pixi --exclude='*.so' . ) | ( cd "/root/cpu-$NAME" && tar xf - )
    cd "/root/cpu-$NAME" || exit 9
    MOJOLEARN_COMMIT=$(cat "/root/t-$NAME/SHIPPED_COMMIT.txt"); export MOJOLEARN_COMMIT
    step identity_cpu 5400 env PYTHONPATH="/root/cpu-$NAME/python:/root/cpu-$NAME/tools" MOJOLEARN_HOST_DIR="/root/hostbins/$NAME" "$P" tools/identity_break.py \
        --require-backend cpu --lanes "$LANES" --fixtures "$FIX5" --repeats 2 --json "$OUT/identity.json"
    : > "$OUT/identity_cpu.done"
    ;;
ab)
    # sh tools/kmeans_linear_body.sh ab AFTER BEFORE [rounds] [lanes] [datasets]
    # The interleaved A/B: `ours` is tree /root/t-AFTER, `ours-base` is
    # /root/t-BEFORE, one worker each, order rotated every round.
    BEFORE=${3:?before arm}; ROUNDS=${4:-7}; ABL=${5:-kmeans}; ABD=${6:-taxi,istella}
    OUT=/root/kls_out/ab-$NAME-vs-$BEFORE; mkdir -p "$OUT/logs"
    cd "/root/t-$NAME" || exit 9
    for ds in $(echo "$ABD" | tr , ' '); do for ln in $(echo "$ABL" | tr , ' '); do
        step "race_${ln}_${ds}" 3600 env MOJOLEARN_CTD_BASE_PY="/root/t-$BEFORE/python" "$P" tools/classical_two_datasets.py race \
            --lane "$ln" --dataset "$ds" --data "$DATA" --out "$OUT" --work "/root/ctd-work-$NAME" --root "/root/t-$NAME" \
            --rounds "$ROUNDS" --arms ours,ours-base --ours-python "$P" --theirs-python "$P"
    done; done
    "$P" tools/classical_two_datasets.py summary --out "$OUT" > "$OUT/summary.txt" 2>&1
    stat -c '%y %n' "/root/t-$NAME"/python/mojolearn/identical/*.so "/root/t-$BEFORE"/python/mojolearn/identical/*.so "/root/t-$NAME/cluster/estimator.mojo" "/root/t-$NAME/cluster/checks/reduce_by_key.mojo" > "$OUT/mtimes.txt" 2>&1
    : > "$OUT/ab.done"
    ;;
diff)
    # sh tools/kmeans_linear_body.sh diff LABEL A.json B.json [...]
    shift 2
    [ "${1:-}" = x ] && shift
    cd "$R" || exit 9
    PYTHONPATH="$R/python:$R/tools" "$P" tools/identity_break.py --diff "$@" > "/root/kls_out/diff.$NAME.txt" 2>&1
    echo "diff $NAME rc=$?" | tee -a /root/kls_out/diff.progress.txt
    ;;
*) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
