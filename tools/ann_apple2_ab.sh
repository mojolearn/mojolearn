#!/bin/sh
# Lane ann-apple2 (2026-09-28): ONE steward speed job that times a BEFORE and
# an AFTER commit on the same Mac, IDENTICAL and FAST, interleaved.
#
#   tools/apple_steward.py submit --kind speed --lane ann-apple2 --commit <after> \
#     --target m3ultra-b --cmd 'sh tools/ann_apple2_ab.sh <before>[,<mid>...] tsne,cagra 2'
#
# Run by the steward in its worktree at the AFTER commit. Each earlier
# commit is checked out as a detached worktree beside it (sharing its .pixi),
# every arm builds x_ann, estimators and ivf in both tiers, then each rep
# runs every arm and tier once (arms alternate), and a last pass prints the
# stage split (MOJOLEARN_ANN_STAGES=1). Every bench line carries its digests.
# Env: ANN_AB_MODES (default "identical fast"), ANN_AB_BENCH_ARGS (extra
# bench/speed/ann_cpu_speed.py flags), ANN_AB_STAGES (default 1), ANN_AB_QUALITY.
set -eu
befores=$1
algos=$2
reps=${3:-2}
modes=${ANN_AB_MODES:-identical fast}
after_wt=$(pwd)
data=$HOME/datasets/gbm-bench/higgs/higgs_speed.npz
[ -f "$data" ] || { echo "ann_apple2_ab: missing $data" >&2; exit 2; }
arms=""
wts=""
cleanup() { cd "$after_wt"; for w in $wts; do git worktree remove --force "$w" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM
# An arm is <commit> or (lane ann-apple3) <commit>+<DEFINE>[+<DEFINE>...]: the
# commit built with -D <DEFINE> ..., through MOJOLEARN_MOJO_BUILD_FLAGS (every
# bindings/build_*.sh passes it to `mojo build`). The same commit may appear
# in several arms with different defines; the steward's own worktree is the
# arm "after", built with no define.
for tok in $(echo "$befores" | tr ',' ' '); do
    rev=${tok%%+*}
    defs=""
    case "$tok" in *+*) for d in $(echo "${tok#*+}" | tr '+' ' '); do defs="$defs -D $d"; done ;; esac
    bw=$HOME/ann-apple2-arm-$(echo "$tok" | tr -c 'A-Za-z0-9_\n' '-')
    git worktree remove --force "$bw" 2>/dev/null || rm -rf "$bw"
    git worktree prune
    git worktree add -q --detach "$bw" "$rev"
    ln -s "$after_wt/.pixi" "$bw/.pixi"
    echo "$defs" > "$bw/.ann_ab_defs"
    wts="$wts $bw"
    arms="$arms $tok=$bw"
done
arms="$arms after=$after_wt"
# ANN_AB_AFTER_ENV: one more arm, the after build run with these VAR=value words
[ -z "${ANN_AB_AFTER_ENV:-}" ] || arms="$arms afterenv=$after_wt"
echo "AB arms:$arms (after=$(git rev-parse --short HEAD)) algos=$algos reps=$reps modes=$modes host=$(hostname)"
# the base binding (all_finite and friends) once, in the after worktree; every
# arm gets a copy (only the ann bindings differ between arms)
for mode in $modes; do
    t0=$(date +%s)
    MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_NUMERIC_MODE=$mode pixi run -e default sh bindings/build.sh > /dev/null 2>"$HOME/ann-apple2-build.err" || {
        echo "BUILD FAIL base $mode" >&2; tail -40 "$HOME/ann-apple2-build.err" >&2; exit 1; }
    echo "BUILT after $mode build.sh $(( $(date +%s) - t0 ))s"
done
for a in $arms; do
    wt=${a#*=}
    [ "${a%%=*}" != afterenv ] || continue
    if [ "$wt" != "$after_wt" ]; then
        mkdir -p "$wt/python/mojolearn/identical"
        cp -p python/mojolearn/_mojolearn.so "$wt/python/mojolearn/" 2>/dev/null || true
        cp -p python/mojolearn/identical/_mojolearn.so "$wt/python/mojolearn/identical/" 2>/dev/null || true
    fi
    for mode in $modes; do
        for b in build_x_ann.sh build_estimators.sh build_ivf.sh; do
            t0=$(date +%s)
            defs=$(cat "$wt/.ann_ab_defs" 2>/dev/null || true)
            (cd "$wt" && MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_NUMERIC_MODE=$mode MOJOLEARN_MOJO_BUILD_FLAGS="$defs" pixi run -e default sh "bindings/$b" > /dev/null 2>"$HOME/ann-apple2-build.err") || {
                echo "BUILD FAIL $wt $mode $b" >&2; tail -40 "$HOME/ann-apple2-build.err" >&2; exit 1; }
            echo "BUILT ${a%%=*} $mode $b$defs $(( $(date +%s) - t0 ))s"
        done
    done
done
run() {  # arm wt mode [stages]
    echo "== $1 $3${4:+ stages}"
    xenv=""
    [ "$1" != afterenv ] || xenv=${ANN_AB_AFTER_ENV:-}
    # the stage pass: the ann drivers' marks and (lane ann-apple3) cluster/'s k-means marks
    [ -z "${4:-}" ] || xenv="$xenv MOJOLEARN_ANN_STAGES=1 MOJOLEARN_KMEANS_STAGES=1"
    (cd "$2" && env $xenv PYTHONPATH="$2/python" MOJOLEARN_NUMERIC_MODE=$3 \
        pixi run -e default python -u bench/speed/ann_cpu_speed.py --data "$data" --algos "$algos" \
        ${ANN_AB_BENCH_ARGS:-} 2>&1 | grep -v '^\s*$' | sed "s/^/[$1 $3] /")
}
r=0
while [ "$r" -lt "$reps" ]; do
    for mode in $modes; do
        for a in $arms; do run "${a%%=*}" "${a#*=}" "$mode"; done
    done
    r=$((r + 1))
done
if [ "${ANN_AB_STAGES:-1}" = 1 ]; then
    for mode in $modes; do
        for a in $arms; do run "${a%%=*}" "${a#*=}" "$mode" 1; done
    done
fi
# ANN_AB_QUALITY: ann_fast_quality.py flags; run FAST on every arm (lane
# ann-apple3; the first and the last arm before)
if [ -n "${ANN_AB_QUALITY:-}" ]; then
    for a in $arms; do
        [ "${a%%=*}" != afterenv ] || continue
        echo "== quality ${a%%=*}"
        (cd "${a#*=}" && PYTHONPATH="${a#*=}/python" MOJOLEARN_NUMERIC_MODE=fast \
            pixi run -e default python -u bench/speed/ann_fast_quality.py $ANN_AB_QUALITY 2>&1 \
            | grep -E "ANN-QUALITY|Error|Traceback" | sed "s/^/[${a%%=*} fast] /")
    done
fi
