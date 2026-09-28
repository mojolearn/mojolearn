#!/bin/sh
# Lane ann-apple2 (2026-09-28): ONE steward speed job that times a BEFORE and
# an AFTER commit on the same Mac, IDENTICAL and FAST, interleaved.
#
#   tools/apple_steward.py submit --kind speed --lane ann-apple2 --commit <after> \
#     --target m3ultra-b --cmd 'sh tools/ann_apple2_ab.sh <before> tsne,cagra 2'
#
# Run by the steward in its worktree at the AFTER commit. BEFORE is checked
# out as a detached worktree beside it (sharing its .pixi), both arms build
# x_ann, estimators and ivf in both tiers, then each rep runs every arm and
# tier once (before/after alternate), and a last pass prints the stage split
# (MOJOLEARN_ANN_STAGES=1). Every bench line carries its digests.
# Env: ANN_AB_MODES (default "identical fast"), ANN_AB_BENCH_ARGS (extra
# bench/speed/ann_cpu_speed.py flags), ANN_AB_STAGES (default 1).
set -eu
before=$1
algos=$2
reps=${3:-2}
modes=${ANN_AB_MODES:-identical fast}
after_wt=$(pwd)
data=$HOME/datasets/gbm-bench/higgs/higgs_speed.npz
[ -f "$data" ] || { echo "ann_apple2_ab: missing $data" >&2; exit 2; }
bw=$HOME/ann-apple2-before-wt
git worktree remove --force "$bw" 2>/dev/null || rm -rf "$bw"
git worktree prune
git worktree add -q --detach "$bw" "$before"
ln -s "$after_wt/.pixi" "$bw/.pixi"
cleanup() { cd "$after_wt"; git worktree remove --force "$bw" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
echo "AB before=$(git -C "$bw" rev-parse --short HEAD) after=$(git rev-parse --short HEAD) algos=$algos reps=$reps modes=$modes host=$(hostname)"
for wt in "$bw" "$after_wt"; do
    for mode in $modes; do
        for b in build_x_ann.sh build_estimators.sh build_ivf.sh; do
            t0=$(date +%s)
            (cd "$wt" && MOJOLEARN_NUMERIC_MODE=$mode pixi run -e default sh "bindings/$b" > /dev/null 2>"$wt/.ab_build.err") || {
                echo "BUILD FAIL $wt $mode $b" >&2; tail -40 "$wt/.ab_build.err" >&2; exit 1; }
            echo "BUILT $(basename "$wt") $mode $b $(( $(date +%s) - t0 ))s"
        done
    done
done
run() {  # arm wt mode [stages]
    echo "== $1 $3${4:+ stages}"
    (cd "$2" && PYTHONPATH="$2/python" MOJOLEARN_NUMERIC_MODE=$3 ${4:+MOJOLEARN_ANN_STAGES=1} \
        pixi run -e default python -u bench/speed/ann_cpu_speed.py --data "$data" --algos "$algos" \
        ${ANN_AB_BENCH_ARGS:-} 2>&1 | grep -v '^\s*$' | sed "s/^/[$1 $3] /")
}
r=0
while [ "$r" -lt "$reps" ]; do
    for mode in $modes; do
        run before "$bw" "$mode"
        run after "$after_wt" "$mode"
    done
    r=$((r + 1))
done
if [ "${ANN_AB_STAGES:-1}" = 1 ]; then
    for mode in $modes; do
        run before "$bw" "$mode" 1
        run after "$after_wt" "$mode" 1
    done
fi
