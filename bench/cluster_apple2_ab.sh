#!/bin/sh
# Lane cluster-apple2 A/B driver for the Apple steward (measurement tooling).
# One arm per call: builds the named bindings from the CURRENT worktree with
# optional extra defines, then runs the named cases with a line prefix.
#   sh bench/cluster_apple2_ab.sh <TAG> "<-D defines or empty>" "<bindings>" "<cases>"
# cases: probe:<list> (kmeans_apple_probe --only), board:<list> (x_cluster_speed
# --only, taxi+higgs, reps 2), dbprobe (budgets), gmmstages (taxi gmm phases)
tag=$1; defs=$2; binds=$3; cases=$4
export MOJOLEARN_MOJO_BUILD_FLAGS="$defs"
for b in $binds; do
    f=bindings/build_$b.sh; [ "$b" = base ] && f=bindings/build.sh
    pixi run -e default sh $f >/tmp/ca2_build_$b.log 2>&1 || { echo "BUILDFAIL $tag $b"; tail -5 /tmp/ca2_build_$b.log; }
done
unset MOJOLEARN_MOJO_BUILD_FLAGS
for c in $cases; do
    case $c in
        probe:*) pixi run python bench/kmeans_apple_probe.py --only "${c#probe:}" | sed "s/^KMPROBE/KM_$tag/" ;;
        board:*) pixi run python bench/x_cluster_speed.py --dataset taxi,higgs --reps 2 --only "${c#board:}" | sed "s/^XCSPEED/XC_$tag/" ;;
        dbprobe) pixi run python bench/dbscan_batch_probe.py --budgets 1000000,0,8000,4000 --dataset taxi,higgs | sed "s/^DBPROBE/DB_$tag/" ;;
        gmmstages) MOJOLEARN_STAGE_TIMES=1 pixi run python bench/x_cluster_speed.py --dataset taxi --reps 1 --only gmm | grep GMM_STAGE | sed "s/^/${tag} /" ;;
    esac
done
