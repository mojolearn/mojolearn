#!/bin/sh
# Lane cluster-apple3 A/B driver for the Apple steward (measurement tooling;
# bench/cluster_apple2_ab.sh plus the phase, profile and quality cases).
# One arm per call: builds the named bindings from the CURRENT worktree with
# optional extra defines, then runs the named cases with a line prefix.
#   sh bench/cluster_apple3_ab.sh <TAG> "<-D defines or empty>" "<bindings>" "<cases>"
# cases: probe:<list> (kmeans_apple_probe --only), board:<list> (x_cluster_speed
# --only, taxi+higgs, reps 2), dbprobe (budgets), gmmstages (taxi gmm phases),
# phases:<list> (MOJOLEARN_XC_PHASES=1 and the other stage timers, reps 1),
# prof:<list> (cProfile of one fit), quality:<list> (FAST against IDENTICAL,
# needs the identical bindings: build them with MODE=identical in <bindings>,
# written as identical/<name>), iboard:<list> and iprobe:<list> (the board and
# the probe on the IDENTICAL bindings, for the digests)
tag=$1; defs=$2; binds=$3; cases=$4
export MOJOLEARN_MOJO_BUILD_FLAGS="$defs"
for b in $binds; do
    mode=${MOJOLEARN_NUMERIC_MODE:-identical}
    case $b in identical/*) mode=identical; b=${b#identical/} ;; esac
    f=bindings/build_$b.sh; [ "$b" = base ] && f=bindings/build.sh
    MOJOLEARN_NUMERIC_MODE=$mode pixi run -e default sh $f >/tmp/ca3_build_$b.log 2>&1 || { echo "BUILDFAIL $tag $b ($mode)"; grep -n -i "error" /tmp/ca3_build_$b.log | head -20; tail -15 /tmp/ca3_build_$b.log; }
done
unset MOJOLEARN_MOJO_BUILD_FLAGS
for c in $cases; do
    case $c in
        probe:*) pixi run python bench/kmeans_apple_probe.py --only "${c#probe:}" | sed "s/^KMPROBE/KM_$tag/" ;;
        board:*) pixi run python bench/x_cluster_speed.py --dataset taxi,higgs --reps 2 --only "${c#board:}" | sed "s/^XCSPEED/XC_$tag/" ;;
        iboard:*) MOJOLEARN_NUMERIC_MODE=identical pixi run python bench/x_cluster_speed.py --dataset taxi,higgs --reps 2 --only "${c#iboard:}" | sed "s/^XCSPEED/XCI_$tag/" ;;
        iprobe:*) MOJOLEARN_NUMERIC_MODE=identical pixi run python bench/kmeans_apple_probe.py --only "${c#iprobe:}" | sed "s/^KMPROBE/KMI_$tag/" ;;
        dbprobe) pixi run python bench/dbscan_batch_probe.py --budgets 1000000,0,8000,4000 --dataset taxi,higgs | sed "s/^DBPROBE/DB_$tag/" ;;
        gmmphases) MOJOLEARN_GMM_MSTEP_TIMES=1 pixi run python bench/x_cluster_speed.py --dataset taxi --reps 1 --only gmm | awk -F"[ =]" -v t="$tag" '/^GMM_MSTEP/{a[$2]+=$3; n[$2]++; next} END{for(k in a) print t, "GMM_MSTEP_SUM", k, a[k]/1000, "ms over", n[k]}' ;;
        phases:*) MOJOLEARN_XC_PHASES=1 MOJOLEARN_STAGE_TIMES=1 MOJOLEARN_KMEANS_STAGES=1 MOJOLEARN_DBSCAN_PHASES=1 pixi run python bench/x_cluster_speed.py --dataset taxi,higgs --reps 1 --no-quality --only "${c#phases:}" | sed "s/^/PH_$tag /" ;;
        prof:*) pixi run python bench/cluster_apple3_prof.py --dataset taxi,higgs --only "${c#prof:}" | sed "s/^CPROF/CPROF_$tag/" ;;
        quality:*) pixi run python bench/cluster_apple3_quality.py --dataset taxi,higgs --only "${c#quality:}" | sed "s/^XCQUAL/XCQUAL_$tag/" ;;
        quality5:*) pixi run python bench/cluster_apple3_quality.py --dataset taxi,higgs --seeds 0,1,2,3,4 --only "${c#quality5:}" | sed "s/^XCQUAL/XCQUAL_$tag/" ;;
        gmmstages) MOJOLEARN_STAGE_TIMES=1 pixi run python bench/x_cluster_speed.py --dataset taxi --reps 1 --only gmm | grep GMM_STAGE | sed "s/^/${tag} /" ;;
    esac
done
