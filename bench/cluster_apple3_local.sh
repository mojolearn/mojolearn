#!/bin/sh
# Lane cluster-apple3 on THE LAPTOP M4 (the brief's 21:25Z update): every build
# through `tools/mac_slot.py run` (nice 19, one compile job), every fit through
# `tools/mac_slot.py metal` (one Metal job on the machine at a time), MAC_SLOTS=2.
#
#   sh bench/cluster_apple3_local.sh build <arm> <base|head|arm:NAME> "<-D defines>" [fast|identical] [binding]
#       builds the binding (default x_cluster) of that source tree and keeps the
#       binary as <arm>.so under $ARMS; x_cluster goes back to HEAD afterwards
#   sh bench/cluster_apple3_local.sh time <rounds> "<cases>" <arm> [<arm> ...]
#       ONE Metal slot: the arms ALTERNATE, `rounds` times over, each run the
#       board's cases (taxi and HIGGS, reps 2); lines `XC_<arm> r<round> ...`
#   sh bench/cluster_apple3_local.sh quality <arm> "<cases>" [seeds]
#       ONE Metal slot: the paired FAST against IDENTICAL check of that arm
#       (needs the arm `identical` built with mode identical)
# BIND=<binding> (default x_cluster) names the binding an arm's binary is, for
# time, quality and phases; an arm written `<arm>:py` in `time` runs the arm's
# binary with MOJOLEARN_HOTPATH=python.
#   sh bench/cluster_apple3_local.sh phases <arm> "<cases>"
# Binaries: $ARMS (default ~/mojolearn-evidence/cluster-apple3/local_arms); delete
# them when their numbers are recorded. Refuses a build under 15 GB of free disk.
set -u
cd "$(dirname "$0")/.."
ARMS=${ARMS:-$HOME/mojolearn-evidence/cluster-apple3/local_arms}
SLOT="python3 tools/mac_slot.py"
export MAC_SLOTS=2
mkdir -p "$ARMS"

place() {  # <arm> <mode>: the arm's binary where the package loads it (BIND names the binding)
    so=_mojolearn_${BIND:-x_cluster}.so
    dst=python/mojolearn; [ "$2" = identical ] && dst=python/mojolearn/identical
    mkdir -p $dst
    cp "$ARMS/$1.so" "$dst/.$so.tmp" && mv "$dst/.$so.tmp" "$dst/$so"
}

case ${1:-} in
build)
    arm=$2; src=$3; defs=$4; mode=${5:-fast}; b=${6:-x_cluster}
    free=$(df -g "$HOME" | awk 'NR==2{print $4}')
    [ "$free" -ge 15 ] || { echo "LOCALFAIL $arm: $free GB free, below 15"; exit 2; }
    [ -z "$(git status --porcelain -- x_cluster)" ] || { echo "LOCALFAIL $arm: x_cluster has uncommitted changes"; exit 2; }
    case $src in
        base) git checkout -q 6856b5f8f -- x_cluster ;;
        head) ;;
        arm:*) git checkout -q 708762e1b -- x_cluster && git apply "bench/cluster_apple3_arms/${src#arm:}.patch" ;;
    esac
    f=bindings/build_$b.sh; [ "$b" = base ] && f=bindings/build.sh
    MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_NUMERIC_MODE=$mode MOJOLEARN_MOJO_BUILD_FLAGS="$defs" \
        $SLOT run -- pixi run -e default sh $f >"$ARMS/$arm.build.log" 2>&1
    rc=$?
    git checkout -q HEAD -- x_cluster
    git status --porcelain -- x_cluster | grep -q . && echo "LOCALWARN $arm: x_cluster is not back at HEAD"
    so=_mojolearn_$b.so; [ "$b" = base ] && so=_mojolearn.so
    dst=python/mojolearn; [ "$mode" = identical ] && dst=python/mojolearn/identical
    if [ $rc -ne 0 ] || [ ! -f "$dst/$so" ]; then
        echo "BUILDFAIL $arm ($mode $b) rc=$rc"; grep -n -i "error" "$ARMS/$arm.build.log" | head -20; tail -15 "$ARMS/$arm.build.log"; exit 1
    fi
    cp "$dst/$so" "$ARMS/$arm.so"
    echo "BUILT $arm ($mode $b, $src, '$defs') $(shasum -a 256 "$ARMS/$arm.so" | cut -c1-16)"
    ;;
time)
    shift; exec $SLOT metal -- sh bench/cluster_apple3_local.sh _time "$@" ;;
_time)
    rounds=$2; cases=$3; shift 3
    for r in $(seq 1 "$rounds"); do
        for arm in "$@"; do
            # `<arm>:py` times the arm's binary with MOJOLEARN_HOTPATH=python
            # (the Python door's reference arm)
            case $arm in *:py) export MOJOLEARN_HOTPATH=python ;; *) unset MOJOLEARN_HOTPATH ;; esac
            place "${arm%%:*}" fast
            pixi run -e default python bench/x_cluster_speed.py --dataset taxi,higgs --reps 2 --only "$cases" \
                | sed "s/^XCSPEED/XC_$arm r$r/"
        done
    done
    ;;
quality)
    shift; exec $SLOT metal -- sh bench/cluster_apple3_local.sh _quality "$@" ;;
_quality)
    arm=$2; cases=$3; seeds=${4:-0}
    place "$arm" fast; place "${IDENT:-identical}" identical
    pixi run -e default python bench/cluster_apple3_quality.py --dataset taxi,higgs --seeds "$seeds" --only "$cases" \
        | sed "s/^XCQUAL/XCQUAL_$arm/"
    ;;
phases)
    shift; exec $SLOT metal -- sh bench/cluster_apple3_local.sh _phases "$@" ;;
_phases)
    arm=$2; cases=$3
    place "$arm" fast
    MOJOLEARN_XC_PHASES=1 MOJOLEARN_STAGE_TIMES=1 pixi run -e default python bench/x_cluster_speed.py \
        --dataset taxi,higgs --reps 1 --no-quality --only "$cases" | sed "s/^/PH_$arm /"
    ;;
*) echo "usage: see the header of bench/cluster_apple3_local.sh"; exit 2 ;;
esac
