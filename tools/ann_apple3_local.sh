#!/bin/bash
# tools/ann_apple3_local.sh (lane ann-apple3, 2026-09-28): the lane's A/B on
# THE LAPTOP GPU (Apple M4, 10 GPU cores, 16 GB), allowed for this round by
# the brief's 21:25Z update after m3ultra-b was terminated
# (~/mojolearn-evidence/apple3_speed_brief.md, Machines).
#
# EVERY build goes through `tools/mac_slot.py run` (one CPU slot, nice 19, one
# compile job) and EVERY fit through ONE `tools/mac_slot.py metal` job, with
# MAC_SLOTS=2. No build starts with less than 15 GB free. Timing arms and
# their quality check only.
#
#   tools/ann_apple3_local.sh build <mode> <arm>=<DEF>+<DEF>... [<arm>=...]
#       builds each arm in THIS worktree, one after the other, and keeps a
#       copy of its package (python/mojolearn, 25 MB) under $ARMS/<arm>.
#       An arm named `default=` has no define. The ivf binding is rebuilt only
#       when an arm's ivf defines differ from the last build's.
#   tools/ann_apple3_local.sh run <mode> <algos> <reps> <arm> [<arm>...]
#       ONE Metal job: every rep runs every arm once (arms alternate), then
#       the stage pass, then (ANN_AB_QUALITY set, FAST) the quality pass per
#       arm. Output is tools/ann_apple2_ab.sh's format
#       (tools/ann_apple3_tab.py reads it).
#   tools/ann_apple3_local.sh clean          deletes $ARMS
#
# ANN_LOCAL_N (default 1000000) is the IVF row count; the row count goes in
# the progress file with every laptop row.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT" || exit 1
ARMS=${ANN_LOCAL_ARMS:-$HOME/mojolearn-evidence/ann-apple3/local-arms}
DATA=${ANN_LOCAL_DATA:-$HOME/datasets/gbm-bench/higgs/higgs_speed.npz}
export MAC_SLOTS=2
SLOT="python3 tools/mac_slot.py"

free_gb() { df -g "$HOME" | awk 'NR==2 {print $4}'; }

build_one() {  # build_one <mode> <script> <defines>
    local g; g=$(free_gb)
    if [ "$g" -lt 15 ]; then echo "NOT BUILDING: $g GB free, under 15"; return 3; fi
    local t0; t0=$(date +%s)
    local log="$ARMS/build.$1.$2.log"
    if MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_NUMERIC_MODE=$1 MOJOLEARN_MOJO_BUILD_FLAGS="$3" \
        $SLOT run -- pixi run -e default sh "bindings/$2" > "$log" 2>&1; then
        echo "BUILT $1 $2 [$3] $(( $(date +%s) - t0 ))s (free ${g} GB)"
    else
        echo "BUILD FAIL $1 $2 [$3]"
        grep -v '^\s*$' "$log" | grep -v '^mac_slot' | tail -60 | sed 's/^/[build] /'
        return 1
    fi
}

ivf_key() {  # the defines the ivf binding reads
    echo "$1" | tr ' ' '\n' | grep -E 'HOST_PASSES|PREPARE|COARSE_SEED|TRAINSET_COPY' | sort | paste -sd' ' -
}

case "${1:-}" in
build)
    mode=$2; shift 2
    mkdir -p "$ARMS"
    [ -e .pixi ] || ln -s "$HOME/CascadeProjects/mojolearn/.pixi" .pixi
    echo "$(date -u +%FT%TZ) laptop $(sysctl -n machdep.cpu.brand_string) build mode=$mode tree $(git log -1 --format=%h) free $(free_gb) GB"
    build_one "$mode" build.sh "" || exit 1
    build_one "$mode" build_estimators.sh "" || exit 1
    last_ivf="(none)"
    for spec in "$@"; do
        name=${spec%%=*}
        defs=""
        for d in $(echo "${spec#*=}" | tr '+' ' '); do defs="$defs -D $d"; done
        key=$(ivf_key "$defs")
        ok=1
        if [ "$key" != "$last_ivf" ]; then
            if build_one "$mode" build_ivf.sh "$defs"; then last_ivf=$key; else ok=0; last_ivf="(none)"; fi
        fi
        [ "$ok" = 0 ] || build_one "$mode" build_x_ann.sh "$defs" || ok=0
        if [ "$ok" = 1 ]; then
            mkdir -p "$ARMS/$name/python"
            rsync -a --delete python/mojolearn/ "$ARMS/$name/python/mojolearn/"
            echo "$defs" > "$ARMS/$name/defs.$mode"
            echo "ARM $name $mode [$defs] kept in $ARMS/$name"
        else
            echo "ARM $name $mode NOT BUILT"
        fi
    done
    ;;
run)
    mode=$2; algos=$3; reps=$4; shift 4
    arms="$*"
    out=${ANN_LOCAL_OUT:-$HOME/mojolearn-evidence/ann-apple3/local_$(date -u +%m%d-%H%M%S)_$mode.txt}
    inner=$ARMS/run_inner.sh
    n=${ANN_LOCAL_N:-1000000}
    cat > "$inner" <<EOF
#!/bin/bash
# written by tools/ann_apple3_local.sh; runs under ONE Metal slot
cd "$ROOT" || exit 1
line="AB arms:"
for a in $arms; do line="\$line \$a=$ARMS/\$a"; done
echo "\$line (after=$(git log -1 --format=%h)) algos=$algos reps=$reps modes=$mode host=laptop-M4 n=$n"
run() {  # arm [stages]
    echo "== \$1 $mode\${2:+ stages}"
    xenv=""
    [ -z "\${2:-}" ] || xenv="MOJOLEARN_ANN_STAGES=1 MOJOLEARN_KMEANS_STAGES=1"
    env \$xenv PYTHONPATH="$ARMS/\$1/python" MOJOLEARN_NUMERIC_MODE=$mode \\
        pixi run -e default python -u bench/speed/ann_cpu_speed.py --data "$DATA" --algos "$algos" --n $n \\
        ${ANN_AB_BENCH_ARGS:-} 2>&1 | grep -v '^\\s*\$' | sed "s/^/[\$1 $mode] /"
}
r=0
while [ "\$r" -lt "$reps" ]; do
    for a in $arms; do run "\$a"; done
    r=\$((r + 1))
done
for a in $arms; do run "\$a" 1; done
if [ -n "${ANN_AB_QUALITY:-}" ] && [ "$mode" = fast ]; then
    for a in $arms; do
        echo "== quality \$a"
        PYTHONPATH="$ARMS/\$a/python" MOJOLEARN_NUMERIC_MODE=fast \\
            pixi run -e default python -u bench/speed/ann_fast_quality.py ${ANN_AB_QUALITY:-} 2>&1 \\
            | grep -E "ANN-QUALITY|Error|Traceback" | sed "s/^/[\$a fast] /"
    done
fi
EOF
    chmod +x "$inner"
    echo "$(date -u +%FT%TZ) laptop run mode=$mode arms=$arms out=$out"
    $SLOT metal -- bash "$inner" > "$out" 2> "$out.err"
    echo "exit $? ; $(wc -l < "$out") lines in $out"
    ;;
clean)
    rm -rf "$ARMS"
    echo "deleted $ARMS"
    ;;
*)
    echo "usage: tools/ann_apple3_local.sh build <mode> <arm>=<DEF>+... | run <mode> <algos> <reps> <arm>... | clean" >&2
    exit 2
    ;;
esac
