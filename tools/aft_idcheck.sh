#!/bin/bash
# aft_idcheck.sh <binding> <lane> <dataset> [rows]: same-bits check for a TREE lane (apple-fast peer, 2026-10-02).
# Builds the IDENTICAL Metal binding and its host binding in this tree, runs the board tree driver with `ours` on
# Metal, then in a second process with `ours` on the host column (MOJOLEARN_VENDOR=cpu --host-digest: hash lines,
# no time; our CPU is never raced or timed) over the first <rows> rows (default 50000), and prints
#   IDCHECK <lane> <ds> rows=<n> metal=<hash> host=<hash> MATCH|DIFFER|ERROR binding: gbdt | rf | trees | svm (iforest).
# AFT_ID_HOST=0: Metal only, prints `IDHASH <lane> <ds> rows=<n> head=<sha> metal=<hash>`; compare that line across two
# branches (a FAST branch vs main) to prove FAST work left IDENTICAL's bits alone. The GBDT host binding trains only
# the declared lanes (no Lossguide/Depthwise/YetiRank), so those lanes use this mode.
# AFT_ID_TREE=<dir>: run against that worktree instead of this script's own (e.g. a main tree without this tool).
set -u
bind=$1 lane=$2 ds=$3 rows=${4:-50000}
here=${AFT_ID_TREE:-$(cd "$(dirname "$0")/.." && pwd)}; cd "$here"
py=${AFT_PY:-python3}
[ -z "${AFT_PY:-}" ] && [ -x "$HOME/board-0834/cache/venv/bin/python" ] && py=$HOME/board-0834/cache/venv/bin/python
export GBM_BENCH_DATA=${GBM_BENCH_DATA:-$HOME/datasets/gbm-bench}
out=${AFT_OUT:-$HOME/aft-id/$(basename "$here")-$lane-$ds}; mkdir -p "$out"
MOJOLEARN_NUMERIC_MODE=identical bash bindings/build.sh > "$out/build_base.log" 2>&1 || echo "AFT-ID build base rc=$?"
MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_$bind.sh > "$out/build_metal.log" 2>&1 || echo "AFT-ID build metal rc=$?"
host=${AFT_ID_HOST:-1}
[ "$host" = 1 ] && MOJOLEARN_HOST_OUTDIR=$here/python/mojolearn/host bash bindings/build_core_host.sh > "$out/build_core_host.log" 2>&1 || echo "AFT-ID build core host rc=$?"
[ "$host" = 1 ] && MOJOLEARN_HOST_OUTDIR=$here/python/mojolearn/host bash bindings/build_${bind}_host.sh > "$out/build_host.log" 2>&1 || echo "AFT-ID build host rc=$?"
MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SPEED_ROUNDS=1 MOJOLEARN_SPEED_SIZE=shipped \
MOJOLEARN_SPEED_EXPECTED_VENDOR=metal PYTHONPATH="$here/python:$here/bench/speed" \
    "$py" -u bench/speed/forest_speed_arm.py --lane "$lane" --dataset "$ds" --ours-only --rows "$rows" \
    > "$out/run.log" 2>&1
[ "$host" = 1 ] && MOJOLEARN_VENDOR=cpu MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SPEED_ROUNDS=1 \
    MOJOLEARN_SPEED_SIZE=shipped PYTHONPATH="$here/python:$here/bench/speed" \
    "$py" -u bench/speed/forest_speed_arm.py --lane "$lane" --dataset "$ds" --ours-only --host-digest \
    --rows "$rows" > "$out/run_host.log" 2>&1
hash_of() { grep -E "^$1 lane=$lane arm=ours " "$2" | grep -o 'hash=[0-9a-f]*' | tail -1 | cut -d= -f2; }
g=$(hash_of FSPEED "$out/run.log")
if [ "$host" != 1 ]; then
    [ -z "$g" ] && grep -m 5 -E 'FSPEED-REFUSED|Traceback|Error' "$out/run.log" | cut -c1-250
    echo "IDHASH $lane $ds rows=$rows head=$(git -C "$here" rev-parse --short HEAD) metal=${g:-none}"; exit 0
fi
h=$(hash_of FSPEED-DIGEST "$out/run_host.log")
st=DIFFER; [ -n "$g" ] && [ "$g" = "$h" ] && st=MATCH; { [ -z "$g" ] || [ -z "$h" ]; } && st=ERROR
[ $st = ERROR ] && grep -h -m 5 -E 'FSPEED-REFUSED|Traceback|Error' "$out/run.log" "$out/run_host.log" | cut -c1-250
echo "IDCHECK $lane $ds rows=$rows metal=${g:-none} host=${h:-none} $st"
