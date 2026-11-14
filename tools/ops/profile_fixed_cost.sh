#!/bin/bash
# Fixed-cost attribution of small paper rows (screening, not a measurement; AGENTS.md "Candidate evaluation": component
# screening). For each race id in PF_RACES (comma list) it re-runs the race's recorded command from PF_SRC/board/board.json
# (the paper board that already timed it) with --rounds PF_ROUNDS (default 100) under py-spy's native sampler
# (`py-spy record --native`), so the Mojo binding frames of the public call are attributed by sample count, then writes
# per race: <id>.collapsed (raw stacks), <id>.summary.txt (tools/ops/profile_fixed_cost_summary.py: frames under the
# public call by self and inclusive samples). cProfile (Python frames only) is the fallback when py-spy cannot ptrace.
# Serial by nature: one profiled process at a time on the one GPU, so the sampled process is the only GPU user
# (busy cores of total are reported by the summary from /proc; the host is otherwise idle: say so in the report).
# Runs under tools/box_persist_run.sh via `tools/nvidia_central.sh submit measure ... tools/ops/profile_fixed_cost.sh`.
set -u
RES=${MOJOLEARN_RESULTS_DIR:?results dir}; mkdir -p "$RES"
SRC=${PF_SRC:?PF_SRC=<results dir of the paper board, e.g. box-out/pc-<commit12>-nvidia-r2>}
[ -f "$SRC/board/board.json" ] || { echo "no $SRC/board/board.json"; exit 5; }
ROUNDS=${PF_ROUNDS:-100}; RATE=${PF_RATE:-250}
PY=$(python3 - "$SRC/board/board.json" <<'PYEOF'
import json,sys; b=json.load(open(sys.argv[1]))["races"]; print(next(iter(b.values()))["command"][0])
PYEOF
)
say() { echo "[$(date +%H:%M:%S) profile-fixed-cost] $*"; }
# the race drivers import the INSTALLED wheel only with the board's child environment (tools/bench_board.py child_env);
# without it `import mojolearn` resolves the source tree and refuses ("no identical binary exists under .../python/mojolearn")
export MOJOLEARN_BENCH_INSTALLED=1 PYTHONUNBUFFERED=1 MOJOLEARN_BOARD_VENDOR=${PF_VENDOR:-nvidia} MOJOLEARN_NUMERIC_MODE=identical
export GBM_BENCH_DATA=${PF_DATA_ROOT:-/root/datasets/gbm-bench} MOJOLEARN_REPO_COMMIT=${PF_SOURCE_COMMIT:-unknown}
[ -x /usr/local/cuda/bin/ptxas ] && export MODULAR_NVPTX_COMPILER_PATH=${MODULAR_NVPTX_COMPILER_PATH:-/usr/local/cuda/bin/ptxas}
say "python $PY rounds=$ROUNDS rate=$RATE data=$GBM_BENCH_DATA races=${PF_RACES:?PF_RACES}"
"$PY" -m pip install -q py-spy 2>&1 | tail -n 1; PYSPY=$(dirname "$PY")/py-spy; [ -x "$PYSPY" ] || PYSPY=$(command -v py-spy || true)
say "py-spy: ${PYSPY:-none}"
IFS=',' read -r -a IDS <<< "$PF_RACES"
for rid in "${IDS[@]}"; do
    name=$(echo "$rid" | tr '/=' '..')
    cmd=$(python3 - "$SRC/board/board.json" "$rid" "$ROUNDS" "$RES" <<'PYEOF'
import json,sys,shlex
b=json.load(open(sys.argv[1]))["races"]; r=b.get(sys.argv[2])
if r is None: sys.exit("no race %s in the board" % sys.argv[2])
c=list(r["command"]); rounds=sys.argv[3]; res=sys.argv[4]
for i,a in enumerate(c):
    if a=="--rounds" and i+1<len(c): c[i+1]=rounds
    if a in ("--out","--work") and i+1<len(c): c[i+1]=res+"/"+("raw" if a=="--out" else "work")
print(" ".join(shlex.quote(x) for x in c))
PYEOF
) || { say "$rid: $cmd"; continue; }
    mkdir -p "$RES/raw" "$RES/work"
    say "race $rid"; t0=$(date +%s)
    if [ -n "${PYSPY:-}" ] && "$PYSPY" record --native --rate "$RATE" --format raw -o "$RES/$name.collapsed" -- bash -c "$cmd" > "$RES/$name.log" 2>&1; then
        mode=py-spy
    else
        say "py-spy failed (tail: $(tail -n 2 "$RES/$name.log" | tr '\n' ' ' | cut -c1-200)); cProfile fallback"
        # the recorded command is "<python> -u <script> ..."; cProfile takes no -u, so it replaces "<python> -u"
        bash -c "${cmd/$PY -u/$PY -m cProfile -o $RES/$name.pstats}" > "$RES/$name.cprof.log" 2>&1; mode=cprofile
    fi
    say "race $rid done mode=$mode $(( $(date +%s) - t0 )) s; $(grep -h -c -E '^(FSPEED|CTD|ALGOS|NEURAL) ' "$RES/raw"/*/*.log 2>/dev/null | paste -sd+ | bc 2>/dev/null) result lines"
    python3 "$(dirname "$0")/profile_fixed_cost_summary.py" --race "$rid" --collapsed "$RES/$name.collapsed" --pstats "$RES/$name.pstats" --out "$RES/$name.summary.txt" 2>&1 | tail -n 3
done
rm -rf "$RES/work"
say "done; summaries: $(ls "$RES"/*.summary.txt 2>/dev/null | wc -l)"
