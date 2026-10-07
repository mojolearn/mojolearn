#!/usr/bin/env bash
# six_lane_grid_run.sh <nvidia|amd> <run-dir> [--phase factorial|pairwise|all] [--dry-run]
#                      [--kit DIR] [--data-dir DIR] [--builds DIR] [--compile-jobs N] [--compile-shards N]
#                      [--retry-failed] [--limit-cells N]
#
# Runs the IDENTICAL switch grid end to end on ONE Linux GPU box (the orchestrator types it; lanes never do):
#   1 grid      python3 tools/six_lane_grid.py --crosses auto --cap 0 --out <run>/grid   (committed grid/ untouched)
#   2 compile   six_lane_ab.py compile, one shard per binding (sm_89 native | gfx942), --keep-going
#   3 kit       six_lane_grid_run.py install-kit: retained variant evidence back at its original /root paths
#   4 stage     six_lane_grid_run.py stage: facts + isolated A/B packages + materialize + queue, one queue per cell,
#               factorial regime first, then pairwise
#   5 run       six_lane_grid_run.py run: performance_full_ab_queue.py one pair at a time; skips measured cells,
#               keeps going past failures, resumes an interrupted cell in a fresh attempt
#   6 evidence  six_lane_timing.py floors --from-pairs (this vendor) + quality rows (tools/af_quality.py)
# Then, with both boxes' run dirs in one place: tools/six_lane_grid_decide.sh <run-nvidia> <run-amd>.
#
# Every step is idempotent: a finished step leaves <run>/state/<step>-<phase>.done and is skipped; compile shards and
# staged configurations are resumed individually. Logs: <run>/logs. One-line status: <run>/status.txt.
# --dry-run prints every command, validates inputs (repo freeze, grid, kit, data, toolchain) and runs nothing
# except the planning-only grid generation into a temporary directory; it works on the laptop.
# Details, inputs and their locations on a box: tools/six_lane_grid_run.md.
set -uo pipefail

usage() { sed -n '2,5p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
[ $# -ge 2 ] || usage
VENDOR=$1; RUN=$2; shift 2
case $VENDOR in nvidia|amd) ;; *) usage;; esac
PHASE=all; DRY=0; KIT=; BUILDS=; COMPILE_JOBS=${GRID_COMPILE_JOBS:-4}; SHARDS=${GRID_COMPILE_SHARDS:-3}
RETRY=; LIMIT=; DATA_DIRS=()
while [ $# -gt 0 ]; do
  case $1 in
    --phase) PHASE=$2; shift 2;;
    --dry-run) DRY=1; shift;;
    --kit) KIT=$2; shift 2;;
    --data-dir) DATA_DIRS+=("$2"); shift 2;;
    --builds) BUILDS=$2; shift 2;;
    --compile-jobs) COMPILE_JOBS=$2; shift 2;;
    --compile-shards) SHARDS=$2; shift 2;;
    --retry-failed) RETRY=--retry-failed; shift;;
    --limit-cells) LIMIT=$2; shift 2;;
    *) echo "unknown option: $1" >&2; usage;;
  esac
done
case $PHASE in factorial|pairwise|all) ;; *) usage;; esac

REPO=$(cd "$(dirname "$0")/.." && pwd)
[ $DRY = 1 ] || mkdir -p "$RUN"
case $RUN in /*) ;; *) RUN=$PWD/$RUN;; esac
KIT=${KIT:-$RUN/kit}
BUILDS=${BUILDS:-$RUN/builds}
[ ${#DATA_DIRS[@]} -eq 0 ] && DATA_DIRS=(/root/six-lane-full-ab-20261006/data)
BENCH_PY=$REPO/.pixi/envs/bench/bin/python
MOJO=$REPO/.pixi/envs/default/bin/mojo
PY3=$(command -v python3)
TOOL=$REPO/tools/six_lane_grid_run.py
STATE=$RUN/state; LOGS=$RUN/logs
if [ $VENDOR = nvidia ]; then TARGET=(--nvidia-target native --nvidia-arch sm_89); else TARGET=(--accelerator gfx942); fi
AUTH=${GRID_AUTHORIZATION:-"Orchestrator ran tools/six_lane_grid_run.sh $VENDOR on $(hostname) at $(date -u +%FT%TZ): IDENTICAL switch grid A/B, one warmup + one scored run per arm (CLAUDE.md measurement process)"}
DATA_ARGS=(); for d in "${DATA_DIRS[@]}"; do DATA_ARGS+=(--data-dir "$d"); done
PROBLEMS=0

say() { echo "grid-run: $*"; }
problem() { echo "grid-run: PROBLEM: $*" >&2; PROBLEMS=$((PROBLEMS + 1)); }
status() {  # one line, overwritten
  [ $DRY = 1 ] && return 0
  echo "grid-run vendor=$VENDOR phase=$PHASE step=$1 state=$2 ${3:-} updated=$(date -u +%FT%TZ)" > "$RUN/status.txt"
}
cmd() {  # cmd <log> <argv...>: print in --dry-run, else run with the log in a file; returns the exit code
  local log=$1; shift
  if [ $DRY = 1 ]; then printf '  $'; printf ' %q' "$@"; printf '  > %s\n' "$log"; return 0; fi
  "$@" >> "$log" 2>&1
}
done_mark() { [ -f "$STATE/$1.done" ]; }
mark() { [ $DRY = 1 ] || date -u +%FT%TZ > "$STATE/$1.done"; }

# ---------------------------------------------------------------- 0 preflight
case $RUN/ in "$REPO"/*) problem "run dir $RUN is inside the checkout; compile/materialize refuse that";; esac
[ $DRY = 1 ] || mkdir -p "$STATE" "$LOGS"
HEAD=$(git -C "$REPO" rev-parse HEAD)
BRANCH=$(git -C "$REPO" branch --show-current)
case $BRANCH in main|integration/*) ;; *) problem "checkout is on '$BRANCH'; six_lane_ab compile/materialize need main or integration/* at the freeze";; esac
DIRTY=$(git -C "$REPO" status --porcelain --untracked-files=all | head -3)
[ -z "$DIRTY" ] || problem "checkout is not clean (first: $(echo "$DIRTY" | head -1)); the freeze must be committed and clean"
if [ $DRY = 0 ]; then
  [ "$(uname -s)" = Linux ] || problem "run on the Linux $VENDOR box, not $(uname -s)"
  [ -x "$BENCH_PY" ] || problem "missing $BENCH_PY (pixi install -e bench)"
  [ -x "$MOJO" ] || [ -d "$BUILDS" ] || problem "missing compiler $MOJO"
  if [ $VENDOR = nvidia ]; then command -v nvidia-smi >/dev/null || problem "no nvidia-smi"; else [ -e /dev/kfd ] || problem "no /dev/kfd"; fi
fi
[ -x "$BENCH_PY" ] || BENCH_PY=$PY3   # dry-run on the laptop: plain python3 for the metadata checks
for f in vendor-workload-facts.json required-bindings.json files.json data-manifest.json; do
  [ -f "$KIT/$f" ] || problem "kit file missing: $KIT/$f (build it on the laptop: python3 tools/six_lane_grid_run.py kit --out <dir>; copy to $KIT)"
done
say "vendor=$VENDOR phase=$PHASE run=$RUN repo=$REPO@${HEAD:0:12} branch=$BRANCH kit=$KIT builds=$BUILDS dry_run=$DRY"
for t in six_lane_grid.py six_lane_ab.py six_lane_materialize.py performance_full_ab_queue.py six_lane_timing.py; do
  [ -f "$REPO/tools/$t" ] || problem "missing tools/$t"
done

# ---------------------------------------------------------------- 1 grid
GRID=$RUN/grid
if [ $DRY = 1 ]; then
  GRID=$(mktemp -d "${TMPDIR:-/tmp}/grid-run-dry.XXXXXX")
  say "1 grid (dry run generates into $GRID to validate; a real run writes $RUN/grid)"
  "$PY3" "$REPO/tools/six_lane_grid.py" --crosses auto --cap 0 --out "$GRID" > "$GRID.log" 2>&1 || problem "grid generation failed: $(tail -1 "$GRID.log")"
  printf '  $ python3 tools/six_lane_grid.py --crosses auto --cap 0 --out %s  > %s\n' "$RUN/grid" "$LOGS/grid.log"
elif [ -f "$GRID/grid-matrix.json.gz" ] && [ "$(cat "$GRID/.source" 2>/dev/null)" = "$HEAD" ]; then
  say "1 grid: present for $HEAD"
elif [ -f "$GRID/grid-matrix.json.gz" ]; then
  problem "$GRID was generated at another commit; use a new run dir per freeze"; exit 1
else
  status grid RUNNING
  "$PY3" "$REPO/tools/six_lane_grid.py" --crosses auto --cap 0 --out "$GRID" > "$LOGS/grid.log" 2>&1 \
    || { status grid FAILED; problem "grid generation failed; see $LOGS/grid.log"; exit 1; }
  echo "$HEAD" > "$GRID/.source"
fi
[ -f "$GRID/grid-matrix.json.gz" ] && say "  $(tail -1 "${GRID}.log" 2>/dev/null | grep -o '"cells_per_vendor": [0-9]*' || grep -o '"cells_per_vendor": [0-9]*' "$GRID/grid-plan.json" | head -1)"
[ $DRY = 1 ] && [ $PROBLEMS -gt 0 ] && [ ! -f "$GRID/grid-matrix.json.gz" ] && { say "dry run stops: $PROBLEMS problem(s)"; exit 1; }

# ---------------------------------------------------------------- 2 compile (one shard per binding, resumable)
SHARDLIST=$( "$PY3" "$TOOL" build-keys --grid "$GRID" --vendor $VENDOR --phase $PHASE )
NSHARDS=$(echo "$SHARDLIST" | grep -c . )
NKEYS=$(echo "$SHARDLIST" | awk '{n+=split($2,a,",")} END {print n+0}')
say "2 compile: $NKEYS builds in $NSHARDS binding shards ($SHARDS at a time, -j $COMPILE_JOBS) -> $BUILDS"
compile_shard() {  # <binding> <keys>
  local binding=$1 keys=$2 name out log moved
  name=$(basename "$binding" .mojo); out=$BUILDS/$name; log=$LOGS/compile-$name.log
  local donefile=$STATE/compile-$name-$(echo "$keys" | cksum | cut -d' ' -f1).done
  [ -f "$donefile" ] && return 0
  if [ -d "$out" ] && [ $DRY = 0 ]; then
    # An interrupted shard: six_lane_ab compile refuses a job dir without a COMPILED receipt. Move those aside
    # (evidence kept) so the rerun recompiles exactly them; COMPILED receipts are reused as they are.
    moved=$BUILDS/.interrupted/$(date -u +%Y%m%dT%H%M%SZ)-$name
    for d in "$out"/*/; do
      [ -d "$d" ] || continue
      grep -q '"status": "COMPILED"' "$d/receipt.json" 2>/dev/null && continue
      mkdir -p "$moved"; mv "$d" "$moved/"
    done
  fi
  local keyargs=(); for k in ${keys//,/ }; do keyargs+=(--key "$k"); done
  cmd "$log" env MOJOLEARN_COMPILE_JOBS=$COMPILE_JOBS "$BENCH_PY" "$REPO/tools/six_lane_ab.py" compile \
    --plan "$GRID/grid-build-plan.json" --vendor $VENDOR "${TARGET[@]}" --compiler "$MOJO" --output "$out" \
    --binding "$binding" "${keyargs[@]}" --keep-going --jobs $COMPILE_JOBS
  local rc=$?
  # rc 1 = finished with recorded FAILED jobs (kept as evidence); 0 = all compiled; anything else = interrupted
  if [ $DRY = 0 ] && { [ $rc = 0 ] || [ $rc = 1 ]; }; then date -u +%FT%TZ > "$donefile"; fi
  return 0
}
if [ -d "$BUILDS" ] && [ -f "$BUILDS/.external" ]; then
  say "  builds supplied externally ($BUILDS/.external); compile skipped"
elif done_mark compile-$PHASE; then
  say "  compile-$PHASE done"
else
  status compile RUNNING "shards=$NSHARDS keys=$NKEYS"
  if [ $DRY = 1 ]; then
    echo "$SHARDLIST" | head -2 | while read -r b k; do compile_shard "$b" "$k"; done
    say "  ... $((NSHARDS > 2 ? NSHARDS - 2 : 0)) more shards like the above"
  else
    mkdir -p "$BUILDS"
    while read -r b k; do
      [ -n "$b" ] || continue
      while [ "$(jobs -rp | wc -l)" -ge "$SHARDS" ]; do sleep 5; done
      compile_shard "$b" "$k" &
    done <<< "$SHARDLIST"
    wait
    PENDING=$(echo "$SHARDLIST" | while read -r b k; do n=$(basename "$b" .mojo); \
      [ -f "$STATE/compile-$n-$(echo "$k" | cksum | cut -d' ' -f1).done" ] || echo "$n"; done | grep -c .)
    COMPILED=$(grep -l '"status": "COMPILED"' "$BUILDS"/*/*/receipt.json 2>/dev/null | wc -l)
    FAILED=$(grep -l '"status": "FAILED"' "$BUILDS"/*/*/receipt.json 2>/dev/null | wc -l)
    say "  compiled=$COMPILED failed=$FAILED unfinished_shards=$PENDING"
    [ "$PENDING" = 0 ] && mark compile-$PHASE || { status compile INTERRUPTED "unfinished_shards=$PENDING"; exit 1; }
  fi
fi

# ---------------------------------------------------------------- 3 kit
say "3 kit: place retained variant evidence at its original paths; check full inputs"
if done_mark kit; then say "  kit done"; else
  if [ $DRY = 1 ]; then
    "$PY3" "$TOOL" install-kit --kit "$KIT" --vendor $VENDOR "${DATA_ARGS[@]}" --dry-run 2>&1 | tail -1 | sed 's/^/  /'
  else
    status kit RUNNING
    "$PY3" "$TOOL" install-kit --kit "$KIT" --vendor $VENDOR "${DATA_ARGS[@]}" > "$LOGS/kit.log" 2>&1 \
      && mark kit || { status kit FAILED; problem "kit install: $(tail -1 "$LOGS/kit.log")"; exit 1; }
  fi
fi

# ---------------------------------------------------------------- 4 stage
say "4 stage: one authorized A/B queue per admissible cell ($PHASE)"
STAGE_ARGS=(stage --vendor $VENDOR --grid "$GRID" --kit "$KIT" --builds "$BUILDS" --run "$RUN" --phase $PHASE "${DATA_ARGS[@]}"
            --cell-timeout "${GRID_CELL_TIMEOUT:-3600}")
if done_mark stage-$PHASE; then say "  stage-$PHASE done"; else
  if [ $DRY = 1 ]; then
    printf '  $ %q %q' "$BENCH_PY" "$TOOL"; printf ' %q' "${STAGE_ARGS[@]}" --authorize "$AUTH"; printf '\n'
    "$PY3" "$TOOL" "${STAGE_ARGS[@]}" --dry-run 2>&1 | tail -1 | sed 's/^/  dry-run admission: /'
  else
    status stage RUNNING
    "$BENCH_PY" "$TOOL" "${STAGE_ARGS[@]}" --authorize "$AUTH" > "$LOGS/stage-$PHASE.log" 2>&1 \
      && mark stage-$PHASE || { status stage FAILED; problem "stage: $(tail -1 "$LOGS/stage-$PHASE.log")"; exit 1; }
    say "  $(tail -1 "$LOGS/stage-$PHASE.log" | cut -c1-400)"
  fi
fi

# ---------------------------------------------------------------- 5 run
say "5 run: serial cells, one warmup + one scored run per arm"
RUN_ARGS=(run --run "$RUN" --phase $PHASE --python "$BENCH_PY" $RETRY ${LIMIT:+--limit $LIMIT})
if done_mark run-$PHASE && [ -z "$RETRY" ]; then say "  run-$PHASE done"; else
  if [ $DRY = 1 ]; then printf '  $ %q %q' "$BENCH_PY" "$TOOL"; printf ' %q' "${RUN_ARGS[@]}"; printf '  > %s\n' "$LOGS/run-$PHASE.log"
  else
    "$BENCH_PY" "$TOOL" "${RUN_ARGS[@]}" >> "$LOGS/run-$PHASE.log" 2>&1
    rc=$?; say "  $(tail -1 "$LOGS/run-$PHASE.log" | cut -c1-300)"
    [ $rc = 0 ] && [ -z "$LIMIT" ] && mark run-$PHASE
  fi
fi

# ---------------------------------------------------------------- 6 evidence (this vendor)
say "6 evidence: incumbent-repeat floors and quality rows for $VENDOR"
EV=$RUN/evidence
if [ $DRY = 1 ]; then
  printf '  $ python3 tools/six_lane_timing.py floors --from-pairs %s --out %s\n' "$RUN/results" "$EV/floors-$VENDOR.json"
  printf '  $ python3 tools/six_lane_grid_run.py quality %s --out %s\n' "$RUN/results" "$EV/quality-$VENDOR.json"
  say "dry run: $PROBLEMS problem(s) found; nothing was run"
  rm -rf "$GRID" "$GRID.log"
  exit $(( PROBLEMS > 0 ))
fi
mkdir -p "$EV"
"$PY3" "$REPO/tools/six_lane_timing.py" floors --from-pairs "$RUN/results" --out "$EV/floors-$VENDOR.json" > "$LOGS/floors.log" 2>&1 \
  || problem "floors: $(tail -1 "$LOGS/floors.log")"
"$PY3" "$TOOL" quality "$RUN/results" --out "$EV/quality-$VENDOR.json" > "$LOGS/quality.log" 2>&1 \
  || problem "quality: $(tail -1 "$LOGS/quality.log")"
say "  $(tail -1 "$LOGS/quality.log" | cut -c1-300)"
"$PY3" "$TOOL" status --run "$RUN" | sed 's/^/  /'
status done "$( [ $PROBLEMS = 0 ] && echo OK || echo PROBLEMS=$PROBLEMS )" "next=tools/six_lane_grid_decide.sh <run-nvidia> <run-amd>"
exit $(( PROBLEMS > 0 ))
