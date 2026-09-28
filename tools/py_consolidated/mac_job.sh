#!/bin/bash
# tools/py_consolidated/mac_job.sh: the ONE light Apple job of lane py-consolidated,
# run as a steward SPEED request (tools/apple_steward.py, one Mac each, --target <mac>)
# in the steward's worktree at the lane's commit (the HEAD tree).
#   base  a SEPARATE worktree at lane/apple2-merged a374c8c08 (`git worktree add`, so
#         the steward's own worktree is never edited and stays clean if this is killed)
#   arms  tools/py_consolidated/check.py arms on the Apple lane subset, base tree then
#         head tree (Metal GPU arm and Mac CPU arm per lane, GPU == CPU); bindings whose
#         sources match come from the steward store (MOJOLEARN_LANE_CHECK_STORE)
#   cross base vs head per lane and column; the py-lm witness base vs head (GPT-3 guard)
#   timing one small interleaved pass on the Metal GPU: base head head base
# The base worktree is removed at the end.
set -u
H=$(pwd)
BASE_REV=${BASE_REV:-a374c8c08}
EV=${EV:-$HOME/mojolearn-evidence/py-consolidated/$(date -u +%m%d-%H%M)}
B=$HOME/mojolearn-evidence/py-consolidated/base-wt
mkdir -p "$EV"
export MOJOLEARN_BUILD_LOCK_HELD=1 MOJOLEARN_COMPILE_JOBS=${MOJOLEARN_COMPILE_JOBS:-4}
export MOJOLEARN_XD_RES_DEV_MIN=1 MOJOLEARN_LANE_CHECK_ARM_TIMEOUT=${MOJOLEARN_LANE_CHECK_ARM_TIMEOUT:-1500}
export MOJOLEARN_LANE_CHECK_STORE=${MOJOLEARN_LANE_CHECK_STORE:-$HOME/mojolearn-evidence/apple-steward/builds}
export CPU_ARMS=${CPU_ARMS:-2} NO_PROBE=1
PIXI=$(command -v pixi || echo ~/.pixi/bin/pixi)
LANES=$(grep -v '^#' "$H/tools/py_consolidated/mac_lanes.txt" | grep . | paste -sd, -)
echo "$(date -u +%FT%TZ) $(hostname) head $(git rev-parse --short HEAD) base $BASE_REV out $EV; $(echo "$LANES" | tr ',' '\n' | wc -l | tr -d ' ') lanes"
sysctl -n machdep.cpu.brand_string 2>/dev/null
cp tools/py_consolidated/check.py tools/py_bugs/probe.py tools/py_lm/witness.py tools/py_consolidated/timing.py \
   tools/py_consolidated/kern_bench.py tools/py_consolidated/svc_bench.py tools/py_consolidated/bench_decomp.py "$EV/"

cleanup() { cd "$H" && git worktree remove --force "$B" >/dev/null 2>&1; rm -rf "$B"; git worktree prune; }
trap cleanup EXIT
cleanup
git worktree add -q --detach "$B" "$BASE_REV" || { echo "BASE WORKTREE FAILED ($BASE_REV not in this clone?)"; exit 1; }

col_env() {  # col_env <tree> <gpu|cpu>
  local e="PYTHONPATH=$1/python MOJOLEARN_NUMERIC_MODE=identical OMP_NUM_THREADS=1"
  [ "$2" = cpu ] && e="$e MOJOLEARN_VENDOR=cpu MOJOLEARN_HOST_DIR=$1/python/mojolearn/host MOJOLEARN_FOREST_HOST_BINARY=$1/python/mojolearn/host/_mojolearn_forest_host.so MOJOLEARN_BYTE_LM_HOST_BINARY=$1/python/mojolearn/host/_mojolearn_byte_lm_host.so"
  echo "$e"
}
tree_phase() {  # tree_phase <tree> <tag>
  local T=$1 tag=$2
  echo "== $tag ($T)"
  (cd "$T" && $PIXI install -e default > "$EV/pixi.$tag.log" 2>&1) || { echo "PIXI FAIL $tag"; tail -3 "$EV/pixi.$tag.log"; }
  (cd "$T" && $PIXI run -e default python -u "$EV/check.py" arms --tree "$T" --out "$EV/$tag" --lanes "$LANES" 2>&1) \
    | tee "$EV/$tag.out" | grep -vE '^\s*$|^== probe|^probe ' | tail -80
  (cd "$T" && $PIXI run -e default python - "$EV/lm_build_$tag.log" <<'PY'
import sys; sys.path.insert(0, "tools")
import algos_lane_check as a
need = {"_mojolearn_transformer", "_mojolearn_training", "_mojolearn_mamba", "_mojolearn_neural_host",
        "_mojolearn_transformer_host", "_mojolearn_mamba_host", "_mojolearn_training_host"}
a.ensure_built(sorted(b for b in need if (a.ROOT / "bindings" / a.script_for(b)).is_file()), sys.argv[1])
PY
  ) 2>&1 | tail -2
  for col in gpu cpu; do
    (cd "$T" && env $(col_env "$T" $col) $PIXI run -e default python -u "$EV/witness.py" --device $col \
      --out "$EV/witness.$tag.$col.json" --timing > "$EV/witness.$tag.$col.log" 2>&1)
    echo "witness $tag $col: exit $? $(tail -1 "$EV/witness.$tag.$col.log")"
  done
}
tree_phase "$B" base
tree_phase "$H" head

echo "== CROSS base vs head"
$PIXI run -e default python -u "$EV/check.py" cross --base "$EV/base" --new "$EV/head" --lanes "$LANES" 2>&1 | tee "$EV/cross.txt" | grep -vE '^\s*$'
for col in gpu cpu; do
  echo "== witness $col base vs head"
  $PIXI run -e default python -u "$EV/witness.py" --compare "$EV/witness.base.$col.json" "$EV/witness.head.$col.json" 2>&1 | tail -12
done

echo "== TIMING (Metal): base head head base"
for tag in base head head base; do
  T=$H; [ $tag = base ] && T=$B
  for s in "timing.py" "kern_bench.py SCALE=cpu" "svc_bench.py NB=10000 NM=3000 K=10 NQ=20000" \
           "bench_decomp.py ONLY=mcd-20k,lda-online-20kx500,mds-nm-1500"; do
    set -- $s
    f=$1; shift
    (cd "$T" && env $(col_env "$T" gpu) "$@" $PIXI run -e default python -u "$EV/$f" 2>&1) \
      | grep -E '^(TIME|BENCH)|FAILED|ERROR' | sed "s/^/T $tag ${f%.py} /"
  done
done
du -sh "$EV" | tail -1
echo "JOB END $(date -u +%FT%TZ)"
exit 0
