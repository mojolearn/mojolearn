#!/bin/bash
# Lane py-sequence: before/after on ONE box in ONE job (docs/lanes/progress/py-sequence.md).
#   BASE = lane/apple2-merged's base commit (a detached worktree beside this tree), NEW = this tree.
#   1. each tree: build the sequence family's bindings once (GPU + CPU host, stamped), then every
#      sequence-family identity lane once per column (merged_check clean: GPU vs CPU AGREE);
#   2. compare_trees.py: BASE == NEW, cell for cell, on both columns;
#   3. bench/py_sequence/ab.py on both trees, both arms: seconds and digests of the touched paths;
#   4. the x_sequence / tsa / arima / parallel forecasting pytest files on NEW.
# Usage (on the box): bash bench/py_sequence/job.sh [OUT] [STAGES]   STAGES default "lanes bench tests"
set -u
cd "$(dirname "$0")/../.."
NEW=$(pwd)
BASE_REV=${BASE_REV:-0a11b50c7}
BASE=${BASE:-/root/ps-base}
OUT=${1:-/root/ev-py-sequence/$(date -u +%Y%m%dT%H%M%SZ)}
STAGES=${2:-"lanes bench tests"}
mkdir -p "$OUT"
export MOJOLEARN_BUILD_LOCK_HELD=1 MOJOLEARN_COMPILE_JOBS=${MOJOLEARN_COMPILE_JOBS:-4}
PIXI=$(command -v pixi || echo ~/.pixi/bin/pixi)
LANES=${LANES:-sequence-lstm,sequence-gru,sequence-rmsprop,sequence-adagrad,sequence-autoarima,sequence-stl,sequence-var,sequence-mlp,sequence-rnn,sequence-lion,sequence-adafactor,sequence-lamb,sequence-adamax,sequence-nadam,sequence-lr-schedulers,sequence-layernorm,sequence-theta,sequence-croston,sequence-ets,sequence-garch,sequence-prophet,sequence-moe,arima,arima-011,arima-seasonal-c,arima-exog,arima-exog-seasonal,holtwinters,holtwinters-multiplicative,kpss}
echo "$(date -u +%FT%TZ) $(hostname) NEW $(git rev-parse --short HEAD) BASE $BASE_REV out $OUT stages $STAGES"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1
lscpu 2>/dev/null | grep -m1 'Model name'

if [ ! -d "$BASE/.git" ] && [ ! -f "$BASE/.git" ]; then
    git worktree add --detach "$BASE" "$BASE_REV" > "$OUT/base_worktree.log" 2>&1 \
        || { echo "BASE WORKTREE FAIL"; tail -5 "$OUT/base_worktree.log"; exit 1; }
fi
[ -e "$BASE/.pixi" ] || ln -s "$NEW/.pixi" "$BASE/.pixi"
mkdir -p "$BASE/bench/py_sequence"
cp bench/py_sequence/ab.py bench/py_sequence/merged_check.py "$BASE/bench/py_sequence/"
echo "BASE at $(git -C "$BASE" rev-parse --short HEAD)"
$PIXI install -e default > "$OUT/pixi_install.log" 2>&1 || { echo "PIXI INSTALL FAIL"; tail -5 "$OUT/pixi_install.log"; exit 1; }

run_lanes() {   # $1 tree, $2 tag
    local T=$1 tag=$2 o="$OUT/lanes_$2"
    mkdir -p "$o"
    ( cd "$T"
      P="$PIXI run -e default python -u bench/py_sequence/merged_check.py"
      [ -f "$o/plan.json" ] || $P plan --out "$o" > "$o/plan.log" 2>&1
      python3 - "$o" "$LANES" <<'PY'
import json, sys
out, only = sys.argv[1], [l for l in sys.argv[2].split(",") if l]
p = json.load(open(out + "/plan.json"))
miss = [l for l in only if l not in p["needed"]]
lanes = [l for l in only if l in p["needed"]]
p["lanes"] = lanes
p["bindings"] = sorted(set().union(*[p["needed"][l] for l in lanes]))
json.dump(p, open(out + "/plan.json", "w"))
print(f"{len(lanes)} lanes, {len(p['bindings'])} bindings: {p['bindings']}; not exposed: {miss}")
PY
      $P build --out "$o" --jobs 4 2>&1 | tail -3
      [ "${3:-}" = build-only ] && exit 0
      $P clean --out "$o" --shard 0/1 --cpu-threads default 2>&1 | grep -E 'RESULT|STALE|not AGREE|ERROR|DISAGREE|REFUSED' )
}

case " $STAGES " in *" build "*)
    echo "== build BASE"; run_lanes "$BASE" base build-only
    echo "== build NEW"; run_lanes "$NEW" new build-only
;; esac

case " $STAGES " in *" lanes "*)
    echo "== lanes BASE"; run_lanes "$BASE" base
    echo "== lanes NEW"; run_lanes "$NEW" new
    python3 bench/py_sequence/compare_trees.py "$OUT/lanes_base/clean_0of1" "$OUT/lanes_new/clean_0of1" | tee "$OUT/compare_trees.txt"
;; esac

case " $STAGES " in *" bench "*)
    # the builds above cover the bindings ab.py runs (x_sequence, tsa, arima, GPU and host)
    for T in BASE NEW; do
        dir=$BASE; old=--old
        [ $T = NEW ] && { dir=$NEW; old=; }
        for arm in gpu cpu; do
            only=rnn,optim,layernorm,forecast,tsa,autoarima,parallel_hw
            [ $arm = cpu ] && only=sched,$only
            env=(MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH="$dir/python" OMP_NUM_THREADS=1)
            [ $arm = cpu ] && env+=(MOJOLEARN_VENDOR=cpu MOJOLEARN_HOST_DIR="$dir/python/mojolearn/host")
            echo "== bench $T $arm ($(date -u +%FT%TZ))"
            ( cd "$dir" && env "${env[@]}" $PIXI run -e default python -u bench/py_sequence/ab.py \
                --arm $arm --out "$OUT/ab_${T}_${arm}.json" --only $only $old ) > "$OUT/ab_${T}_${arm}.log" 2>&1
            grep -E "^(gpu|cpu) |ERROR|Traceback" "$OUT/ab_${T}_${arm}.log" | head -80
        done
    done
    python3 bench/py_sequence/compare_ab.py "$OUT" | tee "$OUT/compare_ab.txt"
;; esac

case " $STAGES " in *" tests "*)
    echo "== pytest NEW"
    tests=$(ls python/mojolearn/tests/test_x_sequence_*.py python/mojolearn/tests/test_holtwinters_*.py \
        python/mojolearn/tests/test_parallel_forecasting.py python/mojolearn/tests/test_arima_surface.py 2>/dev/null)
    MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH="$NEW/python" $PIXI run -e test python -m pytest -q -x -p no:cacheprovider $tests \
        > "$OUT/pytest.log" 2>&1; echo "pytest exit $?"; tail -5 "$OUT/pytest.log"
;; esac
echo "JOB END $(date -u +%FT%TZ)"
