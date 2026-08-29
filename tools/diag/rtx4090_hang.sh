#!/bin/bash
# tools/diag/rtx4090_hang.sh -- the RTX 4090 hang, taken apart on the box.
#
# Runs on a rented NVIDIA pod as phase 9's MOJOLEARN_P9_DIAG hook
# (tools/e1_bootstrap.sh), with $OUT and $REPO in the environment, after the
# identical rf and svm bindings are built. Every step is bounded by
# `timeout -s KILL`, writes its own file under $OUT/diag/, and prints ONE
# verdict line, so a step that hangs is named by the line that never came.
#
# The two hangs it is for (2026-08-29, four 4090 hosts, drivers 550/570/580):
#   H1  IsolationForest.fit never returns; GPU 0 percent, threads in futex.
#   H2  tools/identity_break.py wrote its header and no lane row for 900 s,
#       right after a stability arm whose iforest lane had been killed.
#
#   D0  what is on the GPU before anything runs (a leftover process?)
#   D1  rf-clf step by step (fit, predict, predict_proba, second fit), then
#       identity_break on rf-clf base / hashed and on et-clf x nine fixtures
#   D2  iforest in the background: after 20 s, nvidia-smi PIDS/utilization,
#       any ptxas/nvdisasm child, the wchan histogram, gdb if present; then
#       kill it and ask whether the GPU still answers (H2's hypothesis)
#   D3  the Mojo-level bisect: mojo_only/if_hang_probe.mojo BUILT then RUN
#       once per MOJOLEARN_IF_DIAG_* guard, so "the compile hangs" and "the
#       kernel hangs" are different files, and the TRACE build splits the
#       launch into enqueue (MAX compiles there) and synchronize
set -u
cd "$REPO" || exit 2
export PATH=/root/.pixi/bin:$PATH
D="$OUT/diag"; mkdir -p "$D"
T0=$(date +%s)
elapsed() { echo $(( $(date +%s) - T0 )); }
say() { echo "[diag +$(elapsed)s] $*"; }
PY="pixi run -e gbmbench python3 -u -X faulthandler"
export PYTHONPATH="$REPO/python" MOJOLEARN_NUMERIC_MODE=identical PYTHONUNBUFFERED=1

# run NAME TIMEOUT cmd...   -> $D/NAME.log, one verdict line
run() {
  local name=$1 to=$2; shift 2
  local s=$(date +%s)
  timeout -s KILL "$to" "$@" > "$D/$name.log" 2>&1
  local rc=$?
  local dt=$(( $(date +%s) - s ))
  local v=PASS; [ $rc -eq 137 ] && v="HUNG(killed at ${to}s)"; [ $rc -ne 0 ] && [ $rc -ne 137 ] && v="FAIL(rc=$rc)"
  say "$name: $v in ${dt}s"
  echo "$name $v $dt" >> "$D/verdicts.txt"
  return $rc
}

# ---------------------------------------------------------------- D0
{
  nvidia-smi; nvidia-smi --query-gpu=name,driver_version,utilization.gpu,clocks.sm --format=csv
  echo "--- compute apps"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
  echo "--- python/mojo processes"; ps -eo pid,stat,etime,wchan:20,cmd | grep -E "python|mojo|ptxas" | grep -v grep
  echo "--- gdb: $(command -v gdb || echo none)  py-spy: $(command -v py-spy || echo none)"
  echo "--- MODULAR_NVPTX_COMPILER_PATH=${MODULAR_NVPTX_COMPILER_PATH:-<unset>}"
} > "$D/d0_env.log" 2>&1
say "D0 env recorded"

# ---------------------------------------------------------------- D1
cat > "$D/rf_steps.py" <<'PY'
import faulthandler, sys, time
import numpy as np
faulthandler.dump_traceback_later(240, exit=True)
import mojolearn as ml
rng = np.random.default_rng(0)
X = rng.standard_normal((20000, 16)).astype(np.float32)
y = (X[:, 3] + 0.5 * X[:, 4] > 0).astype(np.int32)
def step(name, fn):
    faulthandler.cancel_dump_traceback_later()
    faulthandler.dump_traceback_later(240, exit=True)
    t = time.time(); r = fn(); print(f"STEP {name}: ok {time.time()-t:.2f}s", flush=True); return r
print("mode", ml.numeric_mode(), flush=True)
m = step("rf_fit", lambda: ml.RandomForestClassifier(n_estimators=16, max_depth=8, random_state=7).fit(X, y))
p = step("rf_predict", lambda: m.predict(X))
pp = step("rf_predict_proba", lambda: m.predict_proba(X))
m2 = step("rf_fit_again", lambda: ml.RandomForestClassifier(n_estimators=16, max_depth=8, random_state=7).fit(X, y))
pp2 = step("rf_predict_proba_again", lambda: m2.predict_proba(X))
print("proba equal across fits:", np.array_equal(pp, pp2), flush=True)
print("RF_STEPS DONE", flush=True)
PY
run d1a_rf_steps 420 $PY "$D/rf_steps.py"
run d1b_break_rf_base 400 $PY tools/identity_break.py --lanes rf-clf --fixtures base --repeats 1
run d1c_break_rf_hashed 400 $PY tools/identity_break.py --lanes rf-clf --fixtures hashed --repeats 1
run d1d_break_et_nine 500 $PY tools/identity_break.py --lanes et-clf --repeats 1

# ---------------------------------------------------------------- D2
cat > "$D/if_bg.py" <<'PY'
import faulthandler, time
import numpy as np
faulthandler.dump_traceback_later(600, exit=True)
import mojolearn as ml
X = np.random.default_rng(0).standard_normal((2000, 8)).astype(np.float32)
t = time.time()
print("IF fit begin", flush=True)
ml.IsolationForest(n_estimators=4, random_state=5).fit(X)
print(f"IF fit returned {time.time()-t:.2f}s", flush=True)
PY
$PY "$D/if_bg.py" > "$D/d2_if_bg.log" 2>&1 &
BG=$!
sleep 20
PID=$(pgrep -f "if_bg.py" | grep -v "^$BG$" | head -1); [ -z "$PID" ] && PID=$BG
# the python process is the one holding the CUDA context: prefer a pid whose cmdline is python3
for p in $(pgrep -f "if_bg.py"); do if tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q "^python3\|/python3"; then PID=$p; fi; done
{
  echo "bg pid=$BG python pid=$PID"; cat "$D/d2_if_bg.log"
  echo "--- nvidia-smi -q -d PIDS"; nvidia-smi -q -d PIDS
  echo "--- utilization"; nvidia-smi --query-gpu=utilization.gpu,clocks.sm --format=csv
  echo "--- dmon"; timeout 6 nvidia-smi dmon -c 2 -s u
  echo "--- ptxas/nvdisasm children"; ps -ef | grep -E "ptxas|nvdisasm|nvlink" | grep -v grep
  echo "--- process state"; ps -o pid,stat,etime,time,wchan:24,cmd -p "$PID"
  echo "--- thread wchan histogram"; cat /proc/$PID/task/*/wchan 2>/dev/null | sort | uniq -c | sort -rn
  echo "--- thread states"; cat /proc/$PID/task/*/stat 2>/dev/null | awk '{print $3}' | sort | uniq -c
  echo "--- non-futex threads"; for t in /proc/$PID/task/*; do w=$(cat $t/wchan 2>/dev/null); case "$w" in *futex*) ;; *) echo "tid=$(basename $t) wchan=$w stat=$(awk '{print $3}' $t/stat 2>/dev/null) name=$(cat $t/comm)"; cat $t/stack 2>&1 | head -12;; esac; done
  if command -v gdb >/dev/null; then echo "--- gdb"; timeout -s KILL 120 gdb -p "$PID" -batch -ex "thread apply all bt 6" 2>&1 | grep -v "^\[New\|^Reading\|^Loaded" | head -400; fi
} > "$D/d2_snapshot.log" 2>&1
say "D2 snapshot taken (pid $PID)"
if kill -0 "$PID" 2>/dev/null; then
  say "D2 iforest still running after 20 s: HUNG as reported"; echo "d2_iforest HUNG 20" >> "$D/verdicts.txt"
  kill -9 "$PID" 2>/dev/null; kill -9 "$BG" 2>/dev/null; pkill -9 -f if_bg.py 2>/dev/null
  sleep 5
  { echo "--- after kill: survivors"; ps -eo pid,stat,etime,wchan:20,cmd | grep -E "if_bg|python3" | grep -v grep
    echo "--- compute apps"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
    nvidia-smi --query-gpu=utilization.gpu --format=csv; } > "$D/d2_after_kill.log" 2>&1
else
  say "D2 iforest FINISHED within 20 s (not hung here)"; echo "d2_iforest PASS 20" >> "$D/verdicts.txt"
fi
# H2's hypothesis: after a killed iforest, does the GPU still answer?
cat > "$D/et_quick.py" <<'PY'
import faulthandler, time
import numpy as np
faulthandler.dump_traceback_later(100, exit=True)
import mojolearn as ml
X = np.random.default_rng(0).standard_normal((20000, 16)).astype(np.float32)
y = (X[:, 3] > 0).astype(np.int32)
t = time.time(); ml.ExtraTreesClassifier(n_estimators=16, max_depth=8, random_state=7).fit(X, y); print(f"ET fit ok {time.time()-t:.2f}s", flush=True)
PY
run d2b_gpu_answers_after_kill 130 $PY "$D/et_quick.py"

# ---------------------------------------------------------------- D3
probe() {  # probe NAME defines...
  local name=$1; shift
  [ "$(elapsed)" -gt 2050 ] && { say "d3_$name SKIPPED (budget)"; echo "d3_${name}_build SKIPPED" >> "$D/verdicts.txt"; return; }
  run "d3_${name}_build" 480 pixi run mojo build -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 "$@" \
      isolation_forest/mojo_only/if_hang_probe.mojo -o "$D/probe_$name" || return
  run "d3_${name}_run" 150 "$D/probe_$name"
}
probe trace -D MOJOLEARN_IF_DIAG_TRACE=1
if [ -x "$D/probe_trace" ]; then
  MODULAR_DEBUG=device-sync-mode run d3_trace_run_syncmode 150 "$D/probe_trace"
fi
probe entry_return -D MOJOLEARN_IF_DIAG_TRACE=1 -D MOJOLEARN_IF_DIAG_ENTRY_RETURN=1
probe gather_only -D MOJOLEARN_IF_DIAG_TRACE=1 -D MOJOLEARN_IF_DIAG_GATHER_ONLY=1
probe no_reject -D MOJOLEARN_IF_DIAG_TRACE=1 -D MOJOLEARN_IF_DIAG_NO_REJECT=1
probe no_record -D MOJOLEARN_IF_DIAG_TRACE=1 -D MOJOLEARN_IF_DIAG_NO_RECORD=1

say "DONE"; echo "=== verdicts"; cat "$D/verdicts.txt"
pkill -9 -f if_bg.py 2>/dev/null; pkill -9 -f probe_ 2>/dev/null
exit 0
