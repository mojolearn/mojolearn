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
#   ORDER on leg 3: D0, D5 (DEVIATION 1946's ordering probe, cheapest and
#   most decisive, needs nothing built), D2 (Python lifetime probes), D4 (the
#   fix through the probe and the stability harness), D3 (Mojo lifetime
#   probes), D1 (leg 1's reproduction, last, budget permitting).
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

# ---------------------------------------------------------------- D5 (leg 3)
# DEVIATION 1946 FIRST. The ordering alone: a context created inside a call
# and its own buffers freed after its last use, twice in one process, in the
# two orders, with no forest, no Python and nothing built. ~30 s for both.
#
#   BAD hangs (rc 137 under timeout -s KILL) and GOOD prints DONE
#       -> the ordering IS the poison; DEVIATION 1946 is the fix and D4's
#          binding lanes below should now pass on this box.
#   BOTH print DONE
#       -> the ordering is NOT the poison here. 1946 stays as a correctness
#          fix, the hunt goes back to the GILReleased block, and D2/D3 below
#          are what say so. WRITE THAT SENTENCE in this file's verdicts.
ORDP=ensemble/mojo_only/rf_ctx_order_probe.mojo
run d5_order_bad_build 480 pixi run mojo build -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D ORDER_BAD=1 "$ORDP" -o "$D/probe_order_bad" \
  && run d5_order_bad_run 120 "$D/probe_order_bad"
run d5_order_good_build 480 pixi run mojo build -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D ORDER_GOOD=1 "$ORDP" -o "$D/probe_order_good" \
  && run d5_order_good_run 120 "$D/probe_order_good"

# ---------------------------------------------------------------- D2 (leg 2)
# Leg 1 (bench/results/e1/2026-08-29_163200-runpod-nvidia/diag) said: rf fit,
# predict and predict_proba pass and the SECOND fit in the process hangs; the
# Python iforest fit hangs with the GPU idle and every host thread in futex;
# the one-context Mojo probe passes with the M4's checksums for every bisect
# guard. So the kernel is innocent and the suspect is DeviceContext lifetime.
cat > "$D/py_probes.py" <<'PY'
import faulthandler, gc, sys, time
import numpy as np
faulthandler.dump_traceback_later(110, exit=True)
import mojolearn as ml
which = sys.argv[1]
rng = np.random.default_rng(0)
X = rng.standard_normal((20000, 16)).astype(np.float32)
y = (X[:, 3] + 0.5 * X[:, 4] > 0).astype(np.int32)
def step(name, fn):
    faulthandler.cancel_dump_traceback_later(); faulthandler.dump_traceback_later(110, exit=True)
    t = time.time(); r = fn(); print(f"STEP {name}: ok {time.time()-t:.2f}s", flush=True); return r
if which == "if_small":      # the probe's shape through the BINDING
    step("if_small_fit", lambda: ml.IsolationForest(n_estimators=4, max_samples=32, random_state=5).fit(X[:64, :4]))
elif which == "if_default":  # the estimator defaults through the binding (the fixed build)
    m = step("if_default_fit", lambda: ml.IsolationForest(n_estimators=16, random_state=5).fit(X))
    step("if_default_score", lambda: m.score_samples(X))
    step("if_default_fit_again", lambda: ml.IsolationForest(n_estimators=16, random_state=5).fit(X))
elif which == "rf_gc":       # second fit after the first model is gone
    m = step("rf_fit1", lambda: ml.RandomForestClassifier(n_estimators=16, max_depth=8, random_state=7).fit(X, y))
    del m; gc.collect()
    step("rf_fit2_after_gc", lambda: ml.RandomForestClassifier(n_estimators=16, max_depth=8, random_state=7).fit(X, y))
elif which == "rf_reg":      # the regressor twice
    step("rfreg_fit1", lambda: ml.RandomForestRegressor(n_estimators=16, max_depth=8, random_state=7).fit(X, X[:, 0]))
    step("rfreg_fit2", lambda: ml.RandomForestRegressor(n_estimators=16, max_depth=8, random_state=7).fit(X, X[:, 0]))
elif which == "rf_small":    # tiny second fit
    step("rf_fit1", lambda: ml.RandomForestClassifier(n_estimators=16, max_depth=8, random_state=7).fit(X, y))
    step("rf_fit2_small", lambda: ml.RandomForestClassifier(n_estimators=2, max_depth=3, random_state=7).fit(X[:500], y[:500]))
elif which == "rf_then_et":  # a different binding's context after rf
    step("rf_fit1", lambda: ml.RandomForestClassifier(n_estimators=16, max_depth=8, random_state=7).fit(X, y))
    step("et_fit_after_rf", lambda: ml.ExtraTreesClassifier(n_estimators=16, max_depth=8, random_state=7).fit(X, y))
elif which == "et_twice":    # sequential contexts in a lane that passes
    step("et_fit1", lambda: ml.ExtraTreesClassifier(n_estimators=16, max_depth=8, random_state=7).fit(X, y))
    step("et_fit2", lambda: ml.ExtraTreesClassifier(n_estimators=16, max_depth=8, random_state=7).fit(X, y))
    step("rf_fit_after_et", lambda: ml.RandomForestClassifier(n_estimators=16, max_depth=8, random_state=7).fit(X, y))
print("PY_PROBE DONE", flush=True)
PY
for w in if_small if_default rf_gc rf_small rf_reg rf_then_et et_twice; do
  run "d2_py_$w" 130 $PY "$D/py_probes.py" $w
done

# ---------------------------------------------------------------- D4 (leg 2)
# The fixed svm binding (DEVIATION 1944) through the probe's own lanes, each
# under its own timeout, then the stability shape.
run d4_break_iforest_rfclf 600 $PY tools/identity_break.py --lanes iforest,rf-clf --json "$D/identity_break.iforest_rfclf.json"
run d4_stability_iforest 400 $PY tools/repeat_run_stability.py --repeats 6 --lanes iforest --json "$D/stability.iforest.json"
run d4_stability_rfclf 400 $PY tools/repeat_run_stability.py --repeats 6 --lanes rf-clf --json "$D/stability.rfclf.json"

# ---------------------------------------------------------------- D3 (leg 2)
# Mojo-level lifetimes. Builds are ~10 s on a box whose cache is warm from
# phase 9; each run is bounded so a hang costs 90 s, not the lease.
mprobe() {  # mprobe NAME SRC defines...
  local name=$1 src=$2; shift 2
  [ "$(elapsed)" -gt 2100 ] && { say "d3_$name SKIPPED (budget)"; echo "d3_${name} SKIPPED" >> "$D/verdicts.txt"; return; }
  run "d3_${name}_build" 480 pixi run mojo build -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 "$@" "$src" -o "$D/probe_$name" || return
  run "d3_${name}_run" 90 "$D/probe_$name"
}
IFP=isolation_forest/mojo_only/if_ctx_probe.mojo
mprobe if_T4_model_on_second_ctx "$IFP" -D MOJOLEARN_IF_DIAG_TRACE=1 -D T4=1
mprobe if_T1_two_alive_fit_second "$IFP" -D MOJOLEARN_IF_DIAG_TRACE=1 -D T1=1
mprobe if_T3_two_alive_fit_first "$IFP" -D MOJOLEARN_IF_DIAG_TRACE=1 -D T3=1
mprobe if_T2_sequential "$IFP" -D MOJOLEARN_IF_DIAG_TRACE=1 -D T2=1
RFP=ensemble/mojo_only/rf_ctx_probe.mojo
mprobe rf_two_ctx "$RFP" -D RF_TWO_CTX=1
mprobe rf_same_ctx "$RFP" -D RF_SAME_CTX=1

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
[ "$(elapsed)" -gt 2000 ] || run d1a_rf_steps 300 $PY "$D/rf_steps.py"
[ "$(elapsed)" -gt 2100 ] || run d1b_break_rf_base 300 $PY tools/identity_break.py --lanes rf-clf --fixtures base --repeats 1
[ "$(elapsed)" -gt 2100 ] || run d1c_break_rf_hashed 300 $PY tools/identity_break.py --lanes rf-clf --fixtures hashed --repeats 1
[ "$(elapsed)" -gt 2100 ] || run d1d_break_et_nine 300 $PY tools/identity_break.py --lanes et-clf --repeats 1

say "DONE"; echo "=== verdicts"; cat "$D/verdicts.txt"
pkill -9 -f py_probes.py 2>/dev/null; pkill -9 -f probe_ 2>/dev/null
exit 0
