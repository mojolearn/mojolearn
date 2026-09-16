#!/bin/bash
# lane/arima-exog: the CPU columns, the sabotage columns, the saved-model gate
# and the statsmodels agreement. One RunPod CPU pod.
set -u
NEW=arima-exog,arima-exog-seasonal
OLD=arima,arima-011,arima-seasonal-c
R=bench/results/identity_break/2026-09-14_166-lanes
C="$R/apple-m4.json $R/nvidia-h100-sm_90a.json $R/amd-mi325x-gfx942.json"
G="--gpu-column $R/apple-m4.json --gpu-column $R/nvidia-h100-sm_90a.json --gpu-column $R/amd-mi325x-gfx942.json"
REC=bench/results/classical_host/2026-09-15-apple-m4-arima-exog
A=bench/results/classical_host/2026-09-15-apple-m4-arima
REC_OLD_BASE="$A/arima/base $A/arima-011/base $A/arima-seasonal-c/base"
SAB="MOJOLEARN_HOST_DIR=python/mojolearn/host-sabotage MOJOLEARN_HOST_ALLOW_SABOTAGE=1"
O="$LEG_OUT"
st() { local name=$1; shift; "$@"; local rc=$?; printf '%s\t%s\n' "$name" "$rc" >> "$O/step_exit_codes.txt"; return 0; }
echo "nproc=$(nproc) python3=$(python3 --version 2>&1) $(uname -m)" > "$O/box.txt"
ls -l python/mojolearn/host/*.so python/mojolearn/host-sabotage/*.so > "$O/so_sizes.txt" 2>&1
sha256sum python/mojolearn/host/*.so python/mojolearn/host-sabotage/*.so > "$O/so_sha256.txt" 2>&1

# No fit in the shipped forecast binding: the reference bindings are the witness.
for b in forecast arima; do
  printf '%s ' "$b"; nm -C python/mojolearn/host/_mojolearn_${b}_host.so 2>/dev/null \
    | grep -ciE 'arima_fit|estimate_x0|exog_regression|lbfgs|oracle_fit'
done > "$O/fit_symbols.txt" 2>&1

# pytest: the new test, the ARIMA surface, CPU training and the host surface.
st pytest env PYTHONPATH=python .pixi/envs/test/bin/python -m pytest -q -p no:cacheprovider \
  python/mojolearn/tests/test_arima_exog.py python/mojolearn/tests/test_cpu_training_arima.py \
  python/mojolearn/tests/test_host_surface.py python/mojolearn/tests/test_cpu_inference_boundary.py \
  -k 'not test_recordings_and_columns_exist' > "$O/pytest.txt" 2>&1

# The two NEW lanes on all nine fixtures; the three OLD lanes on the base fixture (spot check).
st ib_new python3 tools/identity_break.py --lanes $NEW --repeats 2 --json "$O/cpu-x86.json" > "$O/ib-new.log" 2>&1
st ib_new_sab env $SAB python3 tools/identity_break.py --lanes $NEW --repeats 1 --json "$O/cpu-x86.sabotage.json" > "$O/ib-new-sab.log" 2>&1
st ib_new_host env MOJOLEARN_IDENTITY_HOST_INFER=$NEW python3 tools/identity_break.py --lanes $NEW --repeats 1 --json "$O/cpu-x86.host-infer.json" > "$O/ib-new-host.log" 2>&1
st ib_old python3 tools/identity_break.py --lanes $OLD --fixtures base --repeats 1 --json "$O/cpu-x86.arima-base.json" > "$O/ib-old.log" 2>&1
st ib_old_sab env $SAB python3 tools/identity_break.py --lanes $OLD --fixtures base --repeats 1 --json "$O/cpu-x86.arima-base.sabotage.json" > "$O/ib-old-sab.log" 2>&1

# THE EXOG ARITHMETIC'S OWN NEGATIVE CONTROL: the observation intercept alone,
# built here because the leg tool carries one sabotage define set and that one
# is the lane's whole-host control above.
st build_exogsab env MOJOLEARN_HOST_OUTDIR=python/mojolearn/host-exogsab \
  MOJOLEARN_BUILD_EXTRA_DEFINES="-D MOJOLEARN_ARIMA_EXOG_SABOTAGE=1" MOJOLEARN_BUILD_JOBS=8 \
  sh bindings/build_arima_host.sh > "$O/build_exogsab.txt" 2>&1
st build_exogsab_fc env MOJOLEARN_HOST_OUTDIR=python/mojolearn/host-exogsab \
  MOJOLEARN_BUILD_EXTRA_DEFINES="-D MOJOLEARN_ARIMA_EXOG_SABOTAGE=1" MOJOLEARN_BUILD_JOBS=8 \
  sh bindings/build_forecast_host.sh >> "$O/build_exogsab.txt" 2>&1
# the other families the lanes load come from the production set
for f in python/mojolearn/host/*.so; do
  b=$(basename "$f"); [ -e "python/mojolearn/host-exogsab/$b" ] || cp "$f" python/mojolearn/host-exogsab/
done
st ib_new_exogsab env MOJOLEARN_HOST_DIR=python/mojolearn/host-exogsab MOJOLEARN_HOST_ALLOW_SABOTAGE=1 \
  python3 tools/identity_break.py --lanes $NEW --fixtures base --repeats 1 --json "$O/cpu-x86.exog-sabotage.json" > "$O/ib-new-exogsab.log" 2>&1

st diff_new python3 tools/identity_break.py --diff $C "$O/cpu-x86.json" --require-columns 4 --lanes $NEW --owed-json "$O/owed_cells.json" > "$O/diff.record-vs-cpu.txt" 2>&1
st diff_new_sab python3 tools/identity_break.py --diff $C "$O/cpu-x86.sabotage.json" --require-columns 4 --lanes $NEW > "$O/diff.record-vs-cpu-sabotage.txt" 2>&1
st owed python3 tools/cpu_identity_gate_check.py owed "$O/owed_cells.json" --production "$O/cpu-x86.json" --sabotage "$O/cpu-x86.sabotage.json" > "$O/owed_sabotage_check.txt" 2>&1
st diff_new_host python3 tools/identity_break.py --diff $C "$O/cpu-x86.host-infer.json" --require-columns 4 --lanes $NEW --owed-json "$O/owed_cells.host-infer.json" > "$O/diff.record-vs-cpu-host-infer.txt" 2>&1
st diff_exogsab python3 tools/identity_break.py --diff "$O/cpu-x86.json" "$O/cpu-x86.exog-sabotage.json" --lanes $NEW > "$O/diff.cpu-vs-exog-sabotage.txt" 2>&1
st diff_old python3 tools/identity_break.py --diff $C "$O/cpu-x86.arima-base.json" --lanes $OLD > "$O/diff.record-vs-cpu-arima-base.txt" 2>&1
st diff_old_sab python3 tools/identity_break.py --diff $C "$O/cpu-x86.arima-base.sabotage.json" --lanes $OLD > "$O/diff.record-vs-cpu-arima-base-sabotage.txt" 2>&1

# The Metal recordings through the source tree's host bindings, then the sabotage set.
st gate python3 tools/classical_host_gate.py check $REC $REC_OLD_BASE $G --report "$O/classical_gate_cpu.json" > "$O/classical_gate_cpu.txt" 2>&1
st gate_sab env $SAB python3 tools/classical_host_gate.py check $REC --expect-mismatch --every-lane --report "$O/classical_gate_sab.json" > "$O/classical_gate_sab.txt" 2>&1
# The value sabotage on the forecast path alone.
mkdir -p /tmp/fcsab && cp python/mojolearn/host/*.so /tmp/fcsab/ && cp python/mojolearn/host-sabotage/_mojolearn_forecast_host.so /tmp/fcsab/
st gate_fcsab env MOJOLEARN_HOST_DIR=/tmp/fcsab MOJOLEARN_HOST_ALLOW_SABOTAGE=1 \
  python3 tools/classical_host_gate.py check $REC --expect-mismatch --every-lane --report "$O/classical_gate_forecast_only_sab.json" > "$O/classical_gate_forecast_only_sab.txt" 2>&1

# statsmodels agreement: reported, not tuned to.
st sm_install python3 -m pip install --quiet statsmodels > "$O/statsmodels_install.txt" 2>&1
st sm env PYTHONPATH=python python3 bench/results/identity_break/2026-09-15_arima-exog/statsmodels_agreement.py \
  --out "$O/statsmodels_agreement.json" > "$O/statsmodels_agreement.txt" 2>&1

# The installed test wheel: the saved exog models served by the shipped forecast binding.
ALLOWED=$(PYTHONPATH=python python3 -c 'from mojolearn import host_surface as h; print(" ".join(b + ".so" for b in h.wheel_bindings()))')
rm -rf /tmp/wstage && mkdir -p /tmp/wstage && cp -r python/. /tmp/wstage/
rm -rf /tmp/wstage/mojolearn/host-sabotage /tmp/wstage/mojolearn/host-exogsab /tmp/wstage/build /tmp/wstage/dist
for f in /tmp/wstage/mojolearn/host/*.so; do case " $ALLOWED " in *" $(basename "$f") "*) ;; *) rm -f "$f" ;; esac; done
st wheel bash -c 'cd /tmp/wstage && python3 setup.py -q bdist_wheel' > "$O/wheel.txt" 2>&1
st wheel_install bash -c 'python3 -m venv /tmp/wvenv && /tmp/wvenv/bin/pip install --quiet /tmp/wstage/dist/*.whl' >> "$O/wheel.txt" 2>&1
st gate_wheel env MOJOLEARN_PACKAGE_ROOT=/tmp/wvenv /tmp/wvenv/bin/python tools/classical_host_gate.py \
  check $REC $G --package-root "$(/tmp/wvenv/bin/python -c 'import mojolearn,os;print(os.path.dirname(os.path.dirname(mojolearn.__file__)))')" \
  --report "$O/classical_gate_installed_wheel.json" > "$O/classical_gate_installed_wheel.txt" 2>&1
ls -l /tmp/wstage/dist >> "$O/wheel.txt" 2>&1
