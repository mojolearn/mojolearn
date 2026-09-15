#!/bin/bash
# lane/inference-holtwinters: CPU columns, sabotage, installed test wheel, wheel size. One RunPod CPU pod.
set -u
HW=holtwinters,holtwinters-multiplicative
AR=arima,arima-011,arima-seasonal-c
R=bench/results/identity_break/2026-09-14_166-lanes
C="$R/apple-m4.json $R/nvidia-h100-sm_90a.json $R/amd-mi325x-gfx942.json"
G="--gpu-column $R/apple-m4.json --gpu-column $R/nvidia-h100-sm_90a.json --gpu-column $R/amd-mi325x-gfx942.json"
REC_HW=bench/results/classical_host/2026-09-15-apple-m4-holtwinters
A=bench/results/classical_host/2026-09-15-apple-m4-arima
REC_AR_BASE="$A/arima/base $A/arima-011/base $A/arima-seasonal-c/base"
SAB="MOJOLEARN_HOST_DIR=python/mojolearn/host-sabotage MOJOLEARN_HOST_ALLOW_SABOTAGE=1"
O="$LEG_OUT"
st() { local name=$1; shift; "$@"; local rc=$?; printf '%s\t%s\n' "$name" "$rc" >> "$O/step_exit_codes.txt"; return 0; }
echo "nproc=$(nproc) python3=$(python3 --version 2>&1) $(uname -m)" > "$O/box.txt"
ls -l python/mojolearn/host/*.so python/mojolearn/host-sabotage/*.so > "$O/so_sizes.txt" 2>&1
sha256sum python/mojolearn/host/*.so python/mojolearn/host-sabotage/*.so > "$O/so_sha256.txt" 2>&1

# Fit symbols: the forecast inference binding against the reference tsa and arima bindings (the witness).
for b in forecast tsa arima; do
  printf '%s ' "$b"; nm -C python/mojolearn/host/_mojolearn_${b}_host.so 2>/dev/null | grep -ciE 'oracle_fit|holtwinters_fit|host_r1qt|oracle_eval|linesearch|lbfgs|arima_fit|estimate_x0'
done > "$O/fit_symbols.txt" 2>&1

# pytest (source and runtime, host bindings built)
st pytest env PYTHONPATH=python .pixi/envs/test/bin/python -m pytest -q -p no:cacheprovider \
  python/mojolearn/tests/test_holtwinters_inference.py python/mojolearn/tests/test_host_surface.py \
  python/mojolearn/tests/test_cpu_inference_boundary.py python/mojolearn/tests/test_cpu_training_arima.py \
  -k 'not test_recordings_and_columns_exist' > "$O/pytest.txt" 2>&1

# The two new lanes, all nine fixtures; the ARIMA lanes on the base fixture (spot check).
st ib_hw python3 tools/identity_break.py --lanes $HW --repeats 2 --json "$O/cpu-x86.json" > "$O/ib-hw.log" 2>&1
st ib_hw_sab env $SAB python3 tools/identity_break.py --lanes $HW --repeats 1 --json "$O/cpu-x86.sabotage.json" > "$O/ib-hw-sab.log" 2>&1
st ib_hw_host env MOJOLEARN_IDENTITY_HOST_INFER=$HW python3 tools/identity_break.py --lanes $HW --repeats 1 --json "$O/cpu-x86.host-infer.json" > "$O/ib-hw-host.log" 2>&1
st ib_ar python3 tools/identity_break.py --lanes $AR --fixtures base --repeats 1 --json "$O/cpu-x86.arima-base.json" > "$O/ib-ar.log" 2>&1
st ib_ar_sab env $SAB python3 tools/identity_break.py --lanes $AR --fixtures base --repeats 1 --json "$O/cpu-x86.arima-base.sabotage.json" > "$O/ib-ar-sab.log" 2>&1

st diff_hw python3 tools/identity_break.py --diff $C "$O/cpu-x86.json" --require-columns 4 --lanes $HW --owed-json "$O/owed_cells.json" > "$O/diff.record-vs-cpu.txt" 2>&1
st diff_hw_sab python3 tools/identity_break.py --diff $C "$O/cpu-x86.sabotage.json" --require-columns 4 --lanes $HW > "$O/diff.record-vs-cpu-sabotage.txt" 2>&1
st owed python3 tools/cpu_identity_gate_check.py owed "$O/owed_cells.json" --production "$O/cpu-x86.json" --sabotage "$O/cpu-x86.sabotage.json" > "$O/owed_sabotage_check.txt" 2>&1
st diff_hw_host python3 tools/identity_break.py --diff $C "$O/cpu-x86.host-infer.json" --require-columns 4 --lanes $HW --owed-json "$O/owed_cells.host-infer.json" > "$O/diff.record-vs-cpu-host-infer.txt" 2>&1
st diff_ar python3 tools/identity_break.py --diff $C "$O/cpu-x86.arima-base.json" --lanes $AR > "$O/diff.record-vs-cpu-arima-base.txt" 2>&1
st diff_ar_sab python3 tools/identity_break.py --diff $C "$O/cpu-x86.arima-base.sabotage.json" --lanes $AR > "$O/diff.record-vs-cpu-arima-base-sabotage.txt" 2>&1

# The Metal recordings through the source tree's host bindings, then the sabotage set.
st gate python3 tools/classical_host_gate.py check $REC_HW $REC_AR_BASE $G --report "$O/classical_gate_cpu.json" > "$O/classical_gate_cpu.txt" 2>&1
st gate_sab env $SAB python3 tools/classical_host_gate.py check $REC_HW $REC_AR_BASE --expect-mismatch --every-lane --report "$O/classical_gate_sab.json" > "$O/classical_gate_sab.txt" 2>&1
# The value sabotage on the forecast path alone: only the forecast binding sabotaged.
mkdir -p /tmp/fcsab && cp python/mojolearn/host/*.so /tmp/fcsab/ && cp python/mojolearn/host-sabotage/_mojolearn_forecast_host.so /tmp/fcsab/
st gate_fcsab env MOJOLEARN_HOST_DIR=/tmp/fcsab MOJOLEARN_HOST_ALLOW_SABOTAGE=1 python3 tools/classical_host_gate.py check $REC_HW --expect-mismatch --every-lane --report "$O/classical_gate_forecast_only_sab.json" > "$O/classical_gate_forecast_only_sab.txt" 2>&1

# Test wheels: this branch's forecast binding, and origin/main's, the same stage otherwise.
ALLOWED=$(PYTHONPATH=python python3 -c 'from mojolearn import host_surface as h; print(" ".join(b + ".so" for b in h.wheel_bindings()))')
stage() {  # stage <dir> <forecast .so>
  rm -rf "$1"; mkdir -p "$1"; cp -r python/. "$1"/
  rm -rf "$1"/mojolearn/host-sabotage "$1"/build "$1"/dist
  for f in "$1"/mojolearn/host/*.so; do case " $ALLOWED " in *" $(basename "$f") "*) ;; *) rm -f "$f" ;; esac; done
  cp "$2" "$1"/mojolearn/host/_mojolearn_forecast_host.so
}
cat > /tmp/forecast_host_main.mojo <<'MAINSRC'
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The CPU INFERENCE binding for the forecasters (lane/inference-forecast-umap-pca,
2026-09-15): prediction from a saved ARIMA model, with no fit in the binary.

HOST ONLY, IDENTICAL ONLY. It registers the GPU binding's `arima_predict`,
`arima_forecast`, `arima_vendor` and `arima_numeric_mode` names from
`bindings/arima_host_predict.mojo`, the same source the internal reference
binding `bindings/_mojolearn_arima_host.mojo` registers them from, and
nothing that fits. `arima_fit` is absent, so a CPU-only install that holds
only this binding refuses a fit by name twice: `ARIMA.fit` in Python outside
the internal reference context, and the `_HostBinding` proxy for the absent
name.

Why a binding of its own: the reference binding carries the whole fit
(estimate_x0, the least squares, the L-BFGS and the finite-difference
likelihood), and training-only code does not ship in the inference wheels
(docs/lanes/CPU_INFERENCE_BOUNDARY_2026-09-15.md). The manifest declares
this family with `routes=None` and `serves=("_mojolearn_arima",)`:
`_backend` routes `_mojolearn_arima` here on a CPU-only install only when
the reference binding is not built, and `mojolearn.host_model` binds it for
a saved ARIMA model on any machine.

The sabotage arm (`forecast_host_sabotage`) is
`arima/host/arima_oracle.mojo::ARIMA_ORACLE_PREDICT_SABOTAGE`: under
`-D MOJOLEARN_HOST_SABOTAGE=1` every finite predicted value has its lowest
bit flipped, so the saved-model gate must read a mismatch.
"""
from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

from checks.kernel_matrix import (
    COLUMN_CPU,
    TARGET_COLUMN,
    column_name,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from arima.host.arima_oracle import ARIMA_ORACLE_PREDICT_SABOTAGE
from bindings.arima_host_predict import (
    arima_forecast_binding,
    arima_predict_binding,
)


def forecast_host_numeric_mode_binding() raises -> PythonObject:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL, (
        "forecast host: this binding restates the IDENTICAL arms only;"
        " bindings/build_host_family.sh passes -D MOJOLEARN_NUMERIC_IDENTICAL=1"
    )
    return PythonObject(GLOBAL_NUMERIC_MODE)


def forecast_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def forecast_host_column_binding() raises -> PythonObject:
    """`column_name(TARGET_COLUMN)`, the comptime assert's witness: "cpu"."""
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "forecast host: this binding compiles the CPU column only; pass"
        " -D MOJOLEARN_COLUMN_CPU (bindings/build_forecast_host.sh does)"
    )
    return PythonObject(column_name(TARGET_COLUMN))


def forecast_host_sabotage_binding() raises -> PythonObject:
    """Whether this binary flips the lowest bit of every prediction on
    purpose (MOJOLEARN_HOST_SABOTAGE or MOJOLEARN_ARIMA_PREDICT_SABOTAGE)."""
    return PythonObject(ARIMA_ORACLE_PREDICT_SABOTAGE)


def arima_vendor_binding() raises -> PythonObject:
    """"cpu", as every host binding answers."""
    return PythonObject(String("cpu"))


def arima_numeric_mode_binding() raises -> PythonObject:
    """The build's tier as the `NUMERIC_*` code, which `_arima_impl.py`
    cross-checks against the mode the saved model records."""
    return PythonObject(GLOBAL_NUMERIC_MODE)


@export
def PyInit__mojolearn_forecast_host() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_forecast_host")
        module.def_function[forecast_host_numeric_mode_binding]("forecast_host_numeric_mode")
        module.def_function[forecast_host_vendor_binding]("forecast_host_vendor")
        module.def_function[forecast_host_column_binding]("forecast_host_column")
        module.def_function[forecast_host_sabotage_binding]("forecast_host_sabotage")
        module.def_function[arima_vendor_binding]("arima_vendor")
        module.def_function[arima_numeric_mode_binding]("arima_numeric_mode")
        module.def_function[arima_predict_binding]("arima_predict")
        module.def_function[arima_forecast_binding]("arima_forecast")
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_forecast_host: ", error))
MAINSRC
cp bindings/_mojolearn_forecast_host.mojo /tmp/forecast_host_branch.mojo
cp /tmp/forecast_host_main.mojo bindings/_mojolearn_forecast_host.mojo
st build_main_forecast env MOJOLEARN_FORECAST_HOST_OUTDIR=/tmp/oldhost MOJOLEARN_BUILD_JOBS=8 sh bindings/build_forecast_host.sh > "$O/build_main_forecast.txt" 2>&1
cp /tmp/forecast_host_branch.mojo bindings/_mojolearn_forecast_host.mojo
cmp -s /tmp/forecast_host_branch.mojo bindings/_mojolearn_forecast_host.mojo && echo restored >> "$O/build_main_forecast.txt"
stage /tmp/ws_new python/mojolearn/host/_mojolearn_forecast_host.so
stage /tmp/ws_old /tmp/oldhost/_mojolearn_forecast_host.so
python3 -m venv /tmp/wv > "$O/venv.txt" 2>&1 || /usr/bin/python3 -m venv /tmp/wv >> "$O/venv.txt" 2>&1
/tmp/wv/bin/pip install -q --disable-pip-version-check wheel setuptools numpy >> "$O/venv.txt" 2>&1
/tmp/wv/bin/python --version >> "$O/venv.txt" 2>&1
st wheel_new /tmp/wv/bin/pip wheel --no-deps --no-build-isolation -w /tmp/wheel_new /tmp/ws_new > "$O/wheel_new_build.txt" 2>&1
st wheel_old /tmp/wv/bin/pip wheel --no-deps --no-build-isolation -w /tmp/wheel_old /tmp/ws_old > "$O/wheel_old_build.txt" 2>&1
{ ls -l /tmp/wheel_new/*.whl /tmp/wheel_old/*.whl; ls -l python/mojolearn/host/_mojolearn_forecast_host.so /tmp/oldhost/_mojolearn_forecast_host.so
  python3 -c 'import zipfile,glob
for w in glob.glob("/tmp/wheel_new/*.whl")+glob.glob("/tmp/wheel_old/*.whl"):
    z=zipfile.ZipFile(w)
    for i in z.infolist():
        if i.filename.startswith("mojolearn/host/"): print(w.split("/")[2], i.filename, i.file_size, i.compress_size)'
} > "$O/wheel.txt" 2>&1
st install /tmp/wv/bin/pip install --no-deps --force-reinstall /tmp/wheel_new/*.whl > "$O/wheel_install.txt" 2>&1
REPO=$PWD
st gate_installed bash -c "cd /tmp && env -u PYTHONPATH /tmp/wv/bin/python $REPO/tools/classical_host_gate.py --package-root '' check $REPO/$REC_HW $(for d in $REC_AR_BASE; do printf '%s ' $REPO/$d; done) $(for c in $C; do printf -- '--gpu-column %s ' $REPO/$c; done) --report $O/classical_gate_installed_wheel.json" > "$O/classical_gate_installed_wheel.txt" 2>&1
st installed_smoke bash -c "cd /tmp && env -u PYTHONPATH REPO=$REPO /tmp/wv/bin/python -" > "$O/installed_smoke.txt" 2>&1 <<'PY2'
import glob, os, numpy as np
import mojolearn
from mojolearn import ExponentialSmoothing, _backend, host_model
repo = os.environ["REPO"]
print("mojolearn", mojolearn.__file__, "vendor", mojolearn.vendor())
assert "/tmp/wv/" in mojolearn.__file__
print("inference routes", _backend._HOST_INFERENCE_MODULES)
print("host dir", sorted(os.path.basename(p) for p in glob.glob(os.path.join(os.path.dirname(mojolearn.__file__), "host", "*.so"))))
for lane in ("holtwinters", "holtwinters-multiplicative"):
    path = f"{repo}/bench/results/classical_host/2026-09-15-apple-m4-holtwinters/{lane}/base/model.npz"
    plain = ExponentialSmoothing.load(path)
    host = host_model(path)
    n = plain.n
    a = [np.asarray(v).tobytes() for v in (plain.forecast(512), plain.predict(0, n), plain.predict(n - 16, n + 16))]
    b = [np.asarray(v).tobytes() for v in (host.forecast(512), host.predict(0, n), host.predict(n - 16, n + 16))]
    assert a == b, lane
    print(lane, "plain class and host_model agree;", type(host).__name__, "bound to", host._bind().__file__)
    try:
        ExponentialSmoothing(np.ones(48, dtype=np.float32) + np.arange(48) % 12, seasonal_periods=12).fit()
    except NotImplementedError as exc:
        print("fit refuses:", str(exc)[:90])
    else:
        raise AssertionError("fit ran on the installed CPU-only wheel")
try:
    mojolearn.kpss_test(np.ones((64, 2), dtype=np.float32))
except Exception as exc:
    print("kpss_test refuses:", type(exc).__name__, str(exc)[:160])
else:
    raise AssertionError("kpss_test ran without its binding")
print("installed smoke OK")
PY2
cat "$O/step_exit_codes.txt" > "$O/SUMMARY.out"
grep -hE "^summary|require-columns 4|gate verdict|owed verdict|passed|failed|installed smoke OK" \
  "$O"/diff.*.txt "$O"/classical_gate_*.txt "$O"/owed_sabotage_check.txt "$O"/pytest.txt "$O"/installed_smoke.txt >> "$O/SUMMARY.out"
cat "$O/fit_symbols.txt" >> "$O/SUMMARY.out"
exit 0
