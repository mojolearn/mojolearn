# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Workstream E, CPU training for the arima, arima-011 and arima-seasonal-c
lanes (2026-09-14), checked from SOURCE so it runs on a box with nothing
built, plus a binding probe that runs only where the arima host binding is
built and the package took the CPU-only path.

What the source checks hold: the manifest declares the arima family, routes
`_mojolearn_arima`, covers the three ARIMA lanes (and par-arima since
lane/cpu-training-par-classical, 2026-09-15) and no
longer names ARIMA as having no CPU path; the binding registers the GPU
binding's fit, predict and forecast names; the oracle imports no GPU module
and nothing from `arima/` or any other package beyond the
`checks/numerics.mojo` seams; the oracle spells the device constructs a bit
claim rests on (DEVIATION 687's step, the two-rounding Jones association,
test_invparams' one-rounding association, the perturbation and the reset by
copy, the refusals by name); the sabotage define reaches the step and the
binding reads it back.

The binding probe: the loaded host binding is not a sabotage build. The
runtime arithmetic check (the three lane orders fit twice through the host fit
body and return the same bytes, `forecast(h)` equals `predict(n_obs,
n_obs + h)`, an in-sample prediction agrees with one straddling the end of
the series, and a second AR coefficient refuses by name) is native:
checks/native_tests/tsa_training_arima.mojo.

    cd python && python3 -m mojolearn.tests.test_cpu_training_arima
"""

# Gate-runner scope: the binding probe requires the CPU-only route.
GATE_BACKENDS = ("cpu",)
import re
import sys
from pathlib import Path

from mojolearn import _backend, host_surface

ROOT = Path(__file__).resolve().parents[3]

ORACLE = "arima/host/arima_oracle.mojo"
LANES = ("arima", "arima-011", "arima-seasonal-c")
#: lane/arima-exog (2026-09-15).
EXOG_LANES = ("arima-exog", "arima-exog-seasonal")
# The par-arima and par-forecast-arima lanes left the identity harness with
# the multi-GPU verification retirement (868702642c, 2026-10-31); this tuple
# is the arima family's CONTRACT, so a lane appearing in it without a test
# moving is how a family grows a member nobody checked.
GPU_IMPORTS = re.compile(r"^\s*from\s+(max\.gpu|std\.gpu)", re.M)


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def test_manifest_declares_the_arima_family():
    fam = host_surface.family("arima")
    assert fam["routes"] == "_mojolearn_arima"
    assert fam["training_lanes"] == LANES + EXOG_LANES
    assert ORACLE in fam["host_modules"] and (ROOT / ORACLE).is_file()
    assert (ROOT / host_surface.build_shim("arima")).is_file()
    assert (ROOT / host_surface.binding_source("arima")).is_file()
    assert host_surface.routed_modules()["_mojolearn_arima"] == "_mojolearn_arima_host"
    for lane in LANES:
        assert lane in host_surface.covered_lanes(), f"{lane} is not a covered training lane"
    assert "par-arima" not in host_surface.covered_lanes(), "a retired par-* lane is declared again"
    assert "ARIMA" not in host_surface.no_cpu_path_sentence(), host_surface.no_cpu_path_sentence()
    assert "seasonal ARIMA" in host_surface.training_sentence()


def test_binding_registers_the_gpu_names():
    # arima_predict and arima_forecast live in bindings/arima_host_predict_pack.mojo
    # since lane/inference-forecast-umap-pca (2026-09-15), shared with the
    # forecast inference binding; the reference binding imports them.
    # The typed bodies (`_order`, the fit body) moved to
    # arima/host/arima_host_entries.mojo (lane thin-trees-metrics, 2026-10-21).
    src = (_read(host_surface.binding_source("arima")) + _read("bindings/arima_host_predict_pack.mojo")
           + _read("arima/host/arima_host_entries.mojo"))
    exports = host_surface.family("arima")["exports"]
    gpu = _read("bindings/_mojolearn_arima.mojo")
    for name in ("arima_fit", "arima_predict", "arima_forecast", "arima_vendor", "arima_numeric_mode"):
        assert f'("{name}")' in src, f"the arima host binding does not register {name}"
        assert f'("{name}")' in gpu, f"{name} is not a GPU binding name"
        assert name in exports, f"the manifest does not list {name} for arima"
    for slots in ("arima_fit: params must contain 13 values (batch_size, n_obs, p,",
                  " n_obs, start, end, p, d, q, P, D, Q, s, k, n_exog), got ",
                  "params[12] is reserved and must be 0"):
        assert slots in src, f"the host binding does not carry the GPU slot check {slots!r}"
        assert slots in gpu, f"the GPU binding no longer carries {slots!r}"
    assert "validate_order(order)" in src, "the order is not validated by the device's own function"


def test_oracle_imports_no_gpu_and_no_device_module():
    text = _read(ORACLE)
    assert not GPU_IMPORTS.search(text), f"{ORACLE} imports a GPU module"
    assert not re.search(r"^\s*from .*import.*DeviceContext", text, re.M), f"{ORACLE} imports DeviceContext"
    imports = re.findall(r"^from\s+([\w.]+)\s+import", text, re.M)
    # arima/impl/idn_ls_math.mojo is the IDENTICAL least-squares arithmetic
    # the device and host share; it is itself CPU-safe (checked below).
    assert sorted(set(imports)) == ["arima.impl.idn_ls_math", "checks.numerics", "core.host_parallel",
                                    "core.host_predict_threads", "std.math", "std.memory", "std.sys.compile"], imports
    shared = _read("arima/impl/idn_ls_math.mojo")
    assert not GPU_IMPORTS.search(shared), "arima/impl/idn_ls_math.mojo imports a GPU module"
    assert sorted(set(re.findall(r"^from\s+([\w.]+)\s+import", shared, re.M))) == ["checks.numerics", "std.sys.compile"]


def test_oracle_spells_the_bit_carrying_constructs():
    text = _read(ORACLE)
    assert "comptime AH_FIT_H = Float32(0.0009765625)" in text, "DEVIATION 687's step"
    assert "var prod = ftz(a * mine[j - k - 1])" in text, "the Jones recursion rounds a * x on its own"
    assert "identical_mul_add(coef_a, new_params[j - k - 1], new_params[k])" in text, (
        "test_invparams fuses (coef * a) * x, one rounding"
    )
    assert "x_pert[idx] = ftz(ftz(xin[idx]) + h)" in text, "the perturbation"
    assert "x_pert[idx] = xin[idx]" in text, "the reset is a copy, not x + 0.0"
    assert "fout[b] = ftz(ftz(-base[b]) / scale)" in text
    assert "for i in range(q, m1):" in text, "the sigma2 fold starts at q"
    assert "Float32(n_obs - 1)" in text, "the objective scale uses the undifferenced length"
    assert "var neg = (bitcast[DType.uint32](v) >> 31) != 0" in text, "the intercept nudge reads the sign bit"
    assert "identical_mul_add(n_obs_ll_f, inner, b_sum_logFs)" in text
    assert "no CPU implementation of ARIMA with" in text


def test_sabotage_define_moves_the_step():
    text = _read(ORACLE)
    define = host_surface.sabotage_define("arima")
    assert define == "MOJOLEARN_HOST_SABOTAGE"
    assert f'is_defined["{define}"]()' in text
    assert "comptime if ARIMA_ORACLE_HOST_SABOTAGE:" in text
    assert "comptime AH_FIT_H_SABOTAGE = Float32(0.001953125)" in text
    assert "ARIMA_ORACLE_HOST_SABOTAGE" in _read(host_surface.binding_source("arima"))


def test_host_binding_is_not_a_sabotage_build():
    if _backend._CPU_ONLY is None:
        print("SKIP: a GPU set loaded; the host route is not taken here")
        return
    if "_mojolearn_arima_host" not in _backend.host_families_built():
        print("SKIP: the arima host binding is not built")
        return
    module = _backend.load_host_module("_mojolearn_arima_host")
    assert not bool(module.arima_host_sabotage()), "a sabotage build loaded outside the gate"


if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for name in names:
        globals()[name]()
        print("ok", name)
    print(f"{len(names)} passed")
    sys.exit(0)
