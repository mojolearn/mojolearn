# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Workstream E batch 3, CPU training for the gbdt-rmse lane (2026-09-14),
checked from SOURCE so it runs on a box with nothing built, plus a runtime
check that runs only where the gbdt host binding is built and the package
took the CPU-only path.

What the source checks hold: the manifest declares gbdt-rmse on the gbdt
family with the RMSE oracle among its host modules; the binding dispatches
`loss="RMSE"` to the RMSE oracle, lifts the loss refusal for RMSE only,
lifts `boost_from_average=True` for RMSE only, and refuses by name any
`leaf_estimation_iterations` other than 1 under RMSE; the RMSE oracle imports
no GPU module and no module outside the symmetric oracle's reuse set; it
spells the device constructs a bit claim rests on (the Float32 narrowing of
the target average, the flushed der, the negated squared residual, the
Hessian guard and the `1e-20` epsilon, the minimum leaf weight zero, the
fused cursor update, the `bias` record after `losses`); the sabotage define
reaches the RMSE leaf; the CPU identity gate runs by hand since 2026-09-15 (no push trigger).

The runtime check (skipped, and SAID to be skipped, when the binding is
absent or a GPU set loaded): a small RMSE fit runs twice through the host
binding and returns the same model text and predictions, the text carries a
`bias` record, and `leaf_estimation_iterations=2` refuses by name. It is a
plumbing check. The bit claim against the GPU columns is the CPU identity
gate's, not this file's.

    cd python && python3 -m mojolearn.tests.test_cpu_training_e3_gbdt_rmse
"""

# Gate-runner scope: host runtime checks require the CPU-only route.
GATE_BACKENDS = ("cpu",)
import re
import sys
from pathlib import Path

import mojolearn
from mojolearn._cpu_reference import reference_training
from mojolearn import _backend, host_surface

ROOT = Path(__file__).resolve().parents[3]

ORACLE = "gbdt/host/gbdt_oracle_rmse.mojo"
GPU_IMPORTS = re.compile(r"^\s*from\s+(max\.gpu|std\.gpu)", re.M)
REFUSAL = "no CPU implementation of _mojolearn_gbdt.gbdt_fit for "


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def test_manifest_declares_gbdt_rmse():
    fam = host_surface.family("gbdt")
    assert "gbdt-rmse" in fam["training_lanes"]
    assert ORACLE in fam["host_modules"] and (ROOT / ORACLE).is_file()
    assert "gbdt-rmse" in host_surface.covered_lanes()
    assert "RMSE" in host_surface.TRAINING_LANE_NAMES["gbdt-rmse"]
    sentence = host_surface.no_cpu_path_sentence()
    assert "gradient boosting training outside its declared lanes" in sentence, sentence


def test_binding_dispatches_rmse_and_lifts_only_its_refusals():
    src = _read(host_surface.binding_source("gbdt"))
    assert "from gbdt.host.gbdt_oracle_rmse import" in src
    assert 'var is_rmse = loss == String("RMSE")' in src
    assert 'if loss != String("Logloss") and not is_rmse and not is_pointwise and not is_multi:' in src
    # lane/catboost-parity: MAE, Quantile and MAPE joined RMSE's seeded
    # cursor (their CalcSampleQuantile constant)
    assert "if boost_from_average == 1 and not is_rmse and not quantile_family:" in src
    assert "if is_rmse and leaf_iterations >= 0 and leaf_iterations != 1:" in src
    assert "gbdt_rmse_host_fit(" in src and "gbdt_rmse_host_model_text(fit)" in src
    assert "boost_from_average != 0" in src, "unset and True must resolve the seeded cursor"
    assert f'"{REFUSAL}"' in src


def test_oracle_imports_no_gpu_module():
    text = _read(ORACLE)
    assert not GPU_IMPORTS.search(text), f"{ORACLE} imports a GPU module"
    imports = sorted(set(re.findall(r"^from\s+([\w.]+)\s+import", text, re.M)))
    assert imports == [
        "checks.numerics", "gbdt.gpu_data.compressed_index_builder",
        "gbdt.gpu_data.feature_blocks", "gbdt.gpu_data.grid_policy",
        "gbdt.host.gbdt_oracle", "std.memory",
    ], imports
    assert "optimal_const_for_loss" not in "".join(re.findall(r"^from .*$", text, re.M)), (
        "the optimum constant module imports a kernel module; restate it"
    )


def test_oracle_spells_the_bit_carrying_constructs():
    text = _read(ORACLE)
    assert "return Float64(Float32(target_sum / summary_weight))" in text, "the float return"
    assert "var summary_weight = Float64(n_rows)" in text, "the exact unweighted count"
    assert "var der = ftz(weight * (relev - val))" in text, "the flushed der"
    assert "var score = -weight * ((relev - val) * (relev - val))" in text, "the negated score"
    assert "s_g[t] = abs(r.der)" in text, "the der magnitude reads the flushed plane"
    assert "var hessian = w + reg" in text
    assert "if hessian <= Float32(0.0):" in text
    assert "var v = g / (hessian + Float32(1e-20))" in text
    assert "if w < GBDT_RMSE_MIN_LEAF_WEIGHT:" in text
    assert "comptime GBDT_RMSE_MIN_LEAF_WEIGHT = Float32(1e-20)" in text
    assert "identical_mul_add(leaf_values[leaf], lr, cursor[row])" in text
    assert "model_leaves.append(leaf_values[i] * lr)" in text, "the host rescale"
    assert 'String("bias ") + gbdt_f64_token(fit.bias)' in text
    # a zero bias writes no record; zero BY BITS since lane/catboost-parity
    # (a -0.0 bias is theirs to report and is written)
    assert "if bitcast[DType.uint64](fit.bias) == UInt64(0):" in text, "a zero bias writes no record"
    assert "var cursor = List[Float32](length=n_rows, fill=start_value)" in text


def test_level_loop_twins_are_named_in_both_oracles():
    assert "TWIN" in _read(ORACLE) and "gbdt_oracle_rmse.mojo" in _read("gbdt/host/gbdt_oracle.mojo")


def test_sabotage_define_moves_the_rmse_leaf():
    text = _read(ORACLE)
    assert host_surface.sabotage_define("gbdt") == "MOJOLEARN_HOST_SABOTAGE"
    assert "GBDT_ORACLE_HOST_SABOTAGE" in text
    assert "comptime if GBDT_ORACLE_HOST_SABOTAGE:" in text
    assert "reg = reg + Float32(1.0)" in text



@reference_training()
def test_gradient_boosting_rmse_fits_on_the_host_when_built():
    if _backend._CPU_ONLY is None:
        print("SKIP: a GPU set loaded; the host route is not taken here")
        return
    if "_mojolearn_gbdt_host" not in _backend.host_families_built():
        print("SKIP: the gbdt host binding is not built")
        return
    import numpy as np
    module = _backend.load_host_module("_mojolearn_gbdt_host")
    assert not bool(module.gbdt_host_sabotage()), "a sabotage build loaded outside the gate"
    rng = np.random.default_rng(0)
    x = rng.standard_normal((600, 6)).astype(np.float32)
    y = (x[:, 3] + 0.5 * x[:, 4] + 2.0).astype(np.float32)
    outs = []
    for _ in range(2):
        m = mojolearn.GradientBoosting(n_estimators=3, max_depth=3, loss="RMSE").fit(x, y)
        outs.append((m.model_, np.asarray(m.predict(x)).tobytes()))
    assert outs[0] == outs[1], "two host fits returned different bytes"
    assert "\nbias " in str(m.model_), "an unset boost_from_average under RMSE must seed a bias"
    assert m.bias_ != 0.0
    try:
        mojolearn.GradientBoosting(n_estimators=2, max_depth=3, loss="RMSE",
                                   leaf_estimation_iterations=2).fit(x, y)
    except Exception as exc:  # the binding's Error crosses as a Python exception
        assert "no CPU implementation of" in str(exc), str(exc)
    else:
        raise AssertionError("an RMSE fit at two leaf iterations did not refuse on the host binding")


if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for name in names:
        globals()[name]()
        print("ok", name)
    print(f"{len(names)} passed")
    sys.exit(0)
