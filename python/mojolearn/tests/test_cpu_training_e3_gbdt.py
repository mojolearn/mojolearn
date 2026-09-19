# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Workstream E batch 3, CPU training for the gbdt-symmetric lane
(2026-09-14), checked from SOURCE so it runs on a box with nothing built,
plus a runtime check that runs only where the gbdt host binding is built and
the package took the CPU-only path.

What the source checks hold: the manifest declares the gbdt family, routes
`_mojolearn_gbdt`, covers gbdt-symmetric only and names the rest of gradient
boosting training as having no CPU path; the binding registers the GPU
binding's fit, predict, model-dim and sigmoid names and leaves the
multi-dimensional predict and the adapters' binary transforms absent so
they refuse by name (the ordered and FeatureFreq fits are registered since
lane/cpu-training-gbdt-ordered, test_cpu_training_gbdt_ordered.py); every parameter
refusal inside `gbdt_fit` carries the sentence the CPU identity gate's
column check keys on; the oracle imports no GPU module, and the gbdt host
modules it reuses import none either; the oracle spells the device
constructs a bit claim rests on (the pinned 32, the dithered quantizer, the
row-count scale limit, the half-byte block partial flush, the Newton epsilon,
the fused cursor update, the phase B border search's subnormal flush); the sabotage define reaches the leaf walker and
the binding reads it back.

The runtime check (skipped, and SAID to be skipped, when the binding is
absent or a GPU set loaded): a small Logloss fit runs twice through the host
binding and returns the same model text and predictions, and a Lossguide fit
with the Cosine score refuses by name. It is a plumbing check. The bit claim against the GPU
columns is the CPU identity gate's, not this file's.

    cd python && python3 -m mojolearn.tests.test_cpu_training_e3_gbdt
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

ORACLE = "gbdt/host/gbdt_oracle.mojo"
GPU_IMPORTS = re.compile(r"^\s*from\s+(max\.gpu|std\.gpu)", re.M)
REUSED_HOST_MODULES = (
    "gbdt/data/permutation.mojo",
    "gbdt/data/quantization.mojo",
    "gbdt/grid_creator/binarization.mojo",
    "gbdt/gpu_data/compressed_index_builder.mojo",
    "gbdt/gpu_data/feature_blocks.mojo",
    "gbdt/gpu_data/grid_policy.mojo",
    "gbdt/gpu_data/gpu_structures.mojo",
    "gbdt/options/data_processing_options.mojo",
)
REFUSAL = "no CPU implementation of _mojolearn_gbdt.gbdt_fit for "


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def test_manifest_declares_the_gbdt_family():
    fam = host_surface.family("gbdt")
    assert fam["routes"] == "_mojolearn_gbdt"
    assert fam["training_lanes"][0] == "gbdt-symmetric"
    assert ORACLE in fam["host_modules"] and (ROOT / ORACLE).is_file()
    assert (ROOT / host_surface.build_shim("gbdt")).is_file()
    assert (ROOT / host_surface.binding_source("gbdt")).is_file()
    assert host_surface.routed_modules()["_mojolearn_gbdt"] == "_mojolearn_gbdt_host"
    covered = host_surface.covered_lanes()
    assert "gbdt-symmetric" in covered
    for lane in ():
        assert lane not in covered, f"{lane} is declared covered and has no host trainer"
    sentence = host_surface.no_cpu_path_sentence()
    assert "gradient boosting training outside its declared lanes" in sentence, sentence
    # The forest host binding stays loaded by path, never routed.
    assert host_surface.family("forest")["routes"] is None


def test_binding_registers_the_gpu_names():
    src = _read(host_surface.binding_source("gbdt"))
    exports = host_surface.family("gbdt")["exports"]
    for name in ("gbdt_fit", "gbdt_predict", "gbdt_model_dim", "gbdt_sigmoid",
                 "gbdt_vendor", "gbdt_numeric_mode", "gbdt_binary_probabilities",
                 "gbdt_binary_classes", "gbdt_predict_multi"):
        assert f'("{name}")' in src, f"the gbdt host binding does not register {name}"
        assert name in exports, f"the manifest does not list {name} for gbdt"
    for absent in ("gbdt_per_round_paths",
                   "gbdt_parallel_available", "pointwise_parallel_available"):
        assert f'("{absent}")' not in src, f"{absent} must stay absent so it refuses by name"


def test_every_fit_refusal_names_the_missing_cpu_implementation():
    src = _read(host_surface.binding_source("gbdt"))
    assert f'"{REFUSAL}"' in src
    for what in ("loss='", "grow_policy code", "use_pointwise_searcher=True",
                 "score_function code", "leaf_estimation_method code",
                 "bootstrap_type='", '"sample_weight"', '"class_weights outside',
                 '"cat_features or one_hot_features outside SymmetricTree with Logloss"', '"eval_set"',
                 "random_strength=", "boost_from_average=True",
                 "feature_fraction=", "an X carrying NaN"):
        assert f"_refuse(" in src and what in src, f"no by-name refusal for {what}"
    oracle = _read(ORACLE)
    assert "no CPU implementation of _mojolearn_gbdt.gbdt_fit for a" in oracle, (
        "the binary-policy refusal must carry the gate's sentence"
    )


def test_nan_modes_and_adapters_are_declared():
    """lane/cpu-training-gbdt-losses, 2026-09-15: gbdt-nan-modes and the two
    adapter lanes train through the same binding. The binary transforms are
    the device kernel's per-element body, and NaN is accepted only on the
    measured symmetric Logloss fit."""
    fam = host_surface.family("gbdt")
    for lane in ("gbdt-nan-modes", "gbdt-adapter-clf", "gbdt-adapter-reg"):
        assert lane in fam["training_lanes"], lane
        assert lane in host_surface.covered_lanes(), lane
    for cls in ("GradientBoostingClassifier", "GradientBoostingRegressor"):
        assert cls in fam["classes"], cls
    src = _read(host_surface.binding_source("gbdt"))
    assert "var positive = ftz(identical_sigmoid(ftz(margin)))" in src, "the probability body"
    assert "Scalar[dtype](ftz(Float32(1) - positive))" in src, "the flushed complement"
    assert "(bits & UInt32(0x80000000)) == 0 and (bits & UInt32(0x7fffffff)) != 0" in src, (
        "the class code is strict raw > 0 read from the bits"
    )
    assert '"binary prediction: finite Float32 margins required"' in src
    assert "if is_rmse or grow_code != 0 or is_pointwise or is_multi or len(flags) != 0:" in src, "NaN refused outside the measured fit"
    kernel = _read("gbdt/binary_prediction.mojo")
    assert "var positive = ftz(identical_sigmoid(ftz(margin)))" in kernel, (
        "the device kernel moved; the host restatement must move with it"
    )
    assert "either NaN mode and the classifier and regressor adapters" in fam["display"], fam["display"]


def test_oracle_imports_no_gpu_module():
    text = _read(ORACLE)
    assert not GPU_IMPORTS.search(text), f"{ORACLE} imports a GPU module"
    imports = sorted(set(re.findall(r"^from\s+([\w.]+)\s+import", text, re.M)))
    assert imports == [
        "checks.numerics", "gbdt.data.permutation", "gbdt.data.quantization",
        "gbdt.gpu_data.compressed_index_builder", "gbdt.gpu_data.feature_blocks",
        "gbdt.gpu_data.grid_policy", "gbdt.grid_creator.binarization",
        "gbdt.options.data_processing_options",
        "std.math", "std.memory", "std.sys.compile",
    ], imports
    for rel in REUSED_HOST_MODULES:
        body = _read(rel)
        assert not GPU_IMPORTS.search(body), f"{rel} imports a GPU module"
        assert "DeviceContext" not in "".join(re.findall(r"^\s*from .*$", body, re.M)), rel


def test_oracle_spells_the_bit_carrying_constructs():
    text = _read(ORACLE)
    assert "comptime GBDT_PINNED_SM = 32" in text
    assert "comptime GBDT_HB_BLOCK = 512" in text
    assert "UInt32(pos) * UInt32(2654435761)" in text, "the dither hash"
    assert "var scaled = ftz(ftz(val) * fixed_scale)" in text, "the dithered quantizer"
    assert "Int64((1 << 30) - 1) - Int64(row_count)" in text, "the row-count scale limit"
    assert "q = q + Int32(v * fixed_scale)" in text, "the half-byte block partial flush"
    assert "512 * (tid // 32) + (tid & 24) + bin" in text, "the half-byte replica slot"
    assert "comptime EPS_1E20F = Float64(Float32(1e-20))" in text
    assert "identical_mul_add(estimated[leaf], lr, cursor[row])" in text
    assert "if function_value <= next_value:" in text, "AnyImprovement"
    assert "if left_sz < right_sz:" in text, "the sibling tie computes the right child"
    assert "clean.append(ftz(values[i]))" in text, "the phase B border search input flush"
    assert "borders.append(ftz(half_below + half_above))" in text, "the phase B border midpoint flush"
    assert "var q = _calc_quantization_phase_b(col^, border_count, nan_mode, border_type)" in text, (
        "the grid must take the phase B border search, not the imported calc_quantization"
    )


def test_sabotage_define_moves_the_leaf_walker():
    text = _read(ORACLE)
    define = host_surface.sabotage_define("gbdt")
    assert define == "MOJOLEARN_HOST_SABOTAGE"
    assert f'is_defined["{define}"]()' in text
    assert "comptime if GBDT_ORACLE_HOST_SABOTAGE:" in text
    assert "lambda_reg = lambda_reg + 1.0" in text
    assert "GBDT_ORACLE_HOST_SABOTAGE" in _read(host_surface.binding_source("gbdt"))



@reference_training()
def test_gradient_boosting_fits_on_the_host_when_built():
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
    y = (x[:, 3] + 0.5 * x[:, 4] > 0).astype(np.int32)
    outs = []
    for _ in range(2):
        m = mojolearn.GradientBoosting(n_estimators=3, max_depth=3, loss="Logloss").fit(x, y)
        outs.append((m.model_, np.asarray(m.predict(x)).tobytes()))
    assert outs[0] == outs[1], "two host fits returned different bytes"
    proba = np.asarray(m.predict_proba(x))
    assert proba.shape == (600, 2) and float(proba.min()) >= 0.0 and float(proba.max()) <= 1.0
    # the NaN modes and the classifier adapter's transforms
    xn = x.copy()
    xn[::8, 2:4] = np.float32(np.nan)
    for mode in ("Min", "Max"):
        a = mojolearn.GradientBoosting(n_estimators=3, max_depth=3, loss="Logloss", nan_mode=mode).fit(xn, y)
        b = mojolearn.GradientBoosting(n_estimators=3, max_depth=3, loss="Logloss", nan_mode=mode).fit(xn, y)
        assert a.model_ == b.model_, f"two host fits under nan_mode={mode} returned different bytes"
        assert np.asarray(a.predict(xn)).tobytes() == np.asarray(b.predict(xn)).tobytes(), mode
    clf = mojolearn.GradientBoostingClassifier(n_estimators=3, max_depth=3).fit(x, y)
    margins = np.asarray(clf.decision_function(x))
    pp = np.asarray(clf.predict_proba(x))
    assert pp.shape == (600, 2) and pp.dtype == np.float32
    labels = np.asarray(clf.predict(x))
    assert labels.tolist() == [1 if v > 0 else 0 for v in margins.tolist()], "strict raw > 0"
    try:
        mojolearn.GradientBoosting(n_estimators=2, max_depth=3, loss="RMSE").fit(xn, x[:, 0])
    except Exception as exc:
        assert "no CPU implementation of" in str(exc), str(exc)
    else:
        raise AssertionError("an RMSE fit on an X carrying NaN did not refuse on the host binding")
    try:
        mojolearn.GradientBoosting(n_estimators=2, max_depth=3, loss="Logloss",
                                   grow_policy="Lossguide", score_function="Cosine").fit(x, y)
    except Exception as exc:  # the binding's Error crosses as a Python exception
        assert "no CPU implementation of" in str(exc), str(exc)
    else:
        raise AssertionError("a Lossguide Cosine fit did not refuse on the host binding")


if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for name in names:
        globals()[name]()
        print("ok", name)
    print(f"{len(names)} passed")
    sys.exit(0)
