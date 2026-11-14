# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Workstream E batch 3, CPU training for the gbdt-symmetric lane
(2026-09-14), checked from SOURCE so it runs on a box with nothing built.

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

The runtime check (small host fits run twice, the refusals by name) moved to
the native check `checks/native_tests/gbdt_public_fits.mojo`
(`pixi run check-native-gbdt-public-fits-cpu`, lane tmig-trees, 2026-11-14).

    cd python && python3 -m mojolearn.tests.test_cpu_training_e3_gbdt
"""

# Gate-runner scope: host runtime checks require the CPU-only route.
GATE_BACKENDS = ("cpu",)
import re
import sys
from pathlib import Path

from mojolearn import host_surface

ROOT = Path(__file__).resolve().parents[3]

ORACLE = "gbdt/host/gbdt_oracle.mojo"
GPU_IMPORTS = re.compile(r"^\s*from\s+(max\.gpu|std\.gpu)", re.M)
REUSED_HOST_MODULES = (
    "gbdt/data/permutation.mojo",
    "gbdt/data/quantization.mojo",
    "gbdt/grid_creator/binarization.mojo",
    "gbdt/grid_creator/border_types_cells.mojo",
    "gbdt/grid_creator/gls_borders_cells.mojo",
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
    # The forest host binding stays loaded by path, never routed.
    assert host_surface.family("forest")["routes"] is None


def test_binding_registers_the_gpu_names():
    src = _read(host_surface.binding_source("gbdt"))
    exports = host_surface.family("gbdt")["exports"]
    for name in ("gbdt_fit", "gbdt_predict", "gbdt_model_dim", "gbdt_sigmoid_pair",
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
                 '"cat_features or one_hot_features outside SymmetricTree with Logloss"', '"eval_set outside SymmetricTree with Logloss',
                 "random_strength=", "boost_from_average=True",
                 "feature_fraction=", "an X carrying NaN"):
        assert f"_refuse(" in src and what in src, f"no by-name refusal for {what}"
    # The oracle is split by grow policy since 8b0ab3f6f (gbdt_oracle.mojo,
    # _onehot, _depthwise, _pointwise, _ordered); the binary-policy refusal
    # lives in the policies' files, so every oracle file is read.
    oracle = "".join(_read(str(p.relative_to(ROOT))) for p in sorted((ROOT / "gbdt" / "host").glob("gbdt_oracle*.mojo")))
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
    # The transforms' typed bodies moved to gbdt/host/gbdt_host_entries_impl.mojo
    # (lane thin-trees-metrics, 2026-10-21); the shell is the Python door.
    src = _read(host_surface.binding_source("gbdt")) + _read("gbdt/host/gbdt_host_entries_impl.mojo")
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
        "checks.numerics", "core.host_parallel", "gbdt.data.permutation", "gbdt.data.quantization",
        "gbdt.gpu_data.compressed_index_builder", "gbdt.gpu_data.feature_blocks",
        "gbdt.gpu_data.grid_policy", "gbdt.gpu_util.kernel.random_gen",
        "gbdt.grid_creator.binarization", "gbdt.grid_creator.border_types_cells",
        "gbdt.grid_creator.gls_borders_cells", "gbdt.host.gbdt_oracle_eval",
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
    assert "est_p_leaves[leaf], lr, cursors[p][row]" in text
    assert "if function_value <= next_value:" in text, "AnyImprovement"
    assert "if left_sz < right_sz:" in text, "the sibling tie computes the right child"
    # Since 121a82d3b the grid takes the device's own per-column border search
    # (gls_borders_cells.gls_column for GreedyLogSum, border_types_cells.
    # border_type_column for the other six), one shared function per column,
    # instead of the host-only phase B restatement.
    assert "nb = gls_column(" in text and "nb = border_type_column(" in text, (
        "the grid must take the shared per-column border search"
    )
    cells = _read("gbdt/grid_creator/gls_borders_cells.mojo") + _read("gbdt/grid_creator/border_types_cells.mojo")
    assert "def gls_column(" in cells and "def border_type_column(" in cells
    assert "host_parallelize_pool_env(_col, n_features)" in text
    assert "fp + f * out_cap," in text and "cp[f] = nb" in text, (
        "parallel feature searches must publish into disjoint flat slots"
    )
    assert "_ = len(idx)" in text, "the sampled-index owner must outlive the worker join"
    refusal = "There are nan factors and nan values for float features are"
    # The workers mark a Forbidden-NaN column (`mp[f] = -1`) and return; after
    # the join the first such column in feature order raises the public
    # refusal, so the sentence and the column it names are the serial ones.
    launch = text.index("host_parallelize_pool_env(_col, n_features)")
    mark = text.rindex("mp[f] = -1", 0, launch)
    after = text[launch:text.index("\ndef ", launch)]
    assert mark < launch and refusal in after and "if modes[f] < 0:" in after, (
        "Forbidden NaNs must keep their exact serial public refusal, in feature order"
    )


def test_sabotage_define_moves_the_leaf_walker():
    text = _read(ORACLE)
    define = host_surface.sabotage_define("gbdt")
    assert define == "MOJOLEARN_HOST_SABOTAGE"
    assert f'is_defined["{define}"]()' in text
    assert "comptime if GBDT_ORACLE_HOST_SABOTAGE:" in text
    assert "lambda_reg = lambda_reg + 1.0" in text
    assert "GBDT_ORACLE_HOST_SABOTAGE" in _read(host_surface.binding_source("gbdt"))


if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for name in names:
        globals()[name]()
        print("ok", name)
    print(f"{len(names)} passed")
    sys.exit(0)
