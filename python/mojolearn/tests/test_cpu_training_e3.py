# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Workstream E batch 3, CPU training for the rf-clf and rf-reg lanes
(2026-09-14), checked from SOURCE so it runs on a box with nothing built,
plus a runtime check that runs only where the rf host binding is built and
the package took the CPU-only path.

What the source checks hold: the manifest declares the rf family, routes
`_mojolearn_rf`, covers rf-clf and rf-reg and no longer names the random
forests as having no CPU training path; the binding registers the GPU
binding's fit, export and predict names and leaves the shard and
non-resident GPU engine entries absent so they refuse by name (the weighted
fit and the resident parallel_groves entries joined on 2026-09-15, see
test_cpu_training_forest_variants.py); the oracle imports no GPU
module and nothing from `ensemble/` beyond the `checks/numerics.mojo`
seams; the oracle spells the device constructs a bit claim rests on (the
pinned width 32 reduction, the ftz-compared quantile unique, the bootstrap
stride, the pure-node rule); the sabotage define reaches the bootstrap draw
and the binding reads it back.

The runtime check (skipped, and SAID to be skipped, when the binding is
absent or a GPU set loaded): both estimators fit twice through the host
binding on a small draw and return the same bytes, predictions in range.
It is a plumbing check. The bit claim against the GPU columns is the CPU
identity gate's, not this file's.

    cd python && python3 -m mojolearn.tests.test_cpu_training_e3
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

ORACLE = "ensemble/host/rf_oracle.mojo"
GPU_IMPORTS = re.compile(r"^\s*from\s+(max\.gpu|std\.gpu)", re.M)
FIT_NAMES = tuple(
    f"rf_{kind}_fit{suffix}"
    for kind in ("classifier", "regressor")
    for suffix in ("", "_export", "_rowmajor", "_rowmajor_export")
)


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def test_manifest_declares_the_rf_family():
    fam = host_surface.family("rf")
    assert fam["routes"] == "_mojolearn_rf"
    assert fam["training_lanes"][:2] == ("rf-clf", "rf-reg")
    assert ORACLE in fam["host_modules"] and (ROOT / ORACLE).is_file()
    assert (ROOT / host_surface.build_shim("rf")).is_file()
    assert (ROOT / host_surface.binding_source("rf")).is_file()
    assert host_surface.routed_modules()["_mojolearn_rf"] == "_mojolearn_rf_host"
    for lane in ("rf-clf", "rf-reg"):
        assert lane in host_surface.covered_lanes(), f"{lane} is not a covered training lane"
    assert "random forest" not in host_surface.no_cpu_path_sentence(), host_surface.no_cpu_path_sentence()
    assert "the random forest classifier" in host_surface.training_sentence()
    # The forest host binding stays loaded by path, never routed.
    assert host_surface.family("forest")["routes"] is None


def test_binding_registers_the_gpu_names():
    src = _read(host_surface.binding_source("rf"))
    exports = host_surface.family("rf")["exports"]
    for name in FIT_NAMES + ("forest_export", "forest_export_legacy", "forest_export_release",
                             "rf_predict_proba", "rf_predict_reg", "rf_vendor", "rf_numeric_mode",
                             "rf_classifier_fit_shard", "rf_regressor_fit_shard"):
        assert f'("{name}")' in src, f"the rf host binding does not register {name}"
        assert name in exports, f"the manifest does not list {name} for rf"
    for absent in ("rf_predict_proba_gpu_parallel", "rf_predict_reg_gpu_parallel",
                   "forest_predict_resident_gpu", "forest_predict_resident_into_gpu"):
        assert f'("{absent}")' not in src, f"{absent} must stay absent so it refuses by name"


def test_oracle_imports_no_gpu_and_no_device_module():
    text = _read(ORACLE)
    assert not GPU_IMPORTS.search(text), f"{ORACLE} imports a GPU module"
    assert not re.search(r"^\s*from .*import.*DeviceContext", text, re.M), f"{ORACLE} imports DeviceContext"
    imports = re.findall(r"^from\s+([\w.]+)\s+import", text, re.M)
    # THE LIST IS AN ALLOWLIST, NOT A RECORD. It exists so a new import into
    # the oracle is REVIEWED rather than noticed, because this file has to be
    # a bit-exact restatement of the device RF.
    #
    # `std.builtin.sort` was admitted 2026-09-19 (4c80aeb2c) and the argument
    # is worth keeping next to it, since an unstable sort in an oracle is
    # normally exactly the bug the `ties` fixture hunts: the quantile keys are
    # INTEGERS already encoding the complete float total order, so two equal
    # keys are identical BITS. An unstable sort can therefore only reorder
    # elements that are indistinguishable, and stability is unobservable. It
    # replaced an O(n^2) insertion sort.
    #
    # A future import needs its own sentence here before it is added.
    assert sorted(set(imports)) == [
        "checks.numerics", "std.builtin.sort", "std.math", "std.memory", "std.sys.compile",
    ], imports


def test_oracle_spells_the_bit_carrying_constructs():
    text = _read(ORACLE)
    assert "comptime RF_PINNED_LANES = 32" in text, "the split reduction width is pinned to 32"
    assert "comptime RF_TPB = 128" in text
    assert "if ftz(cur) != ftz(prev):" in text, "the quantile unique compares flushed operands"
    assert "comptime RF_RNG_STRIDE = 110592" in text
    assert "var terminal = s.pure != Int32(0)" in text, "a pure node is a leaf (RETRY_PURE_NODES off)"
    assert "x[i] = ftz(x[i])" in text, "X is flushed before the quantiles"
    assert "identical_mul_add(-val, val, gain)" in text


def test_sabotage_define_moves_the_bootstrap():
    text = _read(ORACLE)
    define = host_surface.sabotage_define("rf")
    assert define == "MOJOLEARN_HOST_SABOTAGE"
    assert f'is_defined["{define}"]()' in text
    assert "comptime if RF_ORACLE_HOST_SABOTAGE:" in text
    assert "sub = sub + UInt64(1)" in text, "the sabotage arm does not move the bootstrap subsequence"
    assert "RF_ORACLE_HOST_SABOTAGE" in _read(host_surface.binding_source("rf"))



@reference_training()
def test_random_forests_fit_on_the_host_when_built():
    if _backend._CPU_ONLY is None:
        print("SKIP: a GPU set loaded; the host route is not taken here")
        return
    if "_mojolearn_rf_host" not in _backend.host_families_built():
        print("SKIP: the rf host binding is not built")
        return
    import numpy as np
    module = _backend.load_host_module("_mojolearn_rf_host")
    assert not bool(module.rf_host_sabotage()), "a sabotage build loaded outside the gate"
    rng = np.random.default_rng(0)
    x = rng.standard_normal((600, 6)).astype(np.float32)
    yc = (x[:, 3] + 0.5 * x[:, 4] > 0).astype(np.int32)
    yr = (x[:, 0] - 2.0 * x[:, 1]).astype(np.float32)
    outs = []
    for _ in range(2):
        c = mojolearn.RandomForestClassifier(n_estimators=4, max_depth=5, random_state=7).fit(x, yc)
        r = mojolearn.RandomForestRegressor(n_estimators=4, max_depth=5, random_state=7).fit(x, yr)
        outs.append((np.asarray(c.predict_proba(x)).tobytes(), np.asarray(r.predict(x)).tobytes()))
    assert outs[0] == outs[1], "two host fits returned different bytes"
    proba = np.asarray(mojolearn.RandomForestClassifier(
        n_estimators=4, max_depth=5, random_state=7).fit(x, yc).predict_proba(x))
    assert proba.shape == (600, 2) and float(proba.min()) >= 0.0 and float(proba.max()) <= 1.0


if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for name in names:
        globals()[name]()
        print("ok", name)
    print(f"{len(names)} passed")
    sys.exit(0)
