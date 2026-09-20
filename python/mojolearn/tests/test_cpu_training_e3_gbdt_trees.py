# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Workstream E batch 3, CPU training for the gbdt-depthwise and
gbdt-lossguide lanes (2026-09-14), checked from SOURCE so it runs on a box
with nothing built, plus a runtime check that runs only where the gbdt host
binding is built and the package took the CPU-only path.

What the source checks hold: the manifest covers both lanes in the gbdt
family and lists both new host modules; the two oracles import no GPU
module, and the host modules they reuse import none either; the driver
spells the device constructs a bit claim rests on (the leafwise argmax tie,
the zero-part split, the parent `score_before`, the sibling rule's import,
the `<=` terminal size test, the right child at `leavesCount + i`, the
Lossguide argmin with no sign test, the NewtonL2 plane, the host scale, the
64-bit weight tokens); the binding dispatches on the policy, refuses the
other score functions and the non-symmetric knobs by name, and its predict
parser reads `ntree` and `node` records; the CPU identity gate runs by hand since 2026-09-15 (no push trigger).

The runtime check (skipped, and SAID to be skipped, when the binding is
absent or a GPU set loaded): a small Depthwise fit and a small Lossguide fit
each run twice through the host binding, return the same model text and
predictions, carry `ntree` records, and survive a save and load. It is a
plumbing check. The bit claim against the GPU columns is the CPU identity
gate's, not this file's.

    cd python && python3 -m mojolearn.tests.test_cpu_training_e3_gbdt_trees
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

DRIVER = "gbdt/host/gbdt_oracle_depthwise.mojo"
LOSSGUIDE = "gbdt/host/gbdt_oracle_lossguide.mojo"
GPU_IMPORTS = re.compile(r"^\s*from\s+(max\.gpu|std\.gpu)", re.M)
REUSED_HOST_MODULES = (
    "checks/fixed_point.mojo",
    "gbdt/methods/greedy_subsets_searcher/split_properties_helper.mojo",
    "gbdt/host/gbdt_oracle.mojo",
)
REFUSAL = "no CPU implementation of _mojolearn_gbdt.gbdt_fit for "


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def test_manifest_covers_both_lanes():
    fam = host_surface.family("gbdt")
    assert fam["training_lanes"][:4] == ("gbdt-symmetric", "gbdt-rmse", "gbdt-depthwise", "gbdt-lossguide")
    for rel in (DRIVER, LOSSGUIDE):
        assert rel in fam["host_modules"] and (ROOT / rel).is_file(), rel
    covered = host_surface.covered_lanes()
    assert "gbdt-depthwise" in covered and "gbdt-lossguide" in covered
    # The sentence still says gradient boosting training is what has no CPU
    # route, and it now says WHICH training (lane/close-no-cpu-path-gbdt,
    # 2026-09-20: the entry that ended "among them" became six numbered
    # entries). Pin the two the depthwise and lossguide lanes care about
    # rather than the old prose, which would have gone red on any rewrite
    # and green on a rewrite that said nothing.
    sentence = host_surface.no_cpu_path_sentence()
    assert sentence.startswith("(1) gradient boosting training"), sentence
    assert "(6) gradient boosting training at a (loss, grow_policy" in sentence, sentence
    assert "sample weights" in sentence and "CTR categorical column" in sentence, sentence
    # and it must NOT claim the two lanes above have no CPU route
    assert "Depthwise with Logloss" not in sentence, sentence


def test_oracles_import_no_gpu_module():
    for rel in (DRIVER, LOSSGUIDE):
        text = _read(rel)
        assert not GPU_IMPORTS.search(text), f"{rel} imports a GPU module"
    imports = sorted(set(re.findall(r"^from\s+([\w.]+)\s+import", _read(DRIVER), re.M)))
    assert imports == [
        "checks.fixed_point", "checks.numerics", "gbdt.data.permutation",
        "gbdt.gpu_data.compressed_index_builder", "gbdt.gpu_data.feature_blocks",
        "gbdt.gpu_data.grid_policy", "gbdt.gpu_util.kernel.random_gen",
        "gbdt.host.gbdt_oracle", "gbdt.host.gbdt_oracle_losses",
        "gbdt.host.gbdt_oracle_lossguide",
        "gbdt.methods.greedy_subsets_searcher.split_properties_helper",
        "std.math", "std.memory",
    ], imports
    lg_imports = sorted(set(re.findall(r"^from\s+([\w.]+)\s+import", _read(LOSSGUIDE), re.M)))
    assert lg_imports == ["checks.numerics", "gbdt.host.gbdt_oracle"], lg_imports
    for rel in REUSED_HOST_MODULES:
        body = _read(rel)
        assert not GPU_IMPORTS.search(body), f"{rel} imports a GPU module"


def test_driver_spells_the_bit_carrying_constructs():
    text = _read(DRIVER)
    assert "if gain > blk_gain:" in text, "the per-block argmax, strict, ascending bins"
    assert "(bf // GBDT_LEAFWISE_BLOCK) % argmax_blocks != bx" in text, "the score block stride"
    assert "weight_left < Float32(1e-20) or weight_right < Float32(1e-20)" in text
    assert "_add_leaf_cosine(part_stat, part_weight, lambda_l2, score_b, denum_sqr_b)" in text
    assert "var cand_gain = -our_gain" in text, "the host reduce negates the kernel gain"
    assert "if Float64(leaf.size) <= min_leaf_size:" in text, "IsTerminalLeaf's <="
    assert "var right_id = leaves_count + i" in text
    assert "leaves[i].best_defined and leaves[i].best_gain < Float32(0.0)" in text, "Depthwise sign test"
    assert "Float32(choose_scale(mag, n_rows))" in text, "the host scale"
    assert "build_necessary_histograms(records)" in text
    assert "depth_arg = iteration - 1" in text
    assert "gbdt_f64_token(m.leaf_weights[leaf_lo + i])" in text, "the 64-bit weight records"
    assert "row_index[fill[bins[r]]] = r" in text, "the stable grouping"
    lg = _read(LOSSGUIDE)
    assert "if gains[i] < best_gain:" in lg and "var best_gain = Float32.MAX" in lg
    assert "var plane0 = r.weighted_scale" in lg, "the NewtonL2 plane 0"


def test_binding_dispatches_and_refuses_by_name():
    src = _read(host_surface.binding_source("gbdt"))
    assert re.search(r"gbdt_host_fit_non_symmetric\(\s*x, y, n_rows, n_features, tp, ns_start\s*\)", src)
    assert "gbdt_host_ns_model_text(ns_model)" in src
    for what in ('"min_split_gain="', '"min_child_hessian="', '"min_data_in_leaf="',
                 "under Lossguide (only NewtonL2 and NewtonCosine)", "(only Cosine)"):
        assert what in src, f"no by-name refusal for {what}"
    assert f'"{REFUSAL}"' in src
    assert 'kind == String("ntree")' in src and 'kind == String("node")' in src
    assert "n_trees, 1, non_symmetric," in src, "the predict walk takes the model's shape"
    assert "_ = node_left^" in src and "_ = node_right^" in src, "keep-alives past the walk"


def test_sabotage_reaches_both_policies():
    text = _read(DRIVER)
    assert "_estimate_leaves(" in text, "both policies estimate through the sabotaged walker"
    assert "lambda_reg = lambda_reg + 1.0" in _read("gbdt/host/gbdt_oracle.mojo")



@reference_training()
def test_non_symmetric_fits_on_the_host_when_built():
    if _backend._CPU_ONLY is None:
        print("SKIP: a GPU set loaded; the host route is not taken here")
        return
    if "_mojolearn_gbdt_host" not in _backend.host_families_built():
        print("SKIP: the gbdt host binding is not built")
        return
    import os
    import tempfile
    import numpy as np
    module = _backend.load_host_module("_mojolearn_gbdt_host")
    assert not bool(module.gbdt_host_sabotage()), "a sabotage build loaded outside the gate"
    rng = np.random.default_rng(0)
    x = rng.standard_normal((600, 6)).astype(np.float32)
    y = (x[:, 3] + 0.5 * x[:, 4] > 0).astype(np.int32)
    for kwargs in (dict(max_depth=3, grow_policy="Depthwise"),
                   dict(max_leaves=8, grow_policy="Lossguide")):
        outs = []
        for _ in range(2):
            m = mojolearn.GradientBoosting(n_estimators=3, loss="Logloss", **kwargs).fit(x, y)
            outs.append((m.model_, np.asarray(m.predict(x)).tobytes()))
        assert outs[0] == outs[1], f"two host fits returned different bytes: {kwargs}"
        assert "\nntree 0 " in m.model_, kwargs
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.npz")
            m.save(path)
            back = type(m).load(path)
            assert np.asarray(back.predict(x)).tobytes() == outs[0][1], kwargs


if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for name in names:
        globals()[name]()
        print("ok", name)
    print(f"{len(names)} passed")
    sys.exit(0)
