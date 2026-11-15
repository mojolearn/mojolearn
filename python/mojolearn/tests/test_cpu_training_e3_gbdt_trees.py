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
survive a save and load. Their two-fit repeat and `ntree` records are the
native check `gbdt_public_fits_cpu` (lane tmig-trees, 2026-11-14). It is a
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
    assert "sample weights" in sentence and "categorical or one-hot column" in sentence, sentence
    # and it must NOT claim the two lanes above have no CPU route
    assert "Depthwise with Logloss" not in sentence, sentence


def test_oracles_import_no_gpu_module():
    for rel in (DRIVER, LOSSGUIDE):
        text = _read(rel)
        assert not GPU_IMPORTS.search(text), f"{rel} imports a GPU module"
    # The driver's exact import allowlist is deleted: 524140f57d (sweep trees) deleted T26,
    # gbdt/methods/leaves_estimation/tree_t26_units.mojo and the driver's imports of it and of
    # gbdt.trees_identical_switches. No public-API form; Mojo check: none.
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


# test_binding_dispatches_and_refuses_by_name deleted: a source-text check of
# bindings/_mojolearn_gbdt_host.mojo. 285ac89b42 moved the host binding's RUN (the
# non-symmetric dispatch and its by-name refusals) into gbdt/mojolearn_gbdt_host_run.mojo.
# Depthwise/Lossguide host fits and refusals run natively:
# checks/native_tests/gbdt_public_fits.mojo (`_refuses`, "a Lossguide Cosine fit");
# the public save/load round trip stays below.


def test_sabotage_reaches_both_policies():
    text = _read(DRIVER)
    assert "_estimate_leaves(" in text, "both policies estimate through the sabotaged walker"
    assert "lambda_reg = lambda_reg + 1.0" in _read("gbdt/host/gbdt_oracle.mojo")



@reference_training()
def test_non_symmetric_save_load_on_the_host_when_built():
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
        # The two-fit repeat and the model text moved to the native check
        # gbdt_public_fits_cpu (lane tmig-trees); the archive round trip stays.
        m = mojolearn.GradientBoosting(n_estimators=3, loss="Logloss", **kwargs).fit(x, y)
        expected = np.asarray(m.predict(x)).tobytes()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.npz")
            m.save(path)
            back = type(m).load(path)
            assert np.asarray(back.predict(x)).tobytes() == expected, kwargs


if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for name in names:
        globals()[name]()
        print("ok", name)
    print(f"{len(names)} passed")
    sys.exit(0)
