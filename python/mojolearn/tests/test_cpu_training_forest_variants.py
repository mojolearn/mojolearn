# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU training for the forest variant lanes (lane/cpu-training-forest-variants,
2026-09-15): rf-clf-entropy-log2-noboot, rf-clf-balanced-parallel,
rf-reg-poisson, rf-reg-gamma-ig, et-clf-entropy-bestfirst and
et-reg-bootstrap-parallel. Checked from SOURCE so it runs on a box with
nothing built, plus a runtime check that runs only where the rf and trees
host bindings are built and the package took the CPU-only path.

What the source checks hold: the manifest declares the six lanes on the rf
and trees families with names; both bindings register the resident
parallel_groves names over `core/forest_host_groves.mojo`, and the rf binding
the weighted fit; the groves module imports no GPU module and spells the
grove arithmetic a bit claim rests on (32 lanes, the flushed add, the
shuffle-down from 16, the flushed `identical_div`, the RandomForest input
flush); the rf oracle carries the three regression gains with their
storage-width subtraction and `eps_` guards, the weighted bootstrap's CDF
and `upper_bound`, the refusal of weights without a bootstrap, and a
sabotage arm that also moves a forest without a bootstrap; the ExtraTrees
builder dispatches best-first growth to `train_tree_exact_bestfirst` instead
of refusing it; the CPU identity gate runs by hand since 2026-09-15 (no push trigger).

The runtime check (skipped, and SAID to be skipped, when the bindings are
absent or a GPU set loaded): each variant fits twice through the host
bindings on a small draw and returns the same bytes, and the parallel_groves
engine's predictions are not the sequential engine's bytes (the grove
reduction is reached, not a sequential fallback). It is a plumbing check. The
bit claim against the GPU columns is the CPU identity gate's, not this
file's.

    cd python && python3 -m mojolearn.tests.test_cpu_training_forest_variants
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

RF_LANES = ("rf-clf-entropy-log2-noboot", "rf-clf-balanced-parallel", "rf-reg-poisson", "rf-reg-gamma-ig")
ET_LANES = ("et-clf-entropy-bestfirst", "et-reg-bootstrap-parallel")
GROVES = "core/forest_host_groves.mojo"
GROVES_BINDING = "bindings/forest_host_groves_binding.mojo"
RF_ORACLE = "ensemble/host/rf_oracle.mojo"
ET_BUILDER = "extratrees/impl/decisiontree/batched_levelalgo/builder.mojo"
RESIDENT = ("forest_prepare_gpu", "forest_predict_resident_reuse_gpu", "forest_release_gpu")
GPU_IMPORTS = re.compile(r"^\s*from\s+(max\.gpu|std\.gpu)", re.M)


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def test_manifest_declares_the_variant_lanes():
    rf = host_surface.family("rf")
    trees = host_surface.family("trees")
    for lane in RF_LANES:
        assert lane in rf["training_lanes"], f"{lane} is not an rf training lane"
    for lane in ET_LANES:
        assert lane in trees["training_lanes"], f"{lane} is not a trees training lane"
    covered = host_surface.covered_lanes()
    for lane in RF_LANES + ET_LANES:
        assert lane in covered, f"{lane} is not covered"
        assert host_surface.TRAINING_LANE_NAMES[lane], f"{lane} has no name"
    for fam in (rf, trees):
        assert GROVES in fam["host_modules"] and (ROOT / GROVES).is_file()


def test_bindings_register_the_resident_and_weighted_names():
    rf_src = _read(host_surface.binding_source("rf"))
    et_src = _read(host_surface.binding_source("trees"))
    for name in RESIDENT:
        assert f'("{name}")' in rf_src and f'("{name}")' in et_src, name
        assert name in host_surface.family("rf")["exports"]
        assert name in host_surface.family("trees")["exports"]
    assert "forest_prepare_host_binding[True](" in rf_src, "the rf binding must flush the input (RF_INPUT)"
    assert "forest_predict_resident_host_binding[True](" in rf_src
    assert "forest_prepare_host_binding[False](" in et_src, "the trees binding must not flush the input"
    assert "forest_predict_resident_host_binding[False](" in et_src
    for name in ("rf_classifier_fit_weighted", "rf_classifier_fit_weighted_export"):
        assert f'("{name}")' in rf_src and name in host_surface.family("rf")["exports"], name
    for absent in ("rf_classifier_fit_shard", "rf_predict_proba_gpu_parallel"):
        assert f'("{absent}")' not in rf_src, f"{absent} must stay absent so it refuses by name"
    assert '("et_predict_gpu_parallel")' not in et_src


def test_groves_module_is_host_only_and_spells_the_grove_arithmetic():
    for rel in (GROVES, GROVES_BINDING):
        text = _read(rel)
        assert not GPU_IMPORTS.search(text), f"{rel} imports a GPU module"
        assert "DeviceContext" not in re.sub(r'"""[\s\S]*?"""', "", text), f"{rel} names DeviceContext"
    text = _read(GROVES)
    assert "comptime GROVE_LANES = 32" in text
    assert "return ftz(ftz(a) + ftz(b))" in text, "forest_add flushes both operands and the result"
    assert "var step = GROVE_LANES // 2" in text and "step //= 2" in text
    assert "ftz(identical_div(ftz(sums[c]), divisor))" in text
    assert "comptime if RF_INPUT:\n                value = ftz(value)" in text
    assert "_finite_key(value) <= _finite_key(self.thresholds[node])" in text


def test_rf_oracle_carries_the_new_arms():
    text = _read(RF_ORACLE)
    for gain in ("def host_poisson_gain(", "def host_gamma_gain(", "def host_inverse_gaussian_gain("):
        assert gain in text, gain
    assert "_dequantize_wide(" in text, "the right label sum is subtracted at storage width"
    assert text.count("<= RF_REG_EPS or left_label_sum <= RF_REG_EPS") == 3
    assert "no host restatement yet (ensemble/host/rf_oracle.mojo)" not in text
    assert "if weight_cdf[mid] <= d:" in text, "the weighted draw is an upper_bound over the CDF"
    assert "run += Float64(weights[i])" in text
    assert "class weights without bootstrap" in text, "weights without a bootstrap refuse by name"
    assert "h = h + UInt32(1)" in text, "the sabotage arm does not move the column sample"


def test_extratrees_best_first_is_restated_not_refused():
    text = _read(ET_BUILDER)
    assert "def train_tree_exact_bestfirst(" in text
    assert "is REFUSED BY NAME on the host restatement" not in text
    body = text[text.index("def train_tree_exact_bestfirst("):]
    body = body[:body.index("\ndef ", 10)]
    for step in ("queue.bestfirst_seed()", "queue.bestfirst_admit(", "queue.bestfirst_pop()",
                 "partition_samples(dataset, rec.split, rec.item)", "queue.bestfirst_expand("):
        assert step in body, step
    assert body.index("partition_samples(") < body.index("queue.bestfirst_expand("), "partition before the children"



@reference_training()
def test_variants_fit_on_the_host_when_built():
    if _backend._CPU_ONLY is None:
        print("SKIP: a GPU set loaded; the host route is not taken here")
        return
    built = _backend.host_families_built()
    if "_mojolearn_rf_host" not in built or "_mojolearn_trees_host" not in built:
        print("SKIP: the rf or trees host binding is not built")
        return
    import numpy as np
    for name in ("_mojolearn_rf_host", "_mojolearn_trees_host"):
        module = _backend.load_host_module(name)
        prefix = "rf" if name == "_mojolearn_rf_host" else "trees"
        assert not bool(getattr(module, prefix + "_host_sabotage")()), "a sabotage build loaded outside the gate"
    rng = np.random.default_rng(1)
    x = rng.standard_normal((500, 7)).astype(np.float32)
    yc = (x[:, 2] - 0.4 * x[:, 5] > 0.3).astype(np.int32)
    yr = (x[:, 0] - 2.0 * x[:, 1]).astype(np.float32)
    ypos = (np.abs(yr) + 0.25).astype(np.float32)
    kw = dict(n_estimators=4, max_depth=5, random_state=7)

    def fits(engine):
        return (
            mojolearn.RandomForestClassifier(criterion="entropy", max_features="log2", bootstrap=False,
                                             **kw).fit(x, yc).predict_proba(x),
            mojolearn.RandomForestClassifier(class_weight="balanced", inference_engine=engine,
                                             **kw).fit(x, yc).predict_proba(x),
            mojolearn.RandomForestRegressor(criterion="poisson", **kw).fit(x, ypos).predict(x),
            mojolearn.RandomForestRegressor(criterion="gamma", **kw).fit(x, ypos).predict(x),
            mojolearn.RandomForestRegressor(criterion="inverse_gaussian", **kw).fit(x, ypos).predict(x),
            mojolearn.ExtraTreesClassifier(n_estimators=4, random_state=7, criterion="entropy",
                                           max_leaf_nodes=8).fit(x, yc).predict_proba(x),
            mojolearn.ExtraTreesRegressor(bootstrap=True, max_samples=0.5, inference_engine=engine,
                                          **kw).fit(x, yr).predict(x),
        )

    first = [np.asarray(a).tobytes() for a in fits("parallel_groves")]
    second = [np.asarray(a).tobytes() for a in fits("parallel_groves")]
    assert first == second, "two host fits returned different bytes"
    sequential = [np.asarray(a).tobytes() for a in fits("sequential")]
    assert first[1] != sequential[1], "RF parallel_groves returned the sequential engine's bytes"
    assert first[6] != sequential[6], "ET parallel_groves returned the sequential engine's bytes"


if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for name in names:
        globals()[name]()
        print("ok", name)
    print(f"{len(names)} passed")
    sys.exit(0)
