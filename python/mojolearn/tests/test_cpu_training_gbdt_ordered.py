# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU training for the gbdt-ordered-rmse, gbdt-feature-freq,
gbdt-pointwise-l2-bayesian-eval and gbdt-categorical-ctr lanes (lane/cpu-training-gbdt-ordered,
2026-09-15), checked from SOURCE so it runs
on a box with nothing built, plus a runtime check that runs only where the
gbdt host binding is built and the package took the CPU-only path.

What the source checks hold: the manifest covers both lanes in the gbdt
family, lists both oracles and both exports; the binding registers
`gbdt_fit_ordered_rmse` and `gbdt_fit_two_level_feature_freq` and refuses
`sample_weight` by name on each; neither oracle imports a GPU module, nor do
the host modules they reuse; the oracles spell the device constructs a bit
claim rests on (the fold layout's ceil `IntLog2` and power-of-two stripe,
the 8-bit kernel's document-keyed dither and its `1e-20` write guard, the
half-byte accumulator's window order and two-stage reduce, the dynamic
cosine scorer's `1e-20` seed and fold pairs, the challenger-first record
fold, the stable one-bit sort, the ordered apply through `identical_mul`,
the tensor column's `(count + 0) / (n + 1)` and pinned fold capacity, the
level-two rebuild without subtraction, the host leaf fold); the sabotage
define reaches both; the CPU identity gate workflow triggers on both files.

The runtime check (skipped, and SAID to be skipped, when the binding is
absent or a GPU set loaded): a small OrderedRMSE fit and a small
ExperimentalTwoLevelFeatureFreq fit each run twice through the host binding
and return the same model text and predictions, survive a save and load,
and a weighted fit of each refuses by name. It is a plumbing check. The bit
claim against the GPU columns is the CPU identity gate's, not this file's.

    cd python && python3 -m mojolearn.tests.test_cpu_training_gbdt_ordered
"""

# Gate-runner scope: host runtime checks require the CPU-only route.
GATE_BACKENDS = ("cpu",)
import os
import re
import sys
import tempfile
from pathlib import Path

import mojolearn
from mojolearn._cpu_reference import reference_training
from mojolearn import _backend, host_surface

ROOT = Path(__file__).resolve().parents[3]

ORDERED = "gbdt/host/gbdt_oracle_ordered.mojo"
FEATURE_FREQ = "gbdt/host/gbdt_oracle_feature_freq.mojo"
POINTWISE = "gbdt/host/gbdt_oracle_pointwise.mojo"
ONEHOT = "gbdt/host/gbdt_oracle_onehot.mojo"
GPU_IMPORTS = re.compile(r"^\s*from\s+(max\.gpu|std\.gpu)", re.M)
REUSED_HOST_MODULES = (
    "checks/fixed_point.mojo",
    "gbdt/host/gbdt_oracle.mojo",
    "gbdt/ctrs/ctr_binarization.mojo",
    "gbdt/models/ctr_value_table.mojo",
    "gbdt/grid_creator/binarization.mojo",
    "gbdt/gpu_data/compressed_index_builder.mojo",
    "gbdt/gpu_data/feature_blocks.mojo",
    "gbdt/gpu_util/kernel/random_gen.mojo",
    "gbdt/overfitting_detector/overfitting_detector.mojo",
)


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def test_manifest_covers_both_lanes():
    fam = host_surface.family("gbdt")
    for lane in ("gbdt-ordered-rmse", "gbdt-feature-freq", "gbdt-pointwise-l2-bayesian-eval", "gbdt-categorical-ctr"):
        assert lane in fam["training_lanes"], lane
        assert lane in host_surface.covered_lanes(), lane
    for rel in (ORDERED, FEATURE_FREQ, POINTWISE, ONEHOT):
        assert rel in fam["host_modules"] and (ROOT / rel).is_file(), rel
    for name in ("gbdt_fit_ordered_rmse", "gbdt_fit_two_level_feature_freq"):
        assert name in fam["exports"], name
    for cls in ("OrderedRMSE", "ExperimentalTwoLevelFeatureFreq"):
        assert cls in fam["classes"], cls


def test_binding_registers_and_refuses_weights_by_name():
    src = _read(host_surface.binding_source("gbdt"))
    assert '("gbdt_fit_ordered_rmse")' in src
    assert '("gbdt_fit_two_level_feature_freq")' in src
    assert "gbdt_ordered_rmse_host_fit(" in src and "gbdt_feature_freq_host_fit(" in src
    assert '" for sample_weight; the gbdt host binding trains the"' in src, "ordered weight refusal"
    assert '" sample_weight; the gbdt host binding trains the"' in src, "FeatureFreq weight refusal"
    assert 'raise Error("ordered RMSE params must have nine values")' in src
    assert 'raise Error("two-level FeatureFreq params must have seven values")' in src


def test_oracles_import_no_gpu_module():
    for rel in (ORDERED, FEATURE_FREQ, POINTWISE, ONEHOT):
        text = _read(rel)
        assert not GPU_IMPORTS.search(text), f"{rel} imports a GPU module"
        assert "DeviceContext" not in "".join(re.findall(r"^\s*from .*$", text, re.M)), rel
    for rel in REUSED_HOST_MODULES:
        body = _read(rel)
        assert not GPU_IMPORTS.search(body), f"{rel} imports a GPU module"


def test_ordered_oracle_spells_the_bit_carrying_constructs():
    text = _read(ORDERED)
    assert "var stripe = 1 << Int(ceil(log2(Float32(fold_count))))" in text, "the data partition stripe"
    assert "var u = _hist2_dither(row)" in text, "the dither keyed on the document id"
    assert "var val = Float32(Int(acc[c])) / fixed_scale" in text, "the 8-bit writeback"
    assert "if abs(val) > Float32(1e-20):" in text, "the write guard"
    assert "var at = (t // 32) * SLICE + (t & 16) + (bin << 5) + 2 * j + stat" in text, "the half-byte slot"
    assert "cells[tid] = stage1[32 * fold2 + e] + stage1[32 * fold2 + e + 16]" in text, "the inner copies"
    assert "var denum = List[Float32](length=hist_line, fill=Float32(1e-20))" in text, "the dynamic scorer's seed"
    assert "gain_out[b] = (noisy - score_before) * Float32(1.0)" in text
    assert "if _record_less(loc_gain, loc_fid, loc_bin, best_gain, best_fid, best_bin):" in text
    assert "data_part = left if s.p_sz[left] < s.p_sz[right] else right" in text, "the smaller child"
    assert "var is_left = s.p_sz[left_part] < s.p_sz[right_part]" in text
    assert "row_index[fill[gb[r]]] = r" in text, "the stable counting sort"
    assert "cursor[i] = identical_mul_add(scaled, Float32(1), cursor[i])" in text, "the ordered apply"
    assert "model_leaves.append(identical_mul(leaves[leaf], learning_rate))" in text
    assert "residual_bound *= Float64(1) + Float64(learning_rate)" in text


def test_feature_freq_oracle_spells_the_bit_carrying_constructs():
    text = _read(FEATURE_FREQ)
    assert "values[row] = (Float32(count) + Float32(0.0)) / (" in text, "the FeatureFreq value"
    assert "folds.append(GBDT_FF_GRID_BORDERS)" in text, "the pinned fold capacity"
    assert "var fixed_scale = Float32(choose_scale(magnitude, n_rows))" in text
    assert "_refuse_ff(\"a level-one winner on the FeatureFreq tensor column\")" in text
    assert "leaves.append(learning_rate * total / (total_weight + l2))" in text
    assert "if blocks[b].policy == POLICY_BINARY:" in text


def test_pointwise_oracle_spells_the_bit_carrying_constructs():
    text = _read(POINTWISE)
    assert "var tmp = -identical_log(draw[0] + Float32(1e-20))" in text, "the Bayesian draw"
    assert "bw = identical_pow(tmp, bagging_temperature)" in text
    assert "i += boot_blocks * GBDT_PW_BOOT_BLOCK" in text, "the bootstrap stride"
    assert "var starting_approx = -portable_log64(1.0 / best_probability - 1.0)" in text
    assert "var best_probability = Float64(Float32(target_sum / summary_weight))" in text
    assert "1, 0, scale, l2_leaf_reg, feat_offset, feat_shift, feat_mask,\n            True," in text, (
        "the single-task arm with the plain L2 scorer")
    assert "cursor[row] = identical_mul_add(estimated[leaf], learning_rate, cursor[row])" in text
    assert "if detector.is_need_stop():" in text and "model_leaves.resize(" in text
    ordered = _read(ORDERED)
    assert "l2score += (-sl * sl) / (wl + l2)" in ordered, "TL2ScoreCalcer's leaf score"
    src = _read(host_surface.binding_source("gbdt"))
    assert "return _gbdt_fit_pointwise_arm(" in src
    for what in ('"a fit without sample_weight"', '"a fit without eval_set"',
                 "(only L2)", "(only Bayesian)", '"boost_from_average other than True"'):
        assert what in src, f"no by-name refusal for {what}"


def test_one_hot_arm_spells_the_bit_carrying_constructs():
    text = _read(ONEHOT)
    assert "comptime GBDT_ONE_HOT_MAX_SIZE = 2" in text, "their GPU one_hot_max_size"
    assert "if unique_values > GBDT_ONE_HOT_MAX_SIZE:" in text, "the CTR refusal"
    assert 'line += " split_type take_bin"' in text
    oracle = _read("gbdt/host/gbdt_oracle.mojo")
    assert "bs.append(Float32(c) + Float32(0.5))" in oracle, "the one-hot grid"
    assert "grid.fold_counts[f] = len(bs) + 1 if len(bs) > 0 else 0" in oracle
    src = _read(host_surface.binding_source("gbdt"))
    assert "var one_hot = gbdt_resolve_one_hot(flags, x, n_rows, n_features)" in src
    assert "gbdt_host_model_text_one_hot(r.model, one_hot)" in src


def test_sabotage_reaches_both_oracles():
    for rel in (ORDERED, FEATURE_FREQ, POINTWISE):
        text = _read(rel)
        assert "comptime if GBDT_ORACLE_HOST_SABOTAGE:" in text, rel


def test_workflow_triggers_on_both_oracles():
    # 0affb7c75 removed the gate's push trigger (it runs by hand, weekly or
    # from the release workflow), and its path list with it; the oracles
    # must be listed only while a push trigger exists.
    text = _read(".github/workflows/cpu-identity-gate.yml")
    if not re.search(r"^  push:", text, re.M):
        assert "workflow_dispatch:" in text and "workflow_call:" in text
        return
    for rel in (ORDERED, FEATURE_FREQ, POINTWISE, ONEHOT):
        assert f'- "{rel}"' in text, f"cpu-identity-gate.yml does not trigger on {rel}"


@reference_training()
def test_ordered_and_feature_freq_fit_on_the_host_when_built():
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
    x = rng.standard_normal((1200, 6)).astype(np.float32)
    y = (x[:, 0] + 0.5 * x[:, 1]).astype(np.float32)
    perm = rng.permutation(1200)
    outs = []
    for _ in range(2):
        m = mojolearn.OrderedRMSE(n_estimators=3, max_depth=3).fit(x, y, permutation=perm)
        outs.append((m.model_, np.asarray(m.predict(x)).tobytes()))
    assert outs[0] == outs[1], "two ordered host fits returned different bytes"
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ordered.npz")
        m.save(path)
        back = mojolearn.OrderedRMSE.load(path)
        assert np.asarray(back.predict(x)).tobytes() == outs[0][1]
    try:
        mojolearn.OrderedRMSE(n_estimators=2, max_depth=2).fit(
            x, y, permutation=perm, sample_weight=np.ones(1200, np.float32))
    except Exception as exc:
        assert "no CPU implementation of" in str(exc), str(exc)
    else:
        raise AssertionError("a weighted OrderedRMSE fit did not refuse on the host binding")

    coded = np.column_stack([
        (x[:, 2] > 0).astype(np.float32),
        ((x[:, 3] > 0).astype(np.int32) * 2 + (x[:, 4] > 0)).astype(np.float32),
        x[:, :4],
    ]).astype(np.float32)
    outs = []
    for _ in range(2):
        f = mojolearn.ExperimentalTwoLevelFeatureFreq(sources=[0, 1], random_state=7).fit(coded, y)
        outs.append((f.model_, np.asarray(f.predict(coded)).tobytes()))
    assert outs[0] == outs[1], "two FeatureFreq host fits returned different bytes"
    try:
        mojolearn.ExperimentalTwoLevelFeatureFreq(sources=[0, 1]).fit(
            coded, y, sample_weight=np.ones(1200, np.float32))
    except Exception as exc:
        assert "no CPU implementation of" in str(exc), str(exc)
    else:
        raise AssertionError("a weighted FeatureFreq fit did not refuse on the host binding")

    # the pointwise eval arm, and its by-name refusal without weights
    yc = (x[:, 0] > 0).astype(np.int32)
    xe = rng.standard_normal((300, 6)).astype(np.float32)
    ye = (xe[:, 0] > 0).astype(np.int32)
    wts = rng.uniform(0.5, 1.5, 1200).astype(np.float32)
    kw = dict(n_estimators=4, max_depth=3, loss="Logloss", score_function="L2",
              use_pointwise_searcher=True, bootstrap_type="Bayesian", bagging_temperature=0.5,
              boost_from_average=True, od_type="Iter", od_wait=2, use_best_model=True)
    outs = []
    for _ in range(2):
        g = mojolearn.GradientBoosting(**kw).fit(x, yc, sample_weight=wts, eval_set=(xe, ye))
        outs.append((g.model_, np.asarray(g.predict(x)).tobytes()))
    assert outs[0] == outs[1], "two pointwise host fits returned different bytes"
    try:
        mojolearn.GradientBoosting(**kw).fit(x, yc, eval_set=(xe, ye))
    except Exception as exc:
        assert "no CPU implementation of" in str(exc), str(exc)
    else:
        raise AssertionError("an unweighted pointwise fit did not refuse on the host binding")

    # the one-hot categorical arm, and the CTR refusal above one_hot_max_size
    outs = []
    for _ in range(2):
        c = mojolearn.GradientBoosting(n_estimators=3, max_depth=3, loss="Logloss", cat_features=[0],
                                       one_hot_features=[1]).fit(coded, yc)
        outs.append((c.model_, np.asarray(c.predict(coded)).tobytes()))
    assert outs[0] == outs[1], "two one-hot host fits returned different bytes"
    assert "type cat" in outs[0][0] and "ctr_" not in outs[0][0]
    try:
        mojolearn.GradientBoosting(n_estimators=2, max_depth=3, loss="Logloss", cat_features=[1]).fit(coded, yc)
    except Exception as exc:
        assert "no CPU implementation of" in str(exc), str(exc)
    else:
        raise AssertionError("a CTR categorical fit did not refuse on the host binding")


if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for name in names:
        globals()[name]()
        print("ok", name)
    print(f"{len(names)} passed")
    sys.exit(0)
