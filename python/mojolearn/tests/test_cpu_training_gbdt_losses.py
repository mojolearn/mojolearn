# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU training for the gbdt-parametric-losses, gbdt-exact-mae,
gbdt-lossguide-newtoncosine, gbdt-multiclass and gbdt-onevsall lanes
(lane/cpu-training-gbdt-losses, 2026-09-15), checked from SOURCE so it runs
on a box with nothing built, plus a runtime check that runs only where the
gbdt host binding is built and the package took the CPU-only path.

What the source checks hold: the manifest covers the five lanes in the gbdt
family and lists both new host modules; the two oracles import no GPU
module, and the host modules they reuse import none either; the oracles
spell the device constructs a bit claim rests on (the contracted Poisson
score, the sort on bits [10, 32), the 768 segmented scan, the 1024
need-weights block, the sixteen-step search, splitmix64 and the stride walk
of the bootstrap, the one-per-launch level seed, the child-Hessian bit test,
the 1e-7 probability clip, the pinned MultiClass class, the Cholesky solve);
the binding dispatches the new losses, exports the multi-dimensional predict
and routes the multiclass bootstrap and score noise; the CPU identity gate runs by hand since 2026-09-15 (no push trigger).

The runtime check (skipped, and SAID to be skipped, when the binding is
absent or a GPU set loaded): small fits of each configuration run twice
through the host binding and return the same model text and predictions, and
the multiclass probabilities have the declared widths. A plumbing check; the
bit claim against the GPU columns is the CPU identity gate's.

    cd python && python3 -m mojolearn.tests.test_cpu_training_gbdt_losses
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

LOSSES = "gbdt/host/gbdt_oracle_losses.mojo"
MULTI = "gbdt/host/gbdt_oracle_multiclass.mojo"
DRIVER = "gbdt/host/gbdt_oracle_depthwise.mojo"
LANES = ("gbdt-parametric-losses", "gbdt-exact-mae", "gbdt-lossguide-newtoncosine",
         "gbdt-multiclass", "gbdt-onevsall")
GPU_IMPORTS = re.compile(r"^\s*from\s+(max\.gpu|std\.gpu)", re.M)
REUSED = ("gbdt/gpu_util/kernel/random_gen.mojo", "gbdt/lapack/linear_system.mojo",
          "gbdt/data/permutation.mojo",
          # the learning-to-rank targets the losses oracle routes (2026-09-15)
          "gbdt/data/pairs.mojo", "gbdt/data/yeti_rank_tasks.mojo",
          "gbdt/host/gbdt_oracle_query.mojo", "gbdt/host/gbdt_oracle_pair.mojo",
          "gbdt/host/gbdt_oracle_yeti.mojo")


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def _imports(rel):
    return sorted(set(re.findall(r"^from\s+([\w.]+)\s+import", _read(rel), re.M)))


def test_manifest_covers_the_five_lanes():
    fam = host_surface.family("gbdt")
    for lane in LANES:
        assert lane in fam["training_lanes"], lane
        assert lane in host_surface.covered_lanes(), lane
        assert lane in host_surface.TRAINING_LANE_NAMES, lane
    for rel in (LOSSES, MULTI):
        assert rel in fam["host_modules"] and (ROOT / rel).is_file(), rel
    assert "gbdt_predict_multi" in fam["exports"]
    sentence = host_surface.no_cpu_path_sentence()
    assert "gradient boosting training outside its declared lanes" in sentence, sentence


def test_oracles_import_no_gpu_module():
    for rel in (LOSSES, MULTI) + REUSED:
        assert not GPU_IMPORTS.search(_read(rel)), f"{rel} imports a GPU module"
    assert _imports(LOSSES) == [
        "checks.numerics", "gbdt.data.pairs", "gbdt.data.permutation",
        "gbdt.data.yeti_rank_tasks", "gbdt.gpu_data.compressed_index_builder",
        "gbdt.gpu_data.feature_blocks", "gbdt.gpu_data.grid_policy",
        "gbdt.gpu_util.kernel.random_gen", "gbdt.host.gbdt_oracle",
        "gbdt.host.gbdt_oracle_pair", "gbdt.host.gbdt_oracle_query",
        "gbdt.host.gbdt_oracle_yeti", "std.math", "std.memory",
    ], _imports(LOSSES)
    assert _imports(MULTI) == [
        "checks.numerics", "gbdt.data.permutation", "gbdt.gpu_data.compressed_index_builder",
        "gbdt.gpu_data.feature_blocks", "gbdt.gpu_data.grid_policy",
        "gbdt.gpu_util.kernel.random_gen", "gbdt.host.gbdt_oracle", "gbdt.lapack.linear_system", "std.math",
    ], _imports(MULTI)
    assert _imports("gbdt/lapack/linear_system.mojo") == ["std.math"]
    assert _imports("gbdt/gpu_util/kernel/random_gen.mojo") == ["checks.numerics"]


def test_losses_oracle_spells_the_bit_carrying_constructs():
    text = _read(LOSSES)
    assert "return identical_mul_add(-t, p, identical_exp(p))" in text, "the contracted Poisson score"
    assert "comptime GBDT_EXACT_FIRST_BIT = 10" in text
    assert "(UInt64(key >> UInt32(GBDT_EXACT_FIRST_BIT)) << UInt64(32)) | UInt64(k)" in text, (
        "the stable sort on bits [10, 32)"
    )
    assert "comptime GBDT_SEG_SCAN_BLOCK = 768" in text
    assert "comptime GBDT_NEED_WEIGHTS_BLOCK = 1024" in text
    assert "comptime GBDT_QUANTILE_ITERATIONS = 16" in text
    assert "residuals[pos] = ftz(g_target[pos] - g_cursor[pos])" in text
    assert "weights[pos] = ftz(Float32(1.0) / delta)" in text, "the MAPE quotient"
    # the bootstrap moved to gbdt_oracle.mojo (lane/catboost-parity), where
    # the symmetric Logloss fit's Bayesian arm reads it too
    boot = _read("gbdt/host/gbdt_oracle.mojo")
    assert "x += UInt64(0x9E3779B97F4A7C15)" in boot, "splitmix64 seeds"
    assert "var stride = blocks * GBDT_BOOT_BLOCK" in boot, "the bootstrap stride walk"
    assert "var tmp = -identical_log(draw[0] + Float32(1e-20))" in boot, "the Bayesian weight"
    assert "if function_value <= next_value:" in text, "AnyImprovement"
    assert "out.append(weights_cpu[leaf] + lambda_reg)" in text, "Gradient second derivatives"
    assert "comptime if GBDT_ORACLE_HOST_SABOTAGE:" in text


def test_driver_spells_the_searcher_options():
    text = _read(DRIVER)
    assert "var level_seed = level_rand.next_uniform_l()" in text
    assert "var seed = advance_seed_k(level_seed + UInt64(feature_id), 4)" in text
    assert "bitcast[DType.uint32](min_child_hessian) & UInt32(0x7FFFFFFF)" in text
    assert "if Float64(-leaves[to_split[k]].best_gain) > params.min_split_gain:" in text
    assert "var feature_random = TRandom(base.random_seed ^ UInt64(0x4645415455524553))" in text
    assert "var j = i + Int(random.uniform(UInt64(len(eligible) - i)))" in text
    # `_target_std_dev` moved to gbdt_oracle.mojo (lane/catboost-parity)
    assert "weighted_sum2 = ftz(weighted_sum2 + ftz(ftz(wt * wt) / w))" in _read("gbdt/host/gbdt_oracle.mojo")
    assert "var tree_seed = noise_rand.next_uniform_l()" in text


def test_multiclass_oracle_spells_the_bit_carrying_constructs():
    text = _read(MULTI)
    assert "se += identical_exp(Float32(0.0) - mx)" in text, "the pinned class's term"
    assert "return max(min(p, Float32(1.0) - Float32(1e-7)), Float32(1e-7))" in text
    assert "var max_chunks = (2 * GBDT_PINNED_SM + n_stats - 1) // n_stats" in text
    assert "gradient[bin * sbd + cursor_dim] = -total" in text, "the reconstruction"
    assert "_ = solve_linear_system_cholesky(sigma, solution)" in text
    assert "out.append(point[bin * sbd + d] - point[bin * sbd + cursor_dim])" in text, "the gauge fix"
    assert "var base_count = groups * n_compute * stat_count" in text
    assert "lambda_reg = lambda_reg + 1.0" in text


def test_binding_dispatches_and_refuses_by_name():
    src = _read(host_surface.binding_source("gbdt"))
    # the call carries the group and pairs tails since lane/gbdt-learning-to-rank
    assert "gbdt_losses_host_fit(\n                x, y, n_rows, n_features, p, pw_loss, host_group_sizes," in src
    assert "gbdt_multi_host_fit(" in src and "gbdt_multi_host_model_text(multi_model)" in src
    assert '("gbdt_predict_multi")' in src
    assert "var lg_boot = ns_stochastic and (" in src
    assert "_refuse(\"bootstrap_type='\" + bootstrap_type + \"' under loss='\" + loss + \"'\")" in src
    assert "if is_rmse or grow_code != 0 or is_pointwise or is_multi or len(flags) != 0:" in src



@reference_training()
def test_the_five_configurations_fit_on_the_host_when_built():
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
    yc = (x[:, 3] + 0.5 * x[:, 4] > 0).astype(np.int32)
    yr = (np.abs(x[:, 1] * 2.0) + 1.0).astype(np.float32)
    y3 = np.digitize(x[:, 2], [-0.5, 0.5]).astype(np.float32)

    def twice(make, target):
        outs = []
        for _ in range(2):
            m = make().fit(x, target)
            outs.append((m.model_, np.asarray(m.predict(x)).tobytes()))
        assert outs[0] == outs[1], "two host fits returned different bytes"
        return m

    for loss, kw in (("Quantile", {}), ("MAPE", {}), ("Poisson", {}), ("Tweedie", dict(loss_variance_power=1.5))):
        twice(lambda: mojolearn.GradientBoosting(n_estimators=3, max_depth=3, loss=loss, **kw), yr)
    twice(lambda: mojolearn.GradientBoosting(
        n_estimators=3, max_depth=3, loss="MAE", leaf_estimation_method="Exact",
        bootstrap_type="Poisson", subsample=0.6), yr)
    twice(lambda: mojolearn.GradientBoosting(
        n_estimators=3, max_leaves=8, grow_policy="Lossguide", loss="Logloss",
        score_function="NewtonCosine", min_child_hessian=1.0, min_split_gain=0.01,
        min_data_in_leaf=8, feature_fraction=0.5, random_strength=1.0,
        bootstrap_type="Bernoulli", subsample=0.7, leaf_estimation_method="Gradient",
        leaf_estimation_iterations=3), yc)
    mc = twice(lambda: mojolearn.GradientBoosting(
        n_estimators=3, max_depth=3, loss="MultiClass", class_weights=[1.0, 2.0, 0.5]), y3)
    assert np.asarray(mc.predict_proba(x)).shape == (600, 3)
    ova = twice(lambda: mojolearn.GradientBoosting(n_estimators=3, max_depth=3, loss="MultiClassOneVsAll"), y3)
    assert np.asarray(ova.predict_proba(x)).shape == (600, 3)
    try:
        mojolearn.GradientBoosting(n_estimators=2, max_depth=3, loss="Quantile",
                                   bootstrap_type="Bayesian").fit(x, yr)
    except Exception as exc:
        assert "no CPU implementation of" in str(exc), str(exc)
    else:
        raise AssertionError("a Bayesian bootstrap fit did not refuse on the host binding")


if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for name in names:
        globals()[name]()
        print("ok", name)
    print(f"{len(names)} passed")
    sys.exit(0)
