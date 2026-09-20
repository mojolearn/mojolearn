# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Workstream E batch 2, CPU training for the kmeans, metrics, spectral,
standard-scaler, minmax-scaler and logistic lanes (lane/cpu-training-e2,
2026-09-14), checked from SOURCE so it runs on a box with nothing built,
plus runtime checks that run only where the host bindings are built and
the package took the CPU-only path.

What the source checks hold: the manifest lists kmeans as a core training
lane and names it for the docs, and no longer names k-means as a lane with
no CPU path; the core host binding registers `kmeans_fit` under the GPU
binding's name; the oracle imports no GPU module and reads its fold widths
from the kernel matrix; the sabotage define reaches the oracle through the
quantized accumulation (the arm that moves every centroid) and the binding
reads it back; the oracle spells the device's round seed reassembly (the
sign-extended low half, measured on the M4).

The runtime check (skipped, and SAID to be skipped, when the binding is
absent or a GPU set loaded): `KMeans.fit` runs through the host binding on
a small draw, twice, and returns the same bytes, with `labels_` in range
and one label per row; the binding reads back sabotage False. It is a
plumbing check. The bit claim against the GPU columns is the CPU identity
gate's, not this file's.

    cd python && python3 -m mojolearn.tests.test_cpu_training_e2
"""

# Gate-runner scope: host runtime checks require the CPU-only route.
GATE_BACKENDS = ("cpu",)
import re
import os
import sys
from pathlib import Path

import mojolearn
from mojolearn._cpu_reference import reference_training
from mojolearn import _backend, host_surface

ROOT = Path(__file__).resolve().parents[3]

ORACLE = "cluster/host/kmeans_oracle.mojo"
METRICS_ORACLE = "metrics/host/metrics_oracle.mojo"
METRICS_EXPORTS = ("accuracy_score", "adjusted_rand_score", "entropy", "mutual_info_score",
                   "homogeneity_score", "completeness_score", "v_measure_score", "r2_score",
                   "silhouette", "spectral_fit_predict_dataset", "spectral_fit_predict_graph")
SPECTRAL_ORACLE = "spectral/host/spectral_oracle.mojo"
SCALER_ORACLE = "preprocessing/host/scaler_oracle.mojo"
QN_ORACLE = "glm/host/qn_oracle.mojo"
GPU_IMPORTS = re.compile(r"^\s*from\s+(max\.gpu|std\.gpu)", re.M)


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def test_manifest_covers_kmeans():
    assert "kmeans" in host_surface.covered_lanes(), "kmeans is not a covered training lane"
    assert host_surface.TRAINING_LANE_NAMES["kmeans"] == "k-means"
    core = host_surface.family("core")
    assert "kmeans" in core["training_lanes"], "kmeans is not a core training lane"
    assert "KMeans" in core["classes"]
    assert ORACLE in core["host_modules"], f"{ORACLE} is not a core host module"
    assert (ROOT / ORACLE).is_file(), f"{ORACLE} does not exist"
    assert "k-means" not in host_surface.no_cpu_path_sentence(), host_surface.no_cpu_path_sentence()
    assert "k-means" in host_surface.training_sentence()


def test_accuracy_host_uses_pointer_parallel_count():
    binding = _read("bindings/_mojolearn_metrics_host.mojo")
    oracle = _read(METRICS_ORACLE)
    assert "host_accuracy_score_ptr(yt, yp, n)" in binding
    assert "sync_parallelize(_rows, tasks)" in oracle


def test_binding_registers_kmeans_fit():
    src = _read(host_surface.binding_source("core"))
    assert '("kmeans_fit")' in src, "the core host binding does not register kmeans_fit"
    assert "kmeans_fit" in host_surface.family("core")["exports"], "the manifest does not list kmeans_fit"
    # The ball cover entries joined on lane/cpu-training-batch3 (the radius
    # and knn-rbc lanes); the manifest must list what the binding registers.
    for present in ("rbc_knn_search", "radius_neighbors_count", "radius_neighbors_fill"):
        assert f'("{present}")' in src, f"{present} is not registered"
        assert present in host_surface.family("core")["exports"], f"the manifest does not list {present}"


def test_oracle_imports_no_gpu_and_reads_the_matrix():
    text = _read(ORACLE)
    assert not GPU_IMPORTS.search(text), f"{ORACLE} imports a GPU module"
    assert not re.search(r"^\s*from .*import.*DeviceContext", text, re.M), f"{ORACLE} imports DeviceContext"
    assert "from checks.numerics import" in text
    assert "from checks.fixed_point import choose_scale" in text, "the scale must be the device's choose_scale"
    for row in ("K_LIB_ROW_NORM", "K_LIB_REDUCE_BY_KEY", "K_LIB_PLUS_PLUS"):
        assert f"lib_block_size_for[{row}, TARGET_COLUMN]()" in text, f"{row} is not read from the matrix"
        assert f"lib_block_size_for[{row}, COLUMN_AMD]()" in text, f"{row} is not held to the AMD column"


def test_sabotage_define_moves_the_accumulation():
    text = _read(ORACLE)
    define = host_surface.sabotage_define("core")
    assert f'is_defined["{define}"]()' in text
    assert "comptime if KMEANS_ORACLE_HOST_SABOTAGE:" in text, "the sabotage arm is not a comptime branch"
    assert "q = q + Int32(1)" in text, "the sabotage arm does not add one unit per cell"
    binding = _read(host_surface.binding_source("core"))
    assert "KMEANS_ORACLE_HOST_SABOTAGE" in binding, "the binding does not read the oracle's define back"


def test_oracle_spells_the_device_seed_reassembly():
    text = _read(ORACLE)
    assert "def host_round_seed_as_the_device_reassembles_it(" in text
    assert "lo | UInt64(0xFFFFFFFF00000000)" in text, "the low half is not sign-extended"
    assert "host_round_seed_as_the_device_reassembles_it(\n            rng.next_u64()" in text, \
        "the round seed does not pass through the reassembly"



def test_manifest_covers_metrics():
    assert "metrics" in host_surface.covered_lanes(), "metrics is not a covered training lane"
    fam = host_surface.family("metrics")
    assert fam["routes"] == "_mojolearn_metrics"
    # spectral-precomputed joined the family on lane/cpu-training-batch2-declare,
    # umap on lane/cpu-training-umap-b (tests/test_cpu_training_umap.py),
    # metrics-classification on lane/cpu-training-metrics-classification.
    assert fam["training_lanes"] == ("metrics", "spectral", "spectral-precomputed", "umap", "metrics-classification",
                                     "metrics-fowlkes-mallows", "metrics-homogeneity-completeness")
    assert METRICS_ORACLE in fam["host_modules"]
    assert (ROOT / METRICS_ORACLE).is_file()
    assert (ROOT / "bindings/build_metrics_host.sh").is_file()
    assert (ROOT / host_surface.binding_source("metrics")).is_file()
    assert "_mojolearn_metrics" in host_surface.routed_modules()


def test_metrics_binding_registers_the_lane_entries():
    src = _read(host_surface.binding_source("metrics"))
    exports = host_surface.family("metrics")["exports"]
    for name in METRICS_EXPORTS + ("metrics_vendor", "metrics_numeric_mode"):
        assert f'("{name}")' in src, f"the metrics host binding does not register {name}"
        assert name in exports, f"the manifest does not list {name} for metrics"
    # umap_fit_transform left this list on lane/cpu-training-umap-b; rand_score,
    # trustworthiness, kl_divergence, log_loss and confusion_matrix on the
    # metrics-classification lane.
    for absent in ("graph_parallel_available",):
        assert f'("{absent}")' not in src, f"{absent} must stay absent so it refuses by name"


def test_metrics_oracle_imports_no_gpu_and_carries_the_sabotage_arm():
    text = _read(METRICS_ORACLE)
    assert not GPU_IMPORTS.search(text), f"{METRICS_ORACLE} imports a GPU module"
    assert not re.search(r"^\s*from .*import.*DeviceContext", text, re.M)
    assert "from checks.numerics import" in text
    assert "comptime PINNED_SUM_W = 256" in text, "the slab width is the tree; it must be 256"
    define = host_surface.sabotage_define("metrics")
    assert f'is_defined["{define}"]()' in text
    assert "comptime if METRICS_ORACLE_HOST_SABOTAGE:" in text
    assert "values[(i + 1) % n]" in text, "the sabotage arm does not shift the chunk boundaries"
    assert "METRICS_ORACLE_HOST_SABOTAGE" in _read(host_surface.binding_source("metrics"))


def test_manifest_covers_spectral_and_the_oracle_moved():
    assert "spectral" in host_surface.covered_lanes(), "spectral is not a covered training lane"
    fam = host_surface.family("metrics")
    assert "spectral" in fam["training_lanes"] and "SpectralClustering" in fam["classes"]
    assert SPECTRAL_ORACLE in fam["host_modules"]
    assert "spectral clustering" not in host_surface.no_cpu_path_sentence()
    text = _read(SPECTRAL_ORACLE)
    assert not GPU_IMPORTS.search(text), f"{SPECTRAL_ORACLE} imports a GPU module"
    assert not re.search(r"^\s*from .*import.*DeviceContext", text, re.M)
    assert "from gemm.host.identical_gemm import contract_leaf_size" in text
    assert "def oracle_embedding[" in text and "def host_spectral_fit_predict_dataset(" in text
    assert "def host_coo_symmetrize(" in text
    checks = _read("spectral/checks/spectral_oracle.mojo")
    assert "from spectral.host.spectral_oracle import (" in checks, "the checks file must re-export the host oracle"
    assert "def dense_laplacian_eigenvalues_f64(" in checks
    assert "comptime if SPECTRAL_ORACLE_HOST_SABOTAGE:" in text and "seed + UInt64(1)" in text
    assert "SPECTRAL_ORACLE_HOST_SABOTAGE" in _read(host_surface.binding_source("metrics"))


@reference_training()
def test_spectral_runs_on_the_host_when_built():
    if _backend._CPU_ONLY is None:
        print("SKIP: a GPU set loaded; the host route is not taken here")
        return
    if "_mojolearn_metrics_host" not in _backend.host_families_built():
        print("SKIP: the metrics host binding is not built")
        return
    import numpy as np
    rng = np.random.default_rng(2)
    x = np.concatenate([rng.standard_normal((60, 3)) + 6.0, rng.standard_normal((60, 3)) - 6.0]).astype(np.float32)
    got = [np.asarray(mojolearn.SpectralClustering(n_clusters=2, n_neighbors=8, random_state=5).fit(x).labels_)
           for _ in range(2)]
    assert got[0].tobytes() == got[1].tobytes(), "two host fits returned different labels"
    assert set(got[0].tolist()) <= {0, 1} and got[0].shape == (120,)
    assert (got[0][:60] == got[0][0]).all() and (got[0][60:] == got[0][60]).all() and got[0][0] != got[0][60], \
        "two separated blobs were not split into the two clusters"


def test_manifest_covers_the_scalers():
    for lane in ("standard-scaler", "minmax-scaler"):
        assert lane in host_surface.covered_lanes(), f"{lane} is not a covered training lane"
    fam = host_surface.family("preprocessing")
    assert fam["routes"] == "_mojolearn_preprocessing"
    # The three parameter lanes joined on lane/cpu-training-batch2-declare,
    # par-scaler on lane/cpu-training-par-classical.
    assert fam["training_lanes"] == ("standard-scaler", "minmax-scaler", "standard-scaler-no-mean",
                                     "standard-scaler-no-std", "minmax-scaler-clip", "par-scaler", "par-scaler-minmax")
    assert SCALER_ORACLE in fam["host_modules"] and (ROOT / SCALER_ORACLE).is_file()
    assert (ROOT / "bindings/build_preprocessing_host.sh").is_file()
    assert "_mojolearn_preprocessing" in host_surface.routed_modules()
    src = _read(host_surface.binding_source("preprocessing"))
    for name in ("standard_fit", "standard_transform", "minmax_fit", "minmax_transform",
                 "preprocessing_numeric_mode", "preprocessing_vendor"):
        assert f'("{name}")' in src and name in fam["exports"], name
    text = _read(SCALER_ORACLE)
    assert not GPU_IMPORTS.search(text), f"{SCALER_ORACLE} imports a GPU module"
    assert "from checks.numerics import" in text and "comptime PINNED_SUM_W = 256" in text
    assert "comptime if SCALER_ORACLE_HOST_SABOTAGE:" in text
    assert "values[(i + 1) % n]" in text and "ftz(lower) + ftz(identical_mul(" in text
    assert "SCALER_ORACLE_HOST_SABOTAGE" in src


@reference_training()
def test_scalers_run_on_the_host_when_built():
    if _backend._CPU_ONLY is None:
        print("SKIP: a GPU set loaded; the host route is not taken here")
        return
    if "_mojolearn_preprocessing_host" not in _backend.host_families_built():
        print("SKIP: the preprocessing host binding is not built")
        return
    import numpy as np
    module = _backend.load_host_module("_mojolearn_preprocessing_host")
    assert not bool(module.preprocessing_host_sabotage()), "a sabotage build loaded outside the gate"
    rng = np.random.default_rng(3)
    x = rng.standard_normal((700, 5)).astype(np.float32)
    x[:, 4] = np.float32(2.5)
    s = mojolearn.StandardScaler().fit(x)
    t = np.asarray(s.transform(x[:64]))
    back = np.asarray(s.inverse_transform(t))
    assert np.asarray(s.var_)[4] == 0.0 and np.asarray(s.scale_)[4] == 1.0 and np.asarray(s.mean_)[4] == np.float32(2.5)
    assert np.abs(np.asarray(s.mean_)[:4] - x[:, :4].mean(0)).max() < 1e-4
    assert np.abs(back - x[:64]).max() < 1e-4
    m = mojolearn.MinMaxScaler().fit(x)
    u = np.asarray(m.transform(x[:64]))
    assert np.asarray(m.data_min_).tolist() == x.min(0).tolist() and np.asarray(m.data_max_).tolist() == x.max(0).tolist()
    assert u[:, :4].min() >= 0.0 and u[:, :4].max() <= 1.0
    again = np.asarray(mojolearn.MinMaxScaler().fit(x).transform(x[:64]))
    assert again.tobytes() == u.tobytes(), "two host fits returned different bytes"


def test_manifest_covers_logistic():
    assert "logistic" in host_surface.covered_lanes(), "logistic is not a covered training lane"
    fam = host_surface.family("estimators")
    assert "logistic" in fam["training_lanes"] and "logistic" in fam["inference_lanes"]
    assert QN_ORACLE in fam["host_modules"] and (ROOT / QN_ORACLE).is_file()
    assert "qn_fit" in fam["exports"]
    assert "logistic regression training" not in host_surface.no_cpu_path_sentence()
    src = _read(host_surface.binding_source("estimators"))
    assert '("qn_fit")' in src
    text = _read(QN_ORACLE)
    assert not GPU_IMPORTS.search(text), f"{QN_ORACLE} imports a GPU module"
    assert not re.search(r"^\s*from .*import.*DeviceContext", text, re.M)
    assert "from checks.numerics import" in text
    assert "from glm.host.glm_oracle import host_xty" in text
    assert "from decomposition.host.pca_oracle import STATS_TPB, host_halving_sum" in text
    # The softmax loss and OWL-QN train on the host since
    # lane/cpu-training-batch3; sample_weight still refuses by name.
    for named in ("QN_LOSS_SOFTMAX", "def host_min_owlqn", "sample_weight is NOT IMPLEMENTED"):
        assert named in text, f"{named} is not in {QN_ORACLE}"
    assert "comptime if QN_ORACLE_HOST_SABOTAGE:" in text
    assert "QN_ORACLE_HOST_SABOTAGE" in src
    # Binary training reuses the exact row-parallel inference primitive only
    # above a work threshold; small fits retain the original serial loop.
    assert "if self.n_rows * self.d < (1 << 19):" in text
    assert "self.z = host_qn_decision(" in text


@reference_training()
def test_logistic_runs_on_the_host_when_built():
    if _backend._CPU_ONLY is None:
        print("SKIP: a GPU set loaded; the host route is not taken here")
        return
    if "_mojolearn_estimators_host" not in _backend.host_families_built():
        print("SKIP: the estimators host binding is not built")
        return
    import numpy as np
    rng = np.random.default_rng(4)
    x = rng.standard_normal((400, 3)).astype(np.float32)
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).astype(np.int32)
    fits = [mojolearn.LogisticRegression(max_iter=30).fit(x, y) for _ in range(2)]
    a, b = (np.asarray(f.coef_) for f in fits)
    assert a.tobytes() == b.tobytes(), "two host fits returned different coefficients"
    assert fits[0].retcode_ in (0, 3) and int(np.asarray(fits[0].n_iter_)[0]) >= 1
    pred = np.asarray(fits[0].predict(x))
    assert (pred == y).mean() > 0.9, "the host fit does not separate a separable draw"
    # An l1 penalty takes the host OWL-QN arm (lane/cpu-training-batch3).
    l1 = [np.asarray(mojolearn.LogisticRegression(penalty="l1", C=1.0, max_iter=30).fit(x, y).coef_)
          for _ in range(2)]
    assert l1[0].tobytes() == l1[1].tobytes(), "two host OWL-QN fits returned different coefficients"


@reference_training()
def test_metrics_run_on_the_host_when_built():
    if _backend._CPU_ONLY is None:
        print("SKIP: a GPU set loaded; the host route is not taken here")
        return
    if "_mojolearn_metrics_host" not in _backend.host_families_built():
        print("SKIP: the metrics host binding is not built")
        return
    import numpy as np
    module = _backend.load_host_module("_mojolearn_metrics_host")
    assert not bool(module.metrics_host_sabotage()), "a sabotage build loaded outside the gate"
    mt = mojolearn.metrics
    rng = np.random.default_rng(1)
    x = rng.standard_normal((300, 3)).astype(np.float32)
    yt = (rng.integers(0, 3, 300)).astype(np.int32)
    yp = (yt + (rng.random(300) < 0.2)).astype(np.int32) % 3
    previous_threads = os.environ.get("MOJOLEARN_CPU_THREADS")
    try:
        os.environ["MOJOLEARN_CPU_THREADS"] = "1"
        serial = module.accuracy_score(yt.ctypes.data, yp.ctypes.data, [len(yt)])
        os.environ["MOJOLEARN_CPU_THREADS"] = "7"
        parallel = module.accuracy_score(yt.ctypes.data, yp.ctypes.data, [len(yt)])
    finally:
        if previous_threads is None:
            os.environ.pop("MOJOLEARN_CPU_THREADS", None)
        else:
            os.environ["MOJOLEARN_CPU_THREADS"] = previous_threads
    assert serial == parallel
    got = [(mt.accuracy_score(yt, yp), mt.adjusted_rand_score(yt, yp), mt.v_measure_score(yt, yp),
            mt.r2_score(x[:, 0], x[:, 0] * np.float32(0.9)), mt.silhouette_score(x, yp))
           for _ in range(2)]
    assert got[0] == got[1], "two host runs returned different values"
    acc, ari, vm, r2, sil = got[0]
    assert acc == float((yt == yp).sum()) / 300.0 or abs(acc - (yt == yp).mean()) < 1e-6
    assert -0.5 <= ari <= 1.0 and 0.0 <= vm <= 1.0 and 0.0 <= r2 <= 1.0 and -1.0 <= sil <= 1.0


@reference_training()
def test_kmeans_fit_runs_on_the_host_when_built():
    if _backend._CPU_ONLY is None:
        print("SKIP: a GPU set loaded; the host route is not taken here")
        return
    if "_mojolearn_core_host" not in _backend.host_families_built():
        print("SKIP: the core host binding is not built")
        return
    import numpy as np
    module = _backend.load_host_module("_mojolearn_core_host")
    assert not bool(module.core_host_sabotage()), "a sabotage build loaded outside the gate"
    rng = np.random.default_rng(0)
    x = rng.standard_normal((600, 5)).astype(np.float32)
    outs = []
    for _ in range(2):
        m = mojolearn.KMeans(n_clusters=4, random_state=7).fit(x)
        outs.append((np.asarray(m.cluster_centers_).tobytes(), np.asarray(m.labels_).tobytes(),
                     m.inertia_, m.n_iter_, m.sum_scale_, m.weight_scale_))
    assert outs[0] == outs[1], "two host fits returned different bytes"
    labels = np.asarray(mojolearn.KMeans(n_clusters=4, random_state=7).fit(x).labels_)
    assert labels.shape == (600,) and labels.min() >= 0 and labels.max() < 4
    assert outs[0][3] >= 1 and outs[0][4] > 0 and outs[0][5] > 0


if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for name in names:
        globals()[name]()
        print("ok", name)
    print(f"{len(names)} passed")
    sys.exit(0)
