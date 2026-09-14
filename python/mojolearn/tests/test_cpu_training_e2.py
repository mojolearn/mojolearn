# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Workstream E batch 2, CPU training for the kmeans lane
(lane/cpu-training-e2, 2026-09-14), checked from SOURCE so it runs on a box
with nothing built, plus one runtime check that runs only where the core
host binding is built and the package took the CPU-only path.

What the source checks hold: the manifest lists kmeans as a core training
lane and names it for the docs, and no longer names k-means as a lane with
no CPU path; the core host binding registers `kmeans_fit` under the GPU
binding's name; the oracle imports no GPU module and reads its fold widths
from the kernel matrix; the sabotage define reaches the oracle through the
quantized accumulation (the arm that moves every centroid) and the binding
reads it back; the oracle spells the device's round seed reassembly (the
sign-extended low half, measured on the M4); the CPU identity gate workflow
triggers on the oracle.

The runtime check (skipped, and SAID to be skipped, when the binding is
absent or a GPU set loaded): `KMeans.fit` runs through the host binding on
a small draw, twice, and returns the same bytes, with `labels_` in range
and one label per row; the binding reads back sabotage False. It is a
plumbing check. The bit claim against the GPU columns is the CPU identity
gate's, not this file's.

    cd python && python3 -m mojolearn.tests.test_cpu_training_e2
"""
import re
import sys
from pathlib import Path

import mojolearn
from mojolearn import _backend, host_surface

ROOT = Path(__file__).resolve().parents[3]

ORACLE = "cluster/host/kmeans_oracle.mojo"
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


def test_binding_registers_kmeans_fit():
    src = _read(host_surface.binding_source("core"))
    assert '("kmeans_fit")' in src, "the core host binding does not register kmeans_fit"
    assert "kmeans_fit" in host_surface.family("core")["exports"], "the manifest does not list kmeans_fit"
    for absent in ("rbc_knn_search", "radius_neighbors_count", "radius_neighbors_fill"):
        assert f'("{absent}")' not in src, f"{absent} must stay absent so it refuses by name"


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


def test_workflow_triggers_on_the_oracle():
    text = _read(".github/workflows/cpu-identity-gate.yml")
    assert f'- "{ORACLE}"' in text, f"cpu-identity-gate.yml does not trigger on {ORACLE}"


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
