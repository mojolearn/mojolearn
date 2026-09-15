# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU training for the umap lane (lane/cpu-training-umap-b, 2026-09-14),
checked from SOURCE so it runs on a box with nothing built, plus a runtime
check that runs only where the metrics host binding is built and the
package took the CPU-only path.

What the source checks hold: the manifest covers the umap lane in the
metrics family and no longer names UMAP as having no CPU path; the metrics
host binding registers the GPU binding's umap_fit_transform, umap_transform
and umap_numeric_mode with the same parameter contract; the oracle imports
no GPU module and NOT the serial host optimizer (whose Gauss-Seidel order
gives different bits from the IDENTICAL device fold); the oracle spells the
device epoch kernel's bit-carrying statements character for character; the
sabotage define reaches the negative draw and the binding reads it back; the
CPU identity gate workflow triggers on the oracle and the UMAP host modules
it imports.

The runtime check (skipped, and SAID to be skipped, when the binding is
absent or a GPU set loaded): a small UMAP fits and transforms twice through
the host binding and returns the same bytes. It is a plumbing check. The bit
claim against the GPU columns is the CPU identity gate's, not this file's.

    cd python && python3 -m mojolearn.tests.test_cpu_training_umap
"""
import re
import sys
from pathlib import Path

import mojolearn
from mojolearn import _backend, host_surface

ROOT = Path(__file__).resolve().parents[3]

ORACLE = "umap/host/umap_oracle.mojo"
DEVICE = "umap/optimizer_identical_device.mojo"
GPU_IMPORTS = re.compile(r"^\s*from\s+(max\.gpu|std\.gpu)", re.M)

#: Statements of `umap_identical_epoch_kernel` whose spelling carries the
#: bits; each must appear in the device file and in the oracle.
KERNEL_STATEMENTS = (
    "if Int(next_f * s) <= Int(epoch_f * s):",
    "delta[c] = ftz(x[c] - source",
    "d2 = ftz(identical_mul_add(delta[c], delta[c], d2))",
    "var dp = identical_pow(d2, b)",
    "identical_mul(neg2ab, identical_div(dp, d2)),",
    "ftz(identical_mul_add(a, dp, Float32(1.0))),",
    "acc[c] = ftz(acc[c] + g)",
    "var lane = j & 3",
    "UInt32(j >> 2),",
    "var other = Int(draw[lane] % n_u)",
    "n2 = ftz(identical_mul_add(nd[c], nd[c], n2))",
    "ftz(Float32(0.001) + n2),",
    "ftz(identical_mul_add(a, np_, Float32(1.0))),",
    "var neg2ab = -Float32(2.0) * a * b",
    "var rep2b = Float32(2.0) * repulsion * b",
)


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def test_manifest_covers_the_umap_lane():
    fam = host_surface.family("metrics")
    assert "umap" in fam["training_lanes"]
    assert ORACLE in fam["host_modules"] and (ROOT / ORACLE).is_file()
    assert "UMAP" in fam["classes"]
    assert "umap" in host_surface.covered_lanes()
    assert "UMAP" not in host_surface.no_cpu_path_sentence(), host_surface.no_cpu_path_sentence()
    assert "UMAP" in host_surface.training_sentence()
    for name in ("umap_fit_transform", "umap_transform", "umap_numeric_mode"):
        assert name in fam["exports"], f"the manifest does not list {name} for metrics"


def test_binding_registers_the_gpu_names():
    src = _read(host_surface.binding_source("metrics"))
    gpu = _read("bindings/_mojolearn_metrics.mojo")
    for name in ("umap_fit_transform", "umap_transform", "umap_numeric_mode"):
        assert f'("{name}")' in src, f"the metrics host binding does not register {name}"
        assert f'("{name}")' in gpu, f"{name} is not a GPU binding name"
    for sentence in (
        '_want(String("umap_fit_transform"), params, 13)',
        '_want(String("umap_transform addresses"), addrs, 4)',
        '_want(String("umap_transform parameters"), params, 14)',
        '"UMAP requires positive features and a nonnegative seed"',
        '"UMAP requires 2D/3D output and enough samples for spectral init"',
        '"UMAP transform requires positive dimensions and a nonnegative seed"',
    ):
        assert sentence in src and sentence in gpu, sentence


def test_oracle_imports_no_gpu_and_not_the_host_loop():
    text = _read(ORACLE)
    assert not GPU_IMPORTS.search(text), f"{ORACLE} imports a GPU module"
    assert not re.search(r"^\s*from .*import.*DeviceContext", text, re.M), f"{ORACLE} imports DeviceContext"
    modules = sorted(set(re.findall(r"^from\s+([\w.]+)\s+import", text, re.M)))
    assert modules == [
        "checks.kernel_matrix", "checks.numerics", "core.knn_host_predict",
        "spectral.host.spectral_oracle", "spectral.impl.sparse.coo", "std.math",
        "std.memory", "std.sys.compile", "umap.curve", "umap.graph", "umap.params",
        "umap.sparse_graph",
    ], modules
    for rel in ("umap/sparse_graph.mojo", "umap/graph.mojo", "umap/curve.mojo", "umap/params.mojo"):
        assert not GPU_IMPORTS.search(_read(rel)), f"{rel} imports a GPU module"
        assert "DeviceContext" not in _read(rel), f"{rel} names DeviceContext"


def test_oracle_spells_the_device_kernel():
    oracle = _read(ORACLE)
    device = _read(DEVICE)
    for statement in KERNEL_STATEMENTS:
        assert statement in device, f"the device kernel no longer spells {statement!r}; restate the oracle"
        assert statement in oracle, f"the oracle does not spell {statement!r}"
    assert oracle.count("acc[c] = ftz(acc[c] + g)") == device.count("acc[c] = ftz(acc[c] + g)")
    assert "ftz(Float32(Float64(weight) / Float64(max_weight)))" in oracle
    assert "first.append(ftz(initial[i]))" in oracle, "the initial upload is flushed"
    assert "oracle_embedding[DType.float32](\n        coo, n_components + 1, True, True, Float32(1e-5), seed" in oracle


def test_sabotage_define_moves_the_negative_draw():
    text = _read(ORACLE)
    define = host_surface.sabotage_define("metrics")
    assert define == "MOJOLEARN_HOST_SABOTAGE"
    assert f'is_defined["{define}"]()' in text
    assert text.count("comptime if UMAP_ORACLE_HOST_SABOTAGE:") == 2
    assert "draw_epoch = epoch + 1" in text
    assert "UMAP_ORACLE_HOST_SABOTAGE" in _read(host_surface.binding_source("metrics"))


def test_workflow_triggers_on_the_oracle():
    text = _read(".github/workflows/cpu-identity-gate.yml")
    for rel in (ORACLE, "umap/sparse_graph.mojo", "umap/graph.mojo", "umap/curve.mojo", "umap/params.mojo"):
        assert f'- "{rel}"' in text, f"cpu-identity-gate.yml does not trigger on {rel}"


def test_umap_fits_on_the_host_when_built():
    if _backend._CPU_ONLY is None:
        print("SKIP: a GPU set loaded; the host route is not taken here")
        return
    if "_mojolearn_metrics_host" not in _backend.host_families_built():
        print("SKIP: the metrics host binding is not built")
        return
    import numpy as np
    module = _backend.load_host_module("_mojolearn_metrics_host")
    assert not bool(module.metrics_host_sabotage()), "a sabotage build loaded outside the gate"
    rng = np.random.default_rng(0)
    x = rng.standard_normal((128, 5)).astype(np.float32)
    q = rng.standard_normal((16, 5)).astype(np.float32)
    outs = []
    for _ in range(2):
        m = mojolearn.UMAP(n_neighbors=6, n_components=2, n_epochs=4, random_state=1).fit(x)
        t = m.transform(q)
        emb = np.asarray(m.embedding_)
        assert np.isfinite(emb).all() and np.isfinite(np.asarray(t)).all()
        outs.append((emb.tobytes(), np.asarray(t).tobytes()))
    assert outs[0] == outs[1], "two host fits returned different bytes"


if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for name in names:
        globals()[name]()
        print("ok", name)
    print(f"{len(names)} passed")
    sys.exit(0)
