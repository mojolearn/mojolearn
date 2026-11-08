"""The CPU column serves the x_decomp kernel-call provider to the manifold kit.

On a CPU-only install `_backend.binding("_mojolearn_x_decomp")` is the lazy
`_HostBinding` facade. The UMAP/spectral bridge must retain the exact host
module that owns `x_decomp_native_api_v1`, because the provider admission
(providers/python/neural_descriptor_pyapi.mojo `neural_descriptor_entry`)
accepts an entry only from its owning module. A copy of the entries into a
fresh module made every UMAP fit on the CPU column refuse with
"native provider requires compiled entry x_decomp_native_api_v1".

Written without a run (lane fix-x-decomp-native-api-cpu, 2026-11-08).
"""
from pathlib import Path
from types import BuiltinFunctionType, ModuleType

import numpy as np
import pytest

from mojolearn import _backend, host_surface
from mojolearn import _native_spectral_umap_bridge as bridge

_ROOT = Path(__file__).resolve().parents[3]


def test_host_surface_lists_the_provider_for_vendor_cpu():
    assert host_surface.routed_modules()["_mojolearn_x_decomp"] == "_mojolearn_x_decomp_host"
    family = [f for f in host_surface.FAMILIES if f["binding"] == "_mojolearn_x_decomp_host"]
    assert len(family) == 1
    assert "x_decomp_native_api_v1" in family[0]["exports"]


def test_host_binding_source_registers_the_provider():
    text = (_ROOT / "bindings" / "_mojolearn_x_decomp_host.mojo").read_text()
    assert 'm.def_function[x_decomp_native_api_v1_binding]("x_decomp_native_api_v1")' in text


def test_bridge_resolves_the_host_facade_to_its_exact_module(monkeypatch):
    owner = ModuleType("mojolearn._host._mojolearn_x_decomp_host")
    loaded = []

    def fake_load(basename):
        loaded.append(basename)
        return owner

    facade = _backend._HostBinding("mojolearn._mojolearn_x_decomp", "_mojolearn_x_decomp",
                                   "_mojolearn_x_decomp_host", "test")
    monkeypatch.setattr(_backend, "load_host_module", fake_load)
    monkeypatch.setattr(_backend, "_CPU_ONLY", "test: CPU-only set")
    assert bridge._host_owner(facade) is owner
    assert loaded == ["_mojolearn_x_decomp_host"]


def test_bridge_keeps_an_exact_module_unchanged():
    module = ModuleType("exact")
    assert bridge._host_owner(module) is module


@pytest.mark.skipif(_backend._CPU_ONLY is None, reason="the CPU-only set is not selected")
def test_cpu_only_decomp_route_owns_the_provider():
    routes = bridge.connect()
    module = routes["decomp"]["identical"]
    assert type(module) is ModuleType
    entry = module.__dict__["x_decomp_native_api_v1"]
    assert type(entry) is BuiltinFunctionType and entry.__self__ is module


@pytest.mark.skipif(_backend._CPU_ONLY is None, reason="the CPU-only set is not selected")
def test_cpu_only_umap_fit_runs():
    from mojolearn import UMAP
    rng = np.random.default_rng(0)
    x = rng.standard_normal((80, 6)).astype(np.float32)
    emb = UMAP(n_neighbors=8, n_components=2, n_epochs=20, random_state=0).fit(x).embedding_
    assert emb.shape == (80, 2)
