# SPDX-License-Identifier: Apache-2.0
"""Logical row-shard CPU execution, separately from physical GPU qualification."""
import numpy as np
import pytest

from mojolearn import RBFSampler, _backend, host_surface
from mojolearn._cpu_reference import reference_training
from mojolearn._native_parallel_shard import _native_parallel_shard
from mojolearn.parallel_classical import transform_rbf_sampler

GATE_BACKENDS = ("cpu",)


def test_rbf_route_is_pending_and_only_admits_python_row_shards():
    assert "par-rbf-sampler" in host_surface.covered_lanes()
    assert "par-rbf-sampler" not in host_surface.public_reference_lanes()
    from mojolearn import _verify_all as va, _verify_reference as vr
    from mojolearn import _verification_coverage as coverage
    harness, table = va.load_harness(), vr.load_table()
    with pytest.raises(ValueError, match="par-rbf-sampler"):
        va.select_lanes(harness, table, "cpu", "full", ["par-rbf-sampler"])
    assert va.select_lanes(harness, table, "cpu", "full", ["par-rbf-sampler"],
                           include_pending=True)[0] == ["par-rbf-sampler"]
    row = coverage.inventory(harness, table, "cpu")["lanes"]["par-rbf-sampler"]
    assert row["status"] == "not_applicable", (
        "the inventory stopped saying `excluded` on 2026-09-20: nothing is removed from "
        "the public surface, and a one-device par-* claim is inapplicable, not absent")
    assert row["execution"]["cpu_logical_shards"]
    assert not row["execution"]["requires_gpu_for_execution"]
    assert not row["execution"]["physical_multi_gpu_measured_by_cpu"]
    # the native drivers: row shards run in worker processes on the CPU route,
    # a cooperative driver (here the kernel methods) refuses it by name
    if _backend._CPU_ONLY is not None:
        from mojolearn.kernel_methods import KernelRidge
        from mojolearn.parallel_classical import apply_kernel_method
        with pytest.raises(NotImplementedError, match="no CPU implementation of the cooperative"):
            apply_kernel_method(KernelRidge(), np.zeros((2, 2), np.float32), devices=(0,))


def _require_cpu():
    if _backend._CPU_ONLY is None:
        pytest.skip("requires a CPU-only package")
    from pathlib import Path
    if not Path(_backend.host_module_path("_mojolearn_kernel_methods_host")).exists():
        pytest.skip("kernel methods host binding is not built")


@pytest.mark.parametrize("shard_rows", [1, 7, 19, 32])
def test_cpu_rbf_shards_preserve_exact_rows_and_fitted_state(shard_rows, monkeypatch):
    _require_cpu()
    X = np.random.default_rng(412).standard_normal((19, 7)).astype(np.float32)[::-1]
    with reference_training():
        model = RBFSampler(gamma=0.7, n_components=13, random_state=11).fit(X)
    weights = np.asarray(model.random_weights_).tobytes()
    offsets = np.asarray(model.random_offset_).tobytes()
    expected = np.asarray(model.transform(X))
    _native_parallel_shard("trace_take", ())
    # One worker executes multiple logical shards without introducing a second
    # local numerical worker. The public inference path needs no training scope.
    actual = transform_rbf_sampler(model, X, devices=(0,), rows_per_shard=shard_rows)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()
    trace = _native_parallel_shard("trace_take", ())
    assert len(trace) == 1 and trace[0]["devices"] == [0]
    calls = [call["operation"] for call in trace[0]["calls"]]
    assert calls == ["rbf_rows"] * len(range(0, 19, shard_rows))
    assert np.asarray(model.random_weights_).tobytes() == weights
    assert np.asarray(model.random_offset_).tobytes() == offsets


def test_cpu_rbf_reordered_worker_rows_are_detectable(monkeypatch):
    _require_cpu()
    X = np.random.default_rng(77).standard_normal((15, 7)).astype(np.float32)
    with reference_training():
        model = RBFSampler(n_components=13, random_state=11).fit(X)
    expected = np.asarray(model.transform(X))
    # the transport negative control (bindings/parallel_shard_common.mojo
    # `shard_map`): the first numerical batch comes back reversed, once
    monkeypatch.setenv("MOJOLEARN_PAR_TRANSPORT_FAULT", "reverse_results")
    monkeypatch.setenv("MOJOLEARN_HOST_ALLOW_SABOTAGE", "1")
    _native_parallel_shard("faults_take", ())
    broken = transform_rbf_sampler(model, X, devices=(0,), rows_per_shard=5)
    assert _native_parallel_shard("faults_take", ()) == 1
    # The identity lane compares this ordered output against the ordinary
    # transform; equal shapes or repeated stable hashes alone are insufficient.
    from mojolearn._verify_all import load_harness
    harness = load_harness()
    with pytest.raises(ValueError, match="differ"):
        harness._same_bytes("reordered shards", broken, "plain transform", expected)
