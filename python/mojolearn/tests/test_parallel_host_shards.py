"""The shard and forecasting drivers run a device='cpu' member on host workers.

In a process with a GPU set, a member that says device='cpu' is sharded over
workers on the CPU-only set (the CPU implementation), and its answer has the
member's own CPU bits. The default and device='gpu' keep the GPU workers. On a
CPU-only installation every member already runs there; the GPU-process tests
skip.
"""
import numpy as np
import pytest

import mojolearn
from mojolearn import StandardScaler

GPU_PROCESS = mojolearn.vendor() != "cpu"
gpu_process = pytest.mark.skipif(not GPU_PROCESS, reason="a CPU-only installation has no GPU set")


def data(rows=64, cols=6, seed=3):
    rng = np.random.default_rng(seed)
    return rng.random((rows, cols), dtype=np.float32)


def same_bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


@gpu_process
def test_a_cpu_scaler_is_sharded_on_host_workers_with_its_own_cpu_bits():
    from mojolearn.parallel_preprocessing import fit_scaler, transform_scaler
    x = data()
    own = StandardScaler(device="cpu").fit(x)
    sharded = fit_scaler(StandardScaler(device="cpu"), x, devices=(0,), columns_per_shard=2)
    assert same_bits(sharded.mean_, own.mean_)
    assert same_bits(sharded.var_, own.var_)
    assert same_bits(transform_scaler(sharded, x, devices=(0,), columns_per_shard=2), own.transform(x))
    assert sharded.get_params()["device"] == "cpu"


@gpu_process
def test_the_default_and_a_gpu_scaler_keep_the_gpu_workers():
    from mojolearn.parallel_preprocessing import fit_scaler
    x = data()
    own = StandardScaler().fit(x)
    for device in (None, "gpu"):
        sharded = fit_scaler(StandardScaler(device=device), x, devices=(0,), columns_per_shard=2)
        assert same_bits(sharded.mean_, own.mean_)


def test_a_cpu_scaler_on_any_installation_has_the_cpu_bits():
    from mojolearn.parallel_preprocessing import fit_scaler
    x = data()
    own = StandardScaler(device="cpu").fit(x)
    sharded = fit_scaler(StandardScaler(device="cpu"), x, devices=(0,), columns_per_shard=3)
    assert same_bits(sharded.mean_, own.mean_)
