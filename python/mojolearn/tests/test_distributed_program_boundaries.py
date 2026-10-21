# SPDX-License-Identifier: Apache-2.0
"""Independent gradients and audited lifecycle/checkpoint boundaries.

Python supplies fixtures/oracles only. Missing native entries fail these cases;
physical multi-GPU coverage requires the controller's distinct device IDs.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from mojolearn import ops
from mojolearn._verify_native_runtime import audit_native_call


def native(operation, *args, **kwargs):
    return audit_native_call(operation, *args, **kwargs).assert_native()


def open_model(program, route, optimizer):
    devices = tuple(int(value) for value in os.environ.get(
        "MOJOLEARN_DISTRIBUTED_PROGRAM_DEVICES", "0").split(","))
    if route == "distribute":
        return native(program.distribute, devices, logical_shards=max(2, len(devices)),
                      optimizer=optimizer, seed=91)
    return native(program.partition, devices, stages=max(2, len(devices)), tensor_shards=1,
                  microbatches=2, optimizer=optimizer, seed=91)


def same_state(a, b):
    for key in ("epoch", "step", "seed"):
        assert a[key] == b[key]
    for key in ("parameters", "first_moments", "second_moments"):
        for left, right in zip(a[key], b[key], strict=True):
            np.testing.assert_array_equal(left.view(np.uint32), right.view(np.uint32))


@pytest.mark.parametrize("route", ["distribute", "partition"])
@pytest.mark.parametrize("graph", ["mixed_shared", "mixed_distinct", "detached", "copied"])
def test_detached_paths_independent_gradient(route, graph):
    data = np.array([[1], [2]], np.float32)
    target = np.zeros_like(data)
    with ops.Session() as s:
        x, w = s.tensor(data), s.tensor([[0.25]])
        product = s.matmul(x, w)
        if graph == "mixed_shared":
            out = s.add(product, s.detach(product))
            # mean(2 * (.5*x) * x) = 2.5, with no derivative through detach.
            expected = [np.float32(0.25) - np.float32(0.0625 * 2.5)]
        elif graph == "mixed_distinct":
            v = s.tensor([[0.5]])
            out = s.add(product, s.detach(s.matmul(x, v)))
            expected = [np.float32(0.25) - np.float32(0.0625 * 3.75), np.float32(0.5)]
        else:
            out = s.detach(product) if graph == "detached" else s.copy(product)
            expected = [np.float32(0.25)]
        p = s.capture(x, out)
        opt = s.optimizer("sgd", lr=0.0625, momentum=0)
        d = open_model(p, route, opt)
        try:
            assert native(d.__enter__) is d
            native(d.fit, data, targets=target, epochs=1, batch_size=2, shuffle=False)
            state = native(d.state)
            assert state["step"] == state["epoch"] == 1
            assert len(state["parameters"]) == len(expected)
            for value, wanted in zip(state["parameters"], expected, strict=True):
                np.testing.assert_array_equal(value, np.array([[wanted]], np.float32))
        finally:
            assert native(d.__exit__, None, None, None) is None


@pytest.mark.parametrize("route", ["distribute", "partition"])
def test_zero_adamw_checkpoint_path_and_native_context(route, tmp_path):
    data = np.array([[1], [2]], np.float32)
    target = np.zeros_like(data)
    # With zero gradients and zero initial moments, AdamW only decays the
    # selected parameter. These three dyadic updates have exact FP32 oracles.
    expected = np.float32(0.25)
    for _ in range(3):
        expected = np.float32(expected - np.float32(np.float32(0.0625) *
                                                    np.float32(np.float32(0.125) * expected)))
    with ops.Session() as s:
        x, w = s.tensor(data), s.tensor([[0.25]])
        p = s.capture(x, s.detach(s.matmul(x, w)))
        opt = s.optimizer("adamw", lr=0.0625, weight_decay=0.125)
        path = Path(tmp_path) / (route + ".mjl")
        d = open_model(p, route, opt)
        try:
            assert native(d.__enter__) is d
            native(d.fit, data, targets=target, epochs=1, batch_size=2, shuffle=False)
            saved = native(d.state)
            native(d.save, path)
            native(d.fit, data, targets=target, epochs=2, batch_size=2, shuffle=False)
            final = native(d.state)
            np.testing.assert_array_equal(final["parameters"][0], np.array([[expected]], np.float32))
            for key in ("first_moments", "second_moments"):
                np.testing.assert_array_equal(final[key][0], np.zeros((1, 1), np.float32))
            assert final["step"] == final["epoch"] == 3
        finally:
            native(d.__exit__, None, None, None)
        loader = s.load_distributed if route == "distribute" else s.load_partitioned
        restored = native(loader, path, devices=(0,))
        try:
            assert native(restored.__enter__) is restored
            same_state(native(restored.state), saved)
            native(restored.fit, data, targets=target, epochs=2, batch_size=2, shuffle=False)
            same_state(native(restored.state), final)
        finally:
            native(restored.__exit__, None, None, None)
        audit = audit_native_call(restored.__enter__)
        assert audit.exception is not None
        assert not audit.frames and not audit.wrapper_work and not audit.forbidden_native
