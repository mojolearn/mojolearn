# SPDX-License-Identifier: Apache-2.0
"""Audited lifecycle/checkpoint boundaries of distributed and partitioned programs.

The independent gradients and the checkpoint values are native checks
(checks/native_tests/training_distributed_program.mojo). Missing native entries fail these cases;
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


@pytest.mark.parametrize("route", ["distribute", "partition"])
def test_lifecycle_and_checkpoint_calls_are_native(route, tmp_path):
    # The values (detached-path gradients, zero-gradient AdamW decay and the
    # checkpoint continuation) are checked natively in
    # checks/native_tests/training_distributed_program.mojo; this keeps the
    # audit that every public lifecycle call is one native call.
    data = np.array([[1], [2]], np.float32)
    target = np.zeros_like(data)
    with ops.Session() as s:
        x, w = s.tensor(data), s.tensor([[0.25]])
        p = s.capture(x, s.detach(s.matmul(x, w)))
        opt = s.optimizer("adamw", lr=0.0625, weight_decay=0.125)
        path = Path(tmp_path) / (route + ".mjl")
        d = open_model(p, route, opt)
        try:
            assert native(d.__enter__) is d
            native(d.fit, data, targets=target, epochs=1, batch_size=2, shuffle=False)
            native(d.state)
            native(d.save, path)
        finally:
            assert native(d.__exit__, None, None, None) is None
        loader = s.load_distributed if route == "distribute" else s.load_partitioned
        restored = native(loader, path, devices=(0,))
        try:
            assert native(restored.__enter__) is restored
            native(restored.state)
        finally:
            native(restored.__exit__, None, None, None)
        audit = audit_native_call(restored.__enter__)
        assert audit.exception is not None
        assert not audit.frames and not audit.wrapper_work and not audit.forbidden_native
