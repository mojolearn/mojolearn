# SPDX-License-Identifier: Apache-2.0
"""UNPACK of a member estimator's `device` for the multi-GPU drivers
(bindings/parallel_public_native_door.mojo, bindings/parallel_shard_native_door.mojo).

A multi-GPU driver spreads one fit or query of the estimator it is handed
across the GPUs listed in `devices=`. It has no host form. Before the device
wave it ignored the estimator's own `device`: a `device="cpu"` member was
run on the GPUs, and its state record carried `device="cpu"` to workers that
were started for a GPU. Now the driver reads the member's `device` once,
before any partition, pool or worker request, through the one rule
(core/device_route_dw_msel_parallel.mojo): None, a member without the
parameter and `'gpu'` run as before; `'cpu'` in a process with a GPU set is
refused in words that name the estimator's own method; an unknown value is
refused in the pilot's words. In a CPU-only process `'cpu'` is the
installation's device and nothing changes.

Cost on the default call: one read of the member's instance dictionary with
the held key and one compare, once per driver call.
"""
from std.python import PythonObject
from bindings.py_balanced import py_get
from bindings.parallel_keys_unpack import parallel_keys
from bindings.msel_member_device_unpack import member_device_unpack
from core.device_route_dw_msel_parallel import MEMBER_DEVICE_CPU


def parallel_member_says_cpu(fields: PythonObject) raises -> Bool:
    """True when the estimator whose instance dictionary is `fields` says
    `device='cpu'`. No key, None and `'gpu'` answer False; any other value
    is refused in words."""
    var nk = parallel_keys()
    var value = py_get(fields, nk[].k_device)
    if value is None:
        return False
    return member_device_unpack(value) == MEMBER_DEVICE_CPU
