# SPDX-License-Identifier: Apache-2.0
"""The public-output rule the multi-GPU drivers apply to their native result.

Every multi-GPU driver is one native call (docs/plans/
NATIVE_PARALLEL_DRIVER_20261024.md): a cooperative driver binds a device plan
in its family's binding around the plain native owner
(bindings/family_native.mojo `family_par_run`), a Python-sharded driver runs
its shards on the native local process pool (bindings/parallel_shard_native.mojo,
parallel_forecast_native.mojo). The Python `DevicePool` and its pickle worker
(`_parallel_worker.py`) that did this until 2026-10-24 are removed.

What stays is the one shell rule a driver whose plain method is a public
result method applies to the native result: the plain method's public
conversion for a caller outside the package, the native result otherwise.

THE DRIVER NEGATIVE CONTROL keeps its two variables, now read natively
(bindings/parallel_shard_common.mojo `shard_read_shift`): with
MOJOLEARN_PAR_DRIVER_SABOTAGE=1 and MOJOLEARN_HOST_ALLOW_SABOTAGE=1 every
shard on an owner above rank 0 whose true start is above 0 reads one position
early while it is still placed at its true offset. Inert at one device.
"""

#: the driver negative control's switch (read natively, see above)
PAR_DRIVER_SABOTAGE = 'MOJOLEARN_PAR_DRIVER_SABOTAGE'


def par_driver_sabotage():
    """True when both switch variables are set: what the native drivers read
    at call time (bindings/parallel_shard_common.mojo `shard_read_shift`),
    for a record that states whether the run carried the control."""
    import os
    return (os.environ.get(PAR_DRIVER_SABOTAGE) == '1'
            and os.environ.get('MOJOLEARN_HOST_ALLOW_SABOTAGE') == '1')


def public_method_result(estimator, method, result, frame):
    """`result` as the plain `estimator.<method>` returns it to the caller
    whose frame is `frame`: the public conversion (`_mode._public_result`)
    for a caller outside the package, the native result otherwise. The
    driver passes `sys._getframe(1)`, its own caller.

    Every driver whose plain method is a public result method returns
    through here: the native owner returns the native result (an internal
    caller's), so the driver alone decides what its caller sees, exactly as
    the plain method's `_mode._public_output` wrapper does (2026-10-24,
    par-kernel-ridge dtype flip: a regressor's predict is float32 for the
    identity harness, float64 for a user). Only the names that wrapper
    covers (`_mode._OUTPUTS`) convert; any other method's result is the
    plain method's own."""
    from . import _mode
    if method in _mode._OUTPUTS and _mode._caller_outside(frame):
        return _mode._public_result(estimator, method, result)
    return result
