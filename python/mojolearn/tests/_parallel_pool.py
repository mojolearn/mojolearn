# SPDX-License-Identifier: Apache-2.0
"""The public-output rule the multi-GPU drivers apply to their native result.

Every multi-GPU driver is one native call (docs/plans/
NATIVE_PARALLEL_DRIVER_20261024.md): a cooperative driver binds a device plan
in its family's binding around the plain native owner
(bindings/family_native_door.mojo `family_par_run`), a Python-sharded driver runs
its shards on the native local process pool (bindings/parallel_shard_native_door.mojo,
parallel_forecast_native.mojo). The Python `DevicePool` and its pickle worker
(`_parallel_worker.py`) that did this until 2026-10-24 are removed.

What stays is the one shell rule a driver whose plain method is a public
result method applies to the native result: the plain method's public
conversion for a caller outside the package, the native result otherwise.

THE DRIVER NEGATIVE CONTROL keeps its two variables, now read natively
(bindings/parallel_shard_common_door.mojo `shard_read_shift`): with
MOJOLEARN_PAR_DRIVER_SABOTAGE=1 and MOJOLEARN_HOST_ALLOW_SABOTAGE=1 every
shard on an owner above rank 0 whose true start is above 0 reads one position
early while it is still placed at its true offset. Inert at one device.
"""

#: the driver negative control's switch (read natively, see above)
PAR_DRIVER_SABOTAGE = 'MOJOLEARN_PAR_DRIVER_SABOTAGE'


def par_driver_sabotage():
    """True when both switch variables are set: what the native drivers read
    at call time (bindings/parallel_shard_common_door.mojo `shard_read_shift`),
    for a record that states whether the run carried the control."""
    import os
    return (os.environ.get(PAR_DRIVER_SABOTAGE) == '1'
            and os.environ.get('MOJOLEARN_HOST_ALLOW_SABOTAGE') == '1')
