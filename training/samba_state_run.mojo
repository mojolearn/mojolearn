# SPDX-License-Identifier: Apache-2.0
"""RUN half of the Samba state boundary (lane rs-neural, wave 8 run split,
from bindings/samba_state_native.mojo): the optimizer device plan of one
Samba entry. Typed Mojo only, no Python import
(tools/python_free_dirs.sh). Same body."""
from core.device_plan import DevicePlan
from training.optimizer_device_plan import optimizer_device_plan


def samba_optimizer_device_plan() raises -> DevicePlan:
    """The optimizer device plan of one Samba entry: `optimizer_device_plan`,
    read once here and passed down as a value."""
    return optimizer_device_plan()
