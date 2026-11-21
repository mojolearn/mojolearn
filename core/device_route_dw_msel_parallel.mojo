# SPDX-License-Identifier: Apache-2.0
"""The device of a MEMBER estimator, as an orchestration family reads it.

Model selection, the multi-GPU drivers and the other orchestration owners
hold member estimators. A member's `device` is the member's own constructor
parameter and is passed through unchanged: a clone carries it, and the
member's own fit, predict and score run on the member's own device (its
route owner decides that, core/linear_route.mojo for the linear family).

What an orchestration owner decides ITSELF from a member's device is small
and is all in this file:

* where the owner's fold rows live. A member that says `device="cpu"` in a
  process that has a GPU set takes its folds gathered on the host; rows
  resident on the device are for members that run there.
* whether a multi-GPU driver may take the member at all. Those drivers
  spread one fit or query across the GPUs listed in `devices=` and have no
  host form, so a member that says `device="cpu"` is refused in words.

The codes, the spelling rule and the refusal of an unknown value are the
linear pilot's (core/linear_route.mojo): one rule for every family. A
member whose class has no `device` parameter yet reads as None (this
installation's device).

No Python here. Every function is a compare of two words; nothing to split.
"""
from core.linear_route import (
    LINEAR_DEVICE_CPU, LINEAR_DEVICE_GPU, LINEAR_DEVICE_INSTALL, linear_device_installed,
)

comptime MEMBER_DEVICE_GPU = LINEAR_DEVICE_GPU
comptime MEMBER_DEVICE_CPU = LINEAR_DEVICE_CPU
# `device=None`, and a member with no `device` parameter.
comptime MEMBER_DEVICE_INSTALL = LINEAR_DEVICE_INSTALL

# The vendor word of a process whose base binding recorded no table key: it
# is not 0, so nothing is treated as CPU only on a guess.
comptime MEMBER_VENDOR_UNKNOWN = -1


def member_folds_on_host(device: Int, process_vendor: Int) -> Bool:
    """True when an owner gathers this member's fold rows on the host: the
    member says `cpu` and the process has a GPU set (`process_vendor` is the
    base binding's vendor word, 0 CPU only). In a CPU-only process nothing
    changes: its row owner already works on host rows."""
    return device == MEMBER_DEVICE_CPU and process_vendor != 0


def member_own_device_admit(device: Int, process_vendor: Int) raises:
    """The check of an owner's OWN `device` where it is given (constructor,
    `set_params`): an explicit `gpu` in a CPU-only process is refused in the
    pilot's words. None and `cpu` always pass."""
    if device == MEMBER_DEVICE_GPU and process_vendor == 0:
        _ = linear_device_installed(device, process_vendor)


def member_multi_gpu_refused(device: Int, process_vendor: Int) -> Bool:
    """True when a multi-GPU driver must refuse this member: it says `cpu`
    and the process has a GPU set. In a CPU-only process `cpu` is the
    installation's device and the driver behaves as it does for None."""
    return device == MEMBER_DEVICE_CPU and process_vendor != 0


def member_multi_gpu_refusal(driver: String) raises:
    """The one refusal of a `device="cpu"` member by a multi-GPU driver."""
    raise Error("ValueError: parallel " + driver + " spreads its work across the GPUs listed in devices=, "
        + "and this estimator has device='cpu'. Call the estimator's own method to run it on the CPU, "
        + "or leave device=None to run on this installation's device")
