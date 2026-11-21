# SPDX-License-Identifier: Apache-2.0
"""Typed admission of the native multi-GPU driver doors (RUN): the selected
device indices, the installed vendor's refusals and the numeric-mode word.
`bindings/parallel_plan_door.mojo` unpacks the Python values into these
arguments and binds the plan in the family's binding; no Python object
reaches this file. A handful of words per driver call: nothing to split
across cores."""
from core.device_plan import device_plan_logical
from core.mode_answer import fast_is_answered_by_identical

comptime PLAN_DEVICES_REFUSAL = "ValueError: devices must be distinct nonnegative integer indices"
comptime PLAN_VENDOR_REFUSAL = "ValueError: device selection is unavailable for this vendor/device group"


def plan_indices_admit(devices: List[Int]) raises:
    """At least one index, none negative, none repeated. The logical-device
    test option (core/device_plan.mojo) admits a repeated index; nothing
    else about the route changes."""
    if len(devices) == 0:
        raise Error(PLAN_DEVICES_REFUSAL)
    var logical = device_plan_logical()
    for i in range(len(devices)):  # small-loop(selected device indices)
        if devices[i] < 0:
            raise Error(PLAN_DEVICES_REFUSAL)
        if logical:
            continue
        for j in range(i):  # small-loop(earlier selected device indices)
            if devices[j] == devices[i]:
                raise Error(PLAN_DEVICES_REFUSAL)


def plan_vendor_is_gpu(vendor: String) -> Bool:
    """True on a CUDA, HIP or Metal install (a plan is bound); False on a
    CPU-only install (no plan: the host binding runs the owner)."""
    return vendor == "cuda" or vendor == "hip" or vendor == "metal"


def plan_vendor_admit(devices: List[Int], vendor: String, operation: String, cooperative: Bool) raises:
    """The old pool's route refusals, in its words, for the installed
    `vendor` ("cpu" for a CPU-only install). `cooperative`: the binding
    shards inside one call (refused on a CPU-only install); otherwise the
    door shards and a CPU-only install runs the host binding per shard."""
    if vendor == "cpu" or vendor == "None":
        if cooperative:
            raise Error("NotImplementedError: no CPU implementation of the cooperative multi-GPU driver "
                        + operation + " yet: its shards are device row tiles, chunks or ranges inside "
                        + "the GPU binding, which no host binding restates")
        return
    if vendor == "metal" and (len(devices) != 1 or devices[0] != 0):
        raise Error(PLAN_VENDOR_REFUSAL)
    if not plan_vendor_is_gpu(vendor):
        raise Error(PLAN_VENDOR_REFUSAL)


def plan_mode_admitted(word: String, family: Int) -> Bool:
    """The drivers' `numeric_mode not in (None, 'identical')` rule for a mode
    word: `identical` passes; `fast` passes where the identical set answers
    it for the driver's door `family` (core/mode_answer.mojo; lane
    fx-fast-mask): the driver then runs the identical kernels, as the
    estimator's own fit does."""
    if word == "identical":
        return True
    return word == "fast" and fast_is_answered_by_identical(family)
