# SPDX-License-Identifier: Apache-2.0
"""The device route of one estimator, as native words: the ONE rule, its words
and the ONE route value every family shares.

`device=None | "gpu" | "cpu"` is a constructor parameter of every estimator.
None (the default) is this installation's device: the GPU images where the
installation has a GPU set, else the CPU images. An explicit `"gpu"` on a
CPU-only installation is refused in words; an unknown value is refused in
words; `"cpu"` on a GPU installation names the CPU identical images loaded
beside the device images at import, and is refused in fast mode and for a
class whose family has no CPU implementation.

A route is resolved ONCE per estimator
(providers/python/device_route_boundary_door.mojo holds it in the estimator's
native owner) and every later public operation calls through the held
entries: no operation tests the device, builds a mode word, reads a table
slot or compares a vendor name.

`DeviceRoute` holds up to four resolved thin C entries of the images the
route resolved to, and one word of family flags. What each slot means is the
family's (its resolver names them): Ridge and LinearRegression hold fit,
predict, the weighted solve and the TSQR table (core/linear_route.mojo);
LogisticRegression holds the quasi-Newton entry and two flags
(core/logistic_route.mojo); QNRegressor holds the QN entry and the
resident-fold entry (bindings/qn_boundary_unpack.mojo).

This module replaces the three route values that existed side by side
(`LinearRoute` with its device words in core/linear_route.mojo,
`LogisticRoute`'s owner form, `FamilyRoute` in
core/device_route_dw_classical.mojo).

Resolution is a metadata operation with no independent numerical work to
split; it runs at an estimator's first operation, never per row.
"""
from core.entry_table import EntryRef, entry_ref_unresolved

comptime DEVICE_GPU = 0
comptime DEVICE_CPU = 1
# The request `device=None`: this installation's device. Never a route's
# device: resolution replaces it by DEVICE_GPU or DEVICE_CPU.
comptime DEVICE_INSTALL = 2


def device_code(word: String) -> Int:
    """0 for `gpu`, 1 for `cpu`, -1 for any other spelling (exact words: the
    constructor stores the value verbatim and nothing normalizes it)."""
    if word == "gpu":
        return DEVICE_GPU
    if word == "cpu":
        return DEVICE_CPU
    return -1


def device_name(code: Int) -> String:
    return String("cpu") if code == DEVICE_CPU else String("gpu")


def device_refusal(shown: String) raises:
    """The one refusal of an unknown device; `shown` is the value's repr."""
    raise Error("ValueError: device must be 'gpu', 'cpu' or None, got " + shown)


def device_gpu_absent() raises:
    """The one refusal of an explicit `device="gpu"` where no GPU set exists."""
    raise Error("ValueError: device='gpu' needs a GPU build of mojolearn, and this installation is CPU only. "
        + "Use device='cpu', or leave device=None to run on this installation's device")


def device_cpu_fast() raises:
    """The one refusal of `device="cpu"` in fast mode beside a device set."""
    raise Error("ValueError: device='cpu' runs in identical mode only; numeric_mode='fast' is for Apple GPUs")


def device_cpu_absent(shown: String, reason: String) raises:
    """The one refusal of an explicit `device="cpu"` on a GPU installation
    for a class that cannot run there on the CPU. `shown` is the class name;
    `reason` is the table's own sentence for the class (no CPU
    implementation of its family, or its CPU images are not connected in
    this installation)."""
    raise Error("ValueError: device='cpu' is not available for " + shown + ": " + reason
        + ". Leave device=None to run on this installation's device")


def device_installed(device: Int, process_vendor: Int) raises -> Int:
    """The device a request runs on in a process whose selected vendor word
    is `process_vendor` (0 CPU only): None is the installation's device, an
    explicit `gpu` without a GPU set is refused, `cpu` always runs."""
    if device == DEVICE_INSTALL:
        return DEVICE_CPU if process_vendor == 0 else DEVICE_GPU
    if device == DEVICE_GPU and process_vendor == 0:
        device_gpu_absent()
    return device


def device_mode_code(word: String) -> Int:
    """1 for `identical`, 0 for `fast`, -1 for anything else (the admitted,
    normalised mode word of a door)."""
    if word == "identical":
        return 1
    if word == "fast":
        return 0
    return -1


def device_mode_word(code: Int) -> String:
    return String("identical") if code == 1 else String("fast")


def device_vendor_word(code: Int) raises -> String:
    """The public vendor name of a route's vendor word (`vendor_used`)."""
    if code == 0:
        return String("cpu")
    if code == 1:
        return String("metal")
    if code == 2:
        return String("cuda")
    if code == 3:
        return String("hip")
    raise Error("RuntimeError: unknown compiled vendor")


def device_vendor_code(word: String) -> Int:
    """The vendor word of a public vendor name, -1 for an unknown one."""
    if word == "cpu":
        return 0
    if word == "metal":
        return 1
    if word == "cuda":
        return 2
    if word == "hip":
        return 3
    return -1


struct DeviceRoute(ImplicitlyCopyable, Movable):
    """Resolved entries of one estimator. `mode` (1 identical / 0 fast) and
    `vendor` (0 CPU, 1 Metal, 2 CUDA, 3 HIP) are the words of the image the
    route resolved to, never the request. A slot the family does not use, or
    could not resolve, holds the refusing entry and its `has_*` is False."""
    var device: Int
    var mode: Int
    var vendor: Int
    var entry: EntryRef
    var second: EntryRef
    var third: EntryRef
    var fourth: EntryRef
    var has_second: Bool
    var has_third: Bool
    var has_fourth: Bool
    # Family words (bit meanings are the family resolver's).
    var flags: Int

    def __init__(out self, device: Int):
        self.device = device
        self.mode = -1
        self.vendor = -1
        self.entry = entry_ref_unresolved()
        self.second = entry_ref_unresolved()
        self.third = entry_ref_unresolved()
        self.fourth = entry_ref_unresolved()
        self.has_second = False
        self.has_third = False
        self.has_fourth = False
        self.flags = 0


@fieldwise_init
struct DeviceRoutePlan(ImplicitlyCopyable, Movable):
    """Which table a request resolves against. `device` is the resolved
    device; `cpu_table` says the CPU identical table of the host images
    loaded beside a device set (a GPU installation running `device="cpu"`);
    `expected_vendor` is the vendor word the resolved image must report."""
    var device: Int
    var cpu_table: Bool
    var expected_vendor: Int


def device_route_plan(device_request: Int, process_vendor: Int, mode: Int) raises -> DeviceRoutePlan:
    """The device rule for a family door. `device_request` is the
    constructor value's code (DEVICE_GPU, DEVICE_CPU or DEVICE_INSTALL for
    None), `process_vendor` the selected vendor word of this process (0: CPU
    only), `mode` the selected tier (1 identical, 0 fast). An explicit `gpu`
    without a GPU set and `cpu` with `fast` are refused in words."""
    var device = device_installed(device_request, process_vendor)
    if device == DEVICE_CPU and process_vendor != 0:
        if mode != 1:
            device_cpu_fast()
        return DeviceRoutePlan(device, True, 0)
    return DeviceRoutePlan(device, False, process_vendor)


def device_route_admit(family: String, plan: DeviceRoutePlan, image_mode: Int, image_vendor: Int,
    mode: Int) raises:
    """Admit the image a route resolved to: it reports the tier the estimator
    selected and the vendor the plan names (`family` leads the refusal, as
    each door's own check did: "QN", "classical", ...)."""
    if image_vendor != plan.expected_vendor:
        if plan.cpu_table:
            raise Error("RuntimeError: device='cpu' requires the CPU identical binding")
        raise Error("RuntimeError: " + family + " native vendor differs from selected execution")
    if image_mode != mode:
        raise Error("RuntimeError: " + family + " native mode differs from selected execution")
