# SPDX-License-Identifier: Apache-2.0
"""The device route of one estimator of a family whose compiled images publish
one operations entry per (tier, vendor): the generalisation of
core/linear_route.mojo that lane dw-classical's families share (QNRegressor,
LinearSVC, LinearSVR, the x_linear estimators).

`device=None | "gpu" | "cpu"` is the pilot's rule and the pilot's words: the
device codes, `linear_device_installed`, the unknown-value refusal and the
no-GPU refusal are core/linear_route.mojo's, imported and not copied. This
module adds only the family-neutral route value: the device a request
resolved to, the tier and vendor words of the image it resolved to, the
image's operations entry and one optional second entry of the same image set
(QN: the resident-fold entry; x_linear: the coordinate-descent solver entry).

A route is resolved ONCE per estimator
(providers/python/device_route_dw_classical_door.mojo) and every later public
operation calls through the held entries: no operation tests the device,
builds a mode word, reads a table slot or compares a vendor name.

Resolution is a metadata operation with no independent numerical work to
split; it runs at an estimator's first operation, never per row.
"""
from core.entry_table import EntryRef, entry_ref_unresolved
from core.linear_route import LINEAR_DEVICE_CPU, LINEAR_DEVICE_GPU, linear_device_installed


struct FamilyRoute(ImplicitlyCopyable, Movable):
    """Resolved entries of one estimator. `mode` (1 identical / 0 fast) and
    `vendor` (0 CPU, 1 Metal, 2 CUDA, 3 HIP) are the words of the image the
    route resolved to, never the request."""
    var device: Int
    var mode: Int
    var vendor: Int
    var entry: EntryRef
    var second: EntryRef
    var has_second: Bool

    def __init__(out self, device: Int):
        self.device = device
        self.mode = -1
        self.vendor = -1
        self.entry = entry_ref_unresolved()
        self.second = entry_ref_unresolved()
        self.has_second = False


def family_mode_code(word: String) -> Int:
    """1 for `identical`, 0 for `fast`, -1 for anything else (the admitted,
    normalised mode word of a door)."""
    if word == "identical":
        return 1
    if word == "fast":
        return 0
    return -1


def family_mode_word(code: Int) -> String:
    return String("identical") if code == 1 else String("fast")


def family_vendor_word(code: Int) raises -> String:
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


@fieldwise_init
struct FamilyRoutePlan(ImplicitlyCopyable, Movable):
    """Which table a request resolves against. `device` is the resolved
    device; `cpu_table` says the CPU identical table of the host images
    loaded beside a device set (a GPU installation running `device="cpu"`);
    `expected_vendor` is the vendor word the resolved image must report."""
    var device: Int
    var cpu_table: Bool
    var expected_vendor: Int


def family_route_plan(device_request: Int, process_vendor: Int, mode: Int) raises -> FamilyRoutePlan:
    """The pilot's device rule for a family door. `device_request` is the
    constructor value's code (LINEAR_DEVICE_GPU, _CPU or _INSTALL for None),
    `process_vendor` the selected vendor word of this process (0: CPU only),
    `mode` the selected tier (1 identical, 0 fast). An explicit `gpu` without
    a GPU set and `cpu` with `fast` are refused in the pilot's words."""
    var device = linear_device_installed(device_request, process_vendor)
    if device == LINEAR_DEVICE_CPU and process_vendor != 0:
        if mode != 1:
            raise Error("ValueError: device='cpu' runs in identical mode only; numeric_mode='fast' is for Apple GPUs")
        return FamilyRoutePlan(device, True, 0)
    return FamilyRoutePlan(device, False, process_vendor)


def family_route_admit(family: String, plan: FamilyRoutePlan, image_mode: Int, image_vendor: Int,
    mode: Int) raises:
    """Admit the image a route resolved to: it reports the tier the estimator
    selected and the vendor the plan names (`family` leads the refusal, as
    each door's own check did: "QN", "x_linear", ...)."""
    if image_vendor != plan.expected_vendor:
        if plan.cpu_table:
            raise Error("RuntimeError: device='cpu' requires the CPU identical binding")
        raise Error("RuntimeError: " + family + " native vendor differs from selected execution")
    if image_mode != mode:
        raise Error("RuntimeError: " + family + " native mode differs from selected execution")
