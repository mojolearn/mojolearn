# SPDX-License-Identifier: Apache-2.0
"""The device route of one Ridge or LinearRegression estimator, as native words.

`device="gpu"` and `device="cpu"` name two published entry tables of the same
family (core/linear_entry_table.mojo): the device table of the selected
vendor and tier, and the CPU identical table. `device=None` (the default) is
this installation's device: the GPU table where the installation has a GPU
set, else the CPU table; it is decided at resolution and the route holds the
device it resolved to. A route is resolved ONCE per
estimator (providers/python/linear_route_boundary_door.mojo) and holds the
resolved thin C entries; every later fit, predict and score calls through
them. No public operation tests the device, reads a table slot, builds a
schema String or compares an ABI name.

Resolution is a metadata operation with no independent numerical work to
split; it runs at an estimator's first operation, never per row.
"""
from core.entry_table import EntryRef, entry_ref_unresolved, entry_ref_word
from core.linear_boundary_types import LINEAR_PREDICT_ABI
from core.linear_entry_table import (
    LinearEntryKey, LINEAR_TABLE_ESTIMATORS, LINEAR_TABLE_X_LINEAR, linear_table_entry,
)
from core.linear_fit_types import (
    LINEAR_FIT_ABI, LINEAR_FIT_OPERATION_COUNT, RIDGE_FIT_ABI, LF_MODE, LF_VENDOR,
)
from core.native_descriptor_call import native_descriptor_call

comptime LINEAR_DEVICE_GPU = 0
comptime LINEAR_DEVICE_CPU = 1
# The request `device=None`: this installation's device. Never a route's
# device: resolution replaces it by LINEAR_DEVICE_GPU or LINEAR_DEVICE_CPU.
comptime LINEAR_DEVICE_INSTALL = 2


def linear_device_code(word: String) -> Int:
    """0 for `gpu`, 1 for `cpu`, -1 for any other spelling (exact words: the
    constructor stores the value verbatim and nothing normalizes it)."""
    if word == "gpu":
        return LINEAR_DEVICE_GPU
    if word == "cpu":
        return LINEAR_DEVICE_CPU
    return -1


def linear_device_name(code: Int) -> String:
    return String("cpu") if code == LINEAR_DEVICE_CPU else String("gpu")


def linear_device_refusal(shown: String) raises:
    """The one refusal of an unknown device; `shown` is the value's repr."""
    raise Error("ValueError: device must be 'gpu', 'cpu' or None, got " + shown)


def linear_device_gpu_absent() raises:
    """The one refusal of an explicit `device="gpu"` where no GPU set exists."""
    raise Error("ValueError: device='gpu' needs a GPU build of mojolearn, and this installation is CPU only. "
        + "Use device='cpu', or leave device=None to run on this installation's device")


def linear_device_installed(device: Int, process_vendor: Int) raises -> Int:
    """The device a request runs on in a process whose selected vendor word
    is `process_vendor` (0 CPU only): None is the installation's device, an
    explicit `gpu` without a GPU set is refused, `cpu` always runs."""
    if device == LINEAR_DEVICE_INSTALL:
        return LINEAR_DEVICE_CPU if process_vendor == 0 else LINEAR_DEVICE_GPU
    if device == LINEAR_DEVICE_GPU and process_vendor == 0:
        linear_device_gpu_absent()
    return device


struct LinearRoute(ImplicitlyCopyable, Movable):
    """Resolved entries of one estimator. `mode` and `vendor` are the words
    the compiled fit entry itself reports (1 identical / 0 fast; 0 CPU,
    1 Metal, 2 CUDA, 3 HIP), not the request."""
    var device: Int
    var mode: Int
    var vendor: Int
    var fit: EntryRef
    var predict: EntryRef
    var weighted: EntryRef
    var decomp: EntryRef
    # The R^2 program entry of a CPU route in a GPU installation
    # (providers/python/metrics_route_boundary_door.mojo). Every other route
    # scores on the process's metrics table and holds none.
    var metrics: EntryRef
    var has_weighted: Bool
    var has_decomp: Bool
    var has_metrics: Bool

    def __init__(out self, device: Int):
        self.device = device
        self.mode = -1
        self.vendor = -1
        self.fit = entry_ref_unresolved()
        self.predict = entry_ref_unresolved()
        self.weighted = entry_ref_unresolved()
        self.decomp = entry_ref_unresolved()
        self.metrics = entry_ref_unresolved()
        self.has_weighted = False
        self.has_decomp = False
        self.has_metrics = False


def linear_route_resolve(device: Int, key: LinearEntryKey, expected_vendor: Int,
    expected_mode: Int, explicit_host: Bool) raises -> LinearRoute:
    """Read the family's fit and predict entries of `key` once and admit
    them: the compiled entry must report the vendor and tier the estimator
    selected. `explicit_host` keeps the explicit Host* classes' own wording."""
    var route = LinearRoute(device)
    var fit = linear_table_entry[LINEAR_TABLE_ESTIMATORS](key, 0, String(LINEAR_FIT_ABI), LINEAR_FIT_OPERATION_COUNT)
    var vendor = native_descriptor_call(fit, LF_VENDOR, List[Int](), List[Int](), List[Float64]())
    var mode = native_descriptor_call(fit, LF_MODE, List[Int](), List[Int](), List[Float64]())
    if explicit_host:
        if vendor != 0 or mode != 1:
            raise Error("RuntimeError: the explicit host model requires its CPU identical binding")
    else:
        if vendor != expected_vendor:
            raise Error("RuntimeError: classical native vendor differs from selected execution")
        if mode != expected_mode:
            raise Error("RuntimeError: classical native mode differs from selected execution")
    var predict = linear_table_entry[LINEAR_TABLE_ESTIMATORS](key, 1, String(LINEAR_PREDICT_ABI), 1)
    route.mode = mode
    route.vendor = vendor
    route.fit = EntryRef(fit, entry_ref_word(key.mode, key.vendor, LINEAR_FIT_OPERATION_COUNT))
    route.predict = EntryRef(predict, entry_ref_word(key.mode, key.vendor, 1))
    return route


def linear_route_weighted(mut route: LinearRoute, key: LinearEntryKey, weighted: LinearEntryKey) raises:
    """Ridge's weighted and one-column solver entry (the x_linear table of
    the same tier and vendor)."""
    if weighted.mode != key.mode or weighted.vendor != key.vendor:
        raise Error("native Ridge weighted execution key differs")
    var entry = linear_table_entry[LINEAR_TABLE_X_LINEAR](weighted, 0, String(RIDGE_FIT_ABI), 1)
    route.weighted = EntryRef(entry, entry_ref_word(weighted.mode, weighted.vendor, 1))
    route.has_weighted = True
