# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPython boundary for the resampling lane: `bootstrap`,
`permutation_test`, `monte_carlo_integrate` (workstream D, 2026-09-14).

A separate extension module, for `bindings/_mojolearn_gp.mojo`'s reason.
`resample/estimator.mojo`'s three host surfaces are reached and nothing is
re-decided: every refusal (`validate_positions`, the
statistic and method codes, the BCa refusal DEVIATION 1699, the non-finite
cell, the sort-cell ceiling) is raised one layer down by name. This file
refuses a null address, a list of the wrong length and an integrand id
outside the three compiled arms, because `monte_carlo_integrate_host` takes
its integrand as a COMPTIME parameter and the dispatch below is the only
place a runtime id can become one.

THE OPERATIONS ARE ONE TYPED TABLE (lane dw-decomp, 2026-11-21):
`resample_native_api_v1` publishes "mojolearn.resample.v1"
(resample/resample_native.mojo: addresses, integer words and Float64 scalars,
orders written out there), which the public door
(bindings/linalg_ops_resample_door.mojo) calls with the GIL released. The six
Python entries that took two length-checked lists are deleted. Every result
struct crosses as its arrays and scalars; the scalars widen float32 to
float64 exactly. The row gather of `resample` is operation 8 of the same
table (lane gd-rf-resample, 2026-11-25): the Python-list gather entries
(`resample_gather_gpu`, `resample_permutation_gather_gpu`,
`resample_gpu_gather_enabled`, `resample_gather_narrow`,
`resample_gather_narrow_bytes`) are deleted.
"""

from core.experiment_guards_series import SERIES_GUARDS_OK
from bindings.device_plan_binding_pack import device_plan_enter_binding, device_plan_exit_binding, device_plan_log_binding
from std.os import abort
from core.hostptr import i32_ptr, read_f32

from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from providers.python.neural_descriptor_pyapi_door import neural_descriptor_capsule
from resample.resample_native import RESAMPLE_NATIVE_TABLE
from resample.resample_store import RESAMPLE_STORE_API

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from core.accelerator_target import TARGET_HAS_APPLE_GPU
from checks.vendor import COMPILED_VENDOR
from resample.checks.statistics import MC_DIMS
from resample.mojolearn_resample_run import (
    _f32_ptr,
    _f64_ptr,
    _bootstrap_run,
    _bootstrap_unpaired_run,
    _permutation_run,
    _permutation_samples_run,
    _mc_run,
)
from resample.estimator import (
    resample_indices_replace_into,
    resample_indices_permute_into,
    resample_fast_defines,
)


def resample_numeric_mode_binding() raises -> PythonObject:
    """THE BUILD'S TIER as the `NUMERIC_*` code: 0 FAST, 1 IDENTICAL."""
    return PythonObject(GLOBAL_NUMERIC_MODE)


def resample_fast_defines_binding() raises -> PythonObject:
    """The FAST + Apple candidate defines this build was compiled with, as
    resample/estimator.mojo `resample_fast_defines`' bit mask (0 unless the
    build is FAST on Apple with a `-D MOJOLEARN_RESAMPLE_FAST_*` / `-D
    MOJOLEARN_CV_FAST_*` define). Python switches on this, never on an
    environment variable (lane apple-fast-rec-resample, 2026-10-04)."""
    return PythonObject(resample_fast_defines())


def resample_vendor_binding() raises -> PythonObject:
    """'metal', 'cuda', 'hip' or 'none', from `checks/vendor.mojo`."""
    return PythonObject(String(COMPILED_VENDOR))


# TOMBSTONE: the *_parallel_available multi-GPU capability probes (dead native entries: constant 1, no driver or package code looks them up) deleted 2026-10-26 by b3cad1ef6; code recoverable at a88745022.
# Restore: git apply experiments/removed/dead_entries_parallel_probes.patch


# TOMBSTONE: MOJOLEARN_RESAMPLE_FAST_IDX_DIRECT_OFF (old path, superseded by the FAST+Apple default) deleted 2026-11-15 by lane/expdel-02-decomposition-series; code recoverable at 080e0e6f72.
# Restore: git apply experiments/removed/MOJOLEARN_RESAMPLE_FAST_IDX_DIRECT_OFF.patch; record in docs/TOMBSTONES.md.


# TOMBSTONE: resample_gpu_gather_enabled, resample_gather_gpu, resample_permutation_gather_gpu,
# resample_gather_narrow_bytes, resample_gather_narrow (Python-list entries of the FAST + Apple row gather and its
# dead narrow stub) deleted 2026-11-25 by lane gd-rf-resample: the row gather is operation 8 of the typed table on
# every GPU image (resample/resample_native.mojo). Recoverable at the parent of that commit on
# lane/gd-resident-functions.


# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_resample.mojo
# (the per-binding build) and in the merged roots tools/merged_bindings.py
# writes from bindings/merged_groups.json (one shared file per directory set).
# lane dw-decomp (2026-11-21): the typed operations table of this image
# (resample/resample_native.mojo: bootstrap, bootstrap_unpaired,
# permutation_test, permutation_samples, resample_indices, monte_carlo and the
# numeric mode). The six Python entries that took a list of addresses and a
# list of boxed parameters are deleted: the public door was their one caller
# and it now calls the table.
def resample_native_api_v1_binding() raises -> PythonObject:
    var table = RESAMPLE_NATIVE_TABLE.get_or_create_ptr()
    return neural_descriptor_capsule(table[].name, table[].entries)


def resample_store_v1_binding() raises -> PythonObject:
    """The image's resident store as a scope transport entry
    (resample/resample_store.mojo, lane gd-rf-resample): the public doors
    connect it once, inside a `resident()` scope only."""
    var api = RESAMPLE_STORE_API.get_or_create_ptr()
    return neural_descriptor_capsule(api[].name, api[].entries)


def build__mojolearn_resample() -> PythonObject:
    comptime assert SERIES_GUARDS_OK, "invalid integrated experiment configuration (series guards)"
    try:
        var m = PythonModuleBuilder("_mojolearn_resample")
        # TOMBSTONE: the *_parallel_available multi-GPU capability probes (dead native entries: constant 1, no driver or package code looks them up) deleted 2026-10-26 by b3cad1ef6; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_parallel_probes.patch
        m.def_function[device_plan_enter_binding]("device_plan_enter")
        m.def_function[device_plan_exit_binding]("device_plan_exit")
        m.def_function[device_plan_log_binding]("device_plan_log")
        m.def_function[resample_vendor_binding]("resample_vendor")
        m.def_function[resample_numeric_mode_binding]("resample_numeric_mode")
        m.def_function[resample_native_api_v1_binding]("resample_native_api_v1")
        m.def_function[resample_store_v1_binding]("resample_store_v1")
        m.def_function[resample_fast_defines_binding]("resample_fast_defines")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_resample: ", e))
