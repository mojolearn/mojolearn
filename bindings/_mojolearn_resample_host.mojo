# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for the `_mojolearn_resample` family: `bootstrap`,
`permutation_test`, `monte_carlo_integrate` (the bootstrap,
permutation-test and monte-carlo lanes of tools/identity_break.py;
lane/cpu-training-misc, 2026-09-15).

HOST ONLY. No DeviceContext, no kernel launch, no GPU. The arithmetic is
`resample/host/resample_host.mojo`, `resample/estimator.mojo`'s three entry
points with every device kernel restated on the host and every host stage
called as the device path calls it; that file's header names each original.

THE EXPORTED NAMES AND CONTRACTS ARE THE GPU BINDING'S
(`bindings/_mojolearn_resample.mojo`): the typed operations table
`resample_native_api_v1` ("mojolearn.resample.v1", resample/host/resample_native.mojo:
the same seven operations with the same words in the same orders), plus
`resample_numeric_mode` (1) and `resample_vendor` ("cpu"), so
`python/mojolearn/resample.py` runs unchanged on a CPU-only install through
`_backend._HOST_MODULES` (`"_mojolearn_resample": "_mojolearn_resample_host"`).
The CPU route admits the `resample` operation from a ONE-device cooperative
pool only, where the driver's range split is one shard and the host entry
points here ARE the plain path; two or more devices refuse by name before
reaching this module.
"""

from core.experiment_guards_series import SERIES_GUARDS_OK
from std.os import abort
from core.hostptr import i32_ptr, read_f32
from core.resample_impl import resample_store_indices
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from providers.python.neural_descriptor_pyapi_door import neural_descriptor_capsule
from resample.host.resample_native import RESAMPLE_NATIVE_TABLE

from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN, column_name
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from resample.checks.statistics import MC_DIMS
from resample.host.resample_host import (
    RESAMPLE_HOST_SABOTAGE,
    host_resample_indices,
)
from resample.mojolearn_resample_host_run import (
    _f32_ptr,
    _f64_ptr,
    _bootstrap_run,
    _bootstrap_unpaired_run,
    _permutation_run,
    _permutation_samples_run,
    _mc_run,
)


def resample_host_numeric_mode_binding() raises -> PythonObject:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL, (
        "resample host: this binding restates the IDENTICAL arms only;"
        " bindings/build_host_family.sh passes -D MOJOLEARN_NUMERIC_IDENTICAL=1"
    )
    return PythonObject(GLOBAL_NUMERIC_MODE)


def resample_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def resample_host_column_binding() raises -> PythonObject:
    """`column_name(TARGET_COLUMN)`, the comptime assert's witness: "cpu".
    THE COLUMN IS THE CPU COLUMN, OR THIS DOES NOT BUILD; the assert lives
    in a function body PyInit registers, so it is compiled in every build.
    `bindings/build_host_family.sh` passes -D MOJOLEARN_COLUMN_CPU."""
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "resample host: this binding compiles the CPU column only; pass"
        " -D MOJOLEARN_COLUMN_CPU (bindings/build_resample_host.sh does)"
    )
    return PythonObject(column_name(TARGET_COLUMN))


def resample_host_sabotage_binding() raises -> PythonObject:
    """Whether this binary shifts every per-replicate and per-chunk tree's
    chunk boundaries by one value on purpose (-D MOJOLEARN_HOST_SABOTAGE=1,
    the gate's negative control)."""
    return PythonObject(RESAMPLE_HOST_SABOTAGE)


def resample_numeric_mode_binding() raises -> PythonObject:
    """THE BUILD'S TIER as the `NUMERIC_*` code: always 1 here."""
    return PythonObject(GLOBAL_NUMERIC_MODE)


# TOMBSTONE: the *_parallel_available multi-GPU capability probes (dead native entries: constant 1, no driver or package code looks them up) deleted 2026-10-26 by b3cad1ef6; code recoverable at a88745022.
# Restore: git apply experiments/removed/dead_entries_parallel_probes.patch


def resample_vendor_binding() raises -> PythonObject:
    """"cpu": this binary computes on the host."""
    return PythonObject(String("cpu"))


# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_resample_host.mojo
# (the per-binding build) and in the merged roots tools/merged_bindings.py
# writes from bindings/merged_groups.json (one shared file per directory set).
# lane dw-decomp (2026-11-21): the typed operations table of this image
# (resample/host/resample_native.mojo: bootstrap, bootstrap_unpaired,
# permutation_test, permutation_samples, resample_indices, monte_carlo and the
# numeric mode). The six Python entries that took a list of addresses and a
# list of boxed parameters are deleted: the public door was their one caller
# and it now calls the table.
def resample_native_api_v1_binding() raises -> PythonObject:
    var table = RESAMPLE_NATIVE_TABLE.get_or_create_ptr()
    return neural_descriptor_capsule(table[].name, table[].entries)


def build__mojolearn_resample_host() -> PythonObject:
    comptime assert SERIES_GUARDS_OK, "invalid integrated experiment configuration (series guards)"
    try:
        var m = PythonModuleBuilder("_mojolearn_resample_host")
        m.def_function[resample_host_numeric_mode_binding]("resample_host_numeric_mode")
        m.def_function[resample_host_vendor_binding]("resample_host_vendor")
        m.def_function[resample_host_column_binding]("resample_host_column")
        m.def_function[resample_host_sabotage_binding]("resample_host_sabotage")
        m.def_function[resample_vendor_binding]("resample_vendor")
        # TOMBSTONE: the *_parallel_available multi-GPU capability probes (dead native entries: constant 1, no driver or package code looks them up) deleted 2026-10-26 by b3cad1ef6; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_parallel_probes.patch
        m.def_function[resample_numeric_mode_binding]("resample_numeric_mode")
        m.def_function[resample_native_api_v1_binding]("resample_native_api_v1")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_resample_host: ", e))
