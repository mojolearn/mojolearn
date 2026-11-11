# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for `_mojolearn_x_cluster` (lane/algos-cluster). HOST ONLY: the
GPU binding's export names and contract, over `x_cluster.host.host_ops.HostOps`
(the same `x_cluster/bodies.mojo` bodies in plain loops). The sabotage arm is
`X_CLUSTER_HOST_SABOTAGE` (`-D MOJOLEARN_HOST_SABOTAGE=1`)."""
from core.experiment_guards_cluster import CLUSTER_GUARDS_OK
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from hierarchy.cluster_descriptor_host import HIERARCHY_CLUSTER_HOST_API
from x_cluster.variants_host import KMEANS_VARIANTS_HOST_API
from x_cluster.centroid_fit_host import CENTROID_FIT_HOST_API
from x_cluster.optics_fit_host import OPTICS_FIT_HOST_API
from providers.python.neural_descriptor_pyapi import neural_descriptor_capsule

from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN, column_name
from checks.numerics import GLOBAL_NUMERIC_MODE
from x_cluster.host.execution import cluster_host_execute
from x_cluster.host.host_ops import X_CLUSTER_HOST_SABOTAGE
from x_cluster.out import ClusterOut
from providers.python.x_cluster_out_pyapi import cluster_out_to_py, py_floats, py_ints


from core.shared_slot import SharedSlot
from core.spectral_assignment_api import SpectralAssignmentAPI
from core.spectral_assignment_host import SpectralAssignmentHost
from checks.numerics import NUMERIC_IDENTICAL as _SPECTRAL_IDENTICAL

comptime _SPECTRAL_ASSIGN_SLOT = "MojolearnSpectralAssignHostIdenticalV1" if GLOBAL_NUMERIC_MODE == _SPECTRAL_IDENTICAL else "MojolearnSpectralAssignHostFastV1"
comptime SPECTRAL_ASSIGN_API = SharedSlot[SpectralAssignmentAPI[SpectralAssignmentHost], _SPECTRAL_ASSIGN_SLOT]


def spectral_assign_native_api_v1_binding() raises -> PythonObject:
    var api = SPECTRAL_ASSIGN_API.get_or_create_ptr()
    return neural_descriptor_capsule(api[].name, api[].entries)


def hierarchy_cluster_native_api_v1_binding() raises -> PythonObject:
    var api = HIERARCHY_CLUSTER_HOST_API.get_or_create_ptr()
    return neural_descriptor_capsule(api[].name, api[].entries)


def optics_fit_native_api_v1_binding() raises -> PythonObject:
    var api = OPTICS_FIT_HOST_API.get_or_create_ptr()
    return neural_descriptor_capsule(api[].name, api[].entries)


def cluster_centroid_fit_native_api_v1_binding() raises -> PythonObject:
    var api = CENTROID_FIT_HOST_API.get_or_create_ptr()
    return neural_descriptor_capsule(api[].name, api[].entries)


def kmeans_variants_native_api_v1_binding() raises -> PythonObject:
    var api = KMEANS_VARIANTS_HOST_API.get_or_create_ptr()
    return neural_descriptor_capsule(api[].name, api[].entries)


def call_binding(
    which: PythonObject, x_addr: PythonObject, x_len: PythonObject, a_addr: PythonObject,
    a_len: PythonObject, ip: PythonObject, fp: PythonObject,
) raises -> PythonObject:
    var w = Int(py=which)
    var nx = Int(py=x_len)
    var na = Int(py=a_len)
    var source = Int(py=x_addr) if nx > 0 else 0
    var auxiliary = Int(py=a_addr) if na > 0 else 0
    var ints = py_ints(ip)
    var floats = py_floats(fp)
    var result = ClusterOut()
    with GILReleased(Python()):
        result = cluster_host_execute(w, source, nx, auxiliary, na, ints, floats)
    _ = ints^
    _ = floats^
    return cluster_out_to_py(result)


def x_cluster_host_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def x_cluster_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def x_cluster_host_column_binding() raises -> PythonObject:
    comptime assert TARGET_COLUMN == COLUMN_CPU, "x_cluster host: pass -D MOJOLEARN_COLUMN_CPU"
    return PythonObject(column_name(TARGET_COLUMN))


def x_cluster_host_sabotage_binding() raises -> PythonObject:
    return PythonObject(X_CLUSTER_HOST_SABOTAGE)


def numeric_mode_binding() raises -> PythonObject:
    return PythonObject(Int(GLOBAL_NUMERIC_MODE))


# TOMBSTONE: x_cluster_py2mojo (dead native entry: a constant A/B probe nothing reads) deleted 2026-10-26 by 15b81c461; code recoverable at a88745022.
# Restore: git apply experiments/removed/dead_entries_x_cluster.patch


def vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_x_cluster_host.mojo
# (the per-binding build) and in the merged roots tools/merged_bindings.py
# writes from bindings/merged_groups.json (one shared file per directory set).
def build__mojolearn_x_cluster_host() -> PythonObject:
    comptime assert CLUSTER_GUARDS_OK, "invalid integrated experiment configuration (cluster guards)"
    try:
        var m = PythonModuleBuilder("_mojolearn_x_cluster_host")
        m.def_function[x_cluster_host_numeric_mode_binding]("x_cluster_host_numeric_mode")
        m.def_function[x_cluster_host_vendor_binding]("x_cluster_host_vendor")
        m.def_function[x_cluster_host_column_binding]("x_cluster_host_column")
        m.def_function[x_cluster_host_sabotage_binding]("x_cluster_host_sabotage")
        m.def_function[call_binding]("x_cluster_call")
        m.def_function[hierarchy_cluster_native_api_v1_binding]("hierarchy_cluster_native_api_v1")
        m.def_function[kmeans_variants_native_api_v1_binding]("kmeans_variants_native_api_v1")
        m.def_function[cluster_centroid_fit_native_api_v1_binding]("cluster_centroid_fit_native_api_v1")
        m.def_function[optics_fit_native_api_v1_binding]("optics_fit_native_api_v1")
        m.def_function[spectral_assign_native_api_v1_binding]("spectral_assign_native_api_v1")
        m.def_function[numeric_mode_binding]("x_cluster_numeric_mode")
        m.def_function[vendor_binding]("x_cluster_vendor")
        # TOMBSTONE: x_cluster_py2mojo (dead native entry: a constant A/B probe nothing reads) deleted 2026-10-26 by 15b81c461; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_x_cluster.patch
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_cluster_host: ", e))
