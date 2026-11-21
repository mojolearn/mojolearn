# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE CLUSTER LANE'S GPU BINDING (algorithm expansion, lane/algos-cluster).

One call, `x_cluster_call(which, x_addr, x_len, a_addr, a_len, ip, fp)`, runs
`x_cluster/entries.mojo::run_entry` on `x_cluster.device_ops.DeviceOps`. The
CPU host binding `_mojolearn_x_cluster_host` exports the same names with the
same contract over `HostOps`."""
from core.experiment_guards_cluster import CLUSTER_GUARDS_OK
from bindings.device_plan_binding_pack import device_plan_enter_binding, device_plan_exit_binding, device_plan_log_binding
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from hierarchy.cluster_descriptor_device import HIERARCHY_CLUSTER_DEVICE_API
from x_cluster.variants_device import KMEANS_VARIANTS_DEVICE_API
from x_cluster.centroid_fit_device import CENTROID_FIT_DEVICE_API
from x_cluster.optics_fit_device import OPTICS_FIT_DEVICE_API
from x_cluster.bgmm_native_device import BGMM_NATIVE_DEVICE_API
from providers.python.neural_descriptor_pyapi_door import neural_descriptor_capsule

from core.hostptr import f32_ptr
from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR
from x_cluster.device_execution import cluster_device_execute
from x_cluster.minibatch_ptr import MBK_DEVSCAN_ANY
from x_cluster.out import ClusterOut
from providers.python.x_cluster_out_pyapi_unpack import py_floats, py_ints
from providers.python.x_cluster_out_pyapi_pack import cluster_out_to_py
from x_cluster.minibatch_fast import MBK_CLS2_POOL, MBK_W2_SUMCMP
from x_cluster.resident import xc_centers_count


from core.shared_slot import SharedSlot
from core.spectral_assignment_api import SpectralAssignmentAPI
from core.spectral_assignment_device import SpectralAssignmentDevice
from checks.numerics import NUMERIC_IDENTICAL as _SPECTRAL_IDENTICAL

comptime _SPECTRAL_ASSIGN_SLOT = "MojolearnSpectralAssignDeviceIdenticalV1" if GLOBAL_NUMERIC_MODE == _SPECTRAL_IDENTICAL else "MojolearnSpectralAssignDeviceFastV1"
comptime SPECTRAL_ASSIGN_API = SharedSlot[SpectralAssignmentAPI[SpectralAssignmentDevice], _SPECTRAL_ASSIGN_SLOT]


def spectral_assign_native_api_v1_binding() raises -> PythonObject:
    var api = SPECTRAL_ASSIGN_API.get_or_create_ptr()
    return neural_descriptor_capsule(api[].name, api[].entries)


def hierarchy_cluster_native_api_v1_binding() raises -> PythonObject:
    var api = HIERARCHY_CLUSTER_DEVICE_API.get_or_create_ptr()
    return neural_descriptor_capsule(api[].name, api[].entries)


def bgmm_native_api_v1_binding() raises -> PythonObject:
    # The typed BGMM / GaussianMixture option-parity transport
    # (core/bgmm_native_api.mojo); bindings/mixture_public_native_pack.mojo reads it.
    var api = BGMM_NATIVE_DEVICE_API.get_or_create_ptr()
    return neural_descriptor_capsule(api[].name, api[].entries)


def optics_fit_native_api_v1_binding() raises -> PythonObject:
    var api = OPTICS_FIT_DEVICE_API.get_or_create_ptr()
    return neural_descriptor_capsule(api[].name, api[].entries)


def cluster_centroid_fit_native_api_v1_binding() raises -> PythonObject:
    var api = CENTROID_FIT_DEVICE_API.get_or_create_ptr()
    return neural_descriptor_capsule(api[].name, api[].entries)


def kmeans_variants_native_api_v1_binding() raises -> PythonObject:
    var api = KMEANS_VARIANTS_DEVICE_API.get_or_create_ptr()
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
        result = cluster_device_execute(w, source, nx, auxiliary, na, ints, floats)
    _ = ints^
    _ = floats^
    return cluster_out_to_py(result)

def centers_count_binding() raises -> PythonObject:
    """The number of live resident center handles (tests only)."""
    return PythonObject(xc_centers_count())

def mbk_devscan_binding() raises -> PythonObject:
    """1 when MiniBatchKMeans' NaN/inf scan runs in this binding
    (FAST + Apple default, x_cluster/minibatch_ptr.mojo), else 0."""
    return PythonObject(1 if MBK_DEVSCAN_ANY else 0)


def mbk_policy_binding() raises -> PythonObject:
    """Compiled promoted policies: bit 0 compact center sums, bit 1 input pool.

    F15 qualification is pending. This getter attests compilation only;
    actual public full-fit fixtures must independently establish route reach.
    """
    return PythonObject(Int(MBK_W2_SUMCMP) + 2 * Int(MBK_CLS2_POOL))


def numeric_mode_binding() raises -> PythonObject:
    return PythonObject(Int(GLOBAL_NUMERIC_MODE))


# TOMBSTONE: x_cluster_py2mojo (dead native entry: a constant A/B probe nothing reads) deleted 2026-10-26 by 15b81c461; code recoverable at a88745022.
# Restore: git apply experiments/removed/dead_entries_x_cluster.patch


def vendor_binding() raises -> PythonObject:
    return PythonObject(String(COMPILED_VENDOR))


# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_x_cluster.mojo
# (the per-binding build) and in the merged roots tools/merged_bindings.py
# writes from bindings/merged_groups.json (one shared file per directory set).
def build__mojolearn_x_cluster() -> PythonObject:
    comptime assert CLUSTER_GUARDS_OK, "invalid integrated experiment configuration (cluster guards)"
    try:
        var m = PythonModuleBuilder("_mojolearn_x_cluster")
        m.def_function[call_binding]("x_cluster_call")
        m.def_function[hierarchy_cluster_native_api_v1_binding]("hierarchy_cluster_native_api_v1")
        m.def_function[kmeans_variants_native_api_v1_binding]("kmeans_variants_native_api_v1")
        m.def_function[cluster_centroid_fit_native_api_v1_binding]("cluster_centroid_fit_native_api_v1")
        m.def_function[optics_fit_native_api_v1_binding]("optics_fit_native_api_v1")
        m.def_function[bgmm_native_api_v1_binding]("bgmm_native_api_v1")
        m.def_function[spectral_assign_native_api_v1_binding]("spectral_assign_native_api_v1")
        m.def_function[device_plan_enter_binding]("device_plan_enter")
        m.def_function[device_plan_exit_binding]("device_plan_exit")
        m.def_function[device_plan_log_binding]("device_plan_log")
        m.def_function[numeric_mode_binding]("x_cluster_numeric_mode")
        m.def_function[mbk_devscan_binding]("x_cluster_mbk_devscan")
        m.def_function[mbk_policy_binding]("x_cluster_mbk_policy")
        m.def_function[vendor_binding]("x_cluster_vendor")
        m.def_function[centers_count_binding]("x_cluster_centers_count")
        # TOMBSTONE: x_cluster_py2mojo (dead native entry: a constant A/B probe nothing reads) deleted 2026-10-26 by 15b81c461; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_x_cluster.patch
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_cluster: ", e))
