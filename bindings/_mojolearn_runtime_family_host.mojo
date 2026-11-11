# SPDX-License-Identifier: Apache-2.0
# Host owner of the `family_*` operations of `classical_native_call` (bindings/family_native.mojo) and the host
# SVM attribute doors. Split out of bindings/_mojolearn_runtime_classical_host.mojo (2026-11-07 host split,
# docs/build/HOT_GROUPS.md section 6) so it compiles as its own merged group: the same compiled functions,
# registered under the same names. bindings/_mojolearn_tokenizer_host.mojo keeps a native refusal under each
# name, and import setup replaces it with this module's entry (python/mojolearn/host_surface.py
# HOST_IMPORT_ALIASES). Existing shared Python types stay registered by their
# prior owners; KMeans and density boundary owners are registered once in this family owner.
from core.experiment_guards_linear import LINEAR_GUARDS_OK
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder, lookup_py_type_object
from std.reflection import reflect
from bindings.py_type_registry import py_type_absent, py_type_attach
from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN, column_name
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from std.sys.compile import is_defined
from bindings.classical_family_door import classical_family_c
from bindings.host_svr_native import host_svr_attribute_c
from bindings.host_svc_native import host_svc_attribute_c
from bindings.gp_resident_boundary import GPPreparedLease
from bindings.gp_kernel_storage_boundary import GPPackedKernel
from bindings.gp_kernel_cache_boundary import GPPreparedKernel
from bindings.kmeans_owner_boundary import KMeansResidentOwner, kmeans_resident_owner_pins
from bindings.kde_native_boundary import KDEResidentOwner, kde_resident_owner_pins
from bindings.dbscan_native_boundary import DBSCANReferenceOwner, DBSCANHolder, dbscan_reference_owner_pins, dbscan_holder_pins
from bindings.native_owner_registry import register_native_owner
from bindings.agglo_native_boundary import AggloReferenceOwner, AggloHolder, agglo_reference_owner_pins, agglo_holder_pins
from bindings.kde_admission_boundary import KDEAdmissionOwner

comptime RUNTIME_FAMILY_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()


def runtime_family_host_numeric_mode_binding() raises -> PythonObject:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL, (
        "runtime family host requires MOJOLEARN_NUMERIC_IDENTICAL=1"
    )
    return PythonObject(GLOBAL_NUMERIC_MODE)


def runtime_family_host_vendor_binding() raises -> PythonObject:
    return PythonObject("cpu")


def runtime_family_host_column_binding() raises -> PythonObject:
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "runtime family host requires MOJOLEARN_COLUMN_CPU"
    )
    return PythonObject(column_name(TARGET_COLUMN))


def runtime_family_host_sabotage_binding() raises -> PythonObject:
    return PythonObject(RUNTIME_FAMILY_SABOTAGE)


def register_kmeans_resident_owner(mut module: PythonModuleBuilder) raises:
    var existing = PythonObject(None)
    try:
        existing = lookup_py_type_object[KMeansResidentOwner]()
    except error:
        if not py_type_absent(error, String(reflect[KMeansResidentOwner].name())):
            raise error
        _ = module.add_type[KMeansResidentOwner]("_KMeansResidentOwner")
        register_native_owner(lookup_py_type_object[KMeansResidentOwner](), module.module, kmeans_resident_owner_pins)
        return
    # The same initializer can be reached through both package namespaces.
    # Attach the canonical type; never create a second native Python type.
    py_type_attach(module.module, "_KMeansResidentOwner", existing)
    register_native_owner(existing, module.module, kmeans_resident_owner_pins)



def register_KDEResidentOwner(mut module: PythonModuleBuilder) raises:
    var existing = PythonObject(None)
    try:
        existing = lookup_py_type_object[KDEResidentOwner]()
    except error:
        if not py_type_absent(error, String(reflect[KDEResidentOwner].name())):
            raise error
        _ = module.add_type[KDEResidentOwner]("_KDEResidentOwner")
        register_native_owner(lookup_py_type_object[KDEResidentOwner](), module.module, kde_resident_owner_pins)
        return
    py_type_attach(module.module, "_KDEResidentOwner", existing)
    register_native_owner(existing, module.module, kde_resident_owner_pins)


def register_KDEAdmissionOwner(mut module: PythonModuleBuilder) raises:
    var existing = PythonObject(None)
    try:
        existing = lookup_py_type_object[KDEAdmissionOwner]()
    except error:
        if not py_type_absent(error, String(reflect[KDEAdmissionOwner].name())):
            raise error
        _ = module.add_type[KDEAdmissionOwner]("_KDEAdmissionOwner")
        return
    py_type_attach(module.module, "_KDEAdmissionOwner", existing)


def register_DBSCANReferenceOwner(mut module: PythonModuleBuilder) raises:
    var existing = PythonObject(None)
    try:
        existing = lookup_py_type_object[DBSCANReferenceOwner]()
    except error:
        if not py_type_absent(error, String(reflect[DBSCANReferenceOwner].name())):
            raise error
        _ = module.add_type[DBSCANReferenceOwner]("_DBSCANReferenceOwner")
        register_native_owner(lookup_py_type_object[DBSCANReferenceOwner](), module.module, dbscan_reference_owner_pins)
        return
    py_type_attach(module.module, "_DBSCANReferenceOwner", existing)
    register_native_owner(existing, module.module, dbscan_reference_owner_pins)


def register_DBSCANHolder(mut module: PythonModuleBuilder) raises:
    var existing = PythonObject(None)
    try:
        existing = lookup_py_type_object[DBSCANHolder]()
    except error:
        if not py_type_absent(error, String(reflect[DBSCANHolder].name())):
            raise error
        _ = module.add_type[DBSCANHolder]("_DBSCANHolder")
        register_native_owner(lookup_py_type_object[DBSCANHolder](), module.module, dbscan_holder_pins)
        return
    py_type_attach(module.module, "_DBSCANHolder", existing)
    register_native_owner(existing, module.module, dbscan_holder_pins)

def register_AggloReferenceOwner(mut module: PythonModuleBuilder) raises:
    var existing = PythonObject(None)
    try:
        existing = lookup_py_type_object[AggloReferenceOwner]()
    except error:
        if not py_type_absent(error, String(reflect[AggloReferenceOwner].name())):
            raise error
        _ = module.add_type[AggloReferenceOwner]("_AggloReferenceOwner")
        register_native_owner(lookup_py_type_object[AggloReferenceOwner](), module.module, agglo_reference_owner_pins)
        return
    py_type_attach(module.module, "_AggloReferenceOwner", existing)
    register_native_owner(existing, module.module, agglo_reference_owner_pins)


def register_AggloHolder(mut module: PythonModuleBuilder) raises:
    var existing = PythonObject(None)
    try:
        existing = lookup_py_type_object[AggloHolder]()
    except error:
        if not py_type_absent(error, String(reflect[AggloHolder].name())):
            raise error
        _ = module.add_type[AggloHolder]("_AggloHolder")
        register_native_owner(lookup_py_type_object[AggloHolder](), module.module, agglo_holder_pins)
        return
    py_type_attach(module.module, "_AggloHolder", existing)
    register_native_owner(existing, module.module, agglo_holder_pins)


def register_GPPreparedLease(mut module: PythonModuleBuilder) raises:
    var existing = PythonObject(None)
    try:
        existing = lookup_py_type_object[GPPreparedLease]()
    except error:
        if not py_type_absent(error, String(reflect[GPPreparedLease].name())):
            raise error
        _ = module.add_type[GPPreparedLease]("_GPPreparedLease")
        return
    py_type_attach(module.module, "_GPPreparedLease", existing)


def register_GPPackedKernel(mut module: PythonModuleBuilder) raises:
    var existing = PythonObject(None)
    try:
        existing = lookup_py_type_object[GPPackedKernel]()
    except error:
        if not py_type_absent(error, String(reflect[GPPackedKernel].name())):
            raise error
        _ = module.add_type[GPPackedKernel]("_GPPackedKernel")
        return
    py_type_attach(module.module, "_GPPackedKernel", existing)


def register_GPPreparedKernel(mut module: PythonModuleBuilder) raises:
    var existing = PythonObject(None)
    try:
        existing = lookup_py_type_object[GPPreparedKernel]()
    except error:
        if not py_type_absent(error, String(reflect[GPPreparedKernel].name())):
            raise error
        _ = module.add_type[GPPreparedKernel]("_GPPreparedKernel")
        return
    py_type_attach(module.module, "_GPPreparedKernel", existing)


def build__mojolearn_runtime_family_host() -> PythonObject:
    comptime assert LINEAR_GUARDS_OK, "invalid integrated experiment configuration (linear guards)"
    try:
        var module = PythonModuleBuilder("_mojolearn_runtime_family_host")
        register_kmeans_resident_owner(module)
        register_KDEResidentOwner(module)
        register_KDEAdmissionOwner(module)
        register_GPPreparedLease(module)
        register_GPPackedKernel(module)
        register_GPPreparedKernel(module)
        register_DBSCANReferenceOwner(module)
        register_DBSCANHolder(module)
        register_AggloReferenceOwner(module)
        register_AggloHolder(module)
        module.def_py_c_function(classical_family_c, "family_native_call")
        module.def_py_c_function(host_svr_attribute_c, "host_svr_attribute_call")
        module.def_py_c_function(host_svc_attribute_c, "host_svc_attribute_call")
        module.def_function[runtime_family_host_numeric_mode_binding]("runtime_family_host_numeric_mode")
        module.def_function[runtime_family_host_vendor_binding]("runtime_family_host_vendor")
        module.def_function[runtime_family_host_column_binding]("runtime_family_host_column")
        module.def_function[runtime_family_host_sabotage_binding]("runtime_family_host_sabotage")
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_runtime_family_host: ", error))
