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
from bindings.py_type_registry import py_type_absent, py_type_attach
from bindings.hdbscan_query_boundary_unpack import HdbscanQueryRecord
from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN, column_name
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from std.sys.compile import is_defined
from bindings.classical_family_door import classical_family_c
from bindings.host_svr_native_door import host_svr_attribute_c
from bindings.host_svc_native_door import host_svc_attribute_c
from bindings.gp_resident_boundary_unpack import GPPreparedLease
from bindings.svr_native_boundary_unpack import SvrPreparedLease
from bindings.gp_kernel_storage_boundary import GPPackedKernel
from bindings.gp_kernel_cache_boundary_door import GPPreparedKernel
from bindings.gp_state_cache_boundary_unpack import GPPreparedState
from bindings.gp_class_table_boundary_unpack import GPPreparedClassTable
from bindings.kmeans_owner_boundary_door import KMeansResidentOwner, kmeans_resident_owner_pins
from bindings.kde_native_boundary_unpack import KDEResidentOwner, kde_resident_owner_pins
from bindings.dbscan_native_boundary_door import DBSCANReferenceOwner, DBSCANHolder, dbscan_reference_owner_pins, dbscan_holder_pins
from bindings.native_owner_registry_unpack import register_native_owner_type
from bindings.agglo_native_boundary_door import AggloReferenceOwner, AggloHolder, agglo_reference_owner_pins, agglo_holder_pins
from bindings.kde_admission_boundary_unpack import KDEAdmissionOwner

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


def register_hdbscan_query_record(mut module: PythonModuleBuilder) raises:
    # Prepared weak-table record only, not an estimator field owner. The same
    # canonical native type is attached when another package alias initializes.
    var existing = PythonObject(None)
    try:
        existing = lookup_py_type_object[HdbscanQueryRecord]()
    except error:
        if not py_type_absent(error):
            raise error
        _ = module.add_type[HdbscanQueryRecord]("_HdbscanQueryRecord")
        return
    py_type_attach(module.module, "_HdbscanQueryRecord", existing)


def register_kmeans_resident_owner(mut module: PythonModuleBuilder) raises:
    var existing = PythonObject(None)
    try:
        existing = lookup_py_type_object[KMeansResidentOwner]()
    except error:
        if not py_type_absent(error):
            raise error
        _ = module.add_type[KMeansResidentOwner]("_KMeansResidentOwner")
        return
    # The same initializer can be reached through both package namespaces.
    # Attach the canonical type; never create a second native Python type.
    py_type_attach(module.module, "_KMeansResidentOwner", existing)



def register_KDEResidentOwner(mut module: PythonModuleBuilder) raises:
    var existing = PythonObject(None)
    try:
        existing = lookup_py_type_object[KDEResidentOwner]()
    except error:
        if not py_type_absent(error):
            raise error
        _ = module.add_type[KDEResidentOwner]("_KDEResidentOwner")
        return
    py_type_attach(module.module, "_KDEResidentOwner", existing)


def register_KDEAdmissionOwner(mut module: PythonModuleBuilder) raises:
    var existing = PythonObject(None)
    try:
        existing = lookup_py_type_object[KDEAdmissionOwner]()
    except error:
        if not py_type_absent(error):
            raise error
        _ = module.add_type[KDEAdmissionOwner]("_KDEAdmissionOwner")
        return
    py_type_attach(module.module, "_KDEAdmissionOwner", existing)


def register_DBSCANReferenceOwner(mut module: PythonModuleBuilder) raises:
    var existing = PythonObject(None)
    try:
        existing = lookup_py_type_object[DBSCANReferenceOwner]()
    except error:
        if not py_type_absent(error):
            raise error
        _ = module.add_type[DBSCANReferenceOwner]("_DBSCANReferenceOwner")
        return
    py_type_attach(module.module, "_DBSCANReferenceOwner", existing)


def register_DBSCANHolder(mut module: PythonModuleBuilder) raises:
    var existing = PythonObject(None)
    try:
        existing = lookup_py_type_object[DBSCANHolder]()
    except error:
        if not py_type_absent(error):
            raise error
        _ = module.add_type[DBSCANHolder]("_DBSCANHolder")
        return
    py_type_attach(module.module, "_DBSCANHolder", existing)

def register_AggloReferenceOwner(mut module: PythonModuleBuilder) raises:
    var existing = PythonObject(None)
    try:
        existing = lookup_py_type_object[AggloReferenceOwner]()
    except error:
        if not py_type_absent(error):
            raise error
        _ = module.add_type[AggloReferenceOwner]("_AggloReferenceOwner")
        return
    py_type_attach(module.module, "_AggloReferenceOwner", existing)


def register_AggloHolder(mut module: PythonModuleBuilder) raises:
    var existing = PythonObject(None)
    try:
        existing = lookup_py_type_object[AggloHolder]()
    except error:
        if not py_type_absent(error):
            raise error
        _ = module.add_type[AggloHolder]("_AggloHolder")
        return
    py_type_attach(module.module, "_AggloHolder", existing)


def register_SvrPreparedLease(mut module: PythonModuleBuilder) raises:
    var existing = PythonObject(None)
    try:
        existing = lookup_py_type_object[SvrPreparedLease]()
    except error:
        if not py_type_absent(error):
            raise error
        _ = module.add_type[SvrPreparedLease]("_SvrPreparedLease")
        return
    py_type_attach(module.module, "_SvrPreparedLease", existing)


def register_GPPreparedLease(mut module: PythonModuleBuilder) raises:
    var existing = PythonObject(None)
    try:
        existing = lookup_py_type_object[GPPreparedLease]()
    except error:
        if not py_type_absent(error):
            raise error
        _ = module.add_type[GPPreparedLease]("_GPPreparedLease")
        return
    py_type_attach(module.module, "_GPPreparedLease", existing)


def register_GPPackedKernel(mut module: PythonModuleBuilder) raises:
    var existing = PythonObject(None)
    try:
        existing = lookup_py_type_object[GPPackedKernel]()
    except error:
        if not py_type_absent(error):
            raise error
        _ = module.add_type[GPPackedKernel]("_GPPackedKernel")
        return
    py_type_attach(module.module, "_GPPackedKernel", existing)


def register_GPPreparedKernel(mut module: PythonModuleBuilder) raises:
    var existing = PythonObject(None)
    try:
        existing = lookup_py_type_object[GPPreparedKernel]()
    except error:
        if not py_type_absent(error):
            raise error
        _ = module.add_type[GPPreparedKernel]("_GPPreparedKernel")
        return
    py_type_attach(module.module, "_GPPreparedKernel", existing)


def register_GPPreparedState(mut module: PythonModuleBuilder) raises:
    var existing = PythonObject(None)
    try:
        existing = lookup_py_type_object[GPPreparedState]()
    except error:
        if not py_type_absent(error):
            raise error
        _ = module.add_type[GPPreparedState]("_GPPreparedState")
        return
    py_type_attach(module.module, "_GPPreparedState", existing)


def register_GPPreparedClassTable(mut module: PythonModuleBuilder) raises:
    var existing = PythonObject(None)
    try:
        existing = lookup_py_type_object[GPPreparedClassTable]()
    except error:
        if not py_type_absent(error):
            raise error
        _ = module.add_type[GPPreparedClassTable]("_GPPreparedClassTable")
        return
    py_type_attach(module.module, "_GPPreparedClassTable", existing)


def build__mojolearn_runtime_family_host() -> PythonObject:
    comptime assert LINEAR_GUARDS_OK, "invalid integrated experiment configuration (linear guards)"
    try:
        var module = PythonModuleBuilder("_mojolearn_runtime_family_host")
        register_kmeans_resident_owner(module)
        register_hdbscan_query_record(module)
        register_KDEResidentOwner(module)
        register_KDEAdmissionOwner(module)
        register_GPPreparedLease(module)
        register_SvrPreparedLease(module)
        register_GPPackedKernel(module)
        register_GPPreparedKernel(module)
        register_GPPreparedState(module)
        register_GPPreparedClassTable(module)
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
        # Owner identities need the finalized type objects (Mojo 1.2 creates
        # them in finalize(); see bindings/native_owner_registry.mojo).
        var published = module.finalize()
        register_native_owner_type[KMeansResidentOwner](published, kmeans_resident_owner_pins)
        register_native_owner_type[KDEResidentOwner](published, kde_resident_owner_pins)
        register_native_owner_type[DBSCANReferenceOwner](published, dbscan_reference_owner_pins)
        register_native_owner_type[DBSCANHolder](published, dbscan_holder_pins)
        register_native_owner_type[AggloReferenceOwner](published, agglo_reference_owner_pins)
        register_native_owner_type[AggloHolder](published, agglo_holder_pins)
        return published
    except error:
        abort(String("failed to create _mojolearn_runtime_family_host: ", error))
