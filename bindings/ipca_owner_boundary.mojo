# SPDX-License-Identifier: Apache-2.0
"""Boundary pins around typed IncrementalPCA resident-buffer owners."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from core.ipca_native_state import IPCAResidentState
from bindings.py_balanced import py_call, py_setattr
from bindings.native_loaded_module import NativeModules
from bindings.native_prepared_owner import prepared_get, prepared_put, prepared_drop
from bindings.native_model_records import model_generation, records_open
from bindings.p3nc_fitted_freeze import fitted_is_freezable_any, fitted_freeze_any
from bindings.ipca_descriptor_boundary import ipca_provider, ipca_driver, ipca_run_free


def _slot(index: Int) -> String:
    if index == 0:
        return "ipca_components_native_v1"
    if index == 1:
        return "ipca_singular_native_v1"
    if index == 2:
        return "ipca_mean_native_v1"
    if index == 3:
        return "ipca_variance_native_v1"
    return "ipca_explained_native_v1"


@fieldwise_init
struct IPCAResidentOwner(Movable, Writable):
    var state: IPCAResidentState
    var provider_pin: PythonObject
    var source_pin: PythonObject

    def write_to(self, mut writer: Some[Writer]):
        writer.write("IPCAResidentOwner(open=", self.state.handle >= 0, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    def __del__(deinit self):
        var handle = self.state.take_handle()
        if handle >= 0:
            try:
                with GILReleased(Python()):
                    self.state.driver.free(handle)
            except:
                pass
        _ = self.provider_pin
        _ = self.source_pin


def ipca_owner_drop(model: PythonObject) raises:
    var mods = NativeModules()
    for k in range(5):  # driver-loop: five independently pinned state owners
        prepared_drop(mods, _slot(k), model)


def _generation(model: PythonObject) raises -> Int:
    _ = records_open(model)
    return model_generation(model)


def _frozen(source: PythonObject) raises -> Bool:
    if not fitted_is_freezable_any(source):
        return False
    return Bool(py=source._readonly) and Bool(py=source._mv.readonly)


def _freeze(source: PythonObject) raises -> Bool:
    # Identical to the former decomp resident rule: freeze only an exact
    # owned Array, including its memoryview, never a view or borrowed ndarray.
    if not fitted_is_freezable_any(source) or source._base is not None:
        return False
    _ = fitted_freeze_any(source)
    var view = source._mv
    if not Bool(py=view.readonly):
        py_setattr(source, "_mv", py_call(view.toreadonly))
    if not Bool(py=source._readonly):
        py_setattr(source, "_readonly", PythonObject(True))
    return True


def ipca_owner_cached(model: PythonObject, connection: Int, source: PythonObject,
                      index: Int, check_generation: Bool = True) raises -> PythonObject:
    if model is None or source is None:
        return PythonObject(None)
    var mods = NativeModules()
    var held = prepared_get(mods, _slot(index), model)
    if held is None:
        return held
    var owner = held.downcast_value_ptr[IPCAResidentOwner]()
    if not owner[].state.matches(connection, Int(source._obj_ptr)) or not _frozen(source):
        prepared_drop(mods, _slot(index), model)
        return PythonObject(None)
    if check_generation and not owner[].state.accept_generation(_generation(model)):
        prepared_drop(mods, _slot(index), model)
        return PythonObject(None)
    return held


def _hold(model: PythonObject, connection: Int, source: PythonObject, index: Int,
          rows: Int, cols: Int, handle: Int, pending: Bool) raises -> PythonObject:
    # Takes the handle on entry, including every failure path. Python stores
    # only an opaque registered owner; its state and release driver are Mojo.
    var unowned = handle
    try:
        var generation = -1 if pending else _generation(model)
        var owner = IPCAResidentOwner(IPCAResidentState(connection, generation, Int(source._obj_ptr),
            rows, cols, handle, ipca_driver(connection)), ipca_provider(connection), source)
        unowned = -1
        var object = PythonObject(alloc=owner^)
        var mods = NativeModules()
        prepared_put(mods, _slot(index), model, object)
        # A model without a weak record still pins this object through RUN.
        return object
    except error:
        if unowned >= 0:
            ipca_run_free(connection, unowned)
        raise error^


def ipca_owner_adopt(model: PythonObject, connection: Int, source: PythonObject, index: Int,
                     rows: Int, cols: Int, handle: Int) raises:
    if handle < 0:
        return
    var frozen = False
    try:
        frozen = model is not None and _freeze(source)
    except error:
        ipca_run_free(connection, handle)
        raise error^
    if not frozen:
        ipca_run_free(connection, handle)
        return
    _ = _hold(model, connection, source, index, rows, cols, handle, True)


def ipca_owner_operand(model: PythonObject, connection: Int, source: PythonObject, index: Int,
                       rows: Int, cols: Int, address: Int) raises -> PythonObject:
    var count = rows * cols
    if model is None or count < 1:
        return PythonObject(None)
    var held = ipca_owner_cached(model, connection, source, index)
    if held is not None:
        var state = held.downcast_value_ptr[IPCAResidentOwner]()
        if state[].state.rows == rows and state[].state.cols == cols:
            return held
    if not _freeze(source):
        return PythonObject(None)
    var pin = ipca_provider(connection)
    var driver = ipca_driver(connection)
    var handle = -1
    try:
        with GILReleased(Python()):
            handle = driver.prepare(address, count)
        return _hold(model, connection, source, index, rows, cols, handle, False)
    finally:
        _ = pin
        _ = source
        _ = held


def ipca_owner_handle(held: PythonObject) raises -> Int:
    return held.downcast_value_ptr[IPCAResidentOwner]()[].state.handle
