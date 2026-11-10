# SPDX-License-Identifier: Apache-2.0
"""Optimizer lifetime boundary: a typed control record and its extension pin."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased, PyObjectPtr
from bindings.cpython_table import cpython_table
from bindings.optimizer_admission import opt_value, opt_schedule
from bindings.py_balanced import py_fields, py_get, py_type
from training.optimizer_owner_control import OptimizerOwnerControl
from training.optimizer_native_api import OptimizerNativeDriver


struct OptimizerOwner(Movable, Writable):
    var control: OptimizerOwnerControl
    # This is solely the compiled image/capsule lifetime pin. RUN never reads
    # it; the actual driver, resident handle and decisions are native fields.
    var provider: PythonObject

    def __init__(out self, identity: Int, kind: Int, pristine: Bool):
        self.control = OptimizerOwnerControl(identity, kind, pristine)
        self.provider = PythonObject(None)

    def close(mut self) raises:
        var request = self.control.close_request()
        if request:
            if request.value().driver.host:
                request.value().run()
            else:
                with GILReleased(Python()):
                    request.value().run()
            # Publish native control changes after reacquiring the GIL. Read
            # queries may inspect these fields while the native close runs.
            self.control.finish_close()
        # Drop the compiled image only after its native close has returned.
        self.provider = PythonObject(None)

    def attach(mut self, handle: Int, driver: OptimizerNativeDriver, provider: PythonObject):
        if self.control.handle < 0 and handle >= 0:
            self.provider = provider
            self.control.attach(handle, driver)

    def __del__(deinit self):
        # A live operation pins this object, so destruction cannot race RUN.
        try:
            var request = self.control.close_request()
            if request:
                if request.value().driver.host:
                    request.value().run()
                else:
                    with GILReleased(Python()):
                        request.value().run()
                self.control.finish_close()
        except:
            pass  # same best-effort resource cleanup during interpreter exit
        _ = self.provider

    def write_to(self, mut writer: Some[Writer]):
        writer.write("OptimizerOwner(", self.control.name(), ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def optimizer_owner_new(owner: PythonObject, kind: Int, pristine: Bool) raises -> PythonObject:
    return PythonObject(alloc=OptimizerOwner(Int(owner._obj_ptr), kind, pristine))


def optimizer_owner_idle(owner: PythonObject) raises:
    var value = py_get(py_fields(owner), "_native_optimizer_owner")
    if value is not None:
        var retained = optimizer_owner_retained(value, owner)
        if retained.downcast_value_ptr[OptimizerOwner]()[].control.busy:
            raise Error("RuntimeError: optimizer owner is already in use by another native operation")


def optimizer_owner_retained(value: PythonObject, owner: PythonObject) raises -> PythonObject:
    var record = value.downcast_value_ptr[OptimizerOwner]()
    if record[].control.identity != Int(owner._obj_ptr):
        raise Error("TypeError: optimizer native state belongs to a different owner")
    return value


def optimizer_owner_value(owner: PythonObject) raises -> PythonObject:
    var value = py_get(py_fields(owner), "_native_optimizer_owner")
    if value is None:
        raise Error("RuntimeError: optimizer native state is not initialized")
    return optimizer_owner_retained(value, owner)


struct OptimizerOwnerLease(Movable):
    var value: PythonObject
    var active: Bool

    def __init__(out self, owner: PythonObject) raises:
        self.value = optimizer_owner_value(owner)
        self.active = False
        self.value.downcast_value_ptr[OptimizerOwner]()[].control.begin()
        self.active = True

    def __init__(out self, owner: PythonObject, value: PythonObject) raises:
        self.value = optimizer_owner_retained(value, owner)
        self.active = False
        self.value.downcast_value_ptr[OptimizerOwner]()[].control.begin()
        self.active = True

    def __del__(deinit self):
        if self.active:
            try:
                self.value.downcast_value_ptr[OptimizerOwner]()[].control.finish()
            except:
                pass
        _ = self.value


def optimizer_owner_close(value: PythonObject, owner: PythonObject) raises:
    var retained = optimizer_owner_retained(value, owner)
    retained.downcast_value_ptr[OptimizerOwner]()[].close()


def optimizer_owner_finalize(owner: PythonObject) raises:
    var value = py_get(py_fields(owner), "_native_optimizer_owner")
    if value is None:
        return
    var retained = optimizer_owner_retained(value, owner)
    var record = retained.downcast_value_ptr[OptimizerOwner]()
    if record[].control.busy:
        return
    record[].control.begin()
    try:
        record[].close()
    except error:
        record[].control.finish()
        raise error^
    record[].control.finish()


def optimizer_owner_admit_replacement(owner: PythonObject, fields: PythonObject, context: PythonObject) raises:
    """Admit old field owners before a transactional replacement may drop them."""
    var table = cpython_table()
    var position = List[Int](length=1, fill=0)
    var pair = List[PyObjectPtr](length=2, fill=PyObjectPtr())
    while table[].PyDict_Next(fields._obj_ptr, Int(position.unsafe_ptr()),
            Int(pair.unsafe_ptr()), Int(pair.unsafe_ptr() + 1)):  # driver-loop: public fields retained for rollback
        var key = PythonObject(from_borrowed=pair[0])
        var value = PythonObject(from_borrowed=pair[1])
        if py_type(key) is not context["scalar_types"][5]:
            raise Error("TypeError: optimizer metadata requires native string keys")
        var name = String(key)
        if name == "lr_schedule":
            opt_schedule(value, context)
        elif name == "_native_optimizer_owner":
            _ = optimizer_owner_retained(value, owner)
        else:
            opt_value(value, context)
    _ = position
    _ = pair
