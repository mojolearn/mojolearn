# SPDX-License-Identifier: Apache-2.0
# One shared admission/lifetime boundary for native recursive operation locks.
# Native engines receive a handle from a retained OperationLockLease, never a
# Python lock or callback. Legacy exact RLock instances remain explicit until
# their caller is migrated; arbitrary lock protocols are never invoked.
from std.ffi import _Global
from std.python import Python, PythonObject
from std.python._cpython import GILReleased, PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.py_balanced import py_call, py_type
from bindings.native_error_class import error_without_class
from core.operation_lock import NativeOperationLock, operation_lock_acquire, operation_lock_release, operation_lock_owned


struct ProviderOperationLock(Movable, Writable):
    var native: NativeOperationLock

    def __init__(out self):
        self.native = NativeOperationLock()

    def write_to(self, mut writer: Some[Writer]):
        writer.write("NativeOperationLock")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


struct OperationLockBoundary(Defaultable, Movable):
    var cls: PythonObject
    var getter: PythonObject
    var setter: PythonObject
    var legacy: PythonObject

    def __init__(out self):
        self.cls = PythonObject(None)
        self.getter = PythonObject(None)
        self.setter = PythonObject(None)
        self.legacy = PythonObject(None)


comptime OPERATION_LOCK_BOUNDARY = _Global[StorageType=OperationLockBoundary,
    name="MojoOperationLockBoundaryV1", init_fn=OperationLockBoundary.__init__]


def operation_lock_connect(context: PythonObject) raises -> PythonObject:
    var state = OPERATION_LOCK_BOUNDARY.get_or_create_ptr()
    state[].cls = context["class"]
    state[].getter = context["get_owner"]
    state[].setter = context["set_owner"]
    state[].legacy = context["legacy_class"]
    return PythonObject(None)


def operation_lock_owner(value: PythonObject) raises -> PythonObject:
    var state = OPERATION_LOCK_BOUNDARY.get_or_create_ptr()
    if py_type(value) is not state[].cls:
        raise Error("TypeError: native operation lock requires its exact admitted class")
    # The pinned native slot descriptor reads its storage directly. A replaced
    # Python __getattribute__ or public slot name cannot inject a callback.
    return py_call(state[].getter, value, state[].cls)


def operation_lock_create() raises -> PythonObject:
    var state = OPERATION_LOCK_BOUNDARY.get_or_create_ptr()
    if state[].cls is None:
        raise Error("native operation lock: import-time connection is missing")
    var value = py_call(Python.import_module("builtins").object.__new__, state[].cls)
    var owner = PythonObject(alloc=ProviderOperationLock())
    _ = py_call(state[].setter, value, owner)
    return value


struct OperationLockLease(Movable):
    var value: PythonObject
    var owner: PythonObject
    var handle: Int

    def __init__(out self, value: PythonObject) raises:
        self.value = value
        self.owner = PythonObject(None)
        self.handle = 0
        var state = OPERATION_LOCK_BOUNDARY.get_or_create_ptr()
        if value is not None and py_type(value) is state[].cls:
            self.owner = operation_lock_owner(value)
            self.handle = self.owner.downcast_value_ptr[ProviderOperationLock]()[].native.handle()
        elif value is not None and py_type(value) is not state[].legacy:
            raise Error("TypeError: unsupported operation lock protocol")

    def acquire(self) raises:
        if self.handle != 0:
            with GILReleased(Python()):
                _ = operation_lock_acquire(self.handle)
        else:
            # Retained off/default or CPU/old-state compatibility route. The
            # exact built-in lock is admitted above; no Python override runs.
            _ = py_call(self.value.acquire)

    def release(self) raises:
        if self.handle != 0:
            operation_lock_release(self.handle)
        else:
            _ = py_call(self.value.release)


def operation_lock_c[operation: Int](py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        # Import-time MethodType supplies connection context at index0; the
        # native instance method descriptor supplies the lock at index1.
        if len(args) < 2:
            raise Error("TypeError: native operation lock method requires its owner")
        var value = args[1]
        var boundary = OPERATION_LOCK_BOUNDARY.get_or_create_ptr()
        if py_type(value) is not boundary[].cls:
            raise Error("TypeError: native operation lock requires its exact admitted class")
        var result = PythonObject(None)
        comptime if operation == 0:
            if len(args) != 2:
                raise Error("TypeError: native operation lock constructor takes no arguments")
            _ = py_call(boundary[].setter, value, PythonObject(alloc=ProviderOperationLock()))
        else:
            var owner = operation_lock_owner(value)
            var handle = owner.downcast_value_ptr[ProviderOperationLock]()[].native.handle()
            comptime if operation == 1 or operation == 3:
                var blocking = True
                var timeout = Float64(-1)
                if len(args) > (2 if operation == 3 else 4):
                    raise Error("TypeError: too many native operation lock arguments")
                var bi = Python.import_module("builtins")
                if len(args) > 2:
                    if py_type(args[2]) is not bi.bool and py_type(args[2]) is not bi.int:
                        raise Error("TypeError: native operation lock blocking must be bool or int")
                    blocking = Bool(py=args[2])
                if len(args) > 3:
                    if py_type(args[3]) is not bi.int and py_type(args[3]) is not bi.float:
                        raise Error("TypeError: native operation lock timeout must be numeric")
                    timeout = Float64(py=args[3])
                var acquired = False
                with GILReleased(Python()):
                    acquired = operation_lock_acquire(handle, blocking, timeout)
                result = PythonObject(acquired)
            elif operation == 2 or operation == 4:
                if len(args) != (5 if operation == 4 else 2):
                    raise Error("TypeError: invalid native operation lock release arguments")
                operation_lock_release(handle)
            elif operation == 5:
                if len(args) != 2:
                    raise Error("TypeError: native operation lock ownership takes no arguments")
                result = PythonObject(operation_lock_owned(handle))
            _ = owner
        return result.steal_data()
    except problem:
        if String(problem).startswith("TypeError:"):
            return raise_python_exception(error_without_class(problem, "TypeError"), ExceptionType.TypeError)
        if String(problem).startswith("ValueError:"):
            return raise_python_exception(error_without_class(problem, "ValueError"), ExceptionType.ValueError)
        if String(problem).startswith("OverflowError:"):
            return raise_python_exception(error_without_class(problem, "OverflowError"), ExceptionType("PyExc_OverflowError"))
        if String(problem).startswith("InterruptedError:"):
            return raise_python_exception(error_without_class(problem, "InterruptedError"), ExceptionType("PyExc_InterruptedError"))
        return raise_python_exception(error_without_class(problem, "RuntimeError"), ExceptionType.RuntimeError)
