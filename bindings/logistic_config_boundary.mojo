# SPDX-License-Identifier: Apache-2.0
"""Exact scalar UNPACK and public configuration publication for Logistic."""
from std.memory import bitcast
from std.python import Python, PythonObject
from std.python._cpython import ExternalFunction, PyObjectPtr
from core.shared_slot import SharedSlot
from core.logistic_config import (
    LogisticScalar, LogisticPromotion, LOGISTIC_NO_PENALTY, LOGISTIC_ELASTIC,
    logistic_penalty, logistic_positive, logistic_ratio_valid, logistic_strength_plan,
    logistic_scalar_operation, LogisticArithmetic,
)
from bindings.cpython_table import cpython_table
from bindings.py_balanced import py_type, py_setitem
from bindings.native_scalar_decode import native_scalar_buffer_dtype
from bindings.native_scalar_promotion import native_scalar_weak_promotion
from bindings.optimizer_public_words import stash_python_error
from bindings.sequence_array_warnings import sequence_array_errstate, sequence_array_report

comptime _LogisticRepr = ExternalFunction[
    "PyObject_Repr", def(PyObjectPtr) thin abi("C") -> PyObjectPtr]
comptime _LogisticStr = ExternalFunction[
    "PyObject_Str", def(PyObjectPtr) thin abi("C") -> PyObjectPtr]
comptime _LogisticFormat = ExternalFunction[
    "PyObject_Format", def(PyObjectPtr, PyObjectPtr) thin abi("C") -> PyObjectPtr]


struct LogisticScalarApi(Defaultable, Movable):
    var representation: _LogisticRepr.type
    var text: _LogisticStr.type
    var format: _LogisticFormat.type

    def __init__(out self):
        ref cpy = Python().cpython()
        self.representation = _LogisticRepr.load(cpy.lib.borrow())
        self.text = _LogisticStr.load(cpy.lib.borrow())
        self.format = _LogisticFormat.load(cpy.lib.borrow())


comptime LOGISTIC_SCALAR_API = SharedSlot[LogisticScalarApi, "MojolearnLogisticScalarApiV1"]


def logistic_scalar_text(value: PythonObject, style: Int = 0) raises -> String:
    # Callers already admitted the exact native scalar/container. These
    # cached C APIs never consult mutable builtins or a user conversion hook.
    var api = LOGISTIC_SCALAR_API.get_or_create_ptr()
    var pointer: PyObjectPtr
    if style == 1:
        pointer = api[].text(value._obj_ptr)
    elif style == 2:
        var spec = PythonObject("")
        pointer = api[].format(value._obj_ptr, spec._obj_ptr)
        _ = spec
    else:
        pointer = api[].representation(value._obj_ptr)
    if not pointer:
        var failure = stash_python_error()
        raise failure^
    return String(PythonObject(from_owned=pointer))


def logistic_float(value: PythonObject) raises -> Float64:
    var pointer = cpython_table()[].PyNumber_Float(value._obj_ptr)
    if not pointer:
        var failure = stash_python_error()
        raise failure^
    var number = PythonObject(from_owned=pointer)
    return Float64(py=number)


def logistic_integer(value: PythonObject) raises -> Int:
    var pointer = cpython_table()[].PyNumber_Long(value._obj_ptr)
    if not pointer:
        var failure = stash_python_error()
        raise failure^
    var number = PythonObject(from_owned=pointer)
    return Int(py=number)


def _logistic_numpy_scalar(value: PythonObject) raises -> LogisticScalar:
    var dtype = value.dtype
    var kind = String(dtype.kind)
    var width = Int(py=dtype.itemsize)
    var scalar = native_scalar_buffer_dtype(value, dtype)
    var signaling = scalar.real.special == 2 and (scalar.real.low & (UInt64(1) << UInt64(scalar.real.precision - 2))) == 0
    return LogisticScalar(bitcast[DType.float64](scalar.real.float64_bits()),
        1 if kind == "f" else (3 if kind == "b" else 2), width, signaling)


def _logistic_compare_value(value: PythonObject, operation: String, right: String) raises -> Float64:
    var cls = py_type(value)
    if cls is py_type(PythonObject(0)):
        # Do not convert a huge positive C to float before the old strength
        # stage: penalty=None never needs that conversion. The overflow sign
        # is sufficient for the only comparisons here, against zero and one.
        var overflow = Array[Int32, 1](fill=0)
        var integer = cpython_table()[].PyLong_AsLongLongAndOverflow(value._obj_ptr,
            Int(overflow.unsafe_ptr()))
        if cpython_table()[].PyErr_Occurred():
            var failure = stash_python_error()
            raise failure^
        return Float64(integer) if overflow[0] == 0 else (2.0 if overflow[0] > 0 else -2.0)
    if cls is py_type(PythonObject(0.0)) or cls is py_type(PythonObject(True)):
        return logistic_float(value)
    if value is None or cls is py_type(PythonObject("")):
        var name = String("NoneType") if value is None else String("str")
        raise Error("TypeError: '" + operation + "' not supported between instances of '" + name + "' and '" + right + "'")
    return _logistic_numpy_scalar(value).value


def logistic_validate_configuration(fields: PythonObject) raises -> Int:
    var penalty = fields["penalty"]
    var code = logistic_penalty(String(penalty) if py_type(penalty) is py_type(PythonObject("")) else String(""), penalty is None)
    if code < 0:
        raise Error("`penalty` " + logistic_scalar_text(penalty) + " not supported.")
    var solver = fields["solver"]
    if py_type(solver) is not py_type(PythonObject("")) or String(solver) != "qn":
        raise Error("Only quasi-newton `qn` solver is supported, not " + logistic_scalar_text(solver, 1))
    var c = fields["C"]
    if not logistic_positive(_logistic_compare_value(c, "<=", "int")):
        raise Error("C must be positive, got " + logistic_scalar_text(c, 2))
    # Preserve publication/refusal order, including an elastic-net failure.
    py_setitem(fields, "_l1_ratio", PythonObject(None))
    if code == LOGISTIC_ELASTIC:
        var ratio = fields["l1_ratio"]
        if ratio is None:
            raise Error("l1_ratio has to be specified for loss='elasticnet'")
        if not logistic_ratio_valid(_logistic_compare_value(ratio, "<", "float")):
            raise Error("l1_ratio value has to be between 0.0 and 1.0")
        py_setitem(fields, "_l1_ratio", ratio)
    return code


def _logistic_number(value: PythonObject) raises -> LogisticScalar:
    var cls = py_type(value)
    if cls is py_type(PythonObject(0)) or cls is py_type(PythonObject(0.0)) or cls is py_type(PythonObject(True)):
        var number = logistic_float(value)
        var bits = bitcast[DType.uint64](number) & UInt64(0x7FFFFFFFFFFFFFFF)
        return LogisticScalar(number, 4 if cls is py_type(PythonObject(True)) else 0, 8,
            bits > UInt64(0x7FF0000000000000) and (bits & UInt64(0x0008000000000000)) == 0)
    return _logistic_numpy_scalar(value)


def _logistic_promotion(mut policy: LogisticPromotion, scalar: LogisticScalar,
                        context: PythonObject) raises:
    # Only half/single precision can differ between legacy and weak promotion
    # in this real-valued configuration. Wider/integral scalar cases need no
    # version lookup or native state-getter call.
    if scalar.kind == 1 and scalar.width < 8 and not policy.captured:
        policy.weak = native_scalar_weak_promotion(context, "LogisticRegression")
        policy.captured = True


def _logistic_report(event: LogisticArithmetic, context: PythonObject) raises:
    if event.status == 0 and event.cast_status == 0:
        return
    # Capture native thread-local error metadata only when an operation has
    # a flag, as NumPy does. Native warning delivery supports filters/recording
    # and refuses custom Python warning/error callbacks before invoking them.
    var settings = sequence_array_errstate(context)
    if event.cast_status != 0:
        sequence_array_report(event.cast_status, settings, context, "cast")
    if event.status != 0:
        var name = String("multiply") if event.operation == 0 else (String("subtract") if event.operation == 1 else String("divide"))
        if event.scalar_message:
            var full = "scalar " + name
            name = full^
        sequence_array_report(event.status, settings, context, name)


def logistic_configuration_strengths(fields: PythonObject, penalty: Int,
        mut policy: LogisticPromotion, context: PythonObject) raises -> Tuple[Float64, Float64]:
    if penalty == LOGISTIC_NO_PENALTY:
        return (0.0, 0.0)
    var c = _logistic_number(fields["C"])
    _logistic_promotion(policy, c, context)
    var ratio = LogisticScalar(0.0, 0, 8)
    if penalty == LOGISTIC_ELASTIC:
        ratio = _logistic_number(fields["l1_ratio"])
        _logistic_promotion(policy, ratio, context)
    var plan = logistic_strength_plan(penalty, c, ratio, policy.weak)
    for i in range(4):  # driver-loop: the four original scalar expressions, in refusal order
        _logistic_report(plan.event(i), context)
    return (plan.l1, plan.l2)


def logistic_configuration_tolerance(value: PythonObject, mut policy: LogisticPromotion,
                                     context: PythonObject) raises -> Tuple[Float64, Float64]:
    # float(tol) precedes tol * .01 in the legacy request. A numeric string
    # still fails at the multiplication; None fails at float conversion.
    var tolerance = logistic_float(value)
    if py_type(value) is py_type(PythonObject("")):
        raise Error("TypeError: can't multiply sequence by non-int of type 'float'")
    var cls = py_type(value)
    var bits = bitcast[DType.uint64](tolerance) & UInt64(0x7FFFFFFFFFFFFFFF)
    var scalar = (LogisticScalar(tolerance, 4 if cls is py_type(PythonObject(True)) else 0, 8,
                  bits > UInt64(0x7FF0000000000000) and (bits & UInt64(0x0008000000000000)) == 0) if cls is py_type(PythonObject(0)) or
                  cls is py_type(PythonObject(0.0)) or cls is py_type(PythonObject(True))
                  else _logistic_numpy_scalar(value))
    _logistic_promotion(policy, scalar, context)
    var result = logistic_scalar_operation(scalar, LogisticScalar(0.01, 0, 8), 0, policy.weak)
    _logistic_report(result, context)
    return (tolerance, result.scalar.value)
