# SPDX-License-Identifier: Apache-2.0
"""ONE rounding for an optimizer hyperparameter given as an integer.

An optimizer scalar is read as a float64 and narrowed to float32 where the
kernel uses it. An integer is exact in float64 up to 2^53, so that path
rounds once. Above 2^53 the int64 -> float64 conversion rounds first and the
float64 -> float32 narrowing rounds again, which can land one ulp off the
float32 nearest the integer (fb6-smoke, 2026-11-17: Adagrad's
initial_accumulator_value 2^62 + 2^38 + 1). Every optimizer door reads its
hyperparameters through `optimizer_real` so that every path (training
optimizers, sequence optimizers, the recurrent and MLP fits) narrows an
integer the same way, int64 -> float32 directly, as NumPy's cast does.
Integers beyond int64 keep the float64 value (NumPy would not cast them as
int64 either). A real number (|x| <= 2^53, or any non-integer) is returned
unchanged, after one compare: nothing is added to an ordinary value."""
from std.python import PythonObject
from bindings.py_cpython_table import cpython_table


def optimizer_real(value: PythonObject, real: Float64) raises -> Float64:
    """`real` (the admitted `value` as float64), or, for an integer `value`
    of magnitude above 2^53 within int64, the float32 nearest it (as a
    float64, exact), so the kernel's float32 narrowing is a no-op."""
    if abs(real) <= 9007199254740992.0 or real != real:
        return real
    var table = cpython_table()
    var index = table[].PyNumber_Index(value._obj_ptr)
    if not index:
        table[].PyErr_Clear()
        return real
    var number = PythonObject(from_owned=index)
    var overflow = List[Int32](length=1, fill=0)
    var word = table[].PyLong_AsLongLongAndOverflow(number._obj_ptr, Int(overflow.unsafe_ptr()))
    var wide = overflow[0] != 0
    _ = overflow
    if wide or (word == -1 and table[].PyErr_Occurred()):
        table[].PyErr_Clear()
        return real
    return Float64(word.cast[DType.float32]())
