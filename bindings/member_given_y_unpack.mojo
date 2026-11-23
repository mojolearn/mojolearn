# SPDX-License-Identifier: Apache-2.0
"""UNPACK of THE GIVEN y (lane mf-tail5; core/member_abi.mojo
`MemberCall.y_given`): a regressor's y as the caller was given it, read once
at the boundary with the GIL held, as a typed host view the member RUNs read
by address (no copy, no conversion).

Why: a member fit reads a regressor's targets as float32 words, but the
splitter door stratifies a regressor's y on the values it was GIVEN
(np.unique over a float64 y). Two float64 targets equal in float32 would be
one class on the member path and two at the door; a member whose kind reads
the given y (MEMBER_READS_Y_GIVEN: a stratified splitter in its fit) codes
these words instead (bindings/msel_split_spec_run.mojo
`msel_target_codes_run`).

Only an exact instance of the caller's array class is read (no Python
protocol of another object runs), and only a layout whose words are the
given values themselves: float64, int64 or int32 elements, C order, one row
per fit row (`rows`) and `cols` columns. Anything else answers "not given":
the float32 targets are then the given y, as before (a float32 y, or one the
caller converted anyway)."""
from std.python import PythonObject
from bindings.classical_arrays_door import classical_address
from bindings.py_balanced import py_type
from core.member_abi import MemberMatrix, MEMBER_F64, MEMBER_I64, MEMBER_I32


def _given_dtype(dtype: String) -> Int:
    """MEMBER_F64 / MEMBER_I64 / MEMBER_I32 of a typestr or a dtype name; -1
    for any other element type."""
    if dtype == "<f8" or dtype == "float64":
        return MEMBER_F64
    if dtype == "<i8" or dtype == "int64":
        return MEMBER_I64
    if dtype == "<i4" or dtype == "int32":
        return MEMBER_I32
    return -1


def member_given_y(y: PythonObject, rows: Int, cols: Int, array_class: PythonObject) raises -> MemberMatrix:
    """THE GIVEN y view of `y` (module note), or an empty view (not given).
    The caller keeps `y` alive until every fit that reads the view returned."""
    if rows <= 0 or cols < 1 or y is None or not (py_type(y) is array_class):
        return MemberMatrix()
    try:
        var dtype = _given_dtype(String(py=y.dtype))
        if dtype < 0:
            return MemberMatrix()
        var ndim = Int(py=y.ndim)
        if ndim == 1:
            if cols != 1 or Int(py=y.shape[0]) != rows or String(py=y.order) != "C":
                return MemberMatrix()
        elif ndim == 2:
            if Int(py=y.shape[0]) != rows or Int(py=y.shape[1]) != cols or String(py=y.order) != "C":
                return MemberMatrix()
        else:
            return MemberMatrix()
        if Int(py=y.size) != rows * cols:
            return MemberMatrix()
        var address = Int(py=classical_address(y))
        if address == 0:
            return MemberMatrix()
        return MemberMatrix(address, rows, cols, dtype)
    except:
        return MemberMatrix()
