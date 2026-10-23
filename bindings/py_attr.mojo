# SPDX-License-Identifier: Apache-2.0
# `py_attr`: a Python attribute through CPython's own lookup. PythonObject
# defines Mojo methods named `__getitem__`, `__mul__`, `__contains__`, `copy`,
# `append`, ...; `owner.__getitem__` resolves to that Mojo overload set, which
# cannot be passed as a callable. Ownership as in py_balanced: the result is a
# new reference owned by the returned PythonObject.
from std.python import Python, PythonObject


def py_attr(owner: PythonObject, var name: String) raises -> PythonObject:
    """`getattr(owner, name)`: a Python attribute whose name PythonObject also
    defines as a Mojo method (`__getitem__`, `__mul__`, `copy`, `append`, ...),
    which `owner.name` would resolve to the Mojo overload set."""
    ref cpy = Python().cpython()
    var result = cpy.PyObject_GetAttrString(owner._obj_ptr, name^)
    if not result:
        raise cpy.unsafe_get_error()
    return PythonObject(from_owned=result)

