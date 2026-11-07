# SPDX-License-Identifier: Apache-2.0
# The caller rule of the classical doors, in one leaf module so the classical
# owner (bindings/classical_native.mojo) and the tree door
# (bindings/classical_trees_door.mojo) share it without one importing the
# other (2026-11-07 host split; docs/build/HOT_GROUPS.md section 6). Moved
# here unchanged from bindings/classical_native.mojo.
from std.python import PythonObject
from bindings.native_caller_frame import native_caller_globals
from bindings.py_balanced import py_get, py_type


def classical_caller_outside(context: PythonObject) raises -> Bool:
    """True when the Python frame that called the public method belongs to
    code outside the package (the rule of _mode._caller_outside). Frame and
    globals metadata are read through CPython's native frame APIs only."""
    var globs = PythonObject(None)
    try:
        # Depth 0 is the public method's own one-call frame. A door that
        # forwards to another compiled door (PyObject_Call of a C entry) adds
        # no Python frame, so the depth is the same in either owner.
        globs = native_caller_globals(1)
    except:
        return True
    if globs is None:
        return True
    if py_get(globs, "__mojolearn_user_boundary__") is PythonObject(True):
        return True
    if py_get(globs, "__mojolearn_internal__") is PythonObject(True):
        return False
    var name = py_get(globs, "__name__")
    if py_type(name) is not py_type(PythonObject("")):
        return True
    var module = String(name)
    return module != "mojolearn" and not module.startswith("mojolearn.") and not module.startswith("mojolearn_verify_all_")
