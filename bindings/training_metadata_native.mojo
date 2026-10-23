# SPDX-License-Identifier: Apache-2.0
# Public training metadata: fixed compiled entries, no runtime loader callbacks.
from std.python import PythonObject
from bindings.py_balanced import py_call, py_type
from bindings.py_typed_call import py_call_typed


def training_metadata(operation: String, mode: PythonObject, context: PythonObject) raises -> PythonObject:
    var types = context["training_metadata_types"]
    var selected = mode
    # The former public wrapper treated an empty string as the default too.
    # Check its exact native type before truth/strip/lower can reach a hook.
    if selected is None or (py_type(selected) is types[0] and len(selected) == 0):
        selected = context["backend"]["_DEFAULT_MODE"]
    if py_type(selected) is not types[0]:
        raise Error("ValueError: numeric_mode requires None or a native str ('fast' or 'identical'); Python conversion hooks cannot execute")
    selected = py_call(context["training_metadata_lower"], py_call(context["training_metadata_strip"], selected))
    var requested = String(selected)
    if requested != "identical" and requested != "fast":
        raise Error("ValueError: numeric_mode must be fast, identical or None")
    var entries = context["training_metadata_bindings"][selected]
    if py_type(entries) is types[0]:
        # Import-time binding failures retain their original category/message.
        raise Error(String(entries))
    if py_type(entries) is not types[3] or len(entries) != 2:
        raise Error("TypeError: training metadata requires its connected compiled entries")
    if py_type(entries[0]) is not types[2] or py_type(entries[1]) is not types[2]:
        raise Error("TypeError: training metadata requires its connected compiled entries")
    # These two existing exports read only their build's constants. Recheck
    # the called binary, rather than trusting directory or requested labels.
    var code = py_call_typed(entries[0])
    if py_type(code) is not types[1]:
        raise Error("TypeError: training numeric mode must be a native integer")
    var numeric = Int(py=code)
    var compiled = String("identical") if numeric == 1 else String("fast") if numeric == 0 else String("unknown")
    if compiled != requested:
        if context["backend"]["_CPU_ONLY"] is not None:
            raise Error("ImportError: mojolearn: _mojolearn_training was compiled " + compiled + " but numeric_mode asked for " + requested + " -- a binary is in the wrong directory; rebuild the sets with\n    bash bindings/build_training.sh\n    MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_training.sh")
        raise Error("RuntimeError: mojolearn: _mojolearn_training was compiled for " + compiled + ", but this call requested " + requested + "; rebuild the " + requested + " binding")
    if operation == "training_numeric_mode":
        return PythonObject(compiled)
    var vendor = py_call_typed(entries[1])
    if py_type(vendor) is not types[0]:
        raise Error("TypeError: training vendor must be a native string")
    return vendor
