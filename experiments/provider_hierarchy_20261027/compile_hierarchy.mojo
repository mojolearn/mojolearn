# SPDX-License-Identifier: Apache-2.0
# Compile-only closure probe. The parent compiles this file; do not run it.
# The deliberately empty objects are not valid public-operation inputs.
from std.python import Python, PythonObject
from bindings.cfb_agglo import cfb_agglo_operation


def main() raises:
    var model = PythonObject(None)
    var args = Python.tuple(PythonObject(None), PythonObject(None), PythonObject(None))
    _ = cfb_agglo_operation(model, "family_cfb_agglo_fit", args, Python.dict())
