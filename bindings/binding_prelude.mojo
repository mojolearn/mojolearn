# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The binding prelude: the helpers every Python binding re-declares.

About seventy bindings each define `<family>_vendor_binding`,
`<family>_numeric_mode_binding`, `_f32_ptr` and `_i32_ptr`, identical up to
the family prefix. These are the most common versions:

  vendor_binding        the COMPILED vendor (`checks/vendor.mojo`)
  host_vendor_binding   "cpu", the host bindings' column
  numeric_mode_binding  the build's GLOBAL_NUMERIC_MODE as an int
  f32_ptr, i32_ptr      address -> pointer, refusing a null address

A binding registers them under its own names, e.g.
`module.def_function[vendor_binding]("svm_vendor")`. New binding code
imports these (CURRENT DIRECTIVES); existing bindings keep their copies
until the consolidation pass migrates them. Tested by
`checks/scaffold_check.mojo`.
"""

from std.python import PythonObject

from bindings.hostptr import f32_ptr as _hostptr_f32, i32_ptr as _hostptr_i32
from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR


def compiled_vendor_name() -> String:
    """`metal`, `cuda`, `hip` or `none`, folded in at compile time."""
    return String(COMPILED_VENDOR)


def vendor_binding() raises -> PythonObject:
    return PythonObject(compiled_vendor_name())


def host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def f32_ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    """Refuses address 0 by name (`bindings/hostptr.mojo`)."""
    return _hostptr_f32(addr)


def i32_ptr(addr: Int) raises -> MutPointer[Int32, MutUntrackedOrigin]:
    """Refuses address 0 by name (`bindings/hostptr.mojo`)."""
    return _hostptr_i32(addr)
