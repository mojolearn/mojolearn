# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for the base `_mojolearn` family's HOST HELPERS (the CPU
training lane, phase 1, 2026-09-13; brief
docs/lanes/BRIEF_cpu_training_2026-09-13.md section 3.2).

HOST ONLY, AND NO ARITHMETIC AT ALL. `python/mojolearn/_buffer.py::_native`
resolves the input converters every estimator funnels its arrays through
from the base binding (`bindings/_mojolearn.mojo`): a float64-to-float32
cast, a transpose into column-major order, a finiteness predicate, and the
label helpers of `_labels.py`. On a CPU-only install the base binding is a
by-name stub, so the first Lasso fit refused at `_mojolearn.transpose_f32`
(the Fortran-order move `cdFit` requires) before the solver host binding
was ever called. This module carries those helpers under the base binding's
names, routed by `_backend._HOST_MODULES` (`"_mojolearn":
"_mojolearn_core_host"`), so `_native` finds them through the same
`_backend.binding("_mojolearn")` call it makes on a GPU box. Every one is a
BYTE MOVE (a transpose, a widening or narrowing cast, a predicate, a
gather, an argmax); none carries a fold, so none has a sabotage arm, and
`core_host_sabotage()` reports the define truthfully so a sabotage set
loads as one set.

The base binding's ESTIMATOR entries (`kmeans_fit`, `knn_search`,
`knn_classify`, `knn_regress`, `rbc_knn_search`, `radius_neighbors_*`) and
its other helpers are deliberately ABSENT here, so the kmeans and knn lanes
keep refusing BY NAME through `_HostBinding` until phase 2a lands them.

`transpose_f32` and `cast_colmajor_f64_to_f32` MIRROR
`bindings/_mojolearn.mojo::_tiled_transpose_to_f32` (DEVIATIONS 2471,
2472) element for element, `dst[c * rows + r] = Float32(src[r * cols +
c])`; the tiling there is a cache order, not a value, so a plain loop
writes the same bytes. The seven others are `bindings/host_helpers.mojo`,
the forest host lane's, shared.
"""
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from std.sys.compile import is_defined

from bindings.host_helpers import (
    all_finite_f32_binding,
    all_finite_f64_binding,
    argmax_rows_f32_binding,
    argmax_rows_f64_binding,
    cast_f64_to_f32_binding,
    gather_f64_binding,
    gather_i64_binding,
)
from bindings.hostptr import f32_ptr, f64_ptr
from checks.kernel_matrix import (
    COLUMN_CPU,
    TARGET_COLUMN,
    column_name,
)
from checks.numerics import GLOBAL_NUMERIC_MODE


#: Reported by `core_host_sabotage()`. This binding moves bytes and folds
#: nothing, so the define changes no answer here; it is reported so the
#: gate's sabotage set reads as one set.
comptime CORE_HOST_SABOTAGE_DEFINE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()


def _index(value: PythonObject) raises -> Int:
    var type_name = String(py=value.__class__.__name__)
    if type_name == "bool" or type_name == "bool_":
        raise Error("core host: integers expected, not booleans")
    var operator_module = Python.import_module("operator")
    return Int(py=operator_module.index(value))


def core_host_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def core_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def core_host_column_binding() raises -> PythonObject:
    """`column_name(TARGET_COLUMN)`, the comptime assert's witness: "cpu".
    THE COLUMN IS THE CPU COLUMN, OR THIS DOES NOT BUILD; the assert lives
    in a function body PyInit registers, so it is compiled in every build.
    `bindings/build_core_host.sh` passes -D MOJOLEARN_COLUMN_CPU."""
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "core host: this binding compiles the CPU column only; pass"
        " -D MOJOLEARN_COLUMN_CPU (bindings/build_core_host.sh does)"
    )
    return PythonObject(column_name(TARGET_COLUMN))


# There is no `core_host_detected_column` read-back, for the reason 8d16ce2f
# removed it from the forest and byte LM host bindings: the detected column
# folds to the GPU of the machine that ran the build, so its name would land
# in the vendor-neutral binary. The comptime assert above is the check.


def core_host_sabotage_binding() raises -> PythonObject:
    """Whether this binary was built with -D MOJOLEARN_HOST_SABOTAGE=1. It
    changes NO answer here (byte moves have no fold); reported so the
    gate's sabotage set is refused outside the gate as one set."""
    return PythonObject(CORE_HOST_SABOTAGE_DEFINE)


# The base binding's names, same contract.


def mojolearn_vendor_binding() raises -> PythonObject:
    """"cpu". On a CPU-only install `_backend.vendor()` is "cpu" and the
    read-back cross-check expects that string from every host binding."""
    return PythonObject(String("cpu"))


def mojolearn_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def _transpose_to_f32[
    S: DType
](
    sp: MutPointer[Scalar[S], MutUntrackedOrigin],
    dp: MutPointer[Float32, MutUntrackedOrigin],
    nr: Int,
    nc: Int,
):
    """`dp[c * nr + r] = Float32(sp[r * nc + c])`, the value
    `_tiled_transpose_to_f32` writes, in plain column order."""
    for c in range(nc):
        var dbase = c * nr
        for r in range(nr):
            dp.unsafe_store(dbase + r, sp.unsafe_load(r * nc + c).cast[DType.float32]())


def transpose_f32_binding(
    src_addr: PythonObject,
    dst_addr: PythonObject,
    rows: PythonObject,
    cols: PythonObject,
) raises -> PythonObject:
    """Read a C-contiguous float32 `[rows, cols]` matrix at `src`, write its
    COLUMN-MAJOR layout at `dst`, `dst[c * rows + r] == src[r * cols + c]`
    (DEVIATION 2472). Returns 0. A pure move, no arithmetic. An empty
    matrix writes nothing and reads neither address; a negative dimension
    is refused rather than read. `src` and `dst` must not overlap."""
    var nr = _index(rows)
    var nc = _index(cols)
    if nr < 0:
        raise Error(
            "transpose_f32: rows must be non-negative, got " + String(nr)
        )
    if nc < 0:
        raise Error(
            "transpose_f32: cols must be non-negative, got " + String(nc)
        )
    if nr == 0 or nc == 0:
        return PythonObject(0)
    var sp = f32_ptr(_index(src_addr))
    var dp = f32_ptr(_index(dst_addr))
    with GILReleased(Python()):
        _transpose_to_f32(sp, dp, nr, nc)
    return PythonObject(0)


def cast_colmajor_f64_to_f32_binding(
    src_addr: PythonObject,
    dst_addr: PythonObject,
    rows: PythonObject,
    cols: PythonObject,
) raises -> PythonObject:
    """Read a C-contiguous float64 `[rows, cols]` matrix at `src`, write the
    COLUMN-MAJOR float32 matrix at `dst`, `dst[c * rows + r] ==
    Float32(src[r * cols + c])` (DEVIATION 2471). Returns 0. Every element
    is read once, narrowed once and written once to its transposed
    position. An empty matrix writes nothing; a negative dimension is
    refused rather than read. `src` and `dst` must not overlap."""
    var nr = _index(rows)
    var nc = _index(cols)
    if nr < 0:
        raise Error(
            "cast_colmajor_f64_to_f32: rows must be non-negative, got "
            + String(nr)
        )
    if nc < 0:
        raise Error(
            "cast_colmajor_f64_to_f32: cols must be non-negative, got "
            + String(nc)
        )
    if nr == 0 or nc == 0:
        return PythonObject(0)
    var sp = f64_ptr(_index(src_addr))
    var dp = f32_ptr(_index(dst_addr))
    with GILReleased(Python()):
        _transpose_to_f32(sp, dp, nr, nc)
    return PythonObject(0)


@export
def PyInit__mojolearn_core_host() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_core_host")
        module.def_function[core_host_numeric_mode_binding]("core_host_numeric_mode")
        module.def_function[core_host_vendor_binding]("core_host_vendor")
        module.def_function[core_host_column_binding]("core_host_column")
        module.def_function[core_host_sabotage_binding]("core_host_sabotage")
        module.def_function[mojolearn_vendor_binding]("mojolearn_vendor")
        module.def_function[mojolearn_numeric_mode_binding]("mojolearn_numeric_mode")
        module.def_function[transpose_f32_binding]("transpose_f32")
        module.def_function[cast_colmajor_f64_to_f32_binding]("cast_colmajor_f64_to_f32")
        module.def_function[cast_f64_to_f32_binding]("cast_f64_to_f32")
        module.def_function[all_finite_f32_binding]("all_finite_f32")
        module.def_function[all_finite_f64_binding]("all_finite_f64")
        module.def_function[gather_i64_binding]("gather_i64")
        module.def_function[gather_f64_binding]("gather_f64")
        module.def_function[argmax_rows_f32_binding]("argmax_rows_f32")
        module.def_function[argmax_rows_f64_binding]("argmax_rows_f64")
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_core_host: ", error))
