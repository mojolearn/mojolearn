# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for `mojolearn.identical.gemm.fp32.v1`, the bit-identical
FP32 matrix product (the CPU training lane, phase 1, 2026-09-13; brief
docs/lanes/BRIEF_cpu_training_2026-09-13.md sections 1.1 gemm-pinned and
3.2).

HOST ONLY. No DeviceContext, no kernel, no GPU. The arithmetic is
`gemm/checks/gemm_oracle.mojo::gemm_oracle`, the NORMATIVE answer of the
profile ("logical leaves at contract_leaf_size(k), combined by
fold_balanced_tree's FIXED BALANCED TREE. Not close to; the same bits"),
which already compiles host-only inside the byte LM host binding. The GPU
binding `bindings/_mojolearn_linalg.mojo` computes the same profile through
`gemm/checks/gemm_identical.mojo`; this file computes it through the
definition the kernel is gated against. Nothing here is a third spelling.

THE EXPORTED NAMES ARE THE GPU BINDING'S NAMES, so `python/mojolearn/
_linalg_impl.py` runs unchanged on a CPU-only install through
`_backend._HOST_MODULES` (`"_mojolearn_linalg": "_mojolearn_linalg_host"`):
`gemm`, `linalg_numeric_mode`, `linalg_vendor`, `linalg_profile_version`,
with the SAME address contract (output first, then a, then b, then the
four-value params list, mirrored word for word in `_linalg_impl.py`). The
read-back names the host loader requires are the `linalg_host_*` five.

`linalg_vendor` answers "cpu", not `COMPILED_VENDOR`'s "none": on a
CPU-only install `_backend.vendor()` is "cpu" and the read-back cross-check
expects the same string from every host binding (brief section 3.2).
"""
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from bindings.hostptr import f32_ptr, read_f32
from checks.kernel_matrix import (
    COLUMN_CPU,
    DETECTED_COLUMN,
    TARGET_COLUMN,
    column_name,
)
from checks.numerics import GLOBAL_NUMERIC_MODE
from gemm.checks.gemm_oracle import (
    GEMM_ORACLE_HOST_SABOTAGE,
    OP_NN,
    OP_NT,
    OP_TN,
    gemm_oracle,
)


#: Cells per output, so `m * n` and `m * k` stay far from any Int edge.
comptime LINALG_HOST_MAX_EXTENT = 1073741824


def _index(value: PythonObject) raises -> Int:
    var type_name = String(py=value.__class__.__name__)
    if type_name == "bool" or type_name == "bool_":
        raise Error("linalg host: integers expected, not booleans")
    var operator_module = Python.import_module("operator")
    return Int(py=operator_module.index(value))


def linalg_host_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def linalg_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def linalg_host_column_binding() raises -> PythonObject:
    """`column_name(TARGET_COLUMN)`, the comptime assert's witness: "cpu".

    THE COLUMN IS THE CPU COLUMN, OR THIS DOES NOT BUILD. The assert lives
    in a function body because Mojo takes `comptime assert` there only, and
    PyInit registers this function, so it is compiled in every build of the
    module. `bindings/build_linalg_host.sh` passes -D MOJOLEARN_COLUMN_CPU."""
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "linalg host: this binding compiles the CPU column only; pass"
        " -D MOJOLEARN_COLUMN_CPU (bindings/build_linalg_host.sh does)"
    )
    return PythonObject(column_name(TARGET_COLUMN))


def linalg_host_detected_column_binding() raises -> PythonObject:
    """`column_name(DETECTED_COLUMN)`: what the accelerator predicates fold
    to in THIS build, with no define."""
    return PythonObject(column_name(DETECTED_COLUMN))


def linalg_host_sabotage_binding() raises -> PythonObject:
    """Whether this binary walks every leaf descending on purpose
    (-D MOJOLEARN_HOST_SABOTAGE=1, the gate's negative control)."""
    return PythonObject(GEMM_ORACLE_HOST_SABOTAGE)


# The GPU binding's names, same contract.


def linalg_numeric_mode_binding() raises -> PythonObject:
    """THE BUILD'S TIER as the `NUMERIC_*` code: always 1 here, a host
    binding is IDENTICAL only (the build script refuses any other)."""
    return PythonObject(GLOBAL_NUMERIC_MODE)


def linalg_vendor_binding() raises -> PythonObject:
    """"cpu". See the module docstring."""
    return PythonObject(String("cpu"))


def linalg_profile_version_binding() raises -> PythonObject:
    """The MAJOR VERSION of the GEMM profile this binary implements: 1, for
    `mojolearn.identical.gemm.fp32.v1`. `gemm_oracle` IS that version's
    definition (leaf rule, contract section 7.1; fold topology, 7.2)."""
    return PythonObject(1)


def gemm_binding(
    c_addr: PythonObject,
    a_addr: PythonObject,
    b_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`C = op(A) . op(B)` under `mojolearn.identical.gemm.fp32.v1`, on the
    host, by `gemm_oracle`. Returns `m * n`, the cells written.

    `params` is, in this exact order (mirrored word for word in
    `python/mojolearn/_linalg_impl.py` and in the GPU binding):

        0  m       rows of C
        1  n       columns of C
        2  k       the contracted extent
        3  op      0 = OP_NN, 1 = OP_NT, 2 = OP_TN

    and the row-major element counts, contract section 0.1, are `m * k` for
    A and `n * k` for B in every orientation (`gemm/host_entry.mojo`'s
    table). THE OUTPUT ADDRESS COMES FIRST, as in the GPU binding. The
    degenerate shapes are refused rather than answered, the same rule and
    the same words as `identical_gemm_host`."""
    if len(params) != 4:
        raise Error(
            "gemm: params must contain 4 values (m, n, k, op), got "
            + String(len(params))
        )
    var cp = f32_ptr(_index(c_addr))
    var a_address = _index(a_addr)
    var b_address = _index(b_addr)
    var m = _index(params[0])
    var n = _index(params[1])
    var k = _index(params[2])
    var op = _index(params[3])
    if m <= 0 or n <= 0 or k <= 0:
        raise Error(
            "identical_gemm_host: m, n and k must all be positive, got m="
            + String(m) + " n=" + String(n) + " k=" + String(k)
            + " (contract section 8 specifies the degenerate shapes; this"
            " surface has no gate on them and refuses rather than guesses)"
        )
    if op != OP_NN and op != OP_NT and op != OP_TN:
        raise Error(
            "identical_gemm_host: op must be 0 (OP_NN), 1 (OP_NT) or 2"
            " (OP_TN), got " + String(op)
        )
    if (
        m > LINALG_HOST_MAX_EXTENT or n > LINALG_HOST_MAX_EXTENT
        or k > LINALG_HOST_MAX_EXTENT
    ):
        raise Error("gemm: m, n and k must each be at most 2^30")
    var wrote = 0
    with GILReleased(Python()):
        var a = read_f32(a_address, m * k)
        var b = read_f32(b_address, n * k)
        # THE ONE LINE THAT COMPUTES ANYTHING.
        var c = gemm_oracle(a, b, op, m, n, k)
        for i in range(m * n):
            cp[i] = c[i]
        wrote = m * n
    return PythonObject(wrote)


@export
def PyInit__mojolearn_linalg_host() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_linalg_host")
        module.def_function[linalg_host_numeric_mode_binding]("linalg_host_numeric_mode")
        module.def_function[linalg_host_vendor_binding]("linalg_host_vendor")
        module.def_function[linalg_host_column_binding]("linalg_host_column")
        module.def_function[linalg_host_detected_column_binding]("linalg_host_detected_column")
        module.def_function[linalg_host_sabotage_binding]("linalg_host_sabotage")
        module.def_function[linalg_vendor_binding]("linalg_vendor")
        module.def_function[linalg_numeric_mode_binding]("linalg_numeric_mode")
        module.def_function[linalg_profile_version_binding]("linalg_profile_version")
        module.def_function[gemm_binding]("gemm")
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_linalg_host: ", error))
