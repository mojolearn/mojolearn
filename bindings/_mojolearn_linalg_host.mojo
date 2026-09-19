# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for `mojolearn.identical.gemm.fp32.v1`, the bit-identical
FP32 matrix product (the CPU training lane, phase 1, 2026-09-13; brief
docs/lanes/BRIEF_cpu_training_2026-09-13.md sections 1.1 gemm-pinned and
3.2).

HOST ONLY. No DeviceContext, no kernel, no GPU. The arithmetic is
`gemm/host/gemm_oracle.mojo::gemm_oracle`, the NORMATIVE answer of the
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

THE CHOLESKY DOOR (lane/inference-embedding-ivf-cholesky, 2026-09-15).
`cholesky_profile_jitter`, `cholesky_factor` (3 addresses, 2 params) and
`cholesky_solve` (3 addresses, 6 params) carry the GPU binding
`bindings/_mojolearn_gp.mojo`'s names and contracts over
`cholesky/host/chol_oracle.mojo`, the same entries the gp host binding
exports. They live HERE so that public CPU Cholesky inference (a factor
from a saved model, or a factor of a given matrix, then solve) ships in the
inference wheel: this family ships and the gp family does not. On a
CPU-only install `python/mojolearn/_cholesky_impl.py` binds
`_mojolearn_linalg` for that reason. The sabotage arm reaches the factor
through the trailing update's `gemm_oracle` leaf walk (chol_oracle's
header, THE SABOTAGE).
"""
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from bindings.hostptr import f32_ptr, f64_ptr, read_f32
from checks.kernel_matrix import (
    COLUMN_CPU,
    TARGET_COLUMN,
    column_name,
)
from checks.numerics import GLOBAL_NUMERIC_MODE
from cholesky.host.chol_oracle import (
    CholHostFactor,
    chol_host_jitter_pinned,
    chol_host_potrf,
    chol_host_solve,
)
from gemm.host.identical_gemm import (
    GEMM_ORACLE_HOST_SABOTAGE,
    OP_NN,
    OP_NT,
    OP_TN,
    gemm_oracle,
)
from decomposition.host.linalg_public import (
    host_eigh,
    host_qr_r,
    host_svdvals,
)
from gemm.host.gemm_lowbit_oracle import (
    INT8_MAX_K,
    LOWBIT_PROFILE_VERSION,
    Int8Rows,
    dequantize_rows_int8,
    gemm_bf16_both_oracle,
    gemm_bf16_oracle,
    gemm_int8_oracle,
    narrow_bf16,
    quantize_rows_int8,
    widen_bf16,
)
from bindings.hostptr import i8_ptr, i32_ptr, read_i8, read_i32, read_u16, u16_ptr


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


# There is no `linalg_host_detected_column` read-back, for the reason 8d16ce2f
# removed it from the forest and byte LM host bindings: the detected column
# folds to the GPU of the machine that ran the build, so its name would land
# in the vendor-neutral binary. The comptime assert above is the check.


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


# ===========================================================================
# THE CHOLESKY DOOR, on the host (the GPU gp binding's workstream D entries,
# the same contract word for word; see the module docstring).
# ===========================================================================


def cholesky_profile_jitter_binding() raises -> PythonObject:
    """The profile's pinned ridge, 2^-20, as a Python float."""
    return PythonObject(Float64(chol_host_jitter_pinned()))


def _cholesky_factor_run(
    a: List[Float32],
    n: Int,
    jitter: Float32,
    lp: MutPointer[Float32, MutUntrackedOrigin],
    sp: MutPointer[Float64, MutUntrackedOrigin],
) raises -> Int:
    """The GIL-free half of `cholesky_factor_binding`."""
    var f = chol_host_potrf(a, n, jitter)
    for i in range(n * n):
        lp.unsafe_store(i, f.l[i])
    # info, nb, logdet, jitter -- the GPU binding's order.
    sp.unsafe_store(0, Float64(f.info))
    sp.unsafe_store(1, Float64(f.nb))
    sp.unsafe_store(2, Float64(f.logdet))
    sp.unsafe_store(3, Float64(f.jitter))
    var info = f.info
    _ = f^
    return info


def cholesky_factor_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`cholesky_factor_host(a, n, jitter)` on the host. `addrs`: 0 a,
    1 l_out, 2 scalars_out (info, nb, logdet, jitter). `params`: 0 n,
    1 jitter (unclamped, so the pin refuses by name). Returns `info`."""
    if len(addrs) != 3:
        raise Error(
            "cholesky_factor: addrs must contain 3 addresses (a, l_out,"
            " scalars_out), got "
            + String(len(addrs))
        )
    if len(params) != 2:
        raise Error(
            "cholesky_factor: params must contain 2 values (n, jitter), got "
            + String(len(params))
        )
    var lp = f32_ptr(_index(addrs[1]))
    var sp = f64_ptr(_index(addrs[2]))
    var n = _index(params[0])
    if n < 1 or n > 46340:
        raise Error(
            "cholesky_factor: n must be in [1, 46340] so n * n cells stay"
            " addressable, got " + String(n)
        )
    var jitter = Float32(Float64(py=params[1]))
    var a = read_f32(_index(addrs[0]), n * n)
    var info = 0
    with GILReleased(Python()):
        info = _cholesky_factor_run(a, n, jitter, lp, sp)
    _ = a^
    return PythonObject(info)


def cholesky_solve_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`cholesky_solve_host(factor, b, nrhs)` on the host. `addrs`: 0 l,
    1 b, 2 x_out. `params`: 0 n, 1 nrhs, 2 info (passed through, so the
    refusal to solve against a failed factor fires by name), 3 nb, 4 logdet,
    5 jitter. Returns 0."""
    if len(addrs) != 3:
        raise Error(
            "cholesky_solve: addrs must contain 3 addresses (l, b, x_out),"
            " got "
            + String(len(addrs))
        )
    if len(params) != 6:
        raise Error(
            "cholesky_solve: params must contain 6 values (n, nrhs, info,"
            " nb, logdet, jitter), got "
            + String(len(params))
        )
    var xp = f32_ptr(_index(addrs[2]))
    var n = _index(params[0])
    var nrhs = _index(params[1])
    if n < 1 or n > 46340 or nrhs < 1 or nrhs > LINALG_HOST_MAX_EXTENT // n:
        raise Error(
            "cholesky_solve: n must be in [1, 46340] and nrhs in [1, 2^30 / n],"
            " got n=" + String(n) + " nrhs=" + String(nrhs)
        )
    var info = _index(params[2])
    var nb = _index(params[3])
    var logdet = Float32(Float64(py=params[4]))
    var jitter = Float32(Float64(py=params[5]))
    var l = read_f32(_index(addrs[0]), n * n)
    var b = read_f32(_index(addrs[1]), n * nrhs)
    var factor = CholHostFactor(l^, n, info, logdet, nb, jitter)
    with GILReleased(Python()):
        var x = chol_host_solve(factor, b, nrhs)
        for i in range(n * nrhs):
            xp.unsafe_store(i, x[i])
        _ = x^
    _ = factor^
    _ = b^
    return PythonObject(0)



# ===========================================================================
# THE LOW-BIT PROFILES, on the host (lane/identical-lowbit-inference)
# gemm/IDENTICAL_LOWBIT_CONTRACT.md. The GPU binding's names, the oracles'
# arithmetic; `params` orders mirrored word for word.
# ===========================================================================


def lowbit_profile_version_binding() raises -> PythonObject:
    return PythonObject(LOWBIT_PROFILE_VERSION)


def gemm_bf16_binding(
    c_addr: PythonObject,
    a_addr: PythonObject,
    b_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`params`: 0 m, 1 n, 2 k, 3 op, 4 a_bf16 (1 when A is bf16 bits)."""
    if len(params) != 5:
        raise Error("gemm_bf16: params must contain 5 values (m, n, k, op, a_bf16), got " + String(len(params)))
    var cp = f32_ptr(_index(c_addr))
    var a_address = _index(a_addr)
    var b_address = _index(b_addr)
    var m = _index(params[0])
    var n = _index(params[1])
    var k = _index(params[2])
    var op = _index(params[3])
    var a_bf16 = _index(params[4]) != 0
    if m <= 0 or n <= 0 or k <= 0:
        raise Error("gemm_bf16: m, n and k must all be positive, got m=" + String(m) + " n=" + String(n) + " k=" + String(k))
    if op != OP_NN and op != OP_NT and op != OP_TN:
        raise Error("gemm_bf16: op must be 0 (OP_NN), 1 (OP_NT) or 2 (OP_TN), got " + String(op))
    if m > LINALG_HOST_MAX_EXTENT or n > LINALG_HOST_MAX_EXTENT or k > LINALG_HOST_MAX_EXTENT:
        raise Error("gemm_bf16: m, n and k must each be at most 2^30")
    var wrote = 0
    with GILReleased(Python()):
        var b = read_u16(b_address, n * k)
        var c = List[Float32]()
        if a_bf16:
            var a_bits = read_u16(a_address, m * k)
            c = gemm_bf16_both_oracle(a_bits, b, op, m, n, k)
        else:
            var a = read_f32(a_address, m * k)
            c = gemm_bf16_oracle(a, b, op, m, n, k)
        for i in range(m * n):
            cp[i] = c[i]
        wrote = m * n
    return PythonObject(wrote)


def gemm_int8_binding(
    c_addr: PythonObject,
    qa_addr: PythonObject,
    ea_addr: PythonObject,
    qb_addr: PythonObject,
    eb_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`params`: 0 m, 1 n, 2 k. OP_NT only."""
    if len(params) != 3:
        raise Error("gemm_int8: params must contain 3 values (m, n, k), got " + String(len(params)))
    var cp = f32_ptr(_index(c_addr))
    var qa_address = _index(qa_addr)
    var ea_address = _index(ea_addr)
    var qb_address = _index(qb_addr)
    var eb_address = _index(eb_addr)
    var m = _index(params[0])
    var n = _index(params[1])
    var k = _index(params[2])
    if m <= 0 or n <= 0 or k <= 0:
        raise Error("gemm_int8: m, n and k must all be positive, got m=" + String(m) + " n=" + String(n) + " k=" + String(k))
    if k > INT8_MAX_K:
        raise Error("gemm_int8: k must be at most " + String(INT8_MAX_K) + " (contract L-7), got " + String(k))
    if m > LINALG_HOST_MAX_EXTENT or n > LINALG_HOST_MAX_EXTENT:
        raise Error("gemm_int8: m and n must each be at most 2^30")
    var wrote = 0
    with GILReleased(Python()):
        var qa = read_i8(qa_address, m * k)
        var ea = read_i32(ea_address, m)
        var qb = read_i8(qb_address, n * k)
        var eb = read_i32(eb_address, n)
        var c = gemm_int8_oracle(qa, ea, qb, eb, m, n, k)
        for i in range(m * n):
            cp[i] = c[i]
        wrote = m * n
    return PythonObject(wrote)


def quantize_int8_binding(
    q_addr: PythonObject,
    e_addr: PythonObject,
    x_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`params`: 0 rows, 1 cols."""
    if len(params) != 2:
        raise Error("quantize_int8: params must contain 2 values (rows, cols), got " + String(len(params)))
    var qp = i8_ptr(_index(q_addr))
    var ep = i32_ptr(_index(e_addr))
    var x_address = _index(x_addr)
    var rows = _index(params[0])
    var cols = _index(params[1])
    if rows <= 0 or cols <= 0:
        raise Error("quantize_int8: rows and cols must be positive, got " + String(rows) + " x " + String(cols))
    if rows > LINALG_HOST_MAX_EXTENT or cols > LINALG_HOST_MAX_EXTENT:
        raise Error("quantize_int8: rows and cols must each be at most 2^30")
    var wrote = 0
    with GILReleased(Python()):
        var x = read_f32(x_address, rows * cols)
        var qr = quantize_rows_int8(x, rows, cols)
        for i in range(rows * cols):
            qp[i] = qr.q[i]
        for r in range(rows):
            ep[r] = qr.e[r]
        wrote = rows * cols
    return PythonObject(wrote)


def dequantize_int8_binding(
    y_addr: PythonObject,
    q_addr: PythonObject,
    e_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`params`: 0 rows, 1 cols."""
    if len(params) != 2:
        raise Error("dequantize_int8: params must contain 2 values (rows, cols), got " + String(len(params)))
    var yp = f32_ptr(_index(y_addr))
    var q_address = _index(q_addr)
    var e_address = _index(e_addr)
    var rows = _index(params[0])
    var cols = _index(params[1])
    if rows <= 0 or cols <= 0:
        raise Error("dequantize_int8: rows and cols must be positive, got " + String(rows) + " x " + String(cols))
    if rows > LINALG_HOST_MAX_EXTENT or cols > LINALG_HOST_MAX_EXTENT:
        raise Error("dequantize_int8: rows and cols must each be at most 2^30")
    var wrote = 0
    with GILReleased(Python()):
        var q = read_i8(q_address, rows * cols)
        var e = read_i32(e_address, rows)
        var y = dequantize_rows_int8(Int8Rows(q^, e^, rows, cols))
        for i in range(rows * cols):
            yp[i] = y[i]
        wrote = rows * cols
    return PythonObject(wrote)


def to_bf16_binding(
    dst_addr: PythonObject, src_addr: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`params`: 0 count."""
    if len(params) != 1:
        raise Error("to_bf16: params must contain 1 value (count)")
    var dp = u16_ptr(_index(dst_addr))
    var src_address = _index(src_addr)
    var count = _index(params[0])
    if count <= 0:
        raise Error("to_bf16: count must be positive, got " + String(count))
    if count > LINALG_HOST_MAX_EXTENT:
        raise Error("to_bf16: count must be at most 2^30")
    with GILReleased(Python()):
        var x = read_f32(src_address, count)
        var bits = narrow_bf16(x)
        for i in range(count):
            dp[i] = bits[i]
    return PythonObject(count)


def from_bf16_binding(
    dst_addr: PythonObject, src_addr: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`params`: 0 count."""
    if len(params) != 1:
        raise Error("from_bf16: params must contain 1 value (count)")
    var dp = f32_ptr(_index(dst_addr))
    var src_address = _index(src_addr)
    var count = _index(params[0])
    if count <= 0:
        raise Error("from_bf16: count must be positive, got " + String(count))
    if count > LINALG_HOST_MAX_EXTENT:
        raise Error("from_bf16: count must be at most 2^30")
    with GILReleased(Python()):
        var bits = read_u16(src_address, count)
        var x = widen_bf16(bits)
        for i in range(count):
            dp[i] = x[i]
    return PythonObject(count)

# ---------------------------------------------------------------- linalg door
# THE THREE DECOMPOSITIONS UNDER THEIR OWN NAMES (lane/linalg-public,
# 2026-09-19). Address and param contracts are repeated in each docstring
# and in python/mojolearn/_linalg_impl.py; the arithmetic is
# decomposition/host/linalg_public.mojo, which adds none of its own.


def qr_r_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """`host_qr_r(a, n_rows, n_cols)`. `addrs`: 0 a, 1 r_out (n_cols x
    n_cols). `params`: 0 n_rows, 1 n_cols. Returns n_cols."""
    if len(addrs) != 2:
        raise Error(
            "qr_r: addrs must contain 2 addresses (a, r_out), got "
            + String(len(addrs))
        )
    if len(params) != 2:
        raise Error(
            "qr_r: params must contain 2 values (n_rows, n_cols), got "
            + String(len(params))
        )
    var rp = f32_ptr(_index(addrs[1]))
    var n_rows = _index(params[0])
    var n_cols = _index(params[1])
    var a = read_f32(_index(addrs[0]), n_rows * n_cols)
    with GILReleased(Python()):
        var r = host_qr_r(a, n_rows, n_cols)
        for i in range(n_cols * n_cols):
            rp.unsafe_store(i, r[i])
        _ = r^
    _ = a^
    return PythonObject(n_cols)


def eigh_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """`host_eigh(a, n)`. `addrs`: 0 a, 1 w_out (n, ASCENDING), 2 v_out
    (n x n, eigenvector i in COLUMN i), 3 scalars_out (converged, executed).
    `params`: 0 n. Returns n."""
    if len(addrs) != 4:
        raise Error(
            "eigh: addrs must contain 4 addresses (a, w_out, v_out,"
            " scalars_out), got "
            + String(len(addrs))
        )
    if len(params) != 1:
        raise Error(
            "eigh: params must contain 1 value (n), got " + String(len(params))
        )
    var wp = f32_ptr(_index(addrs[1]))
    var vp = f32_ptr(_index(addrs[2]))
    var sp = f64_ptr(_index(addrs[3]))
    var n = _index(params[0])
    var a = read_f32(_index(addrs[0]), n * n)
    with GILReleased(Python()):
        var got = host_eigh(a, n)
        for i in range(n):
            wp.unsafe_store(i, got.w[i])
        for i in range(n * n):
            vp.unsafe_store(i, got.v[i])
        sp.unsafe_store(0, Float64(1.0) if got.converged else Float64(0.0))
        sp.unsafe_store(1, Float64(got.executed))
        _ = got^
    _ = a^
    return PythonObject(n)


def svdvals_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """`host_svdvals(a, n_rows, n_cols)`. `addrs`: 0 a, 1 s_out (n_cols,
    DESCENDING). `params`: 0 n_rows, 1 n_cols. Returns n_cols."""
    if len(addrs) != 2:
        raise Error(
            "svdvals: addrs must contain 2 addresses (a, s_out), got "
            + String(len(addrs))
        )
    if len(params) != 2:
        raise Error(
            "svdvals: params must contain 2 values (n_rows, n_cols), got "
            + String(len(params))
        )
    var sp = f32_ptr(_index(addrs[1]))
    var n_rows = _index(params[0])
    var n_cols = _index(params[1])
    var a = read_f32(_index(addrs[0]), n_rows * n_cols)
    with GILReleased(Python()):
        var s = host_svdvals(a, n_rows, n_cols)
        for i in range(n_cols):
            sp.unsafe_store(i, s[i])
        _ = s^
    _ = a^
    return PythonObject(n_cols)


@export
def PyInit__mojolearn_linalg_host() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_linalg_host")
        module.def_function[linalg_host_numeric_mode_binding]("linalg_host_numeric_mode")
        module.def_function[linalg_host_vendor_binding]("linalg_host_vendor")
        module.def_function[linalg_host_column_binding]("linalg_host_column")
        module.def_function[linalg_host_sabotage_binding]("linalg_host_sabotage")
        module.def_function[linalg_vendor_binding]("linalg_vendor")
        module.def_function[linalg_numeric_mode_binding]("linalg_numeric_mode")
        module.def_function[linalg_profile_version_binding]("linalg_profile_version")
        module.def_function[gemm_binding]("gemm")
        module.def_function[cholesky_profile_jitter_binding]("cholesky_profile_jitter")
        module.def_function[cholesky_factor_binding]("cholesky_factor")
        module.def_function[cholesky_solve_binding]("cholesky_solve")
        module.def_function[lowbit_profile_version_binding]("lowbit_profile_version")
        module.def_function[gemm_bf16_binding]("gemm_bf16")
        module.def_function[gemm_int8_binding]("gemm_int8")
        module.def_function[quantize_int8_binding]("quantize_int8")
        module.def_function[dequantize_int8_binding]("dequantize_int8")
        module.def_function[to_bf16_binding]("to_bf16")
        module.def_function[from_bf16_binding]("from_bf16")
        module.def_function[qr_r_binding]("qr_r")
        module.def_function[eigh_binding]("eigh")
        module.def_function[svdvals_binding]("svdvals")
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_linalg_host: ", error))
