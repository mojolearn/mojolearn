# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPython boundary for `mojolearn.identical.gemm.fp32.v1`, the bit-identical
FP32 matrix product.

DEVIATION 910 (the host-pointer entry this calls) and DEVIATION 912 (the
mode read-back below). Written 2026-08-24.

WHAT THIS EXTENSION IS FOR, AND WHY IT IS ITS OWN EXTENSION
------------------------------------------------------------
Everything else in this tree that Python can reach is an ESTIMATOR. This is
a numerical primitive, and its audience is anyone who needs a reproducible
matrix product, including people who will never fit a model. It gets its own
extension for the reason `bindings/build.sh` states for all of them: an
independently changing binding must not become a merge point. It also gets
its own for a second reason particular to this one -- it is the only module
in the package whose VALUE is a bit-level claim, so it must be possible to
build, gate and ship it without touching anything that is not the claim.

Arrays cross as BORROWED NumPy addresses. Every device buffer and every
`DeviceContext` lives for exactly one call and no pointer is retained past
the return; `python/mojolearn/_linalg_impl.py` keeps each array alive across
the call, which is the whole point of `_arrays.py` returning the array
beside its address.

SCALARS ARRIVE AS ONE LIST, NOT AS SEPARATE ARGUMENTS.
`PythonModuleBuilder.def_function` infers its signature from arity and stops
working somewhere around nine arguments, so buffer addresses go positionally
and every scalar goes in one `params` list. THE ORDER OF THAT LIST IS
WRITTEN OUT IN A COMMENT ON BOTH SIDES IN THE SAME WORDS. A silent reorder
here is a WRONG ANSWER, not a crash -- swap `m` and `n` on a square shape
and every call still returns a full matrix of plausible floats -- so the two
comments are the only thing standing between a caller and a quiet lie. If
you change one, change the other in the same edit.

THE GIL IS RELEASED AROUND THE DEVICE WORK, and nothing inside a
`GILReleased` block touches a `PythonObject`.

THE MODE READ-BACK IS THE POINT OF `linalg_numeric_mode` (DEVIATION 912)
------------------------------------------------------------------------
`mojolearn.identical.gemm.fp32.v1` is what the IDENTICAL build computes.
The FAST build runs the SAME kernels on the SAME path with the two pins
compiled away (`gemm/checks/gemm_identical.mojo`'s header, "WHAT
`NUMERIC_FAST` DOES HERE"), which is a correct GEMM that makes no identity
claim at all. Nothing in the ANSWER distinguishes the two on Apple at these
seams -- contract section 4.1 measured Metal fused in both modes -- so a
caller cannot tell by looking at the numbers which one it got.

`linalg_numeric_mode` is therefore the only honest way for the Python side
to know what it is holding, and it is a COMPILE-TIME answer from the binary
that actually loaded rather than a restatement of an environment variable.
`python/mojolearn/_linalg_impl.py` refuses to deliver the profile's
guarantee unless this returns 1. This mirrors `gbdt_numeric_mode` in
`bindings/_mojolearn_gbdt.mojo`, which exists for the same reason and whose
wrapper reads it once.
"""

# DEVIATION 2486: shared byte-preserving host copies.
from bindings.hostptr import f32_ptr, f64_ptr, i8_ptr, i32_ptr, read_f32, u16_ptr
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from checks.vendor import COMPILED_VENDOR

from max.gpu.host import DeviceContext

from decomposition.linalg_public_device import (
    device_eigh,
    device_qr_r,
    device_svdvals,
)
from gemm.host_entry import identical_gemm_host
from gemm.checks.gemm_lowbit import (
    LowbitWorkspace,
    bf16_narrow,
    bf16_widen,
    dequantize_rows_int8_device,
    identical_gemm_bf16w_into,
    identical_gemm_int8_into,
    quantize_rows_int8_device,
)
from gemm.host.gemm_lowbit_oracle import INT8_MAX_K, LOWBIT_PROFILE_VERSION
from gemm.host.identical_gemm import OP_NN, OP_NT, OP_TN
from max.gpu.host import DeviceBuffer
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL


def _f32_ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    return f32_ptr(addr)


def linalg_numeric_mode_binding() raises -> PythonObject:
    """THE BUILD'S TIER, as the `NUMERIC_*` code itself: 0 FAST,
    1 IDENTICAL, 2 DETERMINISTIC.

    It returned a BOOLEAN (1 for identical, else 0) until
    2026-08-29, and a boolean stopped being able to tell the truth
    the moment the middle tier existed: a DETERMINISTIC binary
    answered 0 and every reader printed it as "fast". That is the
    mislabelled measurement this read-back exists to make
    impossible. Widening it is backward compatible because the two
    old answers are the two codes they already were.

    A caller gating the CROSS-VENDOR guarantee still tests `== 1`,
    and should: 2 promises reproducibility on one device and says
    nothing about a second."""
    return PythonObject(GLOBAL_NUMERIC_MODE)


def linalg_profile_version_binding() raises -> PythonObject:
    """The MAJOR VERSION of the GEMM profile this binary implements: 1, for
    `mojolearn.identical.gemm.fp32.v1`.

    An Int and not the profile string, because the profile name has no Mojo
    constant anywhere in this tree today -- it lives in
    `gemm/IDENTICAL_FP32_CONTRACT.md` and in docstrings -- so a string
    returned here would be a literal typed twice rather than a fact read
    once, and Int is a conversion the other bindings already prove.

    **The version is part of the claim**, contract preamble: *"a bit-identity
    claim with no version on it is a claim about whichever revision the
    reader happens to be holding."* The leaf rule of section 7.1 and the fold
    topology of section 7.2 are what this number is about. Changing either
    creates v2 and this returns 2; it does not amend v1. The wrapper
    cross-checks it against its own constant, so a stale `.so` beside a newer
    wrapper is a loud error rather than a mislabeled answer."""
    return PythonObject(1)


def gemm_binding(
    c_addr: PythonObject,
    a_addr: PythonObject,
    b_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`C = op(A) . op(B)` under `mojolearn.identical.gemm.fp32.v1`.
    Returns the number of output cells written, `m * n`.

    `params` is, in this exact order (mirrored word for word in
    `python/mojolearn/_linalg_impl.py`):

        0  m       rows of C
        1  n       columns of C
        2  k       the contracted extent
        3  op      0 = OP_NN, 1 = OP_NT, 2 = OP_TN

    and the row-major shapes those name are contract section 0.1:

        OP_NN   C = A . B      A is m x k,  B is k x n
        OP_NT   C = A . B^T    A is m x k,  B is n x k
        OP_TN   C = A^T . B    A is k x m,  B is k x n

    **THE OUTPUT ADDRESS COMES FIRST**, which is the opposite of
    `_mojolearn_estimators.mojo`'s inputs-then-outputs habit and deliberate:
    it mirrors `identical_gemm(ctx, c, a, b, ...)`, the certified function
    this ultimately calls, so the two argument lists read the same way down
    the page. Swapping `c_addr` with `a_addr` writes the device's output over
    the caller's input matrix, which is memory corruption in the caller's
    process and not an exception here.

    `c_addr` is written, `a_addr` and `b_addr` are read. All three are
    float32, row-major and fully contiguous (contract section 2); no leading
    dimension, no stride, no sub-view. `gemm/host_entry.mojo` refuses a
    non-positive extent and an unknown `op` by name."""
    if len(params) != 4:
        raise Error(
            "gemm: params must contain 4 values (m, n, k, op), got "
            + String(len(params))
        )
    var cp = _f32_ptr(Int(py=c_addr))
    var ap = _f32_ptr(Int(py=a_addr))
    var bp = _f32_ptr(Int(py=b_addr))
    var m = Int(py=params[0])
    var n = Int(py=params[1])
    var k = Int(py=params[2])
    var op = Int(py=params[3])
    with GILReleased(Python()):
        var ctx = DeviceContext()
        identical_gemm_host(ctx, cp, ap, bp, m, n, k, op)
        # DEVIATION 3010: the buffers above are gone; DRAIN the frees they
        # enqueued before this block's end destroys the context. Without it
        # the MAX runtime allocator's lock is left held and the NEXT
        # enqueue_create_buffer in the process blocks for ever (DEVIATION
        # 2520's mechanism; the native backtrace of the `transformer-bf16w`
        # hang blames exactly this library). Host-side drain, no arithmetic.
        ctx.synchronize()
    return PythonObject(m * n)


def linalg_vendor_binding() raises -> PythonObject:
    """THE ACCELERATOR API THIS BINARY WAS COMPILED FOR: 'metal', 'cuda',
    'hip' or 'none'. A compile-time constant folded in from
    `checks/vendor.mojo`, the same shape as the tier read-back
    (`gbdt_numeric_mode`): the answer comes from the binary that actually
    loaded, never from the directory it sat in or from the environment.
    `python/mojolearn/_backend.py` refuses at import when this disagrees
    with the vendor directory the set was loaded from."""
    return PythonObject(String(COMPILED_VENDOR))



# ===========================================================================
# THE LOW-BIT PROFILES (lane/identical-lowbit-inference, 2026-09-17)
# gemm/IDENTICAL_LOWBIT_CONTRACT.md. Same conventions as `gemm_binding`:
# output address first, scalars in one `params` list whose order is written
# out on both sides in the same words, the GIL released around the device.
# ===========================================================================


def _dev_f32(ctx: DeviceContext, addr: Int, count: Int) raises -> DeviceBuffer[DType.float32]:
    var n = count
    if n < 1:
        n = 1
    var d = ctx.enqueue_create_buffer[DType.float32](n)
    if count > 0:
        ctx.enqueue_copy(dst_buf=d, src_ptr=f32_ptr(addr))
    return d^


def _dev_u16(ctx: DeviceContext, addr: Int, count: Int) raises -> DeviceBuffer[DType.uint16]:
    var n = count
    if n < 1:
        n = 1
    var d = ctx.enqueue_create_buffer[DType.uint16](n)
    if count > 0:
        ctx.enqueue_copy(dst_buf=d, src_ptr=u16_ptr(addr))
    return d^


def _dev_i8(ctx: DeviceContext, addr: Int, count: Int) raises -> DeviceBuffer[DType.int8]:
    var n = count
    if n < 1:
        n = 1
    var d = ctx.enqueue_create_buffer[DType.int8](n)
    if count > 0:
        ctx.enqueue_copy(dst_buf=d, src_ptr=i8_ptr(addr))
    return d^


def _dev_i32(ctx: DeviceContext, addr: Int, count: Int) raises -> DeviceBuffer[DType.int32]:
    var n = count
    if n < 1:
        n = 1
    var d = ctx.enqueue_create_buffer[DType.int32](n)
    if count > 0:
        ctx.enqueue_copy(dst_buf=d, src_ptr=i32_ptr(addr))
    return d^


def _refuse_lowbit_shape(m: Int, n: Int, k: Int, op: Int, who: String) raises:
    if m <= 0 or n <= 0 or k <= 0:
        raise Error(
            who + ": m, n and k must all be positive, got m=" + String(m)
            + " n=" + String(n) + " k=" + String(k)
        )
    if op != OP_NN and op != OP_NT and op != OP_TN:
        raise Error(who + ": op must be 0 (OP_NN), 1 (OP_NT) or 2 (OP_TN), got " + String(op))


def lowbit_profile_version_binding() raises -> PythonObject:
    """The MAJOR VERSION of the two low-bit profiles this binary implements:
    1, for `mojolearn.identical.gemm.bf16f32.v1` and
    `mojolearn.identical.gemm.int8i32.v1`."""
    return PythonObject(LOWBIT_PROFILE_VERSION)


def gemm_bf16_binding(
    c_addr: PythonObject,
    a_addr: PythonObject,
    b_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`C = op(A) . op(B)` under `mojolearn.identical.gemm.bf16f32.v1`.
    `B` is bf16 bits in a uint16 buffer; `A` is float32, or bf16 bits when
    `params[4]` is 1. `C` is float32. Returns `m * n`.

    `params` is, in this exact order (mirrored word for word in
    `python/mojolearn/_linalg_impl.py` and in the host binding):

        0  m         rows of C
        1  n         columns of C
        2  k         the contracted extent
        3  op        0 = OP_NN, 1 = OP_NT, 2 = OP_TN
        4  a_bf16    1 when A is bf16 bits, 0 when A is float32
    """
    if len(params) != 5:
        raise Error("gemm_bf16: params must contain 5 values (m, n, k, op, a_bf16), got " + String(len(params)))
    var c_address = Int(py=c_addr)
    var a_address = Int(py=a_addr)
    var b_address = Int(py=b_addr)
    var m = Int(py=params[0])
    var n = Int(py=params[1])
    var k = Int(py=params[2])
    var op = Int(py=params[3])
    var a_bf16 = Int(py=params[4]) != 0
    _refuse_lowbit_shape(m, n, k, op, String("gemm_bf16"))
    with GILReleased(Python()):
        var ctx = DeviceContext()
        var db = _dev_u16(ctx, b_address, n * k)
        var dc = ctx.enqueue_create_buffer[DType.float32](m * n)
        var work = LowbitWorkspace(ctx)
        if a_bf16:
            var da_bits = _dev_u16(ctx, a_address, m * k)
            var da = ctx.enqueue_create_buffer[DType.float32](m * k)
            bf16_widen(ctx, da, da_bits, m * k)
            identical_gemm_bf16w_into(ctx, dc, da, db, work, m, n, k, op)
            ctx.synchronize()
            _ = da_bits
            _ = da
        else:
            var da2 = _dev_f32(ctx, a_address, m * k)
            identical_gemm_bf16w_into(ctx, dc, da2, db, work, m, n, k, op)
            ctx.synchronize()
            _ = da2
        ctx.enqueue_copy(dst_ptr=f32_ptr(c_address), src_buf=dc)
        ctx.synchronize()
        _ = db^
        _ = dc^
        _ = work^
        # DEVIATION 3010: the buffers above are gone; DRAIN the frees they
        # enqueued before this block's end destroys the context. Without it
        # the MAX runtime allocator's lock is left held and the NEXT
        # enqueue_create_buffer in the process blocks for ever (DEVIATION
        # 2520's mechanism; the native backtrace of the `transformer-bf16w`
        # hang blames exactly this library). Host-side drain, no arithmetic.
        ctx.synchronize()
    return PythonObject(m * n)


def gemm_int8_binding(
    c_addr: PythonObject,
    qa_addr: PythonObject,
    ea_addr: PythonObject,
    qb_addr: PythonObject,
    eb_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`C[m x n] = Q_a[m x k] . Q_b[n x k]^T` dequantized, under
    `mojolearn.identical.gemm.int8i32.v1` (OP_NT only). `qa`, `qb` are int8
    codes; `ea` (m entries) and `eb` (n entries) are int32 row exponents; `C`
    is float32. Returns `m * n`. `params` is `[m, n, k]`."""
    if len(params) != 3:
        raise Error("gemm_int8: params must contain 3 values (m, n, k), got " + String(len(params)))
    var c_address = Int(py=c_addr)
    var qa_address = Int(py=qa_addr)
    var ea_address = Int(py=ea_addr)
    var qb_address = Int(py=qb_addr)
    var eb_address = Int(py=eb_addr)
    var m = Int(py=params[0])
    var n = Int(py=params[1])
    var k = Int(py=params[2])
    _refuse_lowbit_shape(m, n, k, OP_NT, String("gemm_int8"))
    if k > INT8_MAX_K:
        raise Error("gemm_int8: k must be at most " + String(INT8_MAX_K) + " (contract L-7), got " + String(k))
    with GILReleased(Python()):
        var ctx = DeviceContext()
        var dqa = _dev_i8(ctx, qa_address, m * k)
        var dea = _dev_i32(ctx, ea_address, m)
        var dqb = _dev_i8(ctx, qb_address, n * k)
        var deb = _dev_i32(ctx, eb_address, n)
        var dc = ctx.enqueue_create_buffer[DType.float32](m * n)
        identical_gemm_int8_into(ctx, dc, dqa, dea, dqb, deb, m, n, k)
        ctx.synchronize()
        ctx.enqueue_copy(dst_ptr=f32_ptr(c_address), src_buf=dc)
        ctx.synchronize()
        _ = dqa^
        _ = dea^
        _ = dqb^
        _ = deb^
        _ = dc^
        # DEVIATION 3010: the buffers above are gone; DRAIN the frees they
        # enqueued before this block's end destroys the context. Without it
        # the MAX runtime allocator's lock is left held and the NEXT
        # enqueue_create_buffer in the process blocks for ever (DEVIATION
        # 2520's mechanism; the native backtrace of the `transformer-bf16w`
        # hang blames exactly this library). Host-side drain, no arithmetic.
        ctx.synchronize()
    return PythonObject(m * n)


def quantize_int8_binding(
    q_addr: PythonObject,
    e_addr: PythonObject,
    x_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """Row-wise int8 codes (`q`, rows x cols) and int32 exponents (`e`, rows)
    of a float32 matrix, by contract L-3 and L-4, on the device. `params`
    is `[rows, cols]`. Returns `rows * cols`."""
    if len(params) != 2:
        raise Error("quantize_int8: params must contain 2 values (rows, cols), got " + String(len(params)))
    var q_address = Int(py=q_addr)
    var e_address = Int(py=e_addr)
    var x_address = Int(py=x_addr)
    var rows = Int(py=params[0])
    var cols = Int(py=params[1])
    if rows <= 0 or cols <= 0:
        raise Error("quantize_int8: rows and cols must be positive, got " + String(rows) + " x " + String(cols))
    with GILReleased(Python()):
        var ctx = DeviceContext()
        var dx = _dev_f32(ctx, x_address, rows * cols)
        var dq = ctx.enqueue_create_buffer[DType.int8](rows * cols)
        var de = ctx.enqueue_create_buffer[DType.int32](rows)
        quantize_rows_int8_device(ctx, dq, de, dx, rows, cols)
        ctx.synchronize()
        ctx.enqueue_copy(dst_ptr=i8_ptr(q_address), src_buf=dq)
        ctx.enqueue_copy(dst_ptr=i32_ptr(e_address), src_buf=de)
        ctx.synchronize()
        _ = dx^
        _ = dq^
        _ = de^
        # DEVIATION 3010: the buffers above are gone; DRAIN the frees they
        # enqueued before this block's end destroys the context. Without it
        # the MAX runtime allocator's lock is left held and the NEXT
        # enqueue_create_buffer in the process blocks for ever (DEVIATION
        # 2520's mechanism; the native backtrace of the `transformer-bf16w`
        # hang blames exactly this library). Host-side drain, no arithmetic.
        ctx.synchronize()
    return PythonObject(rows * cols)


def dequantize_int8_binding(
    y_addr: PythonObject,
    q_addr: PythonObject,
    e_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`y = q * 2^e` row by row, exact, on the device. `params` is
    `[rows, cols]`. Returns `rows * cols`."""
    if len(params) != 2:
        raise Error("dequantize_int8: params must contain 2 values (rows, cols), got " + String(len(params)))
    var y_address = Int(py=y_addr)
    var q_address = Int(py=q_addr)
    var e_address = Int(py=e_addr)
    var rows = Int(py=params[0])
    var cols = Int(py=params[1])
    if rows <= 0 or cols <= 0:
        raise Error("dequantize_int8: rows and cols must be positive, got " + String(rows) + " x " + String(cols))
    with GILReleased(Python()):
        var ctx = DeviceContext()
        var dq = _dev_i8(ctx, q_address, rows * cols)
        var de = _dev_i32(ctx, e_address, rows)
        var dy = ctx.enqueue_create_buffer[DType.float32](rows * cols)
        dequantize_rows_int8_device(ctx, dy, dq, de, rows, cols)
        ctx.synchronize()
        ctx.enqueue_copy(dst_ptr=f32_ptr(y_address), src_buf=dy)
        ctx.synchronize()
        _ = dq^
        _ = de^
        _ = dy^
        # DEVIATION 3010: the buffers above are gone; DRAIN the frees they
        # enqueued before this block's end destroys the context. Without it
        # the MAX runtime allocator's lock is left held and the NEXT
        # enqueue_create_buffer in the process blocks for ever (DEVIATION
        # 2520's mechanism; the native backtrace of the `transformer-bf16w`
        # hang blames exactly this library). Host-side drain, no arithmetic.
        ctx.synchronize()
    return PythonObject(rows * cols)


def to_bf16_binding(
    dst_addr: PythonObject, src_addr: PythonObject, params: PythonObject
) raises -> PythonObject:
    """float32 to bf16 bits by contract L-2 (flush, then round to nearest
    even), on the device. `params` is `[count]`. Returns `count`."""
    if len(params) != 1:
        raise Error("to_bf16: params must contain 1 value (count)")
    var dst_address = Int(py=dst_addr)
    var src_address = Int(py=src_addr)
    var count = Int(py=params[0])
    if count <= 0:
        raise Error("to_bf16: count must be positive, got " + String(count))
    with GILReleased(Python()):
        var ctx = DeviceContext()
        var dsrc = _dev_f32(ctx, src_address, count)
        var ddst = ctx.enqueue_create_buffer[DType.uint16](count)
        bf16_narrow(ctx, ddst, dsrc, count)
        ctx.synchronize()
        ctx.enqueue_copy(dst_ptr=u16_ptr(dst_address), src_buf=ddst)
        ctx.synchronize()
        _ = dsrc^
        _ = ddst^
        # DEVIATION 3010: the buffers above are gone; DRAIN the frees they
        # enqueued before this block's end destroys the context. Without it
        # the MAX runtime allocator's lock is left held and the NEXT
        # enqueue_create_buffer in the process blocks for ever (DEVIATION
        # 2520's mechanism; the native backtrace of the `transformer-bf16w`
        # hang blames exactly this library). Host-side drain, no arithmetic.
        ctx.synchronize()
    return PythonObject(count)


def from_bf16_binding(
    dst_addr: PythonObject, src_addr: PythonObject, params: PythonObject
) raises -> PythonObject:
    """bf16 bits to float32 by contract L-1 (exact), on the device.
    `params` is `[count]`. Returns `count`."""
    if len(params) != 1:
        raise Error("from_bf16: params must contain 1 value (count)")
    var dst_address = Int(py=dst_addr)
    var src_address = Int(py=src_addr)
    var count = Int(py=params[0])
    if count <= 0:
        raise Error("from_bf16: count must be positive, got " + String(count))
    with GILReleased(Python()):
        var ctx = DeviceContext()
        var dsrc = _dev_u16(ctx, src_address, count)
        var ddst = ctx.enqueue_create_buffer[DType.float32](count)
        bf16_widen(ctx, ddst, dsrc, count)
        ctx.synchronize()
        ctx.enqueue_copy(dst_ptr=f32_ptr(dst_address), src_buf=ddst)
        ctx.synchronize()
        _ = dsrc^
        _ = ddst^
        # DEVIATION 3010: the buffers above are gone; DRAIN the frees they
        # enqueued before this block's end destroys the context. Without it
        # the MAX runtime allocator's lock is left held and the NEXT
        # enqueue_create_buffer in the process blocks for ever (DEVIATION
        # 2520's mechanism; the native backtrace of the `transformer-bf16w`
        # hang blames exactly this library). Host-side drain, no arithmetic.
        ctx.synchronize()
    return PythonObject(count)

# ---------------------------------------------------------------- linalg door
# THE THREE DECOMPOSITIONS UNDER THEIR OWN NAMES, ON THE DEVICE (2026-09-19).
#
# THE ADDRESS AND PARAM CONTRACTS BELOW ARE THE HOST BINDING'S, WORD FOR
# WORD. `bindings/_mojolearn_linalg_host.mojo` carries the same three names
# over `decomposition/host/linalg_public.mojo`, and
# `python/mojolearn/_linalg_impl.py` calls whichever of the two this install
# has: the device binding on a GPU box, the host binding on a CPU-only one,
# through ONE `_door()` and under ONE set of public names. The two
# signatures must therefore stay identical -- a caller cannot see which
# binding answered, so a parameter that means one thing here and another
# there is a wrong answer with no symptom.
#
# The arithmetic is `decomposition/linalg_public_device.mojo`, which adds
# none of its own: it launches the kernels `PCA(svd_solver='full')` launches
# and shares the host twin's ordering functions.


def qr_r_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """`device_qr_r(a, n_rows, n_cols)`. `addrs`: 0 a, 1 r_out (n_cols x
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
    var rp = f32_ptr(Int(py=addrs[1]))
    var n_rows = Int(py=params[0])
    var n_cols = Int(py=params[1])
    var a = read_f32(Int(py=addrs[0]), max(0, n_rows * n_cols))
    with GILReleased(Python()):
        var r = device_qr_r(a, n_rows, n_cols)
        for i in range(n_cols * n_cols):
            rp.unsafe_store(i, r[i])
        _ = r^
    _ = a^
    return PythonObject(n_cols)


def eigh_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """`device_eigh(a, n)`. `addrs`: 0 a, 1 w_out (n, ASCENDING), 2 v_out
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
    var wp = f32_ptr(Int(py=addrs[1]))
    var vp = f32_ptr(Int(py=addrs[2]))
    var sp = f64_ptr(Int(py=addrs[3]))
    var n = Int(py=params[0])
    var a = read_f32(Int(py=addrs[0]), max(0, n * n))
    with GILReleased(Python()):
        var got = device_eigh(a, n)
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
    """`device_svdvals(a, n_rows, n_cols)`. `addrs`: 0 a, 1 s_out (n_cols,
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
    var sp = f32_ptr(Int(py=addrs[1]))
    var n_rows = Int(py=params[0])
    var n_cols = Int(py=params[1])
    var a = read_f32(Int(py=addrs[0]), max(0, n_rows * n_cols))
    with GILReleased(Python()):
        var s = device_svdvals(a, n_rows, n_cols)
        for i in range(n_cols):
            sp.unsafe_store(i, s[i])
        _ = s^
    _ = a^
    return PythonObject(n_cols)


@export
def PyInit__mojolearn_linalg() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_linalg")
        m.def_function[linalg_vendor_binding]("linalg_vendor")
        m.def_function[gemm_binding]("gemm")
        m.def_function[linalg_numeric_mode_binding]("linalg_numeric_mode")
        m.def_function[linalg_profile_version_binding](
            "linalg_profile_version"
        )
        m.def_function[lowbit_profile_version_binding]("lowbit_profile_version")
        m.def_function[gemm_bf16_binding]("gemm_bf16")
        m.def_function[gemm_int8_binding]("gemm_int8")
        m.def_function[quantize_int8_binding]("quantize_int8")
        m.def_function[dequantize_int8_binding]("dequantize_int8")
        m.def_function[to_bf16_binding]("to_bf16")
        m.def_function[from_bf16_binding]("from_bf16")
        m.def_function[qr_r_binding]("qr_r")
        m.def_function[eigh_binding]("eigh")
        m.def_function[svdvals_binding]("svdvals")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_linalg: ", e))
