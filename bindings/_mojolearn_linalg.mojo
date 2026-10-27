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
from bindings.device_plan_binding import device_plan_enter_binding, device_plan_exit_binding, device_plan_log_binding
from core.hostptr import f32_ptr, f64_ptr, i8_ptr, i32_ptr, read_f32, u16_ptr
from std.os import abort, getenv
from std.time import perf_counter_ns
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from providers.python.neural_descriptor_pyapi import neural_descriptor_capsule
from gemm.linalg_dispatch import LINALG_NATIVE_TABLE
from gemm.linalg_native import _f32_ptr, _dev_f32, _dev_u16, _dev_i8, _dev_i32, _refuse_lowbit_shape, _LowbitSlot, _LOWBIT_POOL, _lowbit_run, _lowbit_tick, gemm_native, gemm_bf16_native, gemm_int8_native, quantize_int8_native, dequantize_int8_native, gemm_int15_native, quantize_int15_native, dequantize_int15_native, to_bf16_native, from_bf16_native, from_f16_native

from checks.vendor import COMPILED_VENDOR

from max.gpu.host import DeviceContext
from core.neural_context import process_ctx
from checks.numerics import GLOBAL_NUMERIC_MODE as _DEVCTX_MODE, NUMERIC_IDENTICAL as _DEVCTX_IDENTICAL

#: This binding's ONE process-lifetime DeviceContext (core/neural_context.mojo,
#: lane/devctx-lifetime): a context per call exhausts Metal command queues.
comptime _DEVCTX_SLOT = "MojoLinalgContextIdentical" if _DEVCTX_MODE == _DEVCTX_IDENTICAL else "MojoLinalgContextFast"


from x_decomp.cells import F32Ptr
from x_decomp.device import DevExec
from decomposition.linalg_public_device import (
    device_qr_r,
    device_svdvals,
)
from gemm.host_entry import identical_gemm_host
from gemm.host_transport import (
    GemmHostLease,
    ROLE_A,
    ROLE_B,
    ROLE_C,
    ROLE_X,
    gemm_down_f32,
    gemm_up_f32,
    gemm_up_u16,
)
from std.ffi import _Global
from gemm.checks.gemm_lowbit import (
    LowbitWorkspace,
    bf16_narrow,
    bf16_widen,
    dequantize_rows_int8_device,
    identical_gemm_bf16w_into,
    identical_gemm_int8_into,
    quantize_rows_int8_device,
)
from gemm.contract import INT8_MAX_K, LOWBIT_PROFILE_VERSION
from gemm.contract import GEMM_NUMERICAL_PROFILE, GEMM_NUMERICAL_PROFILE_VERSION
from gemm.afn_apple_fast import (
    AFN_EPI_BIAS,
    AFN_EPI_BIAS_GELU,
    AFN_EPI_BIAS_RESID,
    AFN_EPI_BIAS_SILU,
    AFN_EPI_NONE,
    AFN_GEMM_BF16_MMA,
    AFN_GEMM_EPILOGUE,
    afn_gemm_bf16_bits_into,
    afn_gemm_fused_into,
)
from decomposition.linalg_door_impl import from_f16_device, qr_r_device_into, svdvals_device_into
from std.gpu import block_dim, block_idx, thread_idx
from gemm.checks.gemm_int15 import (
    Int15QuantWorkspace,
    dequantize_planes_int15_device,
    identical_gemm_int15_planes_into,
    quantize_planes_int15_parallel_device,
)
from gemm.contract import INT15_MAX_K, INT15_PROFILE_VERSION, OP_NN, OP_NT, OP_TN
from max.gpu.host import DeviceBuffer
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from bindings.apple_fast_bf16_public import (
    initialize_fast_bf16_boundary,
    fast_bf16_available_binding,
    fast_to_bf16_binding,
    fast_from_bf16_binding,
    fast_matmul_bf16_binding,
    public_fast_to_bf16_binding,
    public_fast_from_bf16_binding,
)



def linalg_numeric_mode_binding() raises -> PythonObject:
    """THE BUILD'S TIER, as the `NUMERIC_*` code itself: 0 FAST,
    1 IDENTICAL."""
    return PythonObject(GLOBAL_NUMERIC_MODE)


def linalg_profile_version_binding() raises -> PythonObject:
    """The actual shared arithmetic version, including opt-in leaf experiments."""
    return PythonObject(GEMM_NUMERICAL_PROFILE_VERSION)


def linalg_numerical_profile_binding() raises -> PythonObject:
    """Exact contract name; an experimental profile is not certified v1."""
    return PythonObject(String(GEMM_NUMERICAL_PROFILE))


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
        raise Error("gemm: expected 4 scalar parameters")
    var native_c_addr = Int(py=c_addr)
    var native_a_addr = Int(py=a_addr)
    var native_b_addr = Int(py=b_addr)
    var native_params = List[Int]()
    for i in range(4):  # small-loop(4 scalar descriptor words)
        native_params.append(Int(py=params[i]))
    var result = 0
    with GILReleased(Python()):
        result = gemm_native(native_c_addr, native_a_addr, native_b_addr, native_params)
    return PythonObject(result)


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
        raise Error("gemm_bf16: expected 5 scalar parameters")
    var native_c_addr = Int(py=c_addr)
    var native_a_addr = Int(py=a_addr)
    var native_b_addr = Int(py=b_addr)
    var native_params = List[Int]()
    for i in range(5):  # small-loop(5 scalar descriptor words)
        native_params.append(Int(py=params[i]))
    var result = 0
    with GILReleased(Python()):
        result = gemm_bf16_native(native_c_addr, native_a_addr, native_b_addr, native_params)
    return PythonObject(result)


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
        raise Error("gemm_int8: expected 3 scalar parameters")
    var native_c_addr = Int(py=c_addr)
    var native_qa_addr = Int(py=qa_addr)
    var native_ea_addr = Int(py=ea_addr)
    var native_qb_addr = Int(py=qb_addr)
    var native_eb_addr = Int(py=eb_addr)
    var native_params = List[Int]()
    for i in range(3):  # small-loop(3 scalar descriptor words)
        native_params.append(Int(py=params[i]))
    var result = 0
    with GILReleased(Python()):
        result = gemm_int8_native(native_c_addr, native_qa_addr, native_ea_addr, native_qb_addr, native_eb_addr, native_params)
    return PythonObject(result)


def gemm_fused_binding(
    c_addr: PythonObject,
    a_addr: PythonObject,
    b_addr: PythonObject,
    bias_addr: PythonObject,
    resid_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """lane/apple-fast-neural-gemm (2026-10-03), registered only under FAST
    on Apple with MOJOLEARN_AFN_GEMM_EPILOGUE: `C = epi(op(A) . op(B))` with
    the epilogue in the store (gemm/afn_apple_fast.mojo::afn_gemm_fused_into).
    `bias` holds `n` floats; `resid` holds `m * n` floats and is read only by
    epilogue 2. Returns `m * n`. `params` is, in this exact order (mirrored in
    `python/mojolearn/linalg_fast.py`):

        0  m, 1  n, 2  k, 3  op (0 NN, 1 NT, 2 TN)
        4  epi    0 none, 1 bias, 2 bias + residual, 3 bias + SiLU,
                  4 bias + GELU (exact erf)
    """
    if len(params) != 5:
        raise Error("gemm_fused: params must contain 5 values (m, n, k, op, epi), got " + String(len(params)))
    var c_address = Int(py=c_addr)
    var a_address = Int(py=a_addr)
    var b_address = Int(py=b_addr)
    var bias_address = Int(py=bias_addr)
    var resid_address = Int(py=resid_addr)
    var m = Int(py=params[0])
    var n = Int(py=params[1])
    var k = Int(py=params[2])
    var op = Int(py=params[3])
    var epi = Int(py=params[4])
    if m <= 0 or n <= 0 or k <= 0:
        raise Error("gemm_fused: m, n and k must all be positive, got m=" + String(m) + " n=" + String(n) + " k=" + String(k))
    if op != OP_NN and op != OP_NT and op != OP_TN:
        raise Error("gemm_fused: op must be 0 (OP_NN), 1 (OP_NT) or 2 (OP_TN), got " + String(op))
    if epi < AFN_EPI_NONE or epi > AFN_EPI_BIAS_GELU:
        raise Error("gemm_fused: epi must be 0..4, got " + String(epi))
    comptime if not AFN_GEMM_EPILOGUE:
        raise Error("gemm_fused: this binding was built without MOJOLEARN_AFN_GEMM_EPILOGUE")
    else:
        with GILReleased(Python()):
            var ctx = process_ctx[_DEVCTX_SLOT]()
            var da = _dev_f32(ctx, a_address, m * k)
            var db = _dev_f32(ctx, b_address, n * k)
            var dbias = _dev_f32(ctx, bias_address, n if epi != AFN_EPI_NONE else 0)
            var dres = _dev_f32(ctx, resid_address, m * n if epi == AFN_EPI_BIAS_RESID else 0)
            var dc = ctx.enqueue_create_buffer[DType.float32](m * n)
            if not afn_gemm_fused_into(ctx, dc, da, db, dbias, dres, m, n, k, op, epi):
                raise Error("gemm_fused: the fused route declined the shape")
            ctx.enqueue_copy(dst_ptr=f32_ptr(c_address), src_buf=dc)
            ctx.synchronize()
            _ = da^
            _ = db^
            _ = dbias^
            _ = dres^
            _ = dc^
            # DEVIATION 3010: drain the frees before the block ends.
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
        raise Error("quantize_int8: expected 2 scalar parameters")
    var native_q_addr = Int(py=q_addr)
    var native_e_addr = Int(py=e_addr)
    var native_x_addr = Int(py=x_addr)
    var native_params = List[Int]()
    for i in range(2):  # small-loop(2 scalar descriptor words)
        native_params.append(Int(py=params[i]))
    var result = 0
    with GILReleased(Python()):
        result = quantize_int8_native(native_q_addr, native_e_addr, native_x_addr, native_params)
    return PythonObject(result)


def dequantize_int8_binding(
    y_addr: PythonObject,
    q_addr: PythonObject,
    e_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`y = q * 2^e` row by row, exact, on the device. `params` is
    `[rows, cols]`. Returns `rows * cols`."""

    if len(params) != 2:
        raise Error("dequantize_int8: expected 2 scalar parameters")
    var native_y_addr = Int(py=y_addr)
    var native_q_addr = Int(py=q_addr)
    var native_e_addr = Int(py=e_addr)
    var native_params = List[Int]()
    for i in range(2):  # small-loop(2 scalar descriptor words)
        native_params.append(Int(py=params[i]))
    var result = 0
    with GILReleased(Python()):
        result = dequantize_int8_native(native_y_addr, native_q_addr, native_e_addr, native_params)
    return PythonObject(result)


# ===========================================================================
# THE FIFTEEN-BIT PROFILE (lane/lowbit-int15, 2026-09-29)
# gemm/IDENTICAL_LOWBIT_CONTRACT.md section 6. An operand crosses the
# boundary as its two int8 PLANES (clause W-3) and its int32 row exponents:
# the package's Array has no int16, and the planes are what the product
# reads on every column.
# ===========================================================================


def int15_profile_version_binding() raises -> PythonObject:
    """The MAJOR VERSION of the fifteen-bit profile this binary
    implements: 1, for `mojolearn.identical.gemm.int15i64.v1`."""
    return PythonObject(INT15_PROFILE_VERSION)


def gemm_int15_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """`C[m x n] = Q_a[m x k] . Q_b[n x k]^T` dequantized, under
    `mojolearn.identical.gemm.int15i64.v1` (OP_NT only). Returns `m * n`.

    `addrs` is, in this exact order (mirrored word for word in
    `python/mojolearn/_linalg_impl.py` and in the host binding):

        0  c            float32, m x n
        1  a hi         int8, m x k
        2  a lo         int8, m x k
        3  a exponents  int32, m
        4  b hi         int8, n x k
        5  b lo         int8, n x k
        6  b exponents  int32, n

    `params` is `[m, n, k]`."""

    if len(params) != 3:
        raise Error("gemm_int15: expected 3 scalar parameters")
    if len(addrs) != 7:
        raise Error("gemm_int15: expected 7 buffer addresses")
    var native_addrs = List[Int]()
    for i in range(7):  # small-loop(7 buffer descriptor words)
        native_addrs.append(Int(py=addrs[i]))
    var native_params = List[Int]()
    for i in range(3):  # small-loop(3 scalar descriptor words)
        native_params.append(Int(py=params[i]))
    var result = 0
    with GILReleased(Python()):
        result = gemm_int15_native(native_addrs, native_params)
    return PythonObject(result)


def quantize_int15_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """Row-wise planes and int32 exponents of a float32 matrix, by clauses
    W-1 to W-3, on the device (the parallel schedule). `addrs`: 0 hi, 1 lo,
    2 exponents, 3 x. `params` is `[rows, cols]`. Returns `rows * cols`."""

    if len(params) != 2:
        raise Error("quantize_int15: expected 2 scalar parameters")
    if len(addrs) != 4:
        raise Error("quantize_int15: expected 4 buffer addresses")
    var native_addrs = List[Int]()
    for i in range(4):  # small-loop(4 buffer descriptor words)
        native_addrs.append(Int(py=addrs[i]))
    var native_params = List[Int]()
    for i in range(2):  # small-loop(2 scalar descriptor words)
        native_params.append(Int(py=params[i]))
    var result = 0
    with GILReleased(Python()):
        result = quantize_int15_native(native_addrs, native_params)
    return PythonObject(result)


def dequantize_int15_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """`y = (hi * 128 + lo) * 2^e` row by row, on the device. `addrs`:
    0 y, 1 hi, 2 lo, 3 exponents. `params` is `[rows, cols]`."""

    if len(params) != 2:
        raise Error("dequantize_int15: expected 2 scalar parameters")
    if len(addrs) != 4:
        raise Error("dequantize_int15: expected 4 buffer addresses")
    var native_addrs = List[Int]()
    for i in range(4):  # small-loop(4 buffer descriptor words)
        native_addrs.append(Int(py=addrs[i]))
    var native_params = List[Int]()
    for i in range(2):  # small-loop(2 scalar descriptor words)
        native_params.append(Int(py=params[i]))
    var result = 0
    with GILReleased(Python()):
        result = dequantize_int15_native(native_addrs, native_params)
    return PythonObject(result)


def to_bf16_binding(
    dst_addr: PythonObject, src_addr: PythonObject, params: PythonObject
) raises -> PythonObject:
    """float32 to bf16 bits by contract L-2 (flush, then round to nearest
    even), on the device. `params` is `[count]`. Returns `count`."""

    if len(params) != 1:
        raise Error("to_bf16: expected 1 scalar parameters")
    var native_dst_addr = Int(py=dst_addr)
    var native_src_addr = Int(py=src_addr)
    var native_params = List[Int]()
    for i in range(1):  # small-loop(1 scalar descriptor words)
        native_params.append(Int(py=params[i]))
    var result = 0
    with GILReleased(Python()):
        result = to_bf16_native(native_dst_addr, native_src_addr, native_params)
    return PythonObject(result)


def from_bf16_binding(
    dst_addr: PythonObject, src_addr: PythonObject, params: PythonObject
) raises -> PythonObject:
    """bf16 bits to float32 by contract L-1 (exact), on the device.
    `params` is `[count]`. Returns `count`."""

    if len(params) != 1:
        raise Error("from_bf16: expected 1 scalar parameters")
    var native_dst_addr = Int(py=dst_addr)
    var native_src_addr = Int(py=src_addr)
    var native_params = List[Int]()
    for i in range(1):  # small-loop(1 scalar descriptor words)
        native_params.append(Int(py=params[i]))
    var result = 0
    with GILReleased(Python()):
        result = from_bf16_native(native_dst_addr, native_src_addr, native_params)
    return PythonObject(result)



def from_f16_binding(
    dst_addr: PythonObject, src_addr: PythonObject, params: PythonObject
) raises -> PythonObject:
    """IEEE float16 bits to float32, exact, by bit construction
    (`gemm/contract.mojo::f16_bits_to_f32`), on the device. The safetensors
    float16 reader's widening (pyglue-text-io, Oct 3). `params` is
    `[count]`. Returns `count`."""

    if len(params) != 1:
        raise Error("from_f16: expected 1 scalar parameters")
    var native_dst_addr = Int(py=dst_addr)
    var native_src_addr = Int(py=src_addr)
    var native_params = List[Int]()
    for i in range(1):  # small-loop(1 scalar descriptor words)
        native_params.append(Int(py=params[i]))
    var result = 0
    with GILReleased(Python()):
        result = from_f16_native(native_dst_addr, native_src_addr, native_params)
    return PythonObject(result)

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
        qr_r_device_into(a, n_rows, n_cols, rp)
    _ = a^
    return PythonObject(n_cols)


def eigh_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """numpy's eigh: x_decomp's round-robin Jacobi (`DevExec.eigh`, the one eigh
    order of every column; cgr-decomp 2026-10-03 replaced the one-block cyclic
    solver here). `addrs`: 0 a, 1 w_out (n, ASCENDING), 2 v_out (n x n,
    eigenvector i in COLUMN i), 3 scalars_out (converged = 1, executed = 0:
    an unconverged solve raises). `params`: 0 n, 1 (optional) UPLO: 0 the
    whole matrix, 1 the lower triangle, 2 the upper. Returns n."""
    if len(addrs) != 4:
        raise Error(
            "eigh: addrs must contain 4 addresses (a, w_out, v_out,"
            " scalars_out), got "
            + String(len(addrs))
        )
    if len(params) != 1 and len(params) != 2:
        raise Error(
            "eigh: params must contain 1 or 2 values (n, UPLO), got " + String(len(params))
        )
    var n = Int(py=params[0])
    var uplo = Int(py=params[1]) if len(params) == 2 else 0
    if n < 1 or uplo < 0 or uplo > 2:
        raise Error("eigh: n must be >= 1 and UPLO 0, 1 or 2")
    var ap = F32Ptr(unsafe_from_address=Int(f32_ptr(Int(py=addrs[0]))))
    var wp = F32Ptr(unsafe_from_address=Int(f32_ptr(Int(py=addrs[1]))))
    var vp = F32Ptr(unsafe_from_address=Int(f32_ptr(Int(py=addrs[2]))))
    var sp = f64_ptr(Int(py=addrs[3]))
    with GILReleased(Python()):
        DevExec.eigh(ap, wp, vp, n, uplo)
        sp.unsafe_store(0, Float64(1.0))
        sp.unsafe_store(1, Float64(0.0))
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
        svdvals_device_into(a, n_rows, n_cols, sp)
    _ = a^
    return PythonObject(n_cols)


# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_linalg.mojo
# (the per-binding build) and in the merged roots tools/merged_bindings.py
# writes from bindings/merged_groups.json (one shared file per directory set).
def linalg_native_api_v1_binding() raises -> PythonObject:
    var table = LINALG_NATIVE_TABLE.get_or_create_ptr()
    return neural_descriptor_capsule(table[].name, table[].entries)


def build__mojolearn_linalg() -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_linalg")
        m.def_function[linalg_native_api_v1_binding]("linalg_native_api_v1")
        m.def_function[linalg_vendor_binding]("linalg_vendor")
        m.def_function[device_plan_enter_binding]("device_plan_enter")
        m.def_function[device_plan_exit_binding]("device_plan_exit")
        m.def_function[device_plan_log_binding]("device_plan_log")
        m.def_function[gemm_binding]("gemm")
        m.def_function[linalg_numeric_mode_binding]("linalg_numeric_mode")
        m.def_function[linalg_profile_version_binding](
            "linalg_profile_version"
        )
        m.def_function[linalg_numerical_profile_binding]("linalg_numerical_profile")
        m.def_function[lowbit_profile_version_binding]("lowbit_profile_version")
        m.def_function[gemm_bf16_binding]("gemm_bf16")
        initialize_fast_bf16_boundary()
        m.def_function[fast_bf16_available_binding]("fast_bf16_available")
        m.def_py_function[public_fast_to_bf16_binding]("public_fast_to_bf16")
        m.def_py_function[public_fast_from_bf16_binding]("public_fast_from_bf16")
        m.def_function[fast_to_bf16_binding]("fast_to_bf16",
            "Convert a float32 buffer to uint16 BF16 bits using round-to-nearest-even; return a same-shaped mojolearn.Array. Apple fast mode only.")
        m.def_function[fast_from_bf16_binding]("fast_from_bf16",
            "Widen uint16 BF16 bits exactly to a same-shaped float32 mojolearn.Array. Apple fast mode only.")
        m.def_py_function[fast_matmul_bf16_binding]("fast_matmul_bf16",
            "Multiply float32 or BF16-bits a by BF16-bits b with float32 accumulation on Apple in fast mode. Keyword arguments transpose_a=False, transpose_b=False and out=None support NN, NT and TN products. Return a float32 mojolearn.Array, or out when supplied.")
        m.def_function[gemm_int8_binding]("gemm_int8")
        comptime if AFN_GEMM_EPILOGUE:
            m.def_function[gemm_fused_binding]("gemm_fused")
        m.def_function[quantize_int8_binding]("quantize_int8")
        m.def_function[dequantize_int8_binding]("dequantize_int8")
        m.def_function[int15_profile_version_binding]("int15_profile_version")
        m.def_function[gemm_int15_binding]("gemm_int15")
        m.def_function[quantize_int15_binding]("quantize_int15")
        m.def_function[dequantize_int15_binding]("dequantize_int15")
        m.def_function[to_bf16_binding]("to_bf16")
        m.def_function[from_bf16_binding]("from_bf16")
        m.def_function[from_f16_binding]("from_f16")
        m.def_function[qr_r_binding]("qr_r")
        m.def_function[eigh_binding]("eigh")
        m.def_function[svdvals_binding]("svdvals")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_linalg: ", e))
