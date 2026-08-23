"""The OLS identity certificate: one card per fit, one record per step.

DEVIATION 527. IDENTITY_PATHS row 32.

WHY A CARD AND NOT A HASH
--------------------------
Ordinary least squares is the SMALLEST COMPLETE ESTIMATOR in this
repository -- six steps, `covA = A^T A`, `Ab = A^T b`, `Q S Q* = eig(covA)`,
`QS = Q invS`, `covA = QS Q^T`, `w = covA Ab` -- and that is the whole reason
it is worth certifying end to end: if every step is pinned then "the same fit
gives the same model on Metal, CUDA and HIP" becomes a checkable sentence
about an ESTIMATOR rather than about a kernel.

A single hash of the fitted coefficients is much weaker than it looks. It
tells you that something moved and never what, and IDENTITY_PATHS row 9
records the case that proves the point from the other side: at `39a0d88` the
GBDT RMSE prediction hash matched on all three vendors WHILE the score
buffers disagreed, because the argmax survived last-bit wiggle -- an output
identity with no certificate behind it. The card exists so that when this
runs on AMD the FIRST DIVERGING STAGE is named.

THE STAGES, in the order they are emitted:

    ols.input.A         the design matrix, as uploaded
    ols.input.b         the target, as uploaded
    ols.step1.covA      A^T A            (core/gemm.mojo::gemm_tn)
    ols.step2.Ab        A^T b            (core/column_stats.mojo::xty_kernel)
    ols.step3.eigvals   S                (jacobi_eigh_device + diagonal_to_vector)
    ols.step3.eigvecs   Q                (jacobi_eigh_device)
    ols.step3.info      converged / residual / SWEEP COUNT
    ols.step4.rank      HOW MANY DIRECTIONS SURVIVE  <- the integer stage
    ols.step4.QS        Q invS           (divide_columns_by_nonzero_kernel)
    ols.step5.inv       QS Q^T           (core/gemm.mojo::gemm_nt)
    ols.step6.coef      w                (core/gemm.mojo::gemv_n)

Eleven records. Two of them are not floats and are read FIRST, per
`E1_RUNBOOK` Phase 3's ladder: `ols.step3.info` carries the sweep count and
`ols.step4.rank` carries the rank. A card that first differs at either of
those says the two machines disagreed about how much work to do or about how
many directions the model HAS, which is a bigger thing than a rounding and
must be resolved before any float stage is read.

THE FIXTURE IS BUILT WITH NO FLOATING-POINT OPERATION AT ALL
-------------------------------------------------------------
This is not fastidiousness, it is IDENTITY_PATHS row 18's class: cross-vendor
is cross-HOST here. A fixture written the obvious way --

    target = 0.0
    for k in range(d):
        target += v * true_w[k]        # host Float64 multiply-add

-- is a HOST mul-add chain, and a host mul-add chain is a contraction
decision exactly like a device one (row 9). If macOS/arm64 contracts it and
x86-64/glibc does not, the two machines upload DIFFERENT DESIGN MATRICES and
every stage after that differs for a reason that has nothing to do with any
GPU. The existing tolerance checks in `ols_check.mojo` build their fixture
that way and it does not matter there, because they compare against a
tolerance. It would matter here.

So `_hash_f32` below performs no arithmetic: it assembles a sign, an
exponent in [2^-9, 2^-1) and a 23-bit mantissa from a splitmix64 hash and
BITCASTS. Every byte of `ols.input.A` is a pure function of the index and
the salt on any host with any compiler at any optimization level, and the
card's first record proves it before the first kernel runs.

The consequence, said plainly: `b` is NOT `A w*` for any `w*`. This fixture
is not for checking that the answer is right -- `check_ols_exact` and
`check_ols_beats_truth_on_noise` do that, and they are tolerance checks
because that is the right shape for that question. This one is for checking
that the answer is the SAME.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.memory import bitcast

from core.identity_trace import IdentityTrace
from glm.ported.glm.ols import OLS_ALGO_EIG, ols_fit_traced
from glm.ported.linalg.detail.lstsq import OLS_ELEM_TPB
from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL


comptime OLS_CARD_ROWS = 65536
comptime OLS_CARD_COLS = 16
"""The card's shape. `n_cols = 16` is inside the split-K kernel's capacity
(`GRAM_MAX_COLS = 128`), so the IDENTICAL build takes the pinned Gram arm
rather than the refusal; `n_rows = 65536` is large enough that the pinned
k partition (128 chunks) has 512 rows in each, so a fold-order difference
has somewhere to show up."""


def ols_card_mode_name() -> String:
    """The mode THIS BINARY WAS COMPILED IN.

    `GLOBAL_NUMERIC_MODE` is comptime and the flip is an edit to a file four
    sessions share, so a run that compiled inside another session's flip
    window would otherwise write one mode's label over the other mode's
    numbers. Copied from `core/gemm_identity_check.mojo::_mode_name`, which
    is the pattern this repository settled on after that race cost it two
    measurements in one day.
    """
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return String("IDENTICAL")
    return String("FAST")


def _mix(i: Int, salt: Int) -> UInt64:
    """splitmix64. Adjacent indices land nowhere near each other, so a
    permutation of the fixture cannot hide behind a total."""
    var z = (
        UInt64(i + 1) * 0x9E3779B97F4A7C15
        + UInt64(salt + 1) * 0xBF58476D1CE4E5B9
    )
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _hash_f32(i: Int, salt: Int) -> Float32:
    """A normal Float32 in +-[2^-9, 2^-1), ASSEMBLED FROM BITS.

    No floating-point operation is performed: sign, exponent and mantissa
    are integer fields and the result is a bitcast. The exponent range is
    chosen so that nothing here is a denormal, an infinity or a NaN, which
    keeps the fixture out of the one regime where `ftz` is not bitwise inert
    -- a fixture that exercised the denormal policy would be measuring the
    policy rather than the estimator, and row 10 already has its own gate.

    See the module docstring for why this is not `Float64(z) * 1e-6`.
    """
    var h = _mix(i, salt)
    var sign = UInt32((h >> 63) & 1) << 31
    # exponent field 118..126 -> values in [2^-9, 2^-1)
    var expf = UInt32(118 + Int((h >> 40) % 9)) << 23
    var mant = UInt32(h & UInt64(0x007FFFFF))
    return bitcast[DType.float32](sign | expf | mant)


def fill_ols_card_fixture(
    ctx: DeviceContext,
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    mut trace: IdentityTrace,
    n_rows: Int,
    n_cols: Int,
    salt: Int = 527,
) raises:
    """Stage the fixture and record it as the card's first two stages.

    Recording the INPUT is the half of a certificate that is easy to skip
    and expensive to skip. If two machines' cards first differ at
    `ols.input.A`, no kernel on either machine is implicated and the finding
    is in the fixture builder or the upload; if they AGREE there, every
    later difference is the estimator's.
    """
    var ha = ctx.enqueue_create_host_buffer[DType.float32](n_rows * n_cols)
    var hb = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
    ctx.synchronize()
    for i in range(n_rows * n_cols):
        ha.unsafe_ptr().unsafe_store(i, _hash_f32(i, salt))
    for i in range(n_rows):
        hb.unsafe_ptr().unsafe_store(i, _hash_f32(i, salt + 1))
    ctx.enqueue_copy(dst_buf=a, src_ptr=ha.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=b, src_ptr=hb.unsafe_ptr())
    ctx.synchronize()
    trace.record_host("ols.input.A", ha.unsafe_ptr(), n_rows * n_cols)
    trace.record_host("ols.input.b", hb.unsafe_ptr(), n_rows)
    # `[[mojo-buffer-freed-at-last-use]]`: both host buffers are dead at
    # `.unsafe_ptr()` unless something uses them past the copies above.
    _ = ha^
    _ = hb^


def emit_ols_card(
    ctx: DeviceContext,
    mut trace: IdentityTrace,
    n_rows: Int = OLS_CARD_ROWS,
    n_cols: Int = OLS_CARD_COLS,
    elem_tpb: Int = OLS_ELEM_TPB,
    salt: Int = 527,
) raises -> List[Float32]:
    """One fit, eleven records, the coefficients returned.

    Goes through `ols_fit_traced` and therefore through `ols.cuh:112-113`'s
    dispatch guard, not around it: a card taken off `lstsq_eig` directly
    would certify a path no caller reaches. Returns the coefficients so an
    in-process control can compare two fits without re-reading the file.
    """
    var a = ctx.enqueue_create_buffer[DType.float32](n_rows * n_cols)
    var b = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var w = ctx.enqueue_create_buffer[DType.float32](n_cols)
    var cov_a = ctx.enqueue_create_buffer[DType.float32](n_cols * n_cols)
    var q = ctx.enqueue_create_buffer[DType.float32](n_cols * n_cols)
    var qs = ctx.enqueue_create_buffer[DType.float32](n_cols * n_cols)
    var s_vec = ctx.enqueue_create_buffer[DType.float32](n_cols)
    var ab = ctx.enqueue_create_buffer[DType.float32](n_cols)
    var inv = ctx.enqueue_create_buffer[DType.float32](n_cols * n_cols)
    var a_alias = ctx.enqueue_create_buffer[DType.float32](n_rows * n_cols)
    var a_alias2 = ctx.enqueue_create_buffer[DType.float32](n_rows * n_cols)
    ctx.synchronize()

    fill_ols_card_fixture(ctx, a, b, trace, n_rows, n_cols, salt)

    ols_fit_traced(
        ctx, a, b, w, cov_a, q, qs, s_vec, ab, inv, a_alias, a_alias2,
        n_rows, n_cols, trace, OLS_ALGO_EIG, False, False, elem_tpb,
    )

    var hw = ctx.enqueue_create_host_buffer[DType.float32](n_cols)
    ctx.enqueue_copy(dst_ptr=hw.unsafe_ptr(), src_buf=w)
    ctx.synchronize()
    var out = List[Float32]()
    for k in range(n_cols):
        out.append(hw.unsafe_ptr().unsafe_load(k))
    _ = a
    _ = b
    _ = w
    _ = cov_a
    _ = q
    _ = qs
    _ = s_vec
    _ = ab
    _ = inv
    _ = a_alias
    _ = a_alias2
    _ = hw^
    return out^
