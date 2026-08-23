"""ONE traced linear-algebra run per process: the E1 payload for rows 27-31.

`tools/e1_traced_fit.py` is the GBDT/ET/RF driver and goes through the Python
bindings; `bench/unsupervised_trace_main.mojo` is the same thing for rows
19-26. There is no binding for `core/gemm.mojo` and no card for the matrix
products at all, so rows 27 and 28 had no way to be compared across anything
-- which is the same gap the unsupervised lane found and for the same reason.
This main is that way for the products.

ONE ARM PER PROCESS, AND THAT IS THE DIFFER'S CONTRACT, NOT A CONVENIENCE.
`core/identity_trace.mojo`'s records carry a sequence number and a position
tag; `tools/identity_trace_diff.py` refuses a file whose sequence numbers
restart. `MOJOLEARN_LINALG_ARM` selects the arm and the process ends.

    gram      `gemm_tn` at the shipped PCA/OLS aspect. Reaches the pinned
              chunk count (row 27b, DEVIATION 520), the arm resolution
              (row 27a, DEVIATION 521) and the pinned contraction and
              denormal seams inside the split-K pair (row 27c, DEVIATION
              522). The tall-skinny shape is the one the shipped fits have.
    nt        `gemm_nt` at three widths from ONE generator, which is the
              batch-invariance stage: the same logical rows appear at the
              same indices in all three, so the card carries the overlap of
              each. A vendor or a mode that lets the launch width reach the
              answer diverges HERE and at no earlier tag (row 28,
              DEVIATION 526).
    gemv      `gemv_n` at OLS's step-6 shape. Small, and carded anyway,
              because "it is small so it cannot matter" is an argument and
              a card is a measurement.

THE FIXTURE IS AN INTEGER-EXACT FUNCTION OF A CONSTANT SEED, for the same
reason E1's is and the unsupervised card's is: a cross-column or
cross-vendor claim about a product is worth nothing until both sides are
proven to have multiplied the SAME BYTES. Every value here is
`<small integer> / 2^20`, which is exact in Float32 on any backend, and the
driver records a hash of the raw input bits as its FIRST stage. Compare that
tag first; a card diff against different inputs measures nothing.

WHY THE INPUTS ARE NOT THE HASHED `_val_f32` THE CHECKS USE. That generator
multiplies by `1.0e-6`, which is not exact in binary, so its values are a
rounding of a decimal constant. On one backend that is fine and identical;
as an INPUT to a cross-vendor identity claim it puts a float multiply
upstream of the thing being measured. A power-of-two divisor removes it.

    MOJOLEARN_IDENTITY_TRACE=/tmp/gram.card MOJOLEARN_LINALG_ARM=gram \\
        pixi run mojo run -I . bench/linalg_trace_main.mojo

`tools/check_linalg_column_invariance.sh` runs the three arms across the
APPLE, NVIDIA and AMD comptime columns and diffs the cards.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.os import getenv

from core.gemm import gemm_nt, gemm_tn, gemv_n
from core.gram_splitk import gram_splitk_applies, gram_splitk_chunk_count
from core.identity_trace import IdentityTrace
from mojo_only.kernel_matrix import TARGET_COLUMN, column_name
from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL


def _mode_name() -> String:
    """The mode THIS BINARY COMPILED IN, printed so the driver can read it
    BACK rather than assume it from the flip. `mojo_only/numerics.mojo` is a
    shared file in a checkout worked by parallel sessions, and a build that
    lands inside another session's flip window compiles the other arm and
    labels every line in itself consistently with that arm (DEVIATION 514).
    """
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return String("IDENTICAL")
    return String("FAST")


def _exact(i: Int, salt: Int) -> Float32:
    """A value in (-1, 1) that is EXACT in Float32 on every backend.

    splitmix64 for the scatter, then `<integer> / 2^20`: the numerator is
    below 2^21 so it is exact in a Float32 mantissa, and the divisor is a
    power of two, so the division is exact too. No decimal constant, no
    rounding upstream of the product being measured.
    """
    var z = (
        UInt64(i + 1) * 0x9E3779B97F4A7C15
        + UInt64(salt + 1) * 0xBF58476D1CE4E5B9
    )
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    z = z ^ (z >> 31)
    var num = Int(z % 2097151) - 1048575  # (-2^20, 2^20)
    return Float32(num) / Float32(1048576.0)


def _upload(
    ctx: DeviceContext, n: Int, salt: Int
) raises -> DeviceBuffer[DType.float32]:
    var buf = ctx.enqueue_create_buffer[DType.float32](n)
    var h = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.synchronize()
    for i in range(n):
        h.unsafe_ptr().unsafe_store(i, _exact(i, salt))
    ctx.enqueue_copy(dst_buf=buf, src_ptr=h.unsafe_ptr())
    ctx.synchronize()
    return buf


def _gram_arm(ctx: DeviceContext, mut t: IdentityTrace) raises:
    """`gemm_tn` at the shipped PCA/OLS Gram aspect."""
    var m = 32
    var k = 65_536
    var x = _upload(ctx, k * m, 11)
    var z = ctx.enqueue_create_buffer[DType.float32](m * m)
    var xt = ctx.enqueue_create_buffer[DType.float32](k * m)
    var xt2 = ctx.enqueue_create_buffer[DType.float32](k * m)
    t.record_device(ctx, "gram.input", x, k * m)
    gemm_tn(ctx, z, x, xt, xt2, m, m, k)
    ctx.synchronize()
    t.record_device(ctx, "gram.product", z, m * m)
    _ = x
    _ = z
    _ = xt
    _ = xt2


def _nt_arm(ctx: DeviceContext, mut t: IdentityTrace) raises:
    """`gemm_nt` at three widths from ONE generator: the batch-invariance
    stage.

    Row i of X and row j of Y are pure functions of (i, salt) and (j, salt)
    and do NOT depend on m or n, so the same logical rows sit at the same
    indices at every width. Each arm records the OVERLAP -- the first
    `M * N_NARROW` cells of a row-major output -- and under a correct build
    all three overlap tags carry the same hash. A build where the launch
    width reaches the answer diverges at `nt.wide` or `nt.wider` while
    `nt.input.x` and `nt.input.y` still match, which is the diagnosis the
    differ's ladder is for.
    """
    var m = 64
    var k = 512
    var widths_narrow = 4
    var x = _upload(ctx, m * k, 21)
    t.record_device(ctx, "nt.input.x", x, m * k)

    for s in range(3):
        var n = widths_narrow
        var tag = String("nt.narrow")
        if s == 1:
            n = 64
            tag = String("nt.wide")
        elif s == 2:
            n = 256
            tag = String("nt.wider")
        var y = _upload(ctx, n * k, 22)
        if s == 0:
            t.record_device(ctx, "nt.input.y", y, widths_narrow * k)
        var z = ctx.enqueue_create_buffer[DType.float32](m * n)
        gemm_nt(ctx, z, x, y, m, n, k)
        ctx.synchronize()
        # The overlap only. Recording all m*n would make the three tags
        # different lengths and therefore incomparable, which is the
        # opposite of the point.
        var hz = ctx.enqueue_create_host_buffer[DType.float32](m * n)
        var ov = ctx.enqueue_create_host_buffer[DType.float32](
            m * widths_narrow
        )
        ctx.synchronize()
        ctx.enqueue_copy(dst_ptr=hz.unsafe_ptr(), src_buf=z)
        ctx.synchronize()
        for i in range(m):
            for j in range(widths_narrow):
                ov.unsafe_ptr().unsafe_store(
                    i * widths_narrow + j,
                    hz.unsafe_ptr().unsafe_load(i * n + j),
                )
        t.record_host(tag, ov.unsafe_ptr(), m * widths_narrow)
        _ = y
        _ = z
    _ = x


def _gemv_arm(ctx: DeviceContext, mut t: IdentityTrace) raises:
    """`gemv_n` at OLS's step-6 shape."""
    var m = 128
    var k = 128
    var x = _upload(ctx, m * k, 31)
    var y = _upload(ctx, k, 32)
    var z = ctx.enqueue_create_buffer[DType.float32](m)
    t.record_device(ctx, "gemv.input.x", x, m * k)
    t.record_device(ctx, "gemv.input.y", y, k)
    gemv_n(ctx, z, x, y, m, k)
    ctx.synchronize()
    t.record_device(ctx, "gemv.product", z, m)
    _ = x
    _ = y
    _ = z


def main() raises:
    print("mode", _mode_name())
    print("column", column_name(TARGET_COLUMN))
    # THE DISPATCH IS PART OF THE CARD'S PROVENANCE. Two columns that agree
    # on every hash while one of them never entered the pinned kernel would
    # be agreement by coincidence, and on the Apple column that coincidence
    # is available (its FAST arm takes the same kernel the IDENTICAL arm
    # does). Printing both makes the reader check.
    print("gram_chunks", gram_splitk_chunk_count())
    print("gram_splitk_arm", gram_splitk_applies(32, 32, 65_536))

    var arm = String(getenv("MOJOLEARN_LINALG_ARM"))
    if arm == "":
        arm = String("gram")

    var path = String(getenv("MOJOLEARN_IDENTITY_TRACE"))
    if path == "":
        # NOT a warning that is easy to miss. A run with no trace path
        # produces no card and exits 0, and a driver that does not check
        # would record a missing file as agreement.
        raise Error(
            "bench/linalg_trace_main: MOJOLEARN_IDENTITY_TRACE is unset, so"
            " this run would produce NO CARD and still exit 0. Set it, or"
            " use tools/check_linalg_column_invariance.sh."
        )

    var t = IdentityTrace()
    with DeviceContext() as ctx:
        if arm == "gram":
            _gram_arm(ctx, t)
        elif arm == "nt":
            _nt_arm(ctx, t)
        elif arm == "gemv":
            _gemv_arm(ctx, t)
        else:
            raise Error(
                "bench/linalg_trace_main: unknown MOJOLEARN_LINALG_ARM '"
                + arm
                + "'. Known arms: gram, nt, gemv."
            )
    print("arm", arm, "done")
