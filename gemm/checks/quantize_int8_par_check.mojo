# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The gate of the PARALLEL quantizer of `int8i32.v1`: its codes and its
exponents against the host's and the reference device quantizer's.

    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . gemm/checks/quantize_int8_par_check.mojo
    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_QUANT_PAR_SABOTAGE=1 -I . gemm/checks/quantize_int8_par_check.mojo
    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_LOWBIT_SABOTAGE=1 -I . gemm/checks/quantize_int8_par_check.mojo

The first must pass both gates. The second skips the first level of the
block's tree and must FAIL both. The third flips the lowest bit of every
code stored and must FAIL both. `tools/lowbit_mma_speed/quant_gate_job.sh`
runs the three and reads the logs. Runs on every column: the quantizer
needs no matrix unit.

Lane lane/lowbit-mma-speed, 2026-09-29. Kernel
`gemm/checks/quantize_int8_par.mojo`; the reference device quantizer
`gemm/checks/gemm_lowbit.mojo::quantize_rows_int8_kernel`; the answer
`gemm/host/gemm_lowbit_oracle.mojo::quantize_rows_int8`. Contract clauses
L-3 and L-4 and section 2 (the promise covers the codes and the exponents).

WHAT A SCHEDULE CAN GET WRONG, and so what is planted. The absmax of a row
is reduced by many threads, so a thread's maximum can fail to reach the
exponent; a slot can be skipped or written twice at a ragged row end; a
code can be written under another row's exponent. So the fixtures are
quantized floats at widths off the four-wide slot, below and above both
blocks' thread counts and at the transformer widths, and rows planted with
the absmax at the last column, at the first, inside the upper half of the
block's threads, a row of zeros, a row of subnormals, a row holding a NaN,
a row of negative zeros, and a row whose scaled values sit on the rounding
ties and on the clamp. Every output is poisoned first with a value no code
and no exponent of these fixtures takes (the code -128; L-4 clamps to
[-127, 127]), so a cell that was never written shows.

`check_par_quantizer_feeds_the_product` quantizes both operands on the
device with the parallel quantizer and multiplies them on the plan the
profile's dispatcher picks (the reference unit plan where the column has
the unit, the flat plan where it does not), against the oracle on the
host's codes: the whole operation a caller runs.

MAIN RUNS EVERY GATE AND REPORTS EVERY VERDICT before it raises, and a gate
reports every case before it raises, as `gemm_lowbit_check.mojo` does and
for the same reason: under a sabotage build the evidence is which cases
the defect reaches.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.memory import bitcast
from std.sys import has_accelerator

from checks.kernel_matrix import TARGET_COLUMN, column_name
from checks.numerics import numeric_mode_name
from gemm.checks.gemm_lowbit import (
    identical_gemm_int8_into,
    int8_plan_dispatch_name,
    quantize_rows_int8_device,
)
from gemm.checks.gemm_lowbit_check import (
    _download_f32,
    _download_i32,
    _download_i8,
    _fill,
    _first_diff,
    _gate,
    _poisoned,
    _upload_f32,
    _upload_i32,
    _upload_i8,
)
from gemm.checks.quantize_int8_par import (
    QUANT_PAR_BLOCK_NARROW,
    QUANT_PAR_BLOCK_WIDE,
    quantize_par_block_name,
    quantize_par_sabotage_name,
    quantize_rows_int8_par_device,
    quantize_rows_int8_par_with_block,
)
from gemm.host.gemm_lowbit_oracle import gemm_int8_oracle, quantize_rows_int8


struct Tally(Movable):
    """The cases a gate ran, how many differed, and the first difference."""

    var cases: Int
    var failed: Int
    var first: String

    def __init__(out self):
        self.cases = 0
        self.failed = 0
        self.first = String("")

    def note(mut self, e: String):
        """One case's verdict: empty when it agreed."""
        self.cases += 1
        if e.byte_length() > 0:
            self.failed += 1
            print("   !! " + e)
            if self.first.byte_length() == 0:
                self.first = e.copy()


def _diff(got: List[Float32], want: List[Float32], tag: String) -> String:
    """`_first_diff` as a verdict: empty when every cell's bits agree."""
    try:
        _first_diff(got, want, tag)
    except e:
        return String(e)
    return String("")


def _verdict(tally: Tally, what: String) raises:
    print("   " + what + ": " + String(tally.cases) + " cases, " + String(tally.failed) + " failed")
    if tally.failed > 0:
        raise Error(
            String(tally.failed) + " of " + String(tally.cases) + " " + what
            + " cases differ; first: " + tally.first
        )


# ===========================================================================
# THE QUANTIZER'S GATES
# ===========================================================================


comptime QUANT_SHAPE_COUNT = 12


def _quant_shape(i: Int) -> Tuple[Int, Int]:
    """(rows, cols): cols off the four-wide slot, below and above both
    blocks' thread counts, and at the transformer widths."""
    if i == 0:
        return (9, 1)
    if i == 1:
        return (9, 5)
    if i == 2:
        return (10, 32)
    if i == 3:
        return (9, 129)
    if i == 4:
        return (12, 255)
    if i == 5:
        return (9, 1000)
    if i == 6:
        return (9, 1023)
    if i == 7:
        return (16, 1024)
    if i == 8:
        return (9, 4096)
    if i == 9:
        return (11, 4097)
    if i == 10:
        return (9, 14336)
    return (300, 96)


def _quant_fixture(rows: Int, cols: Int, salt: Int) -> List[Float32]:
    """Quantized floats, and eight planted rows (where the shape has them):

        row 1   the absmax at the LAST column
        row 2   the absmax at the FIRST column, negative
        row 3   zeros
        row 4   a NaN at column 0, the absmax in the middle
        row 5   subnormals only: the flush makes the row zero
        row 6   the absmax three quarters along: a slot of the upper half
                of the block's threads wherever the row is long enough
        row 7   absmax 127.75, so the row exponent is 0 and the codes are
                the values rounded: ties at 0.5, 1.5, 2.5 and their
                negatives, and 127.75 itself, which rounds to 128 and
                clamps to 127
        row 8   the absmax at the last column of a row of NEGATIVE zeros
    """
    var x = _fill(rows * cols, salt)
    if rows > 1:
        x[1 * cols + cols - 1] = Float32(1.0e6)
    if rows > 2:
        x[2 * cols] = Float32(-3.0e5)
    if rows > 3:
        for c in range(cols):
            x[3 * cols + c] = Float32(0.0)
    if rows > 4:
        x[4 * cols] = bitcast[DType.float32](UInt32(0x7FC00000))
        x[4 * cols + cols // 2] = Float32(7.0e4)
    if rows > 5:
        for c in range(cols):
            x[5 * cols + c] = bitcast[DType.float32](UInt32(1 + (c * 7919) % 8388607))
    if rows > 6:
        x[6 * cols + (cols * 3) // 4] = Float32(-9.0e5)
    if rows > 7:
        for c in range(cols):
            var v = Float32(0.5) + Float32(c % 3)
            if c % 2 == 1:
                v = -v
            x[7 * cols + c] = v
        x[7 * cols + cols // 3] = Float32(127.75)
    if rows > 8:
        for c in range(cols):
            x[8 * cols + c] = Float32(-0.0)
        x[8 * cols + cols - 1] = Float32(3.0)
    return x^


def _codes_diff(
    got_q: List[Int8],
    got_e: List[Int32],
    want_q: List[Int8],
    want_e: List[Int32],
    rows: Int,
    cols: Int,
    tag: String,
) -> String:
    """Empty when every exponent and every code agrees; else the count and
    the first of each."""
    var bad_e = 0
    var first_e = -1
    for r in range(rows):
        if got_e[r] != want_e[r]:
            bad_e += 1
            if first_e < 0:
                first_e = r
    var bad_q = 0
    var first_q = -1
    for i in range(rows * cols):
        if got_q[i] != want_q[i]:
            bad_q += 1
            if first_q < 0:
                first_q = i
    if bad_e == 0 and bad_q == 0:
        return String("")
    var msg = tag + ": " + String(bad_e) + " of " + String(rows) + " exponents and "
    msg += String(bad_q) + " of " + String(rows * cols) + " codes differ"
    if first_e >= 0:
        msg += "; first exponent at row " + String(first_e) + " got " + String(got_e[first_e])
        msg += " want " + String(want_e[first_e])
    if first_q >= 0:
        msg += "; first code at row " + String(first_q // cols) + " col " + String(first_q % cols)
        msg += " got " + String(Int(got_q[first_q])) + " want " + String(Int(want_q[first_q]))
    return msg


def _poisoned_codes(ctx: DeviceContext, count: Int) raises -> DeviceBuffer[DType.int8]:
    """A code buffer holding -128 everywhere: the one int8 no code is (L-4
    clamps to [-127, 127]), so a code that was never written shows."""
    var h = List[Int8]()
    for _ in range(count):
        h.append(Int8(-128))
    return _upload_i8(ctx, h)


def _poisoned_exponents(ctx: DeviceContext, count: Int) raises -> DeviceBuffer[DType.int32]:
    var h = List[Int32]()
    for _ in range(count):
        h.append(Int32(-987654))
    return _upload_i32(ctx, h)


def check_par_quantizer_matches(ctx: DeviceContext) raises:
    """GATE: the parallel quantizer's codes and exponents are the host's
    and the reference device quantizer's, on BOTH blocks, at every shape.
    The outputs are poisoned first with a value no code and no exponent of
    these fixtures takes."""
    var tally = Tally()
    for s in range(QUANT_SHAPE_COUNT):
        var sh = _quant_shape(s)
        var rows = sh[0]
        var cols = sh[1]
        var x = _quant_fixture(rows, cols, 503 + s)
        var want = quantize_rows_int8(x, rows, cols)
        var dx = _upload_f32(ctx, x)
        var dq_ref = _poisoned_codes(ctx, rows * cols)
        var de_ref = _poisoned_exponents(ctx, rows)
        quantize_rows_int8_device(ctx, dq_ref, de_ref, dx, rows, cols)
        ctx.synchronize()
        var ref_q = _download_i8(ctx, dq_ref, rows * cols)
        var ref_e = _download_i32(ctx, de_ref, rows)
        var tag = "quantize " + String(rows) + "x" + String(cols)
        var before = tally.failed
        tally.note(_codes_diff(ref_q, ref_e, want.q, want.e, rows, cols, tag + " (reference device vs host)"))
        for block in range(2):
            var btag = tag + " " + ("NARROW" if block == QUANT_PAR_BLOCK_NARROW else "WIDE")
            var dq = _poisoned_codes(ctx, rows * cols)
            var de = _poisoned_exponents(ctx, rows)
            quantize_rows_int8_par_with_block(ctx, dq, de, dx, rows, cols, block)
            ctx.synchronize()
            var got_q = _download_i8(ctx, dq, rows * cols)
            var got_e = _download_i32(ctx, de, rows)
            var verdict = _codes_diff(got_q, got_e, want.q, want.e, rows, cols, btag + " (parallel vs host)")
            if verdict.byte_length() == 0:
                verdict = _codes_diff(got_q, got_e, ref_q, ref_e, rows, cols, btag + " (parallel vs reference device)")
            tally.note(verdict)
            _ = dq
            _ = de
        if tally.failed == before:
            print("   ok " + tag + "  e[0]=" + String(want.e[0]))
        _ = dx
        _ = dq_ref
        _ = de_ref
    _verdict(tally, String("quantizer"))


def check_par_quantizer_feeds_the_product(ctx: DeviceContext) raises:
    """GATE: both operands quantized on the device by the parallel
    quantizer (the block its launcher picks), then the product on the plan
    the profile's dispatcher picks, against the oracle on the host's codes:
    the whole operation a caller runs."""
    var tally = Tally()
    for s in range(4):
        var m = 3
        var n = 70
        var k = 1000
        if s == 1:
            m = 33
            n = 129
            k = 4096
        elif s == 2:
            m = 1
            n = 300
            k = 4097
        elif s == 3:
            m = 130
            n = 17
            k = 96
        var ha = _quant_fixture(m, k, 601 + s)
        var hb = _quant_fixture(n, k, 613 + s)
        var qa = quantize_rows_int8(ha, m, k)
        var qb = quantize_rows_int8(hb, n, k)
        var want = gemm_int8_oracle(qa.q, qa.e, qb.q, qb.e, m, n, k)
        var tag = "quantize and multiply " + String(m) + "x" + String(n) + "x" + String(k)
        var da = _upload_f32(ctx, ha)
        var db = _upload_f32(ctx, hb)
        var dqa = _poisoned_codes(ctx, m * k)
        var dea = _poisoned_exponents(ctx, m)
        var dqb = _poisoned_codes(ctx, n * k)
        var deb = _poisoned_exponents(ctx, n)
        var dc = _poisoned(ctx, m * n)
        quantize_rows_int8_par_device(ctx, dqa, dea, da, m, k)
        quantize_rows_int8_par_device(ctx, dqb, deb, db, n, k)
        identical_gemm_int8_into(ctx, dc, dqa, dea, dqb, deb, m, n, k)
        ctx.synchronize()
        var verdict = String("")
        try:
            var got = _download_f32(ctx, dc, m * n, tag)
            verdict = _diff(got, want, tag + " (device vs oracle)")
        except e:
            verdict = String(e)
        tally.note(verdict)
        if verdict.byte_length() == 0:
            print("   ok " + tag)
        _ = da
        _ = db
        _ = dqa
        _ = dea
        _ = dqb
        _ = deb
        _ = dc
    _verdict(tally, String("quantize-and-multiply"))


def main() raises:
    print(
        "== gemm/checks/quantize_int8_par_check.mojo [" + numeric_mode_name()
        + "]  sabotage: " + quantize_par_sabotage_name() + " =="
    )
    print("   profile: mojolearn.identical.gemm.int8i32.v1, the parallel quantizer")
    print("   column: " + column_name(TARGET_COLUMN) + "  int8 dispatch: " + int8_plan_dispatch_name())
    print(
        "   blocks: " + quantize_par_block_name(QUANT_PAR_BLOCK_NARROW) + "; "
        + quantize_par_block_name(QUANT_PAR_BLOCK_WIDE)
    )
    var ran = 0
    var failed = 0
    comptime if not has_accelerator():
        print("   no accelerator: the device gates DID NOT RUN, which is not a pass")
        raise Error("the parallel quantizer's gate needs a device")
    else:
        var ctx = DeviceContext()
        try:
            check_par_quantizer_matches(ctx)
            _gate(String("check_par_quantizer_matches"), ran, failed, String(""))
        except e:
            _gate(String("check_par_quantizer_matches"), ran, failed, String(e))
        try:
            check_par_quantizer_feeds_the_product(ctx)
            _gate(String("check_par_quantizer_feeds_the_product"), ran, failed, String(""))
        except e:
            _gate(String("check_par_quantizer_feeds_the_product"), ran, failed, String(e))
        print("== " + String(ran) + " gates, " + String(failed) + " failed ==")
        if failed > 0:
            raise Error(String(failed) + " of " + String(ran) + " gates failed")
