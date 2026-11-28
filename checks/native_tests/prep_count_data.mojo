# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""The count-data transformers' programs on hand-computed cases (lane sc-prep-1, 2026-11-28): the typed builders of
bindings/prep_scprep_run.mojo (units x_prep/sc_prep.mojo) run on the host column by `prep_enc_host.enc_host_run`,
the requests the door (bindings/prep_scprep_native_door.mojo) makes. NOT RUN by the lane (lanes do not compile).

  row_totals   [[1, 3, 0], [2, 2, 4]] to target 10: [[2.5, 7.5, 0], [2.5, 2.5, 5]]; target None: the median of
               the totals 4 and 8 is 6; the same rows as CSR give the same values word for word.
  rule         [[1, 1, 8], [1, 1, 0]], max_fraction 0.5: column 2 holds 8 of row 0's 10, so it is left out of
               the totals (EXCL = [0, 0, 1]); target 4: [[2, 2, 16], [2, 2, 0]]; the columns [0, 1] given
               explicitly give the same words as the rule (the general subset primitive).
  log1p        [0, 1, 3] -> [0, ln 2, ln 4]; base 2 -> [0, 1, 2].
  scale        one column 1, 2, 3, 4: mean 2.5, var (ddof 1) 5/3, clipped at 1: [-1, -0.3873, 0.3873, 1].
  ranks        keys [3, 1, 2, NaN]: ranks [2, 0, 1, -1]; descending [0, 2, 1, -1].
  chunked      2 x 2500 genes in three groups (three sorted chunks per batch): every rank equals the counted rank.
  percentiles  the sorted 1 .. 11: p10 = 2, p15 = 2.5, p100 = 11 (numpy linear).
  loess        LY an exact quadratic of LX: the local quadratic fit returns LY itself.
  regress      x = 2 c + 1 (one covariate): residuals 0; dense and CSR X alike.
  select       keep columns 0 and 2 of a CSR block: the kept entries renumbered, row order kept.
"""
from std.math import isnan, log, sqrt
from std.memory import bitcast
from bindings.prep_encode_prog_run import EncProgram, enc_input
from bindings.prep_scprep_run import (
    SpInput, RowTotalFit, RowTotalApply, ColBlock, ScaleApply, RegressFit, rowtotal_fit_stage, rowtotal_apply_stage,
    log1p_stage, col_block_stage, scale_apply_stage, regress_fit_stage, keep_stage, select_csr_stage,
    sp_rank_stages,
)
from core.preprocessing_program import prep_program_stage
from x_prep.sc_prep import SP_PERCENTILES, SP_LOESS, SP_PRE_NONE
from checks.native_tests.prep_enc_host import enc_host_run
from checks.native_tests.prep_cls_host import all_close, arena_words
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require


def _ints(values: List[Int]) -> List[Float32]:
    var out = List[Float32](length=len(values), fill=0)
    for i in range(len(values)):  # small-loop(fixture: int words as arena words)
        out[i] = bitcast[DType.float32](Int32(values[i]))
    return out^


def _int_at(arena: List[Float32], i: Int) -> Int:
    return Int(bitcast[DType.int32](arena[i]))


def _close(arena: List[Float32], at: Int, expected: List[Float64], tol: Float64, label: String) raises:
    for i in range(len(expected)):  # small-loop(fixture: the expected words)
        all_close(arena[at + i], expected[i], tol, tol, label + "[" + String(i) + "]")


def _same(a: List[Float32], at_a: Int, b: List[Float32], at_b: Int, count: Int, label: String) raises:
    for i in range(count):  # small-loop(fixture: word-for-word comparison)
        require(bitcast[DType.uint32](a[at_a + i]) == bitcast[DType.uint32](b[at_b + i]),
                label + ": word " + String(i) + " differs")


def _dense(n: Int, d: Int) -> SpInput:
    return SpInput(n, d, False, 0, -1, -1, -1, 0)


def _csr(n: Int, d: Int, nnz: Int) -> SpInput:
    # owners 0, 1, 2: indptr, indices, data
    return SpInput(n, d, True, -1, 0, 1, 2, nnz)


def _apply(var owners: List[List[Float32]], x: SpInput, excl: List[Float32], target: Float64) raises -> List[Float32]:
    var prog = EncProgram(True)
    owners.append(excl.copy())
    var plan = rowtotal_apply_stage(prog, RowTotalApply(x.copy(), len(owners) - 1, target, False))
    var arena = enc_host_run(prog, owners)
    var count = x.nnz if x.csr else x.n * x.d
    return arena_words(arena, plan, count)


def row_totals_unit(unit: Int) raises:
    var x: List[Float32] = [1, 3, 0, 2, 2, 4]
    var owners = List[List[Float32]]()
    owners.append(x.copy())
    var excl: List[Float32] = [0, 0, 0]
    if unit == 0:
        var out = _apply(owners^, _dense(2, 3), excl, 10.0)
        _close(out, 0, [2.5, 7.5, 0.0, 2.5, 2.5, 5.0], 1e-6, "row totals to 10")
        return
    if unit == 1:
        var prog = EncProgram(True)
        var plan = rowtotal_fit_stage(prog, RowTotalFit(_dense(2, 3), -1, 0, False, 0.05, True))
        var arena = enc_host_run(prog, owners)
        all_close(arena[plan.median], 6.0, 0.0, 0.0, "the median of the totals 4 and 8")
        return
    # CSR: the same rows, stored entries only
    var csr = List[List[Float32]]()
    csr.append(_ints([0, 2, 5]))
    csr.append(_ints([0, 1, 0, 1, 2]))
    var data: List[Float32] = [1, 3, 2, 2, 4]
    csr.append(data^)
    var sparse = _apply(csr^, _csr(2, 3, 5), excl, 10.0)
    var dense = _apply(owners^, _dense(2, 3), excl, 10.0)
    var pick: List[Int] = [0, 1, 3, 4, 5]
    for e in range(5):  # small-loop(fixture: the stored entries against their dense words)
        require(bitcast[DType.uint32](sparse[e]) == bitcast[DType.uint32](dense[pick[e]]),
                "CSR row totals: entry " + String(e) + " differs from the dense word")


def test_row_totals() raises:
    check_cases("row_totals", 3, row_totals_unit)


def rule_unit(unit: Int) raises:
    var x: List[Float32] = [1, 1, 8, 1, 1, 0]
    var owners = List[List[Float32]]()
    owners.append(x.copy())
    var prog = EncProgram(True)
    var plan = rowtotal_fit_stage(prog, RowTotalFit(_dense(2, 3), -1, 0, True, 0.5, False))
    var arena = enc_host_run(prog, owners)
    _close(arena, plan.excl, [0.0, 0.0, 1.0], 0.0, "the rule's excluded columns")
    var rule = _apply(owners.copy(), _dense(2, 3), arena_words(arena, plan.excl, 3), 4.0)
    _close(rule, 0, [2.0, 2.0, 16.0, 2.0, 2.0, 0.0], 1e-6, "rule totals to 4")
    # the same subset given as columns [0, 1]
    var with_idx = owners.copy()
    with_idx.append(_ints([0, 1]))
    var prog2 = EncProgram(True)
    var plan2 = rowtotal_fit_stage(prog2, RowTotalFit(_dense(2, 3), 1, 2, False, 0.05, False))
    var arena2 = enc_host_run(prog2, with_idx)
    _same(arena, plan.excl, arena2, plan2.excl, 3, "explicit columns mask")
    var explicit = _apply(owners^, _dense(2, 3), arena_words(arena2, plan2.excl, 3), 4.0)
    _same(rule, 0, explicit, 0, 6, "explicit columns against the rule")


def test_rule() raises:
    check_cases("rule", 1, rule_unit)


def log1p_unit(unit: Int) raises:
    var owners = List[List[Float32]]()
    var x: List[Float32] = [0, 1, 3]
    owners.append(x^)
    var prog = EncProgram(True)
    var o = log1p_stage(prog, _dense(1, 3), 2.0, unit == 1, False)
    var arena = enc_host_run(prog, owners)
    if unit == 0:
        _close(arena, o, [0.0, log(2.0), log(4.0)], 1e-6, "log1p")
    else:
        _close(arena, o, [0.0, 1.0, 2.0], 1e-6, "log1p base 2")


def test_log1p() raises:
    check_cases("log1p", 2, log1p_unit)


def scale_unit(unit: Int) raises:
    var owners = List[List[Float32]]()
    var x: List[Float32] = [1, 2, 3, 4]
    owners.append(x.copy())
    var prog = EncProgram(True)
    var plan = col_block_stage(prog, ColBlock(_dense(4, 1), -1, 1, 0, 1, SP_PRE_NONE, False, -1))
    var arena = enc_host_run(prog, owners)
    all_close(arena[plan.out], 2.5, 1e-7, 1e-7, "mean")
    all_close(arena[plan.out + 1], 5.0 / 3.0, 1e-6, 1e-6, "variance ddof 1")
    var std: List[Float32] = [Float32(sqrt(Float64(5.0) / Float64(3.0)))]
    var mean: List[Float32] = [2.5]
    owners.append(mean^)
    owners.append(std^)
    var prog2 = EncProgram(True)
    var o = scale_apply_stage(prog2, ScaleApply(_dense(4, 1), 1, 2, True, True, 1.0, False))
    var arena2 = enc_host_run(prog2, owners)
    _close(arena2, o, [-1.0, -0.38729833, 0.38729833, 1.0], 1e-5, "clipped scale")


def test_scale() raises:
    check_cases("scale", 1, scale_unit)


def ranks_unit(unit: Int) raises:
    var nan = bitcast[DType.float32](UInt32(0x7FC00000))
    var keys: List[Float32] = [3, 1, 2, nan]
    var owners = List[List[Float32]]()
    owners.append(keys^)
    var prog = EncProgram(True)
    var k = enc_input(prog, 4, 0)
    var r = prog.alloc(4)
    sp_rank_stages(prog, k, -1, 4, 1, unit, r)
    var arena = enc_host_run(prog, owners)
    var want: List[Int] = [2, 0, 1, -1]
    if unit == 1:
        want = [0, 2, 1, -1]
    for i in range(4):  # small-loop(fixture: four ranks)
        require(_int_at(arena, r + i) == want[i], "rank " + String(i) + " desc " + String(unit))


def test_ranks() raises:
    check_cases("ranks", 2, ranks_unit)


def _counted_rank(keys: List[Float32], groups: List[Int], d: Int, b: Int, g: Int, desc: Bool) -> Int:
    """The counted rank the sort-based stages replace (lane sc-leftovers-1): genes of batch b in g's group with a
    non-NaN key before g's, ties (-0.0 == +0.0) by index; -1 for a NaN key or a negative group."""
    var key = keys[b * d + g]
    var mine = groups[b * d + g]
    if isnan(key) or mine < 0:
        return -1
    var r = 0
    for h in range(d):  # small-loop(fixture: the oracle's count over the batch)
        var v = keys[b * d + h]
        if h == g or isnan(v) or groups[b * d + h] != mine:
            continue
        if (v == key and h < g) or (v != key and ((desc and v > key) or (not desc and v < key))):
            r += 1
    return r


def chunked_ranks_unit(unit: Int) raises:
    """Two batches of 2500 genes (three sorted chunks each), three groups, repeated keys, -0.0 beside +0.0, NaN
    keys and a negative group: every rank equals the counted rank, ascending (unit 0) and descending (unit 1)."""
    var d = 2500
    var nb = 2
    var nan = bitcast[DType.float32](UInt32(0x7FC00000))
    var keys = List[Float32](length=nb * d, fill=0)
    var groups = List[Int](length=nb * d, fill=0)
    for t in range(nb * d):  # small-loop(fixture: the keys and groups)
        keys[t] = Float32((t * 7919) % 613) - Float32(300)
        groups[t] = t % 3
    keys[5] = Float32(-0.0)
    keys[8] = Float32(0.0)
    keys[17] = nan
    keys[d + 40] = nan
    groups[23] = -1
    groups[d + 99] = -1
    var owners = List[List[Float32]]()
    owners.append(keys.copy())
    owners.append(_ints(groups))
    var prog = EncProgram(True)
    var k = enc_input(prog, nb * d, 0)
    var gw = enc_input(prog, nb * d, 1)
    var r = prog.alloc(nb * d)
    sp_rank_stages(prog, k, gw, d, nb, unit, r)
    var arena = enc_host_run(prog, owners)
    for t in range(nb * d):  # small-loop(fixture: every rank against the oracle)
        var want = _counted_rank(keys, groups, d, t // d, t % d, unit == 1)
        require(_int_at(arena, r + t) == want, "chunked rank " + String(t) + " desc " + String(unit))


def test_chunked_ranks() raises:
    check_cases("chunked ranks", 2, chunked_ranks_unit)


def percentiles_unit(unit: Int) raises:
    var s = List[Float32](length=11, fill=0)
    for i in range(11):  # small-loop(fixture: the sorted 1 .. 11)
        s[i] = Float32(i + 1)
    var owners = List[List[Float32]]()
    owners.append(s^)
    var prog = EncProgram(True)
    var so = enc_input(prog, 11, 0)
    var o = prog.alloc(19)
    prep_program_stage(prog.program, SP_PERCENTILES, 19, so, 11, o)
    var arena = enc_host_run(prog, owners)
    all_close(arena[o], 2.0, 1e-6, 1e-6, "p10")
    all_close(arena[o + 1], 2.5, 1e-6, 1e-6, "p15")
    all_close(arena[o + 18], 11.0, 0.0, 0.0, "p100")


def test_percentiles() raises:
    check_cases("percentiles", 1, percentiles_unit)


def loess_unit(unit: Int) raises:
    var d = 40
    var lx = List[Float32](length=d, fill=0)
    var ly = List[Float32](length=d, fill=0)
    for i in range(d):  # small-loop(fixture: an exact quadratic, LX descending so the sort matters)
        var x = Float32(d - i) * 0.05
        lx[i] = x
        ly[i] = 0.5 * x * x - x + 2.0
    var sx = List[Float32](length=d, fill=0)
    for i in range(d):  # small-loop(fixture: LX sorted ascending)
        sx[i] = lx[d - 1 - i]
    var owners = List[List[Float32]]()
    owners.append(lx.copy())
    owners.append(ly.copy())
    owners.append(sx^)
    var prog = EncProgram(True)
    var a = enc_input(prog, d, 0)
    var b = enc_input(prog, d, 1)
    var c = enc_input(prog, d, 2)
    var span = prog.put_floats([0.3])
    var est = prog.alloc(d)
    prep_program_stage(prog.program, SP_LOESS, d, a, b, c, d, span, est)
    var arena = enc_host_run(prog, owners)
    for i in range(d):  # small-loop(fixture: every fitted value)
        all_close(arena[est + i], Float64(ly[i]), 1e-4, 1e-4, "loess of a quadratic at " + String(i))


def test_loess() raises:
    check_cases("loess", 1, loess_unit)


def regress_unit(unit: Int) raises:
    var n = 6
    var cov = List[Float32](length=n, fill=0)
    var x = List[Float32](length=n * 2, fill=0)
    for i in range(n):  # small-loop(fixture: x = 2 c + 1 and a zero column)
        cov[i] = Float32(i) * 0.5
        x[i * 2] = 2.0 * cov[i] + 1.0
    var owners = List[List[Float32]]()
    var xin: SpInput
    if unit == 0:
        owners.append(x^)
        xin = _dense(n, 2)
    else:
        var ip = List[Int](length=n + 1, fill=0)
        var ix = List[Int](length=n, fill=0)
        var dv = List[Float32](length=n, fill=0)
        for i in range(n):  # small-loop(fixture: one stored entry a row, column 0)
            ip[i + 1] = i + 1
            dv[i] = x[i * 2]
        owners.append(_ints(ip))
        owners.append(_ints(ix))
        owners.append(dv^)
        xin = _csr(n, 2, n)
    owners.append(cov^)
    var prog = EncProgram(True)
    var plan = regress_fit_stage(prog, RegressFit(xin^, len(owners) - 1, 1, True, False))
    var arena = enc_host_run(prog, owners)
    for i in range(n * 2):  # small-loop(fixture: every residual)
        all_close(arena[plan.out + i], 0.0, 0.0, 1e-5, "residual " + String(i) + " csr " + String(unit))


def test_regress() raises:
    check_cases("regress", 2, regress_unit)


def select_unit(unit: Int) raises:
    # [[1, 2, 3], [0, 4, 5]] as CSR; keep columns 0 and 2
    var owners = List[List[Float32]]()
    owners.append(_ints([0, 3, 5]))
    owners.append(_ints([0, 1, 2, 1, 2]))
    var data: List[Float32] = [1, 2, 3, 4, 5]
    owners.append(data^)
    var hv = _ints([1, 0, 1])
    owners.append(hv.copy())
    var prog = EncProgram(True)
    var h = enc_input(prog, 3, 3)
    var keep = keep_stage(prog, 3, h)
    var arena = enc_host_run(prog, owners)
    require(_int_at(arena, keep.count) == 2, "two kept columns")
    var inv = arena_words(arena, keep.inv, 3)
    owners.append(inv^)
    var prog2 = EncProgram(True)
    var plan = select_csr_stage(prog2, _csr(2, 3, 5), 4)
    var arena2 = enc_host_run(prog2, owners)
    var ptr: List[Int] = [0, 2, 3]
    var idx: List[Int] = [0, 1, 1]
    var val: List[Float64] = [1.0, 3.0, 5.0]
    for i in range(3):  # small-loop(fixture: the result's indptr)
        require(_int_at(arena2, plan.indptr + i) == ptr[i], "indptr " + String(i))
        require(_int_at(arena2, plan.indices + i) == idx[i], "indices " + String(i))
    _close(arena2, plan.data, val, 0.0, "kept values")


def test_select() raises:
    check_cases("select", 1, select_unit)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
