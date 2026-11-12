# SPDX-License-Identifier: Apache-2.0
"""RUN of `bindings/prep_stateless_native.mojo` (wave 8 run split): the x_prep
stage programs of the stateless and fitted preprocessing owners
(PolynomialFeatures term tables, VarianceThreshold, QuantileTransformer and
PowerTransformer fits, column statistics) and their arena layout, as typed
Mojo. Imports no Python. The binding keeps UNPACK (fields, the tier, the
entry words `_fam`/`_pt_flags`, the work switch) and PACK (fitted arrays), and
runs each program through the shared transport as before."""
from std.os import getenv
from core.preprocessing_program import prep_program_stage as _stage

comptime F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime I64P = MutPointer[Int64, MutUntrackedOrigin]

# x_prep stage codes (_expansion_prep._OPS).
comptime OP_BINARIZE = 12
comptime OP_NORMALIZE = 47
comptime OP_POLY = 48
comptime OP_TRANSPOSE = 209
comptime OP_COL_STATS = 1
comptime OP_GATHER_COLS = 61
comptime OP_VAR_PTP = 62
comptime OP_COLB_PART = 124
comptime OP_COLB_FOLD = 125
comptime OP_COLB_SS = 126
comptime OP_COLB_VAR = 127
comptime OP_KEY64 = 253
comptime OP_GT = 255
comptime OP_SORT_COLS = 0
comptime OP_QUANTILE = 2
comptime OP_QT_APPLY = 43
comptime OP_QT_INVERSE = 74
comptime OP_RGATHER = 204
comptime OP_PERM_ROWS = 233
comptime OP_GRID = 263
comptime OP_PT_FIT = 44
comptime OP_PT_APPLY = 45
comptime OP_STD_PARAMS = 46
comptime OP_PT_INVERSE = 75
comptime OP_PT_INIT = 104
comptime OP_PT_MAP = 105
comptime OP_PT_FOLD = 106
comptime OP_PT_LOG = 109
comptime OP_PT_SPTS = 110
comptime OP_PT_SMAP = 111
comptime OP_PT_SFOLD = 112
comptime OP_PT_SRES = 113
comptime OP_PTB_PART1 = 172
comptime OP_PTB_MEAN = 173
comptime OP_PTB_PART2 = 174
comptime OP_PTB_FIN = 175
comptime OP_PTB_STEP = 176
# PowerTransformer's fixed golden-section evaluations and per-column state words.
comptime PT_EVALS = 50
comptime PT_STATE = 10
# x_prep_idn_fam bit 8: IDN_PT_BLOCKED.
comptime IDN_PT_BLOCKED = 8
# Row blocks of the blocked column statistics (_expansion_prep._XB).
comptime ROW_BLOCK = 2048
# x_prep_idn_fam bit 1: IDN_STATS_BLOCKED.
comptime IDN_STATS_BLOCKED = 1
# `_Prog.output`: below this many words the output is plain arena words.
comptime OUT_MIN_WORDS = 134217728


def _emit(mut index: List[Int], mut start: List[Int], terms: List[Int]):
    for i in range(len(terms)):  # driver-loop: over one polynomial term's feature indices
        index.append(terms[i])
    start.append(len(index))


def _combinations(d: Int, k: Int, replacement: Bool, mut index: List[Int], mut start: List[Int]):
    # itertools.combinations(_with_replacement)(range(d), k), in its order.
    if k == 0:
        _emit(index, start, List[Int]())
        return
    if not replacement and k > d:
        return
    var c = List[Int](length=k, fill=0)
    if not replacement:
        for i in range(k):  # driver-loop: over one polynomial term's positions
            c[i] = i
    while True:  # driver-loop: over polynomial terms of feature indices
        _emit(index, start, c)
        var i = k - 1
        while i >= 0 and c[i] >= (d - 1 if replacement else d - k + i):  # driver-loop: over one term's positions
            i -= 1
        if i < 0:
            return
        c[i] += 1
        for j in range(i + 1, k):  # driver-loop: over one term's positions
            c[j] = c[i] if replacement else c[j - 1] + 1


def _poly_terms(d: Int, lo: Int, hi: Int, replacement: Bool, include_bias: Bool, mut index: List[Int],
               mut start: List[Int]):
    # The PolynomialFeatures term table (`_fit`): index words and term starts.
    start.append(0)
    if include_bias:
        _combinations(d, 0, replacement, index, start)
    for k in range(max(1, lo), hi + 1):  # driver-loop: over polynomial degrees
        _combinations(d, k, replacement, index, start)


def _poly_powers(table: I64P, d: Int, nout: Int, index: List[Int], start: List[Int]):
    # powers_[t, j]: how often feature j occurs in term t (`_fit`).
    for t in range(nout):  # driver-loop: over polynomial terms
        for i in range(start[t], start[t + 1]):  # driver-loop: over one term's feature indices
            table[t * d + index[i]] += 1


struct PrepLayout(Movable):
    """A program's arena words and device scratch words, in allocation order
    (`_Prog.alloc` / `_Prog.work`): scratch offsets are taken past the host
    words when the program runs, so they are resolved with `scratch_at`."""
    var host: Int
    var scratch: Int
    var work: Bool

    def __init__(out self, host: Int, work: Bool):
        self.host = host
        self.scratch = 0
        self.work = work

    def alloc(mut self, words: Int) -> Int:
        var offset = self.host
        self.host += max(words, 0)
        return offset

    def scratch_words(mut self, words: Int) -> Int:
        # `_Prog.scratch`: device-only words, whatever the `work` switch.
        var offset = self.scratch
        self.scratch += max(words, 0)
        return -(offset + 2)

    def work_words(mut self, words: Int) -> Int:
        # A stage word -(k + 2) marks scratch word k (-1 is the stages' "none").
        if not self.work:
            return self.alloc(words)
        var offset = self.scratch
        self.scratch += max(words, 0)
        return -(offset + 2)


def _col_stats_run(mut program: List[Int32], mut layout: PrepLayout, fam: Int, xo: Int, n: Int, d: Int,
                   out_buf: Int, variance: Bool = True) raises:
    """`_col_stats` with the family word `fam` read by the door (`_fam`: the
    IDENTICAL binding's x_prep_idn_fam, 0 on another tier)."""
    var blocked = String(getenv("MOJOLEARN_XPREP_BLOCKED", "1").strip()) != "0"
    if blocked and (fam & IDN_STATS_BLOCKED) != 0:
        var nb = (n + ROW_BLOCK - 1) // ROW_BLOCK
        var part = layout.work_words(5 * nb * d)
        _stage(program, OP_COLB_PART, nb * d, xo, n, d, part, nb)
        _stage(program, OP_COLB_FOLD, d, part, nb, d, out_buf)
        if variance:
            var ss = layout.work_words(nb * d)
            _stage(program, OP_COLB_SS, nb * d, xo, n, d, out_buf, ss, nb)
            _stage(program, OP_COLB_VAR, d, ss, nb, d, out_buf)
    else:
        _stage(program, OP_COL_STATS, d, xo, n, d, out_buf)


def _place_scratch(mut program: List[Int32], host: Int):
    # A stage word -(k + 2) is scratch word k, at host + k.
    for i in range(len(program) // 16):  # driver-loop: over the program's stages
        for k in range(2, 16):  # small-loop(params: the fixed fourteen-word stage ABI)
            var word = Int(program[i * 16 + k])
            if word < -1:
                program[i * 16 + k] = Int32(host - word - 2)


def _variance_program(mut program: List[Int32], mut layout: PrepLayout, fam: Int, n: Int, d: Int,
                      zero_threshold: Bool) raises -> Tuple[Int, Int, Int]:
    """VarianceThreshold.fit's stages (`_variance_fit`): column statistics,
    var_ptp, then variance > threshold in binary64 (c2_key64, c2_gt).
    Returns the variance, mask and threshold-word offsets."""
    var st = layout.alloc(6 * d)
    var variance = layout.alloc(d)
    _col_stats_run(program, layout, fam, 0, n, d, st)
    _stage(program, OP_VAR_PTP, d, st, d, variance, 1 if zero_threshold else 0)
    var key = layout.work_words(2 * d)
    var mask = layout.alloc(d)
    var limit = layout.alloc(2)
    _stage(program, OP_KEY64, d, variance, 0, key)
    _stage(program, OP_GT, d, key, limit, mask)
    return (variance, mask, limit)


def _quantile_program(mut program: List[Int32], mut layout: PrepLayout, fam: Int, xo: Int, n: Int, d: Int,
                      nq: Int) raises -> Tuple[Int, Int, Int, Int]:
    """QuantileTransformer.fit's stages after the optional row subsample
    (`_quantile_fit`): the reference grid (c2_grid), sort_cols, column
    statistics, quantile and the p2m_transpose of quantiles_. Returns the
    grid, grid-level word, quantile and transposed-quantile offsets; the door
    supplies the grid-level word (nq - 1) as an input at the returned offset."""
    var sorted = layout.work_words(n * d)
    var st = layout.alloc(6 * d)
    var grid = layout.alloc(nq)
    var nb = layout.alloc(1)
    _stage(program, OP_GRID, nq, nb, nq, 1, grid)
    var qo = layout.alloc(nq * d)
    _stage(program, OP_SORT_COLS, d, xo, n, d, sorted, 0)
    _col_stats_run(program, layout, fam, xo, n, d, st, False)
    _stage(program, OP_QUANTILE, nq * d, sorted, n, d, grid, nq, qo, st)
    var qt = layout.alloc(nq * d)
    _stage(program, OP_TRANSPOSE, nq * d, qo, d, nq, qt)
    return (grid, nb, qo, qt)


def _pt_spec_depth(n: Int, d: Int) -> Int:
    """`_expansion_prep._pt_spec_depth`: MOJOLEARN_XPREP_PT_SPEC (default 3,
    clamped to [0, 6]), lowered until the candidates' transforms fit in 2^28
    words (a size range, not a shape); 0 when even two do not."""
    var s = 3
    try:
        s = Int(String(getenv("MOJOLEARN_XPREP_PT_SPEC", "3")).strip())
    except:
        s = 3
    s = max(0, min(s, 6))
    while s > 1 and ((1 << s) - 1) * n * d > 268435456:  # driver-loop: over speculation depths
        s -= 1
    if s == 1 and 2 * n * d > 268435456:
        s = 0
    return s


def _power_program(mut program: List[Int32], mut layout: PrepLayout, fam: Int, flags: Int, host: Bool,
                   identical: Bool, n: Int, d: Int, method: Int, standardize: Bool) raises -> Tuple[Int, Int, Int, Int, Int, Int, Bool]:
    """PowerTransformer.fit's stages (`_power_fit`): the host column's pt_fit,
    or the device's golden-section search as stages (staged, or speculated
    `spec` deep, with the blocked folds under IDN_PT_BLOCKED), then the
    standardization statistics. The door reads `fam` (`_fam`), `flags`
    (`_pt_flags`), `host` (the host-column binding) and the tier. Returns the
    statistics, lambda, anchor, anchor-kind, mean and scale offsets and
    whether the centered coordinates are on."""
    var xo = 0
    var st = layout.alloc(6 * d)
    var lam = layout.alloc(d)
    # PT_SCORE_STABLE (bit 32): the device's centered coordinates; never on the host binding
    var centered = (flags & 32) != 0 and not host
    var anchor = layout.alloc(d) if centered else -1
    var anchor_kind = layout.alloc(d) if centered else -1
    _col_stats_run(program, layout, fam, xo, n, d, st)
    var blocked = String(getenv("MOJOLEARN_XPREP_BLOCKED", "1").strip()) != "0"
    var fam_pt = blocked and (fam & IDN_PT_BLOCKED) != 0
    var nb = (n + ROW_BLOCK - 1) // ROW_BLOCK
    var unit = 1 if standardize else 0
    if not fam_pt and host:
        _stage(program, OP_PT_FIT, d, xo, n, d, method, st, lam)
    else:
        var tiled = (flags & 1) != 0
        var spec = _pt_spec_depth(n, d) if identical else 0
        if spec:
            var mmax = max((1 << spec) - 1, 2)
            var il = 0
            if not tiled and String(getenv("MOJOLEARN_XPREP_PT_INTERLEAVE", "0")) == "1":
                il = 1
            var state = layout.alloc(PT_STATE * d)
            var leval = layout.alloc(d)
            var spl = layout.alloc(d * mmax)
            var vals = layout.alloc(d * mmax)
            var lg = layout.scratch_words(1 if tiled else n * d)
            var tv = layout.scratch_words(1 if tiled else n * d * mmax)
            _stage(program, OP_PT_INIT, d, method, st, d, lam, state, leval, anchor, anchor_kind, unit)
            if not tiled:
                _stage(program, OP_PT_LOG, n * d, xo, n, d, method, lg)
            var bps = 0
            var bpc = 0
            var bss = 0
            var bpj = 0
            var bmean = 0
            var bcnt = 0
            if fam_pt:
                bps = layout.work_words(nb * d * mmax)
                bpc = layout.work_words(nb * d * mmax)
                bss = layout.work_words(nb * d * mmax)
                bpj = layout.work_words(nb * d)
                bmean = layout.work_words(d * mmax)
                bcnt = layout.work_words(d * mmax)
            var k0 = 0
            while k0 <= PT_EVALS - 2:  # driver-loop: over the fixed golden-section evaluations
                var steps = 2 if k0 == 0 else min(spec, PT_EVALS - 1 - k0)
                var m = 2 if k0 == 0 else (1 << steps) - 1
                _stage(program, OP_PT_SPTS, d, state, leval, spl, m, k0)
                _stage(program, OP_PT_SMAP, m * n * d, xo, n, d, method, spl, m, tv, lg, il)
                if fam_pt:
                    var first = 1 if k0 == 0 else 0
                    _stage(program, OP_PTB_PART1, nb * d * m, xo, n, d, method, tv, m, state, bps, bpc, bpj, nb, first, il)
                    _stage(program, OP_PTB_MEAN, d * m, bps, bpc, bpj, nb, d, m, state, bmean, bcnt, first)
                    _stage(program, OP_PTB_PART2, nb * d * m, tv, n, d, m, state, bmean, bcnt, bss, nb, il)
                    _stage(program, OP_PTB_FIN, d * m, bss, nb, d, m, state, spl, vals, bcnt)
                else:
                    _stage(program, OP_PT_SFOLD, d * m, xo, n, d, method, tv, m, state, spl, vals, 1 if k0 == 0 else 0, il)
                _stage(program, OP_PT_SRES, d, state, leval, m, vals, k0, steps, lam)
                k0 += steps
        else:
            var state = layout.alloc(PT_STATE * d)
            var leval = layout.alloc(d)
            var tv = layout.alloc(1 if tiled else n * d)
            var lg = layout.alloc(1 if tiled else n * d)
            _stage(program, OP_PT_INIT, d, method, st, d, lam, state, leval, anchor, anchor_kind, unit)
            if not tiled:
                _stage(program, OP_PT_LOG, n * d, xo, n, d, method, lg)
            var bps = 0
            var bpc = 0
            var bss = 0
            var bpj = 0
            var bmean = 0
            var bcnt = 0
            if fam_pt:
                bps = layout.work_words(nb * d)
                bpc = layout.work_words(nb * d)
                bss = layout.work_words(nb * d)
                bpj = layout.work_words(nb * d)
                bmean = layout.work_words(d)
                _ = d  # keep the Python object alive through the native call above
                bcnt = layout.work_words(d)
                _ = d  # keep the Python object alive through the native call above
            for k in range(PT_EVALS):  # driver-loop: over the fixed golden-section evaluations
                # tiled: the device fuses pt_map into pt_fold (LG1 = 0 either way)
                _stage(program, OP_PT_MAP, n * d, xo, n, d, method, leval, tv, 0 if tiled else lg + 1)
                if fam_pt:
                    var first = 1 if k == 0 else 0
                    _stage(program, OP_PTB_PART1, nb * d, xo, n, d, method, tv, 1, state, bps, bpc, bpj, nb, first, 0)
                    _stage(program, OP_PTB_MEAN, d, bps, bpc, bpj, nb, d, 1, state, bmean, bcnt, first)
                    _stage(program, OP_PTB_PART2, nb * d, tv, n, d, 1, state, bmean, bcnt, bss, nb, 0)
                    _stage(program, OP_PTB_STEP, d, xo, n, d, method, tv, k, state, leval, lam, bss, nb, bcnt)
                else:
                    _stage(program, OP_PT_FOLD, d, xo, n, d, method, tv, k, state, leval, lam)
    var mean = layout.alloc(d)
    var scale = layout.alloc(d)
    if standardize:
        var tx = layout.alloc(n * d)
        var st2 = layout.alloc(6 * d)
        _stage(program, OP_PT_APPLY, n * d, xo, n, d, lam, method, -1, -1, tx, anchor, anchor_kind)
        _col_stats_run(program, layout, fam, tx, n, d, st2)
        _stage(program, OP_STD_PARAMS, d, st2, d, mean, scale)
    return (st, lam, anchor, anchor_kind, mean, scale, centered)
