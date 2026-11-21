# SPDX-License-Identifier: Apache-2.0
"""RUN of SimpleImputer: the typed requests of fit and transform and the
stage builders of their x_prep programs (lane dw-prep-encode, 2026-11-21).

The door (bindings/prep_encode_native_door.mojo `_imputer_fit`,
`_imputer_transform`) is UNPACK (the parameters and fitted fields read once
into `ImputerFit` / `ImputerApply`; X and the fitted fill vector become
owners of the door's `EncProg`), then one builder here on the typed
`EncProgram`, then the typed run, then PACK (statistics, kept columns and the
result read back from the arena). No builder reads a field, a mode or an
entry table. The same allocations and stages in the same order as the door's
former builders, so the bits do not move. No Python import
(tools/python_free_dirs.sh)."""
from bindings.prep_encode_prog_run import EncProgram, enc_input, enc_col_stats, enc_mark_missing, enc_floats_of
from bindings.x_prep_native_owner_run import xprep_r3
from core.preprocessing_program import prep_program_stage

comptime IMP_NONE = -1
# x_prep stage codes (_expansion_prep._OPS).
comptime IMP_OP_SORT_COLS = 0
comptime IMP_OP_QUANTILE = 2
comptime IMP_OP_MODE_COLS = 6
comptime IMP_OP_FILL = 24
comptime IMP_OP_NAN_MASK = 60
comptime IMP_OP_HCAT = 140
comptime IMP_OP_IMP_STATS = 252
# `ImputerFit.strategy`
comptime IMP_MEAN = 0
comptime IMP_MEDIAN = 1
comptime IMP_MOST_FREQUENT = 2
comptime IMP_CONSTANT = 3


@fieldwise_init
struct ImputerFit(Copyable, Movable):
    """SimpleImputer.fit of the (n, d) block of owner `x`. `mark`: a numeric
    missing_values (`missing`) is marked NaN first. `qselect`: the median by
    radix select (no sort). `stats_blocked`: the image's blocked column
    statistics. `fill`: the constant strategy's value."""
    var n: Int
    var d: Int
    var x: Int
    var mark: Bool
    var missing: Float64
    var strategy: Int
    var qselect: Bool
    var stats_blocked: Bool
    var fill: Float64
    var keep_empty: Bool


@fieldwise_init
struct ImputerFitPlan(Copyable, Movable):
    """Where the fit leaves its results: the six statistic rows (row 0: the
    per-column non-missing counts), the strategy's row, and the statistics /
    fill rows of a constant strategy or an empty column."""
    var stats: Int
    var source: Int
    var statistics: Int
    var fill: Int


@fieldwise_init
struct ImputerApply(Movable):
    """SimpleImputer.transform of the (n, d) block of owner `x`: the fitted
    fill vector (owner `fill`, `fill_words` words), the kept columns and the
    indicator columns."""
    var n: Int
    var d: Int
    var x: Int
    var mark: Bool
    var missing: Float64
    var fill: Int
    var fill_words: Int
    var keep: List[Int]
    var indicator: List[Int]


def _one(value: Float64) -> List[Float64]:
    var out = List[Float64]()
    out.append(value)
    return out^


def imputer_fit_stage(mut prog: EncProgram, r: ImputerFit) -> ImputerFitPlan:
    var n = r.n
    var d = r.d
    var x_in = prog.input(n * d, r.x)
    var xo = enc_mark_missing(prog, x_in, n * d, r.mark, r.missing)
    # only the median and the mode read the sorted columns (`imputer_nosort`);
    # the median by radix select needs no sort (`_prep2_qselect`)
    var median = r.strategy == IMP_MEDIAN
    var ordered = median or r.strategy == IMP_MOST_FREQUENT
    var qsel = median and r.qselect
    var sorts = (ordered or not xprep_r3("imputer_nosort")) and not qsel
    var so = 0
    if sorts:
        so = prog.work(n * d) if ordered else prog.alloc(n * d)
    var st = prog.alloc(6 * d)
    var med = prog.alloc(d)
    var mf = prog.alloc(d)
    var half = prog.put_floats(_one(0.5))
    if sorts:
        prep_program_stage(prog.program, IMP_OP_SORT_COLS, d, xo, n, d, so, 0)
    enc_col_stats(prog, r.stats_blocked, xo, n, d, st, False)
    if median and qsel:
        prep_program_stage(prog.program, IMP_OP_QUANTILE, d, xo, n, d, half, 1, med, st, 1)
    elif median:
        prep_program_stage(prog.program, IMP_OP_QUANTILE, d, so, n, d, half, 1, med, st)
    if r.strategy == IMP_MOST_FREQUENT:
        prep_program_stage(prog.program, IMP_OP_MODE_COLS, d, so, n, d, mf, IMP_NONE)
    var konst = r.strategy == IMP_CONSTANT
    var src = 0
    var fv = IMP_NONE
    if konst:
        fv = prog.put_floats(_one(r.fill))
    if konst or r.strategy == IMP_MEAN:
        src = st + d
    elif median:
        src = med
    else:
        src = mf
    var so2 = prog.alloc(d)
    var fo2 = prog.alloc(d)
    prep_program_stage(prog.program, IMP_OP_IMP_STATS, d, st, src, d, 1 if r.keep_empty else 0, 1 if konst else 0, fv,
                       so2, fo2)
    return ImputerFitPlan(st, src, so2, fo2)


def imputer_transform_stage(mut prog: EncProgram, r: ImputerApply, region: Bool) raises -> Int:
    """The filled kept columns, and beside them the missing indicator
    columns when there are any. Returns the offset of the (n, kept +
    indicator) float32 result."""
    var n = r.n
    var d = r.d
    var dout = len(r.keep)
    var x_in = prog.input(n * d, r.x)
    var xo = enc_mark_missing(prog, x_in, n * d, r.mark, r.missing)
    var so = enc_input(prog, r.fill_words, r.fill)
    var ko = prog.put_floats(enc_floats_of(r.keep))
    var out = prog.alloc(n * dout)
    prep_program_stage(prog.program, IMP_OP_FILL, n * dout, xo, n, d, so, out, ko, dout)
    var m = len(r.indicator)
    if m == 0:
        return out
    var io = prog.put_floats(enc_floats_of(r.indicator))
    var mo = prog.alloc(n * m)
    prep_program_stage(prog.program, IMP_OP_NAN_MASK, n * m, xo, n, d, io, m, mo)
    var hc = prog.output(n * (dout + m), region)
    prep_program_stage(prog.program, IMP_OP_HCAT, n * (dout + m), out, dout, mo, m, hc)
    return hc
