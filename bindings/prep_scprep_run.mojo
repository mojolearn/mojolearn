# SPDX-License-Identifier: Apache-2.0
"""RUN of the count-data transformers (lane sc-prep-1, 2026-11-28): the typed requests and the stage builders of
their x_prep programs (units in x_prep/sc_prep.mojo).

The door (bindings/prep_scprep_native_door.mojo) is UNPACK (X dense or CSR, the parameter words, the fitted
vectors), then the programs built here and run, then PACK. No builder reads a field, a mode or an entry table; no
Python import (tools/python_free_dirs.sh).

Input. `SpInput` names X: a dense owner (n * d float32 words) or the CSR owners (INDPTR n + 1, INDICES nnz int32
words, DATA nnz float32 words). `sp_input` records it in a program and returns the arena offsets every unit reads.

Gene blocks. The column statistics read a gene-major, batch-ordered copy of X, n x dc words per program; dc is the
most genes whose copy stays within SP_SCRATCH_WORDS (`sp_chunk_width`). It is a memory bound, never a shape rule:
every n and d take the same formula, and the blocks change no bit (each gene's units read only its column).
"""
from core.preprocessing_program import prep_program_stage
from bindings.prep_encode_prog_run import EncProgram, enc_input
from x_prep.sc_prep import (
    SP_ZERO, SP_INCLUDE, SP_ROW_SUM, SP_EXCL_RULE, SP_MEDIAN_POS, SP_SCALE_ROWS, SP_LOG1P, SP_ORDER,
    SP_GATHER_DENSE, SP_GATHER_CSR, SP_SEG_STATS, SP_SEG_CLIP, SP_DISP, SP_MINMAX, SP_BIN_SEURAT, SP_PERCENTILES,
    SP_BIN_CR, SP_BIN_STATS, SP_RANK, SP_GROUP_START, SP_SCATTER, SP_SEG_MEDIAN, SP_ABSDEV, SP_NORM_DISP,
    SP_HV_CUT, SP_COMBINE, SP_HV_BATCH, SP_V3_KEY, SP_LOESS, SP_COUNT, SP_V3_CLIPVAL, SP_V3_NORMVAR,
    SP_V3_SELECT, SP_V3_HV, SP_SCAN_PART, SP_SCAN_TOP, SP_SCAN_FILL, SP_COMPACT, SP_COL_SELECT, SP_CSR_SELECT,
    SP_SCALE_APPLY, SP_SCALE_CSR, SP_COL_MEANSTD, SP_STANDARDIZE, SP_PINV, SP_DENSIFY, SP_CENTER, SP_RESID,
    SP_DENSE, SP_CSR, SP_PRE_NONE, SP_PRE_EXPM1, SP_SEURAT, SP_CELL_RANGER, SP_SEURAT_V3, SP_SCAN_BLOCK,
    SP_CR_EDGES,
)

#: x_prep ops reused from x_prep/units.mojo.
comptime OP_SORT_COLS = 0
comptime OP_MATMUL = 13
comptime OP_EIGH = 18

#: The gene-major block's words (2^27 float32 words, 512 MiB of device scratch per program).
comptime SP_SCRATCH_WORDS = 134217728
#: Seurat's mean bins when the caller gives none (Scanpy's n_bins default).
comptime SP_DEFAULT_BINS = 20


def sp_chunk_width(n: Int, d: Int) -> Int:
    """Genes per block program: all of them when n x d fits SP_SCRATCH_WORDS, else the most that fit (at least
    one)."""
    if n <= 0 or d <= 0:
        return max(d, 1)
    return max(1, min(d, SP_SCRATCH_WORDS // n))


def _blocks(n: Int) -> Int:
    return (n + SP_SCAN_BLOCK - 1) // SP_SCAN_BLOCK


@fieldwise_init
struct SpInput(Copyable, Movable):
    """X of one call: `csr` False: the owner `x` of n * d float32 words; True: the owners `indptr`, `indices`,
    `data` (nnz words each but indptr, n + 1)."""
    var n: Int
    var d: Int
    var csr: Bool
    var x: Int
    var indptr: Int
    var indices: Int
    var data: Int
    var nnz: Int


@fieldwise_init
struct SpPlaced(Copyable, Movable):
    """X's arena offsets: VALUES (dense X, or CSR DATA), INDPTR and INDICES (-1 for dense)."""
    var mode: Int
    var values: Int
    var indptr: Int
    var indices: Int


def sp_input(mut prog: EncProgram, x: SpInput) -> SpPlaced:
    if x.csr:
        var ip = enc_input(prog, x.n + 1, x.indptr)
        var ix = enc_input(prog, x.nnz, x.indices)
        var dv = enc_input(prog, x.nnz, x.data)
        return SpPlaced(SP_CSR, dv, ip, ix)
    var xo = enc_input(prog, x.n * x.d, x.x)
    return SpPlaced(SP_DENSE, xo, -1, -1)


# ---------------------------------------------------------------- the row-total normalizer


@fieldwise_init
struct RowTotalFit(Copyable, Movable):
    """The fit: `columns` the owner of the n_columns int32 indices the totals sum over (-1: every column);
    `rule` Scanpy's exclude_highly_expressed with `max_fraction`; `median` True when target_sum is None."""
    var x: SpInput
    var columns: Int
    var n_columns: Int
    var rule: Bool
    var max_fraction: Float64
    var median: Bool


@fieldwise_init
struct RowTotalFitPlan(Copyable, Movable):
    """Host words: EXCL (d words, 1.0 for a column left out of the totals) and MEDIAN (one word: the median of
    the positive totals, NaN when none is)."""
    var excl: Int
    var median: Int


def _row_sum(mut prog: EncProgram, x: SpInput, at: SpPlaced, excl: Int, sums: Int):
    prep_program_stage(prog.program, SP_ROW_SUM, x.n, at.mode, at.values, at.indptr, at.indices, x.n, x.d, excl,
                       sums)


def rowtotal_fit_stage(mut prog: EncProgram, r: RowTotalFit) -> RowTotalFitPlan:
    var x = r.x.copy()
    var at = sp_input(prog, x)
    var excl = prog.alloc(x.d)
    if r.columns >= 0:
        var idx = enc_input(prog, r.n_columns, r.columns)
        prep_program_stage(prog.program, SP_ZERO, x.d, excl, 1)
        if r.n_columns > 0:
            prep_program_stage(prog.program, SP_INCLUDE, r.n_columns, idx, excl)
    else:
        prep_program_stage(prog.program, SP_ZERO, x.d, excl, 0)
    if r.rule:
        var first = prog.work(x.n)
        _row_sum(prog, x, at, excl, first)
        var frac = prog.put_floats([r.max_fraction])
        prep_program_stage(prog.program, SP_EXCL_RULE, x.n, at.mode, at.values, at.indptr, at.indices, x.n, x.d,
                           first, frac, excl)
    var med = prog.alloc(1)
    if r.median:
        var sums = prog.work(x.n)
        _row_sum(prog, x, at, excl, sums)
        var srt = prog.work(x.n)
        prep_program_stage(prog.program, OP_SORT_COLS, 1, sums, x.n, 1, srt, 0)
        prep_program_stage(prog.program, SP_MEDIAN_POS, 1, srt, x.n, med)
    return RowTotalFitPlan(excl, med)


@fieldwise_init
struct RowTotalApply(Copyable, Movable):
    """The transform: `excl` the owner of the fitted d-word EXCL mask, `target` the fitted target_sum."""
    var x: SpInput
    var excl: Int
    var target: Float64
    var region: Bool


def rowtotal_apply_stage(mut prog: EncProgram, r: RowTotalApply) raises -> Int:
    """Returns OUT: n x d words (dense) or nnz words (CSR values, the input's structure)."""
    var x = r.x.copy()
    var at = sp_input(prog, x)
    var excl = enc_input(prog, x.d, r.excl)
    var sums = prog.work(x.n)
    _row_sum(prog, x, at, excl, sums)
    var tw = prog.put_floats([r.target])
    var o = prog.output(x.nnz if x.csr else x.n * x.d, r.region)
    prep_program_stage(prog.program, SP_SCALE_ROWS, x.n, at.mode, at.values, at.indptr, x.n, x.d, sums, tw, o)
    return o


# ---------------------------------------------------------------- the log transform


def log1p_stage(mut prog: EncProgram, x: SpInput, base: Float64, has_base: Bool, region: Bool) raises -> Int:
    """OUT = log1p of every value (dense) or of the stored values (CSR), over log(base) when given."""
    var at = sp_input(prog, x)
    var count = x.nnz if x.csr else x.n * x.d
    var bw = prog.put_floats([base]) if has_base else -1
    var o = prog.output(count, region)
    if count > 0:
        prep_program_stage(prog.program, SP_LOG1P, count, at.values, bw, o)
    return o


# ---------------------------------------------------------------- the gene-major block programs


@fieldwise_init
struct ColBlock(Copyable, Movable):
    """One block program: genes [c0, c0 + dc). `codes` the owner of the n int32 batch codes in [0, batches)
    (-1: one batch); `pre` SP_PRE_NONE or SP_PRE_EXPM1; `all_rows` adds the whole-data segment after the batches;
    `clip` (>= 0, the owner of batches x d clip values) runs seurat_v3's clipped sums instead of the moments."""
    var x: SpInput
    var codes: Int
    var batches: Int
    var c0: Int
    var dc: Int
    var pre: Int
    var all_rows: Bool
    var clip: Int


@fieldwise_init
struct ColBlockPlan(Copyable, Movable):
    """Host words: START (batches + 1 int words) and OUT (two segment-major blocks of segments x dc words: the
    means and variances, or the clipped sums and sums of squares)."""
    var start: Int
    var out: Int
    var segments: Int


def col_block_stage(mut prog: EncProgram, r: ColBlock) -> ColBlockPlan:
    var x = r.x.copy()
    var n = x.n
    var dc = r.dc
    var codes = enc_input(prog, n, r.codes) if r.codes >= 0 else -1
    var start = prog.alloc(r.batches + 1)
    var perm = prog.work(n)
    prep_program_stage(prog.program, SP_ORDER, r.batches if r.codes >= 0 else 1, codes, n, r.batches, start, perm)
    var g = prog.work(n * dc)
    var at = sp_input(prog, x)
    if x.csr:
        prep_program_stage(prog.program, SP_GATHER_CSR, n, at.indptr, at.indices, at.values, n, r.c0, dc, perm, g,
                           r.pre)
    else:
        prep_program_stage(prog.program, SP_GATHER_DENSE, n * dc, at.values, n, x.d, r.c0, perm, g, r.pre)
    var segments = r.batches + (1 if r.all_rows and r.clip < 0 else 0)
    var o = prog.alloc(2 * segments * dc)
    if r.clip >= 0:
        var clip = enc_input(prog, r.batches * x.d, r.clip)
        prep_program_stage(prog.program, SP_SEG_CLIP, r.batches * dc, g, n, dc, start, r.batches, clip, r.c0, x.d,
                           o)
    else:
        prep_program_stage(prog.program, SP_SEG_STATS, segments * dc, g, n, dc, start, r.batches,
                           1 if r.all_rows else 0, o)
    return ColBlockPlan(start, o, segments)


# ---------------------------------------------------------------- seurat and cell_ranger


@fieldwise_init
struct HvgDispersion(Copyable, Movable):
    """The whole-gene program of seurat and cell_ranger: `mean`, `var` the owners of batches x d words; `n_top`
    (-1: the cutoffs `cuts` = min_mean, max_mean, min_disp, max_disp)."""
    var d: Int
    var batches: Int
    var flavor: Int
    var n_bins: Int
    var mean: Int
    var var_: Int
    var n_top: Int
    var cuts: List[Float64]


@fieldwise_init
struct HvgDispersionPlan(Copyable, Movable):
    """Host words, each d words (int words for HV and NBATCHES): the means, dispersions and normalized dispersions
    the attributes report (one batch: its own; several: the merged ones), HV, NBATCHES (-1: one batch)."""
    var means: Int
    var disp: Int
    var disp_norm: Int
    var hv: Int
    var nbatches: Int


def hvg_dispersion_stage(mut prog: EncProgram, r: HvgDispersion) -> HvgDispersionPlan:
    var d = r.d
    var nb = r.batches
    var total = nb * d
    var mean = enc_input(prog, total, r.mean)
    var var_ = enc_input(prog, total, r.var_)
    var single = nb == 1
    var m = prog.alloc(total) if single else prog.work(total)
    var disp = prog.alloc(total) if single else prog.work(total)
    prep_program_stage(prog.program, SP_DISP, total, mean, var_, r.flavor, m, disp)
    var bins = prog.work(total)
    var ng: Int
    if r.flavor == SP_SEURAT:
        ng = r.n_bins
        var mm = prog.work(2 * nb)
        prep_program_stage(prog.program, SP_MINMAX, nb, m, d, mm)
        prep_program_stage(prog.program, SP_BIN_SEURAT, total, m, d, mm, ng, bins)
    else:
        ng = SP_CR_EDGES + 1
        var srt = prog.work(total)
        for b in range(nb):  # driver-loop: over the batches (one sort stage each)
            prep_program_stage(prog.program, OP_SORT_COLS, 1, m + b * d, d, 1, srt + b * d, 0)
        var pct = prog.work(nb * SP_CR_EDGES)
        prep_program_stage(prog.program, SP_PERCENTILES, nb * SP_CR_EDGES, srt, d, pct)
        prep_program_stage(prog.program, SP_BIN_CR, total, m, d, pct, bins)
    var avg = prog.work(nb * ng)
    var dev = prog.work(nb * ng)
    if r.flavor == SP_SEURAT:
        prep_program_stage(prog.program, SP_BIN_STATS, nb * ng, disp, bins, d, ng, avg, dev)
    else:
        # the bin medians of the dispersions, then of the absolute deviations (statsmodels `mad`)
        var rank = prog.work(total)
        var start = prog.work(nb * (ng + 1))
        var seg = prog.work(total)
        prep_program_stage(prog.program, SP_RANK, total, disp, bins, d, 0, rank)
        prep_program_stage(prog.program, SP_GROUP_START, nb * (ng + 1), disp, bins, d, ng, start)
        prep_program_stage(prog.program, SP_SCATTER, total, disp, bins, rank, d, ng, start, seg)
        prep_program_stage(prog.program, SP_SEG_MEDIAN, nb * ng, seg, d, ng, start, avg)
        var dabs = prog.work(total)
        prep_program_stage(prog.program, SP_ABSDEV, total, disp, bins, avg, d, ng, dabs)
        var rank2 = prog.work(total)
        var start2 = prog.work(nb * (ng + 1))
        var seg2 = prog.work(total)
        prep_program_stage(prog.program, SP_RANK, total, dabs, bins, d, 0, rank2)
        prep_program_stage(prog.program, SP_GROUP_START, nb * (ng + 1), dabs, bins, d, ng, start2)
        prep_program_stage(prog.program, SP_SCATTER, total, dabs, bins, rank2, d, ng, start2, seg2)
        prep_program_stage(prog.program, SP_SEG_MEDIAN, nb * ng, seg2, d, ng, start2, dev)
    var dn = prog.alloc(total) if single else prog.work(total)
    prep_program_stage(prog.program, SP_NORM_DISP, total, disp, bins, avg, dev, d, ng, r.flavor, dn)
    var cuts = prog.put_floats(r.cuts.copy())
    if single:
        var hv = prog.alloc(d)
        prep_program_stage(prog.program, SP_HV_CUT, d, m, dn, d, r.n_top, cuts, hv)
        return HvgDispersionPlan(m, disp, dn, hv, -1)
    var hvb = prog.work(total)
    prep_program_stage(prog.program, SP_HV_CUT, total, m, dn, d, r.n_top, cuts, hvb)
    var om = prog.alloc(d)
    var od = prog.alloc(d)
    var odn = prog.alloc(d)
    var nbat = prog.alloc(d)
    prep_program_stage(prog.program, SP_COMBINE, d, m, disp, dn, hvb, d, nb, om, od, odn, nbat)
    var hv = prog.alloc(d)
    prep_program_stage(prog.program, SP_HV_BATCH, d, nbat, odn, om, d, r.n_top, cuts, hv)
    return HvgDispersionPlan(om, od, odn, hv, nbat)


# ---------------------------------------------------------------- seurat_v3


@fieldwise_init
struct HvgV3Loess(Copyable, Movable):
    """seurat_v3's first whole-gene program: `mean`, `var` the owners of batches x d moments, `start` the owner of
    the batches + 1 row starts; `span` the loess span."""
    var d: Int
    var batches: Int
    var mean: Int
    var var_: Int
    var start: Int
    var span: Float64


@fieldwise_init
struct HvgV3LoessPlan(Copyable, Movable):
    """Host words: REGSTD and CLIP (batches x d words each) and NCNT (batches int words)."""
    var regstd: Int
    var clip: Int
    var ncnt: Int


def hvg_v3_loess_stage(mut prog: EncProgram, r: HvgV3Loess) -> HvgV3LoessPlan:
    var d = r.d
    var nb = r.batches
    var total = nb * d
    var mean = enc_input(prog, total, r.mean)
    var var_ = enc_input(prog, total, r.var_)
    var start = enc_input(prog, nb + 1, r.start)
    var lx = prog.work(total)
    var ly = prog.work(total)
    prep_program_stage(prog.program, SP_V3_KEY, total, mean, var_, lx, ly)
    var sx = prog.work(total)
    for b in range(nb):  # driver-loop: over the batches (one sort stage each)
        prep_program_stage(prog.program, OP_SORT_COLS, 1, lx + b * d, d, 1, sx + b * d, 0)
    var span = prog.put_floats([r.span])
    var est = prog.work(total)
    prep_program_stage(prog.program, SP_LOESS, total, lx, ly, sx, d, span, est)
    var ncnt = prog.alloc(nb)
    prep_program_stage(prog.program, SP_COUNT, nb, start, ncnt)
    var reg = prog.alloc(total)
    var clip = prog.alloc(total)
    prep_program_stage(prog.program, SP_V3_CLIPVAL, total, mean, est, ncnt, d, reg, clip)
    return HvgV3LoessPlan(reg, clip, ncnt)


@fieldwise_init
struct HvgV3Select(Copyable, Movable):
    """seurat_v3's choice: the owners of the batch means, REGSTD, the clipped SUM and SQ (batches x d each) and
    NCNT (batches int words)."""
    var d: Int
    var batches: Int
    var mean: Int
    var regstd: Int
    var sums: Int
    var squares: Int
    var ncnt: Int
    var n_top: Int


@fieldwise_init
struct HvgV3SelectPlan(Copyable, Movable):
    """Host words, d each: VARNORM, RANK (the median rank, NaN when never in a batch's top), NBATCHES and HV (int
    words)."""
    var varnorm: Int
    var rank: Int
    var nbatches: Int
    var hv: Int


def hvg_v3_select_stage(mut prog: EncProgram, r: HvgV3Select) -> HvgV3SelectPlan:
    var d = r.d
    var nb = r.batches
    var total = nb * d
    var mean = enc_input(prog, total, r.mean)
    var reg = enc_input(prog, total, r.regstd)
    var sm = enc_input(prog, total, r.sums)
    var sq = enc_input(prog, total, r.squares)
    var ncnt = enc_input(prog, nb, r.ncnt)
    var nv = prog.work(total)
    prep_program_stage(prog.program, SP_V3_NORMVAR, total, mean, reg, sm, sq, ncnt, d, nv)
    var rank = prog.work(total)
    prep_program_stage(prog.program, SP_RANK, total, nv, -1, d, 1, rank)
    var orank = prog.alloc(d)
    var onb = prog.alloc(d)
    var ovn = prog.alloc(d)
    prep_program_stage(prog.program, SP_V3_SELECT, d, rank, nv, d, nb, r.n_top, orank, onb, ovn)
    var hv = prog.alloc(d)
    prep_program_stage(prog.program, SP_V3_HV, d, orank, onb, d, r.n_top, hv)
    return HvgV3SelectPlan(ovn, orank, onb, hv)


# ---------------------------------------------------------------- the kept columns


@fieldwise_init
struct KeepPlan(Copyable, Movable):
    """Host words: IDX (d int words, the kept columns ascending in the first k), INV (d int words, -1 for a
    dropped column) and COUNT (one int word, k)."""
    var idx: Int
    var inv: Int
    var count: Int


def keep_stage(mut prog: EncProgram, d: Int, hv: Int) -> KeepPlan:
    """The kept-column tables of the int-word mask HV (an arena offset of this program)."""
    var nblk = _blocks(d)
    var part = prog.work(nblk + 1)
    var pos = prog.alloc(d + 1)
    prep_program_stage(prog.program, SP_SCAN_PART, nblk, hv, d, part)
    prep_program_stage(prog.program, SP_SCAN_TOP, 1, part, nblk)
    prep_program_stage(prog.program, SP_SCAN_FILL, nblk, hv, d, part, pos)
    var idx = prog.alloc(d)
    var inv = prog.alloc(d)
    prep_program_stage(prog.program, SP_COMPACT, d, hv, pos, idx, inv)
    return KeepPlan(idx, inv, pos + d)


def select_dense_stage(mut prog: EncProgram, x: SpInput, idx: Int, k: Int, region: Bool) raises -> Int:
    """OUT (n x k words): the kept columns IDX (the owner of k int words) of a dense X."""
    var at = sp_input(prog, x)
    var ix = enc_input(prog, k, idx)
    var o = prog.output(x.n * k, region)
    if x.n * k > 0:
        prep_program_stage(prog.program, SP_COL_SELECT, x.n * k, at.values, x.d, ix, k, o)
    return o


@fieldwise_init
struct CsrSelectPlan(Copyable, Movable):
    """Host words: OPTR (n + 1 int words), OIDX (int words) and ODATA, both sized nnz (the first OPTR[n] hold
    the result)."""
    var indptr: Int
    var indices: Int
    var data: Int


def select_csr_stage(mut prog: EncProgram, x: SpInput, inv: Int) -> CsrSelectPlan:
    """The kept columns of a CSR X: INV (the owner of d int words) renumbers them, -1 drops one."""
    var at = sp_input(prog, x)
    var iv = enc_input(prog, x.d, inv)
    var cnt = prog.work(x.n)
    prep_program_stage(prog.program, SP_CSR_SELECT, x.n, at.indptr, at.indices, at.values, iv, -1, -1, -1, cnt)
    var nblk = _blocks(x.n)
    var part = prog.work(nblk + 1)
    var optr = prog.alloc(x.n + 1)
    prep_program_stage(prog.program, SP_SCAN_PART, nblk, cnt, x.n, part)
    prep_program_stage(prog.program, SP_SCAN_TOP, 1, part, nblk)
    prep_program_stage(prog.program, SP_SCAN_FILL, nblk, cnt, x.n, part, optr)
    var oidx = prog.alloc(max(x.nnz, 1))
    var odat = prog.alloc(max(x.nnz, 1))
    prep_program_stage(prog.program, SP_CSR_SELECT, x.n, at.indptr, at.indices, at.values, iv, optr, oidx, odat,
                       -1)
    return CsrSelectPlan(optr, oidx, odat)


# ---------------------------------------------------------------- the clipped scaler


@fieldwise_init
struct ScaleApply(Copyable, Movable):
    """The transform: `mean`, `std` the owners of the d fitted words; `zero_center`; `max_value` (`clip` False:
    none). A CSR X takes only zero_center False (the door refuses the other)."""
    var x: SpInput
    var mean: Int
    var std: Int
    var zero_center: Bool
    var clip: Bool
    var max_value: Float64
    var region: Bool


def scale_apply_stage(mut prog: EncProgram, r: ScaleApply) raises -> Int:
    var x = r.x.copy()
    var at = sp_input(prog, x)
    var mean = enc_input(prog, x.d, r.mean)
    var std = enc_input(prog, x.d, r.std)
    var mv = prog.put_floats([r.max_value]) if r.clip else -1
    var count = x.nnz if x.csr else x.n * x.d
    var o = prog.output(count, r.region)
    if count > 0:
        if x.csr:
            prep_program_stage(prog.program, SP_SCALE_CSR, count, at.values, at.indices, std, mv, o)
        else:
            prep_program_stage(prog.program, SP_SCALE_APPLY, count, at.values, x.d, mean, std,
                               1 if r.zero_center else 0, mv, o)
    return o


# ---------------------------------------------------------------- covariate regression


@fieldwise_init
struct RegressFit(Copyable, Movable):
    """The fit: `cov` the owner of the n x k float32 covariates; `residuals` also writes the residual block (the
    fit_transform)."""
    var x: SpInput
    var cov: Int
    var k: Int
    var residuals: Bool
    var region: Bool


@fieldwise_init
struct RegressFitPlan(Copyable, Movable):
    """Host words: CMEAN, CSTD (k each), XMEAN (d), COEF (k x d) and OUT (n x d residuals, -1 when not asked)."""
    var cmean: Int
    var cstd: Int
    var xmean: Int
    var coef: Int
    var out: Int


def _standardized(mut prog: EncProgram, cov: Int, n: Int, k: Int, cmean: Int, cstd: Int) -> Int:
    var z = prog.work(n * k)
    prep_program_stage(prog.program, SP_STANDARDIZE, n * k, cov, k, cmean, cstd, z)
    return z


def regress_fit_stage(mut prog: EncProgram, r: RegressFit) raises -> RegressFitPlan:
    var x = r.x.copy()
    var n = x.n
    var d = x.d
    var k = r.k
    var cov = enc_input(prog, n * k, r.cov)
    var cmean = prog.alloc(k)
    var cstd = prog.alloc(k)
    prep_program_stage(prog.program, SP_COL_MEANSTD, k, cov, n, k, cmean, cstd)
    var z = _standardized(prog, cov, n, k, cmean, cstd)
    # the normal equations of the standardized covariates, one solve for every gene: G = Z'Z, its pseudo-inverse
    var gram = prog.work(k * k)
    prep_program_stage(prog.program, OP_MATMUL, k * k, z, 1, k, z, k, 1, gram, k, n, -1, -1)
    var eval = prog.work(k)
    var evec = prog.work(k * k)
    prep_program_stage(prog.program, OP_EIGH, 1, gram, k, k * k, eval, evec, 0)
    var pinv = prog.work(k * k)
    prep_program_stage(prog.program, SP_PINV, k * k, eval, evec, k, pinv)
    # B = G^+ Z' (k x n)
    var bm = prog.work(k * n)
    prep_program_stage(prog.program, OP_MATMUL, k * n, pinv, k, 1, z, 1, k, bm, n, k, -1, -1)
    # the centred X, in place in the residual block
    var at = sp_input(prog, x)
    var o = prog.output(n * d, r.region) if r.residuals else prog.work(n * d)
    if x.csr:
        prep_program_stage(prog.program, SP_DENSIFY, n, at.mode, at.values, at.indptr, at.indices, d, o)
    else:
        prep_program_stage(prog.program, SP_DENSIFY, n * d, at.mode, at.values, at.indptr, at.indices, d, o)
    var xmean = prog.alloc(d)
    var xsd = prog.work(d)
    prep_program_stage(prog.program, SP_COL_MEANSTD, d, o, n, d, xmean, xsd)
    prep_program_stage(prog.program, SP_CENTER, n * d, o, d, xmean)
    # every gene's coefficients at once: COEF = B Xc (k x d)
    var coef = prog.alloc(k * d)
    prep_program_stage(prog.program, OP_MATMUL, k * d, bm, n, 1, o, d, 1, coef, d, n, -1, -1)
    if r.residuals:
        prep_program_stage(prog.program, SP_RESID, n * d, o, d, z, k, coef)
        return RegressFitPlan(cmean, cstd, xmean, coef, o)
    return RegressFitPlan(cmean, cstd, xmean, coef, -1)


@fieldwise_init
struct RegressApply(Copyable, Movable):
    """The transform: the owners of the covariates (n x k) and the fitted CMEAN, CSTD (k), XMEAN (d), COEF."""
    var x: SpInput
    var cov: Int
    var k: Int
    var cmean: Int
    var cstd: Int
    var xmean: Int
    var coef: Int
    var region: Bool


def regress_apply_stage(mut prog: EncProgram, r: RegressApply) raises -> Int:
    var x = r.x.copy()
    var n = x.n
    var d = x.d
    var k = r.k
    var cov = enc_input(prog, n * k, r.cov)
    var cmean = enc_input(prog, k, r.cmean)
    var cstd = enc_input(prog, k, r.cstd)
    var xmean = enc_input(prog, d, r.xmean)
    var coef = enc_input(prog, k * d, r.coef)
    var z = _standardized(prog, cov, n, k, cmean, cstd)
    var at = sp_input(prog, x)
    var o = prog.output(n * d, r.region)
    if x.csr:
        prep_program_stage(prog.program, SP_DENSIFY, n, at.mode, at.values, at.indptr, at.indices, d, o)
    else:
        prep_program_stage(prog.program, SP_DENSIFY, n * d, at.mode, at.values, at.indptr, at.indices, d, o)
    prep_program_stage(prog.program, SP_CENTER, n * d, o, d, xmean)
    prep_program_stage(prog.program, SP_RESID, n * d, o, d, z, k, coef)
    return o
