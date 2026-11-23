# SPDX-License-Identifier: Apache-2.0
"""RUN of the feature-selection door (bindings/prep_select_native_door.mojo):
the typed builders and checks of mutual_info_classif / mutual_info_regression
and of RFE's elimination rounds (lane bm-rest-sel, 2026-11-23;
docs/plans/PYTHON_AT_THE_BOUNDARY.md). No Python import
(tools/python_free_dirs.sh): no Python object parameter, field, local or
import.

The door is UNPACK (the arguments' admission, X and the targets as owners of
its program, the selected image's words read once), one builder here per
program on the typed program (`EncProgram` for the mutual information,
`ClsProgram` for RFE), the door's typed run of that program, then PACK (the
returned Array, the fitted attributes). The builders make the allocations,
input records and stages the door's builders made, in the same order, so no
bit moves.

Mutual information, in the door's order:
  1. `mi_mask_of_bools` / `mi_mask_of_indices`: the `discrete_features` mask
     from its admitted values, `mi_columns` the continuous and discrete
     column ids.
  2. A discrete target: `mi_class_count_stage` (one program), then
     `mi_int_words` and `mi_class_check`.
  3. `mi_gather_stage` (one program each): the continuous block when a
     discrete column is present, the discrete block when one is present;
     the discrete block's categories are the encoders' category fit.
  4. `mi_stage`: the one scoring program; `mi_scatter` the scores to their
     columns. Every feature is an independent unit of each stage (the x_prep
     runner spreads a stage's `total` units over the device or the host
     cores); nothing here loops over rows.

RFE, per elimination round (serial by nature: a round's block is chosen from
the previous round's importances):
  1. With a native member (core/member_abi.mojo): `rfe_member_importances`
     fits a new state on the round's block and exports its importance block
     as native words (the caller holds the state's call and frees it).
     Without one, the door reads the estimator's own attribute.
  2. `rfe_round_stage`: the importance keys (squared and summed over rows
     for a coefficient block), c2_rfe_step on the support, c2_rfe_rank, then
     the next block (`rfe_block_stage`: c2_rfe_keep and gather_cols).
`rfe_first_stage` stages the first block from the all-ones support,
`rfe_select_count` / `rfe_step_count` resolve n_features_to_select and step,
`rfe_member_fit` is the final fit over a native member.
"""
from std.memory import unsafe_memcpy
from core.preprocessing_program import prep_program_stage
from bindings.prep_encode_prog_run import EncProgram, enc_col_stats, enc_floats_of
from bindings.prep_encode_cats_run import CatBlock, cat_codes_stage, cat_i2f_stage

comptime SEL_F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime SEL_I32P = MutPointer[Int32, MutUntrackedOrigin]

# x_prep stage codes (_expansion_prep._OPS).
comptime SEL_OP_GATHER_COLS = 61
comptime SEL_OP_MI_COLSCALE = 66
comptime SEL_OP_MI_NOISE = 67
comptime SEL_OP_MI_CC = 68
comptime SEL_OP_MI_CD = 69
comptime SEL_OP_MI_REDUCE = 70
comptime SEL_OP_CODE_COUNTS = 88
comptime SEL_OP_I2F = 10
comptime SEL_OP_MI_DC = 94
comptime SEL_OP_MI_DD = 95
comptime SEL_OP_P2M_CCOUNT = 200
comptime SEL_OP_P2M_CSCAN = 201
# `_II_CH`: the p2m units' chunk rows.
comptime SEL_II_CH = 1024
comptime SEL_INT32_MAX = 2147483647


# ---------------------------------------------------------------- mutual information
def mi_check_neighbors(k: Int) raises:
    if not (1 <= k and k <= 32):
        raise Error("NotImplementedError: mojolearn: mutual_info supports 1 <= n_neighbors <= 32")


def mi_check_shape(n: Int, d: Int) raises:
    if n == 0 or d == 0:
        raise Error("mojolearn: X must be a nonempty two-dimensional array")
    if n * d > SEL_INT32_MAX:
        raise Error("mojolearn: X exceeds the native Int32 indexing bound")


def mi_mask_of_bools(values: List[Bool], d: Int) raises -> List[Bool]:
    """`discrete_features` as a bool mask: one entry per column of X."""
    if len(values) != d:
        raise Error("mojolearn: discrete_features mask has " + String(len(values)) + " entries; X has " + String(d)
                    + " features")
    return values.copy()


def mi_mask_of_indices(indices: List[Int], d: Int) raises -> List[Bool]:
    """`discrete_features` as column indices (negative ones from the end);
    the first index out of bounds is refused."""
    var mask = List[Bool](length=d, fill=False)
    for i in range(len(indices)):  # driver-loop: over the discrete_features indices
        var j = indices[i]
        if not (-d <= j and j < d):
            raise Error("IndexError: mojolearn: discrete_features index " + String(j) + " is out of bounds for "
                        + String(d) + " features")
        mask[j + d if j < 0 else j] = True
    return mask^


def mi_columns(mask: List[Bool]) -> Tuple[List[Int], List[Int]]:
    """(continuous column ids, discrete column ids), each ascending."""
    var cont = List[Int]()
    var disc = List[Int]()
    for j in range(len(mask)):  # driver-loop: over the column kinds (feature count)
        if mask[j]:
            disc.append(j)
        else:
            cont.append(j)
    return (cont^, disc^)


def mi_gather_stage(mut prog: EncProgram, x: Int, n: Int, d: Int, cols: List[Int]) -> Int:
    """`_gather(arr, cols, mode)`: the columns `cols` of the (n, d) block of
    owner `x` as an (n, len(cols)) block. Returns its offset."""
    var k = len(cols)
    var xo = prog.input(n * d, x)
    var ko = prog.put_floats(enc_floats_of(cols))
    var dst = prog.alloc(n * k)
    prep_program_stage(prog.program, SEL_OP_GATHER_COLS, n * k, xo, n, d, ko, k, dst)
    return dst


def mi_class_count_stage(mut prog: EncProgram, codes: Int, words: Int, n: Int, K: Int) -> Int:
    """`_class_counts(codes, K)`: the int32 class codes of owner `codes`
    (`words` words) as floats (i2f), then p2m_ccount / p2m_cscan. Returns the
    offset of the K int32 totals."""
    var co = cat_i2f_stage(prog, words, codes)
    # `_p2m_chunks(n, K)`
    var ch = max(SEL_II_CH, (n + SEL_II_CH - 1) // SEL_II_CH, (n * K + 16777215) // 16777216, 1)
    var nch = max(1, (n + ch - 1) // ch)
    var cnt = prog.alloc(nch * K)
    var tot = prog.alloc(K)
    prep_program_stage(prog.program, SEL_OP_P2M_CCOUNT, nch, co, n, K, ch, cnt)
    prep_program_stage(prog.program, SEL_OP_P2M_CSCAN, K, cnt, nch, K, tot)
    return tot


def mi_int_words(arena: Int, offset: Int, count: Int) -> List[Int]:
    """`count` int32 words of a run's host arena at `offset`."""
    var src = SEL_I32P(unsafe_from_address=arena)
    var out = List[Int](capacity=max(count, 0))
    for i in range(count):  # driver-loop: over the class counts (class count)
        out.append(Int(src[offset + i]))
    return out^


def mi_class_check(counts: List[Int], continuous: Bool) raises:
    """The reference's neighbour search over the classes with more than one
    sample: a continuous column needs one class with two samples."""
    var most = 0
    for i in range(len(counts)):  # driver-loop: over the class counts (class count)
        most = max(most, counts[i])
    if continuous and most < 2:
        raise Error("mojolearn: mutual_info: every class has one sample (the reference's "
                    + "neighbour search over the classes with more than one finds 0 samples)")


def mi_discrete_check(kx: List[Int], disc: List[Int], n: Int) raises:
    """A continuous target: a discrete column with one sample per value is
    refused (the reference's neighbour search finds 0 samples)."""
    for c in range(len(kx)):  # driver-loop: over the per-column category counts (discrete feature count)
        if kx[c] == n:
            raise Error("mojolearn: mutual_info: discrete feature " + String(disc[c]) + " has one sample per "
                        + "value (the reference's neighbour search finds 0 samples)")


@fieldwise_init
struct MiProgram(Copyable, Movable):
    """One scoring program's request. `x`: the owner of the (n, len(cont))
    continuous block (X itself when no column is discrete); `y`: the owner
    of the int32 class codes (`y_words` words, a discrete target) or of the
    n float32 target words; `counts`: the class counts (a discrete target);
    `xd`: the owner of the (n, len(disc)) discrete block, `cats` its fitted
    category block and `kx` its per-column category counts. `seed`: the
    30-bit noise seed. `scratch`: the FAST tier's work words
    (MOJOLEARN_MI_WORK); `stats_blocked`: the image's IDN_STATS_BLOCKED word."""
    var n: Int
    var k: Int
    var seed: Int
    var discrete_target: Bool
    var scratch: Bool
    var stats_blocked: Bool
    var cont: List[Int]
    var disc: List[Int]
    var x: Int
    var y: Int
    var y_words: Int
    var counts: List[Int]
    var xd: Int
    var cats: CatBlock
    var kx: List[Int]


@fieldwise_init
struct MiStaged(ImplicitlyCopyable, Movable):
    """The offsets of the continuous columns' scores and of the discrete
    columns' scores (-1: none)."""
    var outc: Int
    var outd: Int


def _mi_work(mut prog: EncProgram, words: Int, scratch: Bool) -> Int:
    # `work`: pr.work under MOJOLEARN_MI_WORK on the FAST tier, else pr.alloc.
    return prog.work(words) if scratch else prog.alloc(words)


def mi_stage(mut prog: EncProgram, r: MiProgram) -> MiStaged:
    """`_mutual_info`'s program: continuous columns scaled and noised (the
    1e-10-scaled noise kept as a second word), then per column Kraskov, Ross
    or the contingency table."""
    var n = r.n
    var k = r.k
    var dc = len(r.cont)
    var dd = len(r.disc)
    var yo = -1
    var lc = -1
    if r.discrete_target:
        yo = cat_i2f_stage(prog, r.y_words, r.y)
        lc = prog.put_floats(enc_floats_of(r.counts))
    var z = -1
    var zs = -1
    if dc > 0:
        var xo = prog.input(n * dc, r.x)
        var st = prog.alloc(6 * dc)
        var sc = prog.alloc(dc)
        var ma = prog.alloc(dc)
        z = _mi_work(prog, n * dc, r.scratch)
        zs = _mi_work(prog, n * dc, r.scratch)
        enc_col_stats(prog, r.stats_blocked, xo, n, dc, st)
        prep_program_stage(prog.program, SEL_OP_MI_COLSCALE, dc, xo, n, dc, st, sc, ma)
        prep_program_stage(prog.program, SEL_OP_MI_NOISE, n * dc, xo, n, dc, sc, ma, 2 * r.seed, z, zs + 1)
    var zy = -1
    var zys = -1
    if not r.discrete_target:
        yo = prog.input(n, r.y)
        var sty = prog.alloc(6)
        var scy = prog.alloc(1)
        var may = prog.alloc(1)
        zy = _mi_work(prog, n, r.scratch)
        zys = _mi_work(prog, n, r.scratch)
        enc_col_stats(prog, r.stats_blocked, yo, n, 1, sty)
        prep_program_stage(prog.program, SEL_OP_MI_COLSCALE, 1, yo, n, 1, sty, scy, may)
        prep_program_stage(prog.program, SEL_OP_MI_NOISE, n, yo, n, 1, scy, may, 2 * r.seed + 1, zy, zys + 1)
    var outc = -1
    if dc > 0:
        var term = _mi_work(prog, n * dc, r.scratch)
        outc = prog.alloc(dc)
        if r.discrete_target:
            prep_program_stage(prog.program, SEL_OP_MI_CD, n * dc, z, n, dc, yo, lc, k, term, zs + 1)
            prep_program_stage(prog.program, SEL_OP_MI_REDUCE, dc, term, n, dc, 3, k, 0, outc, lc, len(r.counts))
        else:
            prep_program_stage(prog.program, SEL_OP_MI_CC, n * dc, z, n, dc, zy, k, term, zs + 1, zys + 1)
            prep_program_stage(prog.program, SEL_OP_MI_REDUCE, dc, term, n, dc, 0, k, n, outc)
    var outd = -1
    if dd > 0:
        var kmax = 0
        for c in range(dd):  # driver-loop: over the per-column category counts (discrete feature count)
            kmax = max(kmax, r.kx[c])
        var coded = cat_codes_stage(prog, n, dd, r.xd, r.cats, False)
        var xc = coded.codes
        outd = prog.alloc(dd)
        if r.discrete_target:
            var ky = len(r.counts)
            var stride = kmax * ky + kmax + ky
            var tb = prog.alloc(dd * stride)
            var kxo = prog.put_floats(enc_floats_of(r.kx))
            prep_program_stage(prog.program, SEL_OP_MI_DD, dd, xc, n, dd, yo, ky, kxo, tb, stride, outd)
        else:
            var cnti = prog.alloc(dd * kmax)
            var cntf = prog.alloc(dd * kmax)
            var term = _mi_work(prog, n * dd, r.scratch)
            prep_program_stage(prog.program, SEL_OP_CODE_COUNTS, dd, xc, n, dd, kmax, cnti)
            prep_program_stage(prog.program, SEL_OP_I2F, dd * kmax, cnti, cntf)
            prep_program_stage(prog.program, SEL_OP_MI_DC, n * dd, zy, n, dd, xc, cntf, kmax, k, term, zys + 1)
            prep_program_stage(prog.program, SEL_OP_MI_REDUCE, dd, term, n, dd, 2, k, 0, outd, cntf, kmax)
    return MiStaged(outc, outd)


def mi_scatter(arena: Int, staged: MiStaged, cont: List[Int], disc: List[Int], dst: Int):
    """The run's scores scattered to their columns of the d float32 words at
    `dst` (the caller keeps both buffers alive)."""
    var src = SEL_F32P(unsafe_from_address=arena)
    var out = SEL_F32P(unsafe_from_address=dst)
    for i in range(len(cont)):  # driver-loop: scatters the scores to their columns (continuous feature count)
        out[cont[i]] = src[staged.outc + i]
    for i in range(len(disc)):  # driver-loop: scatters the scores to their columns (discrete feature count)
        out[disc[i]] = src[staged.outd + i]
