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
  1. Over the native member (core/member_abi.mojo): `rfe_member_importances`
     fits a new state on the round's block and exports its importance block
     as native words (the caller holds the state's call and frees it).
     (Lane mf-rfe7: every RFE fit runs over a member; the door's rounds over
     the estimator's own attribute reads are deleted.)
  2. `rfe_round_stage`: the importance keys (squared and summed over rows
     for a coefficient block), c2_rfe_step on the support, c2_rfe_rank, then
     the next block (`rfe_block_stage`: c2_rfe_keep and gather_cols).
`rfe_first_stage` stages the first block from the all-ones support,
`rfe_select_count` / `rfe_step_count` resolve n_features_to_select and step,
`rfe_member_fit` is the final fit over a native member. `rfe_member_source` / `rfe_member_source_named` pick a
member kind's importance block for the getter (`rfe_named_block`: a named feature-wide attribute), or the
refusal of a kind with none (lanes mf-shaprfe, mf-gbdt5, mf-rfe7). Since lane mf-gbdt5 every member fit also takes the fit's extra inputs
(core/member_set_run.mojo `MemberFitExtras`: RFE's fit parameters as the estimator's fit door admits them, a
held-out X restricted to the round's columns, the round's column ids at `columns`).
"""
from std.memory import unsafe_memcpy
from core.preprocessing_program import prep_program_stage
from bindings.prep_encode_prog_run import EncProgram, enc_col_stats, enc_floats_of
from bindings.prep_encode_cats_run import CatBlock, cat_codes_stage, cat_i2f_stage
from bindings.prep_classify_prog_run import (
    ClsProgram, CLS_IN_BUFFER, CLS_IN_CONST_F, ClsRunPlan, ClsBorrowed, ClsBorrowedDone, cls_borrowed_run,
)
from bindings.arena_prog_run import _switch
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_new, member_free, member_own,
    MEMBER_OP_FIT, MEMBER_OP_EXPORT, MEMBER_F32, MEMBER_F64, MEMBER_I32,
    MEMBER_CAN_COEF, MEMBER_CAN_IMPORTANCES, MEMBER_CAN_CLASSIFIER, MEMBER_BLOCK_COEF,
    MEMBER_BLOCK_FEATURE_IMPORTANCES, MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_SCALARS,
)
from core.member_families import (
    MEMBER_FAMILY_NAIVE_BAYES, MEMBER_FAMILY_DISCRIMINANT, MEMBER_FAMILY_KERNEL_SVM, MEMBER_FAMILY_KERNEL_RIDGE,
    MEMBER_FAMILY_KERNEL_APPROX, MEMBER_FAMILY_GP, MEMBER_FAMILY_KMEANS, MEMBER_FAMILY_XCLUSTER, MEMBER_FAMILY_MIXTURE,
    MEMBER_FAMILY_DECOMPOSITION, MEMBER_FAMILY_XDECOMP, MEMBER_FAMILY_SCALER, MEMBER_FAMILY_PREP, MEMBER_FAMILY_SELECT,
    MEMBER_FAMILY_XLINEAR, MEMBER_FAMILY_PLS, MEMBER_FAMILY_XNEIGHBORS,
)
from core.member_notices import member_notices_take
from x_linear.door_run import xl_bayes_sigma_run
from core.member_set_run import MemberFitExtras, member_fit_extras_apply, member_fit_extras_clear

comptime SEL_F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime SEL_I32P = MutPointer[Int32, MutUntrackedOrigin]

# x_prep stage codes (_expansion_prep._OPS).
comptime SEL_OP_GATHER_COLS = 61
comptime SEL_OP_MI_COLSCALE = 66
comptime SEL_OP_MI_NOISE = 67
comptime SEL_OP_MI_CC = 68
comptime SEL_OP_MI_CD = 69
comptime SEL_OP_MI_REDUCE = 70
comptime SEL_OP_SQSUM_COLS = 71
comptime SEL_OP_CODE_COUNTS = 88
comptime SEL_OP_I2F = 10
comptime SEL_OP_MI_DC = 94
comptime SEL_OP_MI_DD = 95
comptime SEL_OP_P2M_CCOUNT = 200
comptime SEL_OP_P2M_CSCAN = 201
comptime SEL_OP_C2_KEY64 = 253
comptime SEL_OP_C2_RFE_STEP = 256
comptime SEL_OP_C2_RFE_RANK = 257
comptime SEL_OP_C2_RFE_KEEP = 271
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


# ---------------------------------------------------------------- RFE
comptime RFE_SELECT_HALF = 0      # n_features_to_select None: d // 2
comptime RFE_SELECT_FRACTION = 1  # a float in (0, 1): int(fraction * d)
comptime RFE_SELECT_COUNT = 2     # anything else: int(value)


def rfe_select_count(d: Int, kind: Int, fraction: Float64, count: Int) -> Int:
    """n_features_to_select: None -> d // 2; a float in (0, 1) ->
    int(fraction * d); else int(value); then at least 1."""
    var nsel = count
    if kind == RFE_SELECT_HALF:
        nsel = d // 2
    elif kind == RFE_SELECT_FRACTION:
        nsel = Int(fraction * Float64(d))
    return max(1, nsel)


def rfe_step_count(d: Int, is_fraction: Bool, fraction: Float64, count: Int) raises -> Int:
    """step: a float below 1 -> int(max(1, step * d)); else int(step)."""
    var step = count
    if is_fraction:
        var v = fraction * Float64(d)
        step = 1 if v <= 1.0 else Int(v)
    if step <= 0:
        raise Error("mojolearn: step must be > 0")
    return step


def _cls_stage(mut prog: ClsProgram, op: Int, total: Int, *params: Int):
    # `_Prog.stage`: op, total, the parameters, zero padding.
    var words = List[Int](capacity=len(params))
    for i in range(len(params)):  # driver-loop: over at most fourteen stage parameters
        words.append(params[i])
    prog.stage(op, total, words)


def _cls_buffer(mut prog: ClsProgram, words: Int, in_out: Bool = False) -> Int:
    # `_Prog.put` of a caller buffer of `words` words (its owner is the caller's, in record order).
    return prog.input(words, CLS_IN_BUFFER, -1, in_out)


@fieldwise_init
struct RfeBlock(ImplicitlyCopyable, Movable):
    """Where a staged block lands: the int32 ids of its columns (`ids`, d
    words, the first k used) and the (n, k) float32 block (`block`)."""
    var ids: Int
    var block: Int


def rfe_block_stage(mut prog: ClsProgram, x_words: Int, n: Int, d: Int, sup: Int, k: Int) -> RfeBlock:
    """The next round's block: the supported columns of the support mask at
    `sup` compacted on the device (c2_rfe_keep) and those k columns of X (a
    caller buffer of `x_words` words, the next record) gathered into an
    (n, k) block (gather_cols, bit for bit)."""
    var xo = _cls_buffer(prog, x_words)
    var kf = prog.alloc(d)
    var ki = prog.alloc(d)
    _cls_stage(prog, SEL_OP_C2_RFE_KEEP, d, sup, d, kf, ki)
    var dst = prog.alloc(n * k)
    _cls_stage(prog, SEL_OP_GATHER_COLS, n * k, xo, n, d, kf, k, dst)
    return RfeBlock(ki, dst)


def rfe_first_stage(mut prog: ClsProgram, x_words: Int, n: Int, d: Int) -> RfeBlock:
    """Program 0: the first block from the all-ones support. Caller buffers,
    in record order: the d support words, X."""
    var sup = _cls_buffer(prog, d)
    return rfe_block_stage(prog, x_words, n, d, sup, d)


@fieldwise_init
struct RfeKeys(ImplicitlyCopyable, Movable):
    """One round's importances, the float32 words a member's EXPORT gave (they travel beside it as the
    program's own words). `squared`: a coefficient block of `rows` rows of `m` words, squared and summed
    over its rows; else m float32 scores (c2_key64 kind 0).
    REMOVED (lane mf-rfe7, with the door's rounds): the `kind` word (int32 / int64 / float64 scores of the
    estimator's own attribute, `_score_words`) and the caller-buffer form of the words (`_keys_of`'s owned
    buffer); every round now reads a member's float32 block."""
    var m: Int
    var rows: Int
    var squared: Bool


def _rfe_key_words(mut prog: ClsProgram, var words: List[Float32], count: Int) -> Int:
    # The round's importance words as the program's own constant input (`count` of them).
    prog.consts_f.append(words^)
    return prog.input(count, CLS_IN_CONST_F, len(prog.consts_f) - 1, False)


def rfe_keys_stage(mut prog: ClsProgram, keys: RfeKeys, var words: List[Float32]) -> Int:
    """The ordered keys (c2_key64) of m column importances. Returns the keys' offset."""
    var m = keys.m
    var key = prog.work(2 * m)
    if not keys.squared:
        var wo = _rfe_key_words(prog, words^, m)
        _cls_stage(prog, SEL_OP_C2_KEY64, m, wo, 0, key)
        return key
    var summed = prog.work(m)
    var co = _rfe_key_words(prog, words^, keys.rows * m)
    _cls_stage(prog, SEL_OP_SQSUM_COLS, m, co, keys.rows, m, summed)
    _cls_stage(prog, SEL_OP_C2_KEY64, m, summed, 0, key)
    return key


@fieldwise_init
struct RfeRoundStaged(ImplicitlyCopyable, Movable):
    """A round program's in-out support and ranking (d words each) and the
    next block."""
    var support: Int
    var ranking: Int
    var next: RfeBlock


def rfe_round_stage(mut prog: ClsProgram, keys: RfeKeys, var words: List[Float32], x_words: Int, n: Int, d: Int, nsup: Int,
                    drop: Int, k: Int) -> RfeRoundStaged:
    """One elimination round: the importance keys (`words`, the program's own), c2_rfe_step dropping
    `drop` of the `nsup` supported columns from the support mask, c2_rfe_rank, then the next (n, k)
    block. Caller buffers, in record order: the d support words (in-out), the d ranking words (in-out),
    the nsup int32 column ids, X."""
    var key = rfe_keys_stage(prog, keys, words^)
    var sup = _cls_buffer(prog, d, True)
    var rk = _cls_buffer(prog, d, True)
    var fo = _cls_buffer(prog, nsup)
    _cls_stage(prog, SEL_OP_C2_RFE_STEP, nsup, key, fo, nsup, drop, sup)
    _cls_stage(prog, SEL_OP_C2_RFE_RANK, d, sup, rk)
    var nxt = rfe_block_stage(prog, x_words, n, d, sup, k)
    return RfeRoundStaged(sup, rk, nxt)


#: The refusal of an estimator with neither coef_ nor feature_importances_ under importance_getter='auto' (the
#: door's own words, before and after lane mf-shaprfe).
comptime RFE_NO_IMPORTANCES = "mojolearn: RFE needs an estimator with coef_ or feature_importances_"


struct RfeSource(Copyable, Movable):
    """Where a native member's round importances come from: `block` (0: no block this kind exports; the door
    keeps the estimator's own doors) and `squared` (a coefficient block, or a named attribute path, squared and
    summed over its rows); or `refusal`, non-empty: the words RFE raises before any round is fitted."""
    var block: Int
    var squared: Bool
    var refusal: String

    def __init__(out self, block: Int, squared: Bool, var refusal: String):
        self.block = block
        self.squared = squared
        self.refusal = refusal^


def rfe_member_source(family: Int, mask: Int, getter: String, estimator: String) -> RfeSource:
    """The importance block of a member KIND (`mask`: its kind-exact capabilities) for importance_getter
    `getter` ('auto', 'coef_' or 'feature_importances_'): 'auto' takes MEMBER_BLOCK_COEF (squared) when the
    kind exports it, else MEMBER_BLOCK_FEATURE_IMPORTANCES as given; a named attribute takes that block, squared,
    as the attribute path squares it. A kind with neither block is REFUSED, and the caller raises the words
    only when an elimination round reads importances (scikit-learn reads none when n_features_to_select keeps
    every column): 'auto' in the door's words (RFE_NO_IMPORTANCES: neither attribute exists on the fitted
    estimator, scikit-learn's `_get_feature_importances` raises there too), a named attribute in scikit-learn's
    attrgetter's (`estimator` the class name). The kernel SVM kinds (SVC, SVR) export MEMBER_BLOCK_COEF only
    with kernel='linear' (lane mf-rfe7); any other kernel answers 'auto' and 'coef_' in the SVM coef_
    operation's words (bindings/svc_kernel_query_unpack.mojo `svc_coef_unpack`, as the door's attribute read
    raised them; scikit-learn's `coef_` property raises an AttributeError there too).
    History: lane mf-shaprfe refused the naive Bayes and discriminant kinds, mf-meta5b the XNeighbors kinds,
    mf-imp6 gave the Gbdt kinds their importance block. Lane mf-rfe7: every other kind without either block
    is refused in the same words (block 0 with no words, which kept the estimator's own doors, is gone with
    them: the doors' attribute reads found neither attribute on those kinds either)."""
    var auto = getter == "auto"
    if getter != "feature_importances_" and (mask & MEMBER_CAN_COEF) != 0:
        return RfeSource(MEMBER_BLOCK_COEF, True, String(""))
    if getter != "coef_" and (mask & MEMBER_CAN_IMPORTANCES) != 0:
        return RfeSource(MEMBER_BLOCK_FEATURE_IMPORTANCES, not auto, String(""))
    if family == MEMBER_FAMILY_KERNEL_SVM and getter != "feature_importances_":
        var svm = String("SVC") if (mask & MEMBER_CAN_CLASSIFIER) != 0 else String("SVR")
        return RfeSource(0, False, "AttributeError: mojolearn " + svm + ": coef_ is only available for kernel='linear'")
    if auto:
        return RfeSource(0, False, String(RFE_NO_IMPORTANCES))
    return RfeSource(0, False, "AttributeError: '" + estimator + "' object has no attribute '" + getter + "'")


# A fitted attribute named by importance_getter that a member exports as one block, or one row of a block
# (lanes mf-rfe7, mf-rfe7b): THE NAMED-ARRAY TABLE, one entry per family, built from the attributes each
# family's PACK writes from an exported block (providers/python/member_<family>_boundary_door.mojo). The block
# names are mirrored here so this RUN imports no member module; each line names its source. An entry is a
# BLOCK CODE: the block, plus (row + 1) << 16 when the attribute is one row of a multi-row block
# (`rfe_block_code`, decoded by `rfe_member_importances`). A width other than the round's columns raises the
# deleted door's words there (`_keys_of`: "RFE importances have k columns, expected m").
comptime _ROW_SHIFT = 65536
comptime _F = MEMBER_BLOCK_FAMILY
# Round b (lane mf-member8): attributes a PACK computes rather than copies, served natively by the getter (no
# block code): BayesianRidge's sigma_ (V diag(1 / (alpha_ ev + lambda_)) V', formed from the factor block and the
# scalars as its first read forms it: x_linear/door_run.mojo `xl_bayes_sigma_run`), and the X_scale_ of
# BayesianRidge / ARDRegression (ones, as PACK writes them).
comptime RFE_COMPUTED_BAYES_SIGMA = 1099511627776   # 2^40
comptime RFE_COMPUTED_ONES = 2199023255552          # 2^41
comptime _XL_BLOCK_FACTOR = MEMBER_BLOCK_FAMILY + 10  # core/xlinear_member.mojo XL_MEMBER_BLOCK_FACTOR


def rfe_block_code(block: Int, row: Int) -> Int:
    """The block code of `block` (row -1: the whole block; else that row of it)."""
    return block + (row + 1) * _ROW_SHIFT


def rfe_block_of(code: Int) -> Int:
    """The member block of a block code."""
    return code % _ROW_SHIFT


def rfe_row_of(code: Int) -> Int:
    """The row of a block code (-1: the whole block)."""
    return code // _ROW_SHIFT - 1


def _nb_named(kind: Int, name: String) -> Int:
    # core/naive_bayes_member.mojo NB_KIND_* / NB_BLOCK_*; member_naive_bayes_boundary_door PACK.
    if kind == 0:  # GaussianNB
        if name == "theta_":
            return _F  # NB_BLOCK_THETA
        if name == "var_":
            return _F + 1  # NB_BLOCK_VAR
        return 0
    if kind == 4:  # CategoricalNB: its arrays are lists per feature in scikit-learn
        return 0
    if name == "feature_log_prob_":
        return _F + 6  # NB_BLOCK_LOG_PROB
    if name == "feature_count_":
        return _F + 8  # NB_BLOCK_FEATURE_COUNT
    return 0


def _da_named(kind: Int, name: String) -> Int:
    # core/discriminant_member.mojo DA_BLOCK_*: means_ both kinds, LDA (kind 0) xbar_.
    if name == "means_":
        return _F  # DA_BLOCK_MEANS
    if kind == 0 and name == "xbar_":
        return _F + 2  # DA_BLOCK_XBAR
    if kind == 0 and name == "covariance_":
        # Round b (lane mf-member8): DA_BLOCK_COV, d x d, published by PACK for solver 'lsqr' / 'eigen' or
        # store_covariance=True (as scikit-learn's LDA); otherwise refused at admission in attrgetter's words.
        return _F + 7
    return 0


def _xdecomp_named(kind: Int, name: String) -> Int:
    # core/xdecomp_member.mojo XDECOMP_KIND_* / XDECOMP_BLOCK_COMPONENTS (+ k); member_xdecomp_boundary_door PACK.
    if kind == 14:  # EllipticEnvelope / MinCovDet
        if name == "location_":
            return _F
        if name == "raw_location_":
            return _F + 2
        if name == "raw_covariance_":
            return _F + 3
        if name == "covariance_":
            return _F + 4
        if name == "precision_":
            # Round b (lane mf-member8): the stored precision block (d x d); store_precision=False publishes None,
            # refused at admission in the words scikit-learn's `_get_feature_importances` fails with.
            return _F + 1
        return 0
    if kind == 0 and name == "mean_":  # IncrementalPCA (lane mf-member8): XDECOMP_BLOCK_MEAN, 1 x features
        return _F + 2
    if kind == 0 and name == "var_":  # XDECOMP_BLOCK_VAR, 1 x features
        return _F + 3
    # IncrementalPCA, NMF, Gaussian / Sparse RandomProjection, FastICA, FactorAnalysis, (MiniBatch)DictionaryLearning,
    # (MiniBatch)SparsePCA, LatentDirichletAllocation: components_ (components x features).
    if name == "components_" and (kind == 0 or kind == 3 or kind == 5 or kind == 6 or kind == 7 or kind == 8
                                  or kind == 9 or kind == 10 or kind == 12 or kind == 13 or kind == 15):
        return _F
    return 0


def _scaler_named(kind: Int, name: String) -> Int:
    # core/scaler_member.mojo SCALER_BLOCK_STATS (MEMBER_BLOCK_FAMILY): StandardScaler 3 x features (mean_, var_,
    # scale_); MinMaxScaler 5 x features (data_min_, data_max_, data_range_, scale_, min_).
    if kind == 0:
        if name == "mean_":
            return rfe_block_code(_F, 0)
        if name == "var_":
            return rfe_block_code(_F, 1)
        if name == "scale_":
            return rfe_block_code(_F, 2)
        return 0
    if kind == 1:
        if name == "data_min_":
            return rfe_block_code(_F, 0)
        if name == "data_max_":
            return rfe_block_code(_F, 1)
        if name == "data_range_":
            return rfe_block_code(_F, 2)
        if name == "scale_":
            return rfe_block_code(_F, 3)
        if name == "min_":
            return rfe_block_code(_F, 4)
    return 0


def _prep_named(kind: Int, name: String) -> Int:
    # core/prep_member.mojo PREP_KIND_* / PREP_BLOCK_*; member_prep_boundary_door PACK.
    if kind == 0:  # RobustScaler
        if name == "center_":
            return _F  # PREP_BLOCK_CENTER
        if name == "scale_":
            return _F + 1  # PREP_BLOCK_SCALE
        return 0
    if kind == 1:  # MaxAbsScaler
        if name == "max_abs_":
            return _F  # PREP_BLOCK_MAX_ABS
        if name == "scale_":
            return _F + 1
        return 0
    if kind == 7 and name == "statistics_":  # SimpleImputer (round b, lane mf-member8): PREP_BLOCK_CENTER, 1 x features
        return _F
    if kind == 5 and name == "quantiles_":  # QuantileTransformer
        return _F + 2  # PREP_BLOCK_EXTRA
    if kind == 6 and name == "lambdas_":  # PowerTransformer
        return _F + 2
    return 0


def _select_named(kind: Int, name: String) -> Int:
    # core/select_member.mojo SELECT_KIND_* / SELECT_BLOCK_*.
    if kind == 0 and name == "variances_":
        return _F  # SELECT_BLOCK_VARIANCES
    if kind == 1 and name == "scores_":
        return _F  # SELECT_BLOCK_SCORES
    if kind == 1 and name == "pvalues_":
        return _F + 2  # SELECT_BLOCK_PVALUES
    return 0


def rfe_named_block(family: Int, kind: Int, name: String) -> Int:
    """THE NAMED-ARRAY TABLE (module note above): the block code a kind exports under the public attribute
    `name` when its PACK writes that attribute from the block (or one row of it), else 0. scikit-learn's RFE
    reads any attribute by name (attrgetter), squares it and sums it over rows; the deleted door read the
    fitted estimator's fields the same way. Families: naive Bayes, discriminant, kernel SVM (support_vectors_),
    kernel ridge (X_fit_), Nystroem (components_), Gaussian processes (X_train_), KMeans and MiniBatchKMeans /
    BisectingKMeans / MeanShift (cluster_centers_), mixtures (means_), PCA / TruncatedSVD (components_, PCA
    mean_), the extra decompositions and MinCovDet, StandardScaler / MinMaxScaler (their rows), RobustScaler,
    MaxAbsScaler, QuantileTransformer, PowerTransformer, VarianceThreshold / SelectKBest, ARD / BayesianRidge
    (lambda_, X_offset_), PLS (_x_mean, _x_std)."""
    if family == MEMBER_FAMILY_NAIVE_BAYES:
        return _nb_named(kind, name)
    if family == MEMBER_FAMILY_DISCRIMINANT:
        return _da_named(kind, name)
    if family == MEMBER_FAMILY_KERNEL_SVM:
        return _F + 1 if name == "support_vectors_" else 0  # core/kernel_svm_member.mojo _B_VECTORS
    if family == MEMBER_FAMILY_KERNEL_RIDGE:
        return _F if name == "X_fit_" else 0  # core/kernel_ridge_member.mojo _B_X_FIT
    if family == MEMBER_FAMILY_KERNEL_APPROX:
        return _F if (kind == 0 and name == "components_") else 0  # Nystroem FAMILY+0
    if family == MEMBER_FAMILY_GP:
        return _F if name == "X_train_" else 0  # core/gp_member.mojo _B_X
    if family == MEMBER_FAMILY_KMEANS:
        return _F if name == "cluster_centers_" else 0  # core/kmeans_member.mojo _BLOCK_CENTERS
    if family == MEMBER_FAMILY_XCLUSTER:
        # MiniBatchKMeans, BisectingKMeans, MeanShift, AffinityPropagation: PACK's cluster_centers_ is FAMILY+0
        # (lane mf-member8: AffinityPropagation's centers are the centroid-fit entry's exemplar rows, kept in the
        # state and exported as that block, core/xcluster_member.mojo `_affinity_fit`; an affinity='precomputed'
        # fit publishes none: providers/python/member_rfe_boundary_door.mojo refuses the name there in
        # attrgetter's words, and a fit with no exemplar exports a 0 x features block, squared and summed to
        # zeros as scikit-learn's `_get_feature_importances` sums its empty cluster_centers_).
        if kind == 5 and name == "means_":
            # Round b (lane mf-member8): BayesianGaussianMixture (and GaussianMixture on this kind's plain words):
            # means_ is the entry's f[1] block (_BLOCK_BGMM + 1 = FAMILY+9), n_components x features.
            return _F + 9
        return _F if ((kind == 0 or kind == 1 or kind == 2 or kind == 4) and name == "cluster_centers_") else 0
    if family == MEMBER_FAMILY_MIXTURE:
        return _F + 1 if name == "means_" else 0  # member_mixture_boundary_door PACK: means_ (k x d)
    if family == MEMBER_FAMILY_DECOMPOSITION:
        if name == "components_":
            return _F  # DECOMP_BLOCK_COMPONENTS
        return _F + 1 if (kind == 0 and name == "mean_") else 0  # PCA DECOMP_BLOCK_MEAN
    if family == MEMBER_FAMILY_XDECOMP:
        return _xdecomp_named(kind, name)
    if family == MEMBER_FAMILY_SCALER:
        return _scaler_named(kind, name)
    if family == MEMBER_FAMILY_PREP:
        return _prep_named(kind, name)
    if family == MEMBER_FAMILY_SELECT:
        return _select_named(kind, name)
    if family == MEMBER_FAMILY_XLINEAR:
        if kind == 5 and name == "lambda_":
            return _F + 1  # XL_MEMBER_BLOCK_LAMBDA (ARDRegression)
        if kind == 5 and name == "sigma_":
            return _F + 8  # XL_MEMBER_BLOCK_SIGMA (ARDRegression, kept x kept: the width check decides)
        if kind == 4 and name == "sigma_":
            return RFE_COMPUTED_BAYES_SIGMA
        if (kind == 4 or kind == 5) and name == "X_scale_":
            return RFE_COMPUTED_ONES
        return _F + 9 if ((kind == 4 or kind == 5) and name == "X_offset_") else 0  # XL_MEMBER_BLOCK_X_OFFSET
    if family == MEMBER_FAMILY_XNEIGHBORS and kind == 0:
        # Round b (lane mf-member8): NearestCentroid (core/xneighbors_member.mojo NC_BLOCK_*): centroids_ and
        # deviations_ (classes x features), within_class_std_dev_ (features); scikit-learn publishes all three.
        if name == "centroids_":
            return _F
        if name == "within_class_std_dev_":
            return _F + 1
        return _F + 2 if name == "deviations_" else 0
    if family == MEMBER_FAMILY_PLS:
        if name == "_x_mean":
            return _F
        return _F + 1 if name == "_x_std" else 0
    return 0


def rfe_member_source_named(family: Int, kind: Int, mask: Int, getter: String, estimator: String) -> RfeSource:
    """`rfe_member_source` for any importance_getter attribute name (lane mf-rfe7): 'auto', 'coef_' and
    'feature_importances_' as there; any other name the block `rfe_named_block` gives it (squared, as the
    attribute path squares it), else scikit-learn's attrgetter words, raised before the first round fits."""
    if getter == "auto" or getter == "coef_" or getter == "feature_importances_":
        return rfe_member_source(family, mask, getter, estimator)
    var block = rfe_named_block(family, kind, getter)
    if block != 0:
        return RfeSource(block, True, String(""))
    return RfeSource(0, False, "AttributeError: '" + estimator + "' object has no attribute '" + getter + "'")


def rfe_member_fit(plan: MemberPlan, mut call: MemberCall, x: Int, n: Int, k: Int, y: Int, y_dtype: Int,
                   extras: MemberFitExtras, columns: Int) raises:
    """FIT of the caller's new member state (`call`, from `member_new(plan)`)
    on the (n, k) float32 host block at `x` and the n target words at `y`,
    with the fit's extra inputs (`extras`: RFE's fit parameters; the
    held-out X restricted to the block's k columns, whose int32 ids into the
    caller's X are at `columns`, as scikit-learn's RFE fits every round on
    `X[:, features]`). Needs no GIL: the caller releases it only for a kind
    with MEMBER_GIL_FREE or MEMBER_GIL_DOOR (THE GIL RULE) and frees the
    state."""
    call.x = MemberMatrix(x, n, k)
    call.y = MemberMatrix(y, n, 1, y_dtype)
    var held = List[Float32]()
    member_fit_extras_apply(extras, call, columns, k, False, held)
    try:
        _ = member_call(plan.member, MEMBER_OP_FIT, call)
    finally:
        member_fit_extras_clear(call)
        _ = len(held)  # the gathered held-out block lives until the fit returned


def rfe_member_importances(plan: MemberPlan, mut call: MemberCall, x: Int, n: Int, m: Int, y: Int, y_dtype: Int,
                           block: Int, squared: Bool, mut notices: List[Float64], mut rows: Int,
                           extras: MemberFitExtras, columns: Int) raises -> List[Float32]:
    """`rfe_member_importances` with no nested path (the importance block is the member's own)."""
    return rfe_member_importances(plan, call, x, n, m, y, y_dtype, block, squared, notices, rows, extras, columns,
                                  List[Float64]())


def rfe_member_importances(plan: MemberPlan, mut call: MemberCall, x: Int, n: Int, m: Int, y: Int, y_dtype: Int,
                           block: Int, squared: Bool, mut notices: List[Float64], mut rows: Int,
                           extras: MemberFitExtras, columns: Int, hops: List[Float64]) raises -> List[Float32]:
    """`hops` non-empty (lane mf-meta5b: a dotted importance_getter such as 'best_estimator_.coef_'):
    `block` is the META kind's forwarded-export block and `hops` its words (each nested member's index along
    the path, then the leaf's block), read by the EXPORT through `call.params` (core/ensemble_member.mojo
    `META_BLOCK_FORWARD`). A round over a native member: fit the caller's new state on the
    round's (n, m) block (with the fit's `extras` over the round's column ids
    at `columns`, `rfe_member_fit`), take its fit notices, read its
    importance block `block` by EXPORT and copy it out (the view dies with
    the state, which the caller frees). m words, or (rows, m) for a
    coefficient block with one row per class or output (`squared`); `rows`
    is set to its row count. Raises when the member refuses the round;
    nothing was staged."""
    rfe_member_fit(plan, call, x, n, m, y, y_dtype, extras, columns)
    member_notices_take(plan, call.state, notices)
    if block == RFE_COMPUTED_ONES:
        # X_scale_: m ones (squared, m ones), as PACK writes it.
        rows = 1
        return List[Float32](length=m, fill=1.0)
    if block == RFE_COMPUTED_BAYES_SIGMA:
        return _rfe_bayes_sigma(plan, call, m, rows)
    # A block code (lane mf-rfe7b, `rfe_block_code`): the block, and the row a named attribute is of it.
    var row = rfe_row_of(block)
    call.block = rfe_block_of(block)
    call.flags = 0
    if len(hops) > 0:
        call.params = Int(hops.unsafe_ptr())
        call.param_count = len(hops)
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    _ = len(hops)  # read through the export above
    call.params = 0
    call.param_count = 0
    var view = call.dst
    var first = 0
    rows = view.rows
    if row >= 0:
        # One row of the block (a StandardScaler's var_ of its stats block, ...).
        if row >= view.rows:
            raise Error("mojolearn: RFE importances have 0 columns, expected " + String(m))
        first = row
        rows = 1
    if view.cols != m:
        # The deleted door's words for a named attribute of another width (`_keys_of`).
        raise Error("mojolearn: RFE importances have " + String(view.cols) + " columns, expected " + String(m))
    if rows == 0 and squared and row < 0:
        # Lane mf-member8: an empty (0 x m) attribute (an AffinityPropagation fit with no exemplar): scikit-learn
        # squares it and sums it over its rows (sklearn/feature_selection/_base.py `_get_feature_importances`),
        # m zeros.
        rows = 1
        return List[Float32](length=m, fill=0)
    if rows < 1 or (rows > 1 and not squared):
        raise Error("mojolearn: RFE importances are " + String(view.rows) + " x " + String(view.cols)
                    + " words, expected " + String(m) + " columns")
    var count = rows * m
    var words = List[Float32](length=count, fill=0)
    if view.dense_host(MEMBER_F32):
        unsafe_memcpy(dest=words.unsafe_ptr(), src=SEL_F32P(unsafe_from_address=view.data + 4 * first * m),
                      count=count)
    elif view.dense_host(MEMBER_F64):
        # A float64 block (a mixture's means_): rounded to float32, as the door's float32 admission of the
        # attribute rounded it. Serial: one block per round, features x rows words.
        var src = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=view.data + 8 * first * m)
        for i in range(count):  # small-loop(one exported block per round)
            words[i] = Float32(src[i])
    elif view.dense_host(MEMBER_I32):
        var src = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=view.data + 4 * first * m)
        for i in range(count):  # small-loop(one exported block per round)
            words[i] = Float32(src[i])
    else:
        raise Error("TypeError: mojolearn: RFE importances must be a numeric host block")
    return words^


def _rfe_bayes_sigma(plan: MemberPlan, mut call: MemberCall, m: Int, mut rows: Int) raises -> List[Float32]:
    """BayesianRidge's sigma_ of the round's fitted state (m x m words, `rows` = m), formed from its factor block
    ((m + 1) x m: V, then the clamped eigenvalues) and its scalars [intercept, alpha_, lambda_, n_iter] by the
    property's own RUN (`xl_bayes_sigma_run`, parallel over the cells)."""
    call.flags = 0
    call.block = MEMBER_BLOCK_SCALARS
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    if not call.dst.dense_host(MEMBER_F64) or call.dst.rows * call.dst.cols < 3:
        raise Error("mojolearn: RFE: a BayesianRidge member's scalars are [intercept, alpha_, lambda_, n_iter]")
    var scalars = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call.dst.data)
    var alpha = scalars[1]
    var lam = scalars[2]
    call.block = _XL_BLOCK_FACTOR
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var view = call.dst
    if view.cols != m or view.rows != m + 1 or not view.dense_host(MEMBER_F32):
        raise Error("mojolearn: RFE importances have " + String(view.cols) + " columns, expected " + String(m))
    var words = List[Float32](length=m * m, fill=0)
    xl_bayes_sigma_run(view.data, view.data + 4 * m * m, m, alpha, lam, Int(words.unsafe_ptr()))
    rows = m
    return words^


def rfe_final_plan(plan: MemberPlan, importance_word: Bool) -> MemberPlan:
    """The final fit's plan: the estimator's plain fit, with the importance
    word the rounds set cleared (nobody reads the final fit's importances,
    and computing them costs a column-major copy of the block)."""
    var final = plan.copy()
    if importance_word:
        final.params[10] = 0.0
    return final^


# ---------------------------------------------------------------- the RFE fit RUN (lane mf-meta5)
# THE ONE BODY of RFE.fit over a native member (lane mf-meta5, 2026-11-23): the elimination rounds and
# the final fit, typed. The door (bindings/prep_select_native_door.mojo `_rfe_rounds`, a member it admitted)
# and THE RFE KIND (core/ensemble_member.mojo `_fit_rfe`: RFE as a member of another meta-estimator, a
# search, a calibration or an imputer) both run it. Each round's program is the one the door built
# (`rfe_first_stage`, `rfe_round_stage`: the same stages, records and order), run by the typed caller's
# run (bindings/prep_classify_prog_run.mojo `cls_borrowed_run`: the door's route words and input choices),
# so no bit moves. The rounds are serial by nature (a round's block is chosen from the previous round's
# importances); each round's member fit splits over the host cores inside its own family.


struct RfeFitPlan(Movable):
    """RFE.fit's typed request. `member`: the rounds' member plan (the importance word set by the
    admission, `importance_word`); `run`: the selected x_prep image of the rounds' programs, `work_on` its
    `work` switch, read once at UNPACK; `x`: n x d float32 host rows; `y`: n target words (`y_dtype`:
    MEMBER_I32 class codes of `member.classes` classes, or MEMBER_F32 values); `nsel` / `step` the resolved
    counts (`rfe_select_count`, `rfe_step_count`); `block` / `squared` the importance source
    (`rfe_member_source`); `refusal` non-empty: the words the first round raises before it fits;
    `extras` the fit's extra inputs (sample weights, a held-out set); `hops` a dotted getter's forwarded-export
    words (`rfe_member_importances`; empty: the member's own block). `scope` / `pin`: the caller's
    resident scope word (negative: none) and X's pin token (0: none); `kept_at`: the address of the
    caller's Int the run writes the pin token into the moment a scope's cache takes it (0: none)."""
    var member: MemberPlan
    var run: ClsRunPlan
    var work_on: Bool
    var x: Int
    var n: Int
    var d: Int
    var y: Int
    var y_dtype: Int
    var nsel: Int
    var step: Int
    var block: Int
    var squared: Bool
    var importance_word: Bool
    var refusal: String
    var extras: MemberFitExtras
    var hops: List[Float64]
    var scope: Int
    var pin: Int
    var kept_at: Int
    # THE DEVICE ROWS ROUTE (lane mf-rfe6): `x_index` nonzero: X is a fold's rows, the n int64 row numbers at
    # `x_index` of the base at `x` (`x_base_words` float32 words, d per row), gathered on the device by each
    # round's program (`rfe_rows_gather`: the door's DeviceRows route, the same programs and gather); `pin`
    # then pins the base.
    var x_index: Int
    var x_base_words: Int

    def __init__(out self, var member: MemberPlan, run: ClsRunPlan, work_on: Bool, x: Int, n: Int, d: Int,
                 y: Int, y_dtype: Int):
        self.member = member^
        self.run = run
        self.work_on = work_on
        self.x = x
        self.n = n
        self.d = d
        self.y = y
        self.y_dtype = y_dtype
        self.nsel = 1
        self.step = 1
        self.block = 0
        self.squared = False
        self.importance_word = False
        self.refusal = String("")
        self.extras = MemberFitExtras()
        self.hops = List[Float64]()
        self.scope = -1
        self.pin = 0
        self.kept_at = 0
        self.x_index = 0
        self.x_base_words = 0


def rfe_rows_gather(run: ClsRunPlan) -> Bool:
    """Whether every RFE program on the image of `run` gathers a fold's rows on the device (THE DEVICE ROWS
    ROUTE, lane mf-rfe6): the take-rows entries, the ranges entry and switch, and a device scratch for any
    scratch the program has (`cls_run_route`'s `ranges`, true for every program size). False: the caller
    hands the fold's host rows (the door's programs then copied them, `_values(..., True)`)."""
    return (run.take_rows and run.has_ranges and (run.has_out or run.has_scratch)
            and _switch("MOJOLEARN_ARENA_RANGES"))


struct RfeFitResult(Movable):
    """What RFE.fit publishes, typed: the float32 support (1.0 kept) and ranking words (d each), the kept
    column ids (`selected` of them, in column order), the final estimator's plan (`rfe_final_plan`) and its
    fitted state (OWNED by the caller, which packs or keeps it and frees it; it owns its fit rows), and the
    fit notices of every member fit in fit order (each round's, then the final's)."""
    var support: List[Float32]
    var ranking: List[Float32]
    var columns: List[Int32]
    var selected: Int
    var final_plan: MemberPlan
    var final_state: Int
    var notices: List[List[Float64]]

    def __init__(out self):
        self.support = List[Float32]()
        self.ranking = List[Float32]()
        self.columns = List[Int32]()
        self.selected = 0
        self.final_plan = MemberPlan()
        self.final_state = 0
        self.notices = List[List[Float64]]()


def _rfe_take_f32(done: ClsBorrowedDone, offset: Int, count: Int) -> List[Float32]:
    # `count` float32 words of a finished program's arena (the door's `pr.get`).
    var out = List[Float32](length=count, fill=0.0)
    if count > 0:
        unsafe_memcpy(dest=out.unsafe_ptr(), src=SEL_F32P(unsafe_from_address=done.arena.address + 4 * offset),
                      count=count)
    return out^


def _rfe_take_i32(done: ClsBorrowedDone, offset: Int, count: Int) -> List[Int32]:
    # `count` int32 words of a finished program's arena (the door's `pr.get_i32`: the words as they are).
    var out = List[Int32](length=count, fill=Int32(0))
    if count > 0:
        unsafe_memcpy(dest=out.unsafe_ptr(), src=SEL_I32P(unsafe_from_address=done.arena.address + 4 * offset),
                      count=count)
    return out^


def _rfe_pin_settle(plan: RfeFitPlan, kept: Int):
    # THE PIN RULE: the caller learns the moment a scope's cache took X's pin reference.
    if plan.kept_at != 0 and kept != 0:
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=plan.kept_at)[] = kept


def _rfe_run_settled(plan: RfeFitPlan, program: ClsProgram, sources: List[ClsBorrowed],
                     mut kept: Int) raises -> ClsBorrowedDone:
    # One program run with THE PIN RULE settled on BOTH paths (lane open-fixes, 2026-11-25).
    # `cls_borrowed_run` writes `kept` the moment the scope's cache takes X's pin and may raise after it
    # (a program refused after the put, e.g. "device store: id N is not live" in `upload_ranges`). The
    # settle used to run only after a return, so on that raise the door
    # (bindings/prep_select_native_door.mojo `_rfe_native`) read kept == 0 and unpinned the reference the
    # cache still held; the scope's exit unpinned it a second time: X lost one reference too many and
    # was freed while referenced (SIGSEGV in the garbage collector, importlib and str(exc) after the
    # store errors of x-prep-rfe, x-prep-score-edges-rfe-*, generic-*-prep and the RFE tests, L40S
    # smoke of main 3cd641b37e).
    try:
        var done = cls_borrowed_run(plan.run, program, sources, plan.scope, kept)
        _rfe_pin_settle(plan, kept)
        return done^
    except error:
        _rfe_pin_settle(plan, kept)
        raise error^


def rfe_fit_run(plan: RfeFitPlan) raises -> RfeFitResult:
    """RFE.fit over a native member (module note, THE ONE BODY): program 0 stages the first block from the
    all-ones support; each round fits a new member state on the block (`rfe_member_importances`, its
    notices kept), frees it and runs the round's program (`rfe_round_stage`: keys, c2_rfe_step, c2_rfe_rank,
    the next block); the final estimator is fitted on the selected block (`rfe_member_fit` over
    `rfe_final_plan`) and made to own its rows (`member_own`, THE LIFETIME RULE: the caller keeps the state
    past this frame). No Python: the caller holds the GIL only for a member kind with neither
    MEMBER_GIL_FREE nor MEMBER_GIL_DOOR (THE GIL RULE), and releases it otherwise. Raises the first
    failure (a member's or a program's words); no state is left behind."""
    var n = plan.n
    var d = plan.d
    var x_words = n * d
    var kept = 0
    var support = List[Float32](length=d, fill=1.0)
    var ranking = List[Float32](length=d, fill=1.0)
    # THE DEVICE ROWS ROUTE: the base and the fold's row numbers, gathered by each program on the device.
    var x_source = ClsBorrowed(plan.x, x_words, plan.pin, 0, 0, 0)
    if plan.x_index != 0:
        x_source = ClsBorrowed(plan.x, plan.x_base_words, plan.pin, plan.x_index, n, d)
    var first_program = ClsProgram(plan.work_on)
    var first = rfe_first_stage(first_program, x_words, n, d)
    var first_sources: List[ClsBorrowed] = [ClsBorrowed(Int(support.unsafe_ptr()), d, 0, 0, 0, 0), x_source]
    var first_done = _rfe_run_settled(plan, first_program, first_sources, kept)
    _ = len(support)  # read through the run above
    var feat = _rfe_take_i32(first_done, first.ids, d)
    var block = _rfe_take_f32(first_done, first.block, n * d)
    _ = first_done^
    var result = RfeFitResult()
    var nsup = d
    while nsup > plan.nsel:  # driver-loop: over elimination rounds (serial by nature: module note)
        if plan.refusal.byte_length() > 0:
            # A kind with no importance block: refused before the round fits (`rfe_member_source`).
            raise Error(plan.refusal.copy())
        var drop = min(plan.step, nsup - plan.nsel)
        var k = nsup - drop
        var notices = List[Float64]()
        var rows = 1
        var words = List[Float32]()
        var columns = Int(feat.unsafe_ptr()) if plan.extras.eval_rows > 0 else 0
        var call = member_new(plan.member)
        try:
            words = rfe_member_importances(plan.member, call, Int(block.unsafe_ptr()), n, nsup, plan.y,
                                           plan.y_dtype, plan.block, plan.squared, notices, rows, plan.extras,
                                           columns, plan.hops)
        finally:
            member_free(plan.member, call)
        _ = len(block)  # read through the member's fit above
        if len(notices) > 0:
            result.notices.append(notices^)
        var keys = RfeKeys(nsup, rows, plan.squared)
        var program = ClsProgram(plan.work_on)
        var staged = rfe_round_stage(program, keys, words^, x_words, n, d, nsup, drop, k)
        # Caller buffers in record order (the door's `pr.own` order): support, ranking, the round's
        # column ids, X (the keys are the program's own words).
        var sources: List[ClsBorrowed] = [
            ClsBorrowed(Int(support.unsafe_ptr()), d, 0, 0, 0, 0),
            ClsBorrowed(Int(ranking.unsafe_ptr()), d, 0, 0, 0, 0),
            ClsBorrowed(Int(feat.unsafe_ptr()), nsup, 0, 0, 0, 0),
            x_source,
        ]
        var done = _rfe_run_settled(plan, program, sources, kept)
        _ = len(support)  # read through the run above
        _ = len(ranking)
        _ = len(feat)
        support = _rfe_take_f32(done, staged.support, d)
        ranking = _rfe_take_f32(done, staged.ranking, d)
        feat = _rfe_take_i32(done, staged.next.ids, k)
        block = _rfe_take_f32(done, staged.next.block, n * k)
        _ = done^
        nsup = k
    var final_plan = rfe_final_plan(plan.member, plan.importance_word)
    var columns = Int(feat.unsafe_ptr()) if plan.extras.eval_rows > 0 else 0
    var call = member_new(final_plan)
    try:
        rfe_member_fit(final_plan, call, Int(block.unsafe_ptr()), n, nsup, plan.y, plan.y_dtype, plan.extras,
                       columns)
        var notices = List[Float64]()
        member_notices_take(final_plan, call.state, notices)
        if len(notices) > 0:
            result.notices.append(notices^)
        member_own(final_plan, call.state)
    except error:
        member_free(final_plan, call)
        raise error^
    _ = len(block)  # read through the final fit above (the state owns its rows since `member_own`)
    result.final_state = call.state
    result.final_plan = final_plan^
    result.support = support^
    result.ranking = ranking^
    result.selected = nsup
    result.columns = feat^
    return result^
