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
from bindings.prep_classify_prog_run import ClsProgram, CLS_IN_BUFFER, CLS_IN_CONST_F
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, MEMBER_OP_FIT, MEMBER_OP_EXPORT, MEMBER_F32,
)
from core.member_notices import member_notices_take

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
    """One round's importances. `squared`: a coefficient block of `rows`
    rows of `m` float32 words, squared and summed over its rows; else m
    scores of c2_key64 kind `kind` (0 float32, 1 int32, 3 int64, 4 float64;
    `m * (1 or 2)` words). The words travel beside it: empty, they are a
    caller buffer (the next record); else they are the program's own words."""
    var m: Int
    var rows: Int
    var squared: Bool
    var kind: Int


def _rfe_key_words(mut prog: ClsProgram, var words: List[Float32], count: Int) -> Int:
    if len(words) == 0:
        return _cls_buffer(prog, count)
    prog.consts_f.append(words^)
    return prog.input(count, CLS_IN_CONST_F, len(prog.consts_f) - 1, False)


def rfe_keys_stage(mut prog: ClsProgram, keys: RfeKeys, var words: List[Float32]) -> Int:
    """`_keys_of`: the ordered keys (c2_key64) of m column importances.
    Returns the keys' offset."""
    var m = keys.m
    var key = prog.work(2 * m)
    if not keys.squared:
        var per = 1 if keys.kind <= 1 else 2
        var wo = _rfe_key_words(prog, words^, m * per)
        _cls_stage(prog, SEL_OP_C2_KEY64, m, wo, keys.kind, key)
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
    """One elimination round: the importance keys, c2_rfe_step dropping
    `drop` of the `nsup` supported columns from the support mask,
    c2_rfe_rank, then the next (n, k) block. Caller buffers, in record
    order: the importance words (when `words` is empty), the d support
    words (in-out), the d ranking words (in-out), the nsup int32 column ids,
    X."""
    var key = rfe_keys_stage(prog, keys, words^)
    var sup = _cls_buffer(prog, d, True)
    var rk = _cls_buffer(prog, d, True)
    var fo = _cls_buffer(prog, nsup)
    _cls_stage(prog, SEL_OP_C2_RFE_STEP, nsup, key, fo, nsup, drop, sup)
    _cls_stage(prog, SEL_OP_C2_RFE_RANK, d, sup, rk)
    var nxt = rfe_block_stage(prog, x_words, n, d, sup, k)
    return RfeRoundStaged(sup, rk, nxt)


def rfe_member_fit(plan: MemberPlan, mut call: MemberCall, x: Int, n: Int, k: Int, y: Int, y_dtype: Int) raises:
    """FIT of the caller's new member state (`call`, from `member_new(plan)`)
    on the (n, k) float32 host block at `x` and the n target words at `y`.
    Needs no GIL: the caller releases it only for a kind with
    MEMBER_GIL_FREE or MEMBER_GIL_DOOR (THE GIL RULE) and frees the state."""
    call.x = MemberMatrix(x, n, k)
    call.y = MemberMatrix(y, n, 1, y_dtype)
    _ = member_call(plan.member, MEMBER_OP_FIT, call)


def rfe_member_importances(plan: MemberPlan, mut call: MemberCall, x: Int, n: Int, m: Int, y: Int, y_dtype: Int,
                           block: Int, squared: Bool, mut notices: List[Float64], mut rows: Int) raises -> List[Float32]:
    """A round over a native member: fit the caller's new state on the
    round's (n, m) block, take its fit notices, read its importance block
    `block` by EXPORT and copy it out (the view dies with the state, which
    the caller frees). m words, or (rows, m) for a coefficient block with
    one row per class or output (`squared`); `rows` is set to its row
    count. Raises when the member refuses the round; nothing was staged."""
    rfe_member_fit(plan, call, x, n, m, y, y_dtype)
    member_notices_take(plan, call.state, notices)
    call.block = block
    call.flags = 0
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var view = call.dst
    rows = view.rows
    if not view.dense_host(MEMBER_F32) or view.cols != m or rows < 1 or (rows > 1 and not squared):
        raise Error("mojolearn: RFE importances are " + String(view.rows) + " x " + String(view.cols)
                    + " words, expected " + String(m) + " columns")
    var words = List[Float32](length=rows * m, fill=0)
    unsafe_memcpy(dest=words.unsafe_ptr(), src=SEL_F32P(unsafe_from_address=view.data), count=rows * m)
    return words^


def rfe_final_plan(plan: MemberPlan, importance_word: Bool) -> MemberPlan:
    """The final fit's plan: the estimator's plain fit, with the importance
    word the rounds set cleared (nobody reads the final fit's importances,
    and computing them costs a column-major copy of the block)."""
    var final = plan.copy()
    if importance_word:
        final.params[10] = 0.0
    return final^
