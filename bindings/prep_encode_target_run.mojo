# SPDX-License-Identifier: Apache-2.0
"""RUN of TargetEncoder: the typed requests of the fit (with its folds and
the fit_transform's apply) and of transform, and the stage builders of
their x_prep programs (lane dw-prep-encode, 2026-11-21). The category fit
before them is the shared builder (bindings/prep_encode_cats_run.mojo).

The door (bindings/prep_encode_native_door.mojo `_target_run`,
`_target_transform`) is UNPACK (the parameters and fitted fields read once
into `TargetFit` / `TargetApply`; X, the C08 inverse, the target words, the
given folds and the fitted encodings become owners of the door's
`EncProg`), then one builder here on the typed `EncProgram`, then the typed
run, then PACK. No builder reads a field, a mode or an entry table. The
same allocations and stages in the same order as the door's former
builders, so the bits do not move. The per-bucket pass and the gather are
the only device route (their off-arms were deleted on 2026-11-21). No
Python import (tools/python_free_dirs.sh)."""
from bindings.prep_encode_prog_run import EncProgram, enc_input, enc_floats_of
from bindings.prep_encode_kbins_run import enc_seed_words
from bindings.prep_encode_cats_run import CatBlock, cat_codes_stage, cat_i2f_stage
from core.preprocessing_program import prep_program_stage

comptime TE_NONE = -1
# x_prep stage codes (_expansion_prep._OPS).
comptime TE_OP_I2F = 10
comptime TE_OP_TE_GLOBAL = 20
comptime TE_OP_TE_ENC = 21
comptime TE_OP_TE_APPLY = 22
comptime TE_OP_LABEL_BINARIZE = 51
comptime TE_OP_TE_GATHER = 114
comptime TE_OP_TE_HIST = 115
comptime TE_OP_TE_HSUM = 116
comptime TE_OP_TE_HSTART = 117
comptime TE_OP_TE_HSCATTER = 118
comptime TE_OP_CAT_ZERO = 157
comptime TE_OP_CCOUNT = 200
comptime TE_OP_CSCAN = 201
comptime TE_OP_CSTART = 202
comptime TE_OP_CWRITE = 203
comptime TE_OP_KFOLD = 243
comptime TE_OP_STRAT_META = 244
comptime TE_OP_STRAT_FLAG = 245
comptime TE_OP_STRAT_FOLD = 246
# `_p2m_chunks`' minimum rows per chunk.
comptime TE_SEL_CHUNKS = 1024
# `TargetFit.folds_kind`: every row in no fold (the plain fit); the folds
# staged in the program (KFold, or StratifiedKFold over the class codes);
# the given int32 fold of each row (owner `folds`, a user cv).
comptime TE_FOLDS_NONE = 0
comptime TE_FOLDS_STAGED = 1
comptime TE_FOLDS_GIVEN = 2


def _class_rows_chunks(n: Int, per: Int) -> Tuple[Int, Int]:
    # `_p2m_chunks(n, per)`.
    var ch = max(TE_SEL_CHUNKS, (n + TE_SEL_CHUNKS - 1) // TE_SEL_CHUNKS, (n * per + 16777215) // 16777216, 1)
    return (ch, max(1, (n + ch - 1) // ch))


def te_folds_stage(mut prog: EncProgram, n: Int, n_folds: Int, lo: Int, hi: Int, shuffle: Bool, codes: Int,
                   n_classes: Int) -> Tuple[Int, Int]:
    """`_stage_folds`: KFold (c2_kfold) without class codes (`codes`
    TE_NONE), else StratifiedKFold over them (p2m_ccount .. p2m_cwrite,
    c2_strat_meta, c2_strat_flag, c2_strat_fold). Returns (folds, flag or -1)."""
    var fo = prog.alloc(max(n, 1))
    if codes == TE_NONE:
        var seed = prog.put_ints(enc_seed_words(lo, hi))
        prep_program_stage(prog.program, TE_OP_KFOLD, n, seed, n, n_folds, 1 if shuffle else 0, fo)
        return (fo, -1)
    var K = n_classes
    var chunks = _class_rows_chunks(n, K)
    var ch = chunks[0]
    var nch = chunks[1]
    var cnt = prog.alloc(nch * K)
    var tot = prog.alloc(K)
    prep_program_stage(prog.program, TE_OP_CCOUNT, nch, codes, n, K, ch, cnt)
    prep_program_stage(prog.program, TE_OP_CSCAN, K, cnt, nch, K, tot)
    var start = prog.alloc(K)
    var ro = prog.alloc(n)
    prep_program_stage(prog.program, TE_OP_CSTART, 1, tot, K, start, TE_NONE)
    prep_program_stage(prog.program, TE_OP_CWRITE, nch, codes, n, K, ch, cnt, start, ro)
    var meta = prog.alloc(2 * K)
    var flag = prog.alloc(1)
    prep_program_stage(prog.program, TE_OP_STRAT_META, K, start, tot, ro, K, meta)
    prep_program_stage(prog.program, TE_OP_STRAT_FLAG, 1, tot, K, n_folds, flag)
    var seed = prog.put_ints(enc_seed_words(lo, hi))
    prep_program_stage(prog.program, TE_OP_STRAT_FOLD, n, ro, start, tot, codes, meta, n_folds, seed,
                       1 if shuffle else 0, fo)
    return (fo, flag)


@fieldwise_init
struct TargetFit(Movable):
    """TargetEncoder `_run` over the (n, d) block of owner `x` looked up in
    `cats` (`device_codes`: the codes stay in device scratch), or
    (`from_inverse`) over the C08 fit's (n, d) inverse (owner `inverse`).
    The target: n float32 words (`continuous`) or `target_words` int32 class
    codes of owner `target`, with `outputs` 1 or the class count of a
    multiclass target and `n_classes` classes. The folds: `folds_kind`,
    `n_folds`, the uint64 seed, `shuffle`, and for given folds owner `folds`
    of `folds_words` int32 words. `sizes` / `cmax`: the per-column category
    counts and their maximum. `smooth`: -1.0 for 'auto'. `host_column`: the
    host image groups te_enc its own way. `apply`: te_apply over the
    training rows (fit_transform)."""
    var n: Int
    var d: Int
    var outputs: Int
    var x: Int
    var cats: CatBlock
    var device_codes: Bool
    var from_inverse: Bool
    var inverse: Int
    var continuous: Bool
    var target: Int
    var target_words: Int
    var n_classes: Int
    var folds_kind: Int
    var n_folds: Int
    var folds: Int
    var folds_words: Int
    var seed_lo: Int
    var seed_hi: Int
    var shuffle: Bool
    var sizes: List[Int]
    var cmax: Int
    var smooth: Float64
    var host_column: Bool
    var apply: Bool


@fieldwise_init
struct TargetFitPlan(Copyable, Movable):
    """Where the fit leaves its results: the (n_folds + 1, d, cmax, outputs)
    encodings, the (n_folds + 1, outputs, 2) target statistics, the
    stratified folds' refusal flag (-1: none) and the (n, d * outputs)
    applied block (-1 without `apply`)."""
    var enc: Int
    var meta: Int
    var flag: Int
    var applied: Int


def target_fit_stage(mut prog: EncProgram, r: TargetFit, region: Bool) raises -> TargetFitPlan:
    var n = r.n
    var d = r.d
    var T = r.outputs
    var F = r.n_folds
    var cmax = r.cmax
    var codes = 0
    if r.from_inverse:
        codes = enc_input(prog, n * d, r.inverse)
    else:
        codes = cat_codes_stage(prog, n, d, r.x, r.cats, r.device_codes).codes
    var yo = 0
    var co = 0
    if r.continuous:
        yo = enc_input(prog, n, r.target)
    elif T == 1:
        yo = cat_i2f_stage(prog, r.target_words, r.target)
    else:
        # the one-hot rows: word 1.0f (bits 0x3F800000) where row i's code is c
        co = cat_i2f_stage(prog, r.target_words, r.target)
        yo = prog.alloc(n * T)
        prep_program_stage(prog.program, TE_OP_LABEL_BINARIZE, n * T, co, n, T, 0, 0, 0x3F800000, T, yo)
    var fo = 0
    var flag = -1
    if r.folds_kind == TE_FOLDS_STAGED:
        var codes_off = TE_NONE if r.continuous else (yo if T == 1 else co)
        var staged = te_folds_stage(prog, n, F, r.seed_lo, r.seed_hi, r.shuffle, codes_off, r.n_classes)
        fo = staged[0]
        flag = staged[1]
    elif r.folds_kind == TE_FOLDS_NONE:
        # every row in no fold: max(n, 1) program-owned int32 words -1, then their floats
        var m = max(n, 1)
        var words = List[Int32](length=m, fill=-1)
        # The inner list's buffer does not move when `consts_i` grows.
        var address = Int(words.unsafe_ptr())
        prog.consts_i.append(words^)
        var bits = prog.input(m, -1, address)
        fo = prog.alloc(m)
        prep_program_stage(prog.program, TE_OP_I2F, m, bits, fo)
    else:
        fo = cat_i2f_stage(prog, r.folds_words, r.folds)
    var nco = prog.put_floats(enc_floats_of(r.sizes))
    var meta = prog.alloc(2 * (F + 1) * T)
    var smooth = List[Float64]()
    smooth.append(r.smooth)
    var smo = prog.put_floats(smooth)
    var enc = prog.alloc((F + 1) * d * cmax * T)
    prep_program_stage(prog.program, TE_OP_TE_GLOBAL, (F + 1) * T, yo, n, T, fo, meta)
    if r.host_column:
        # the host binding groups te_enc its own way
        prep_program_stage(prog.program, TE_OP_TE_ENC, (F + 1) * d * cmax * T, codes, n, d, yo, T, fo, cmax, nco, meta,
                           smo, enc)
    else:
        # each category's rows, ascending: te_enc walks one bucket, not every row
        var bstart = 0
        var brows = 0
        if r.device_codes:
            bstart = prog.scratch(d * (cmax + 1))
            brows = prog.scratch(n * d)
            # te_gather also visits unused bucket tails: keep their zero row indices
            prep_program_stage(prog.program, TE_OP_CAT_ZERO, n * d, brows)
        else:
            bstart = prog.alloc(d * (cmax + 1))
            brows = prog.alloc(n * d)
        var ch = max(1, min(256, (n + 4095) // 4096))
        var hh = prog.scratch(d * ch * cmax)
        var tot = prog.scratch(d * cmax)
        prep_program_stage(prog.program, TE_OP_TE_HIST, d * ch, codes, n, d, cmax, ch, hh)
        prep_program_stage(prog.program, TE_OP_TE_HSUM, d * cmax, cmax, ch, hh, tot)
        prep_program_stage(prog.program, TE_OP_TE_HSTART, d, cmax, bstart, tot)
        prep_program_stage(prog.program, TE_OP_TE_HSCATTER, d * ch, codes, n, d, cmax, ch, hh, bstart, brows)
        var gb = prog.scratch(n * d * (1 + T))
        prep_program_stage(prog.program, TE_OP_TE_GATHER, n * d, brows, n, d, yo, T, fo, gb)
        prep_program_stage(prog.program, TE_OP_TE_ENC, (F + 1) * d * cmax * T, codes, n, d, yo, T, fo, cmax, nco, meta,
                           smo, enc, bstart + 1, brows, gb + 1)
    var applied = TE_NONE
    if r.apply:
        applied = prog.output(n * d * T, region)
        prep_program_stage(prog.program, TE_OP_TE_APPLY, n * d * T, codes, n, d, T, fo, enc, cmax, meta, F, applied)
    return TargetFitPlan(enc, meta, flag, applied)


@fieldwise_init
struct TargetApply(Copyable, Movable):
    """TargetEncoder.transform of the (n, d) block of owner `x` looked up in
    `cats`: te_apply over the fitted encodings (owner `enc`) and target
    statistics (owner `meta`)."""
    var n: Int
    var d: Int
    var outputs: Int
    var cmax: Int
    var x: Int
    var cats: CatBlock
    var device_codes: Bool
    var enc: Int
    var enc_words: Int
    var meta: Int
    var meta_words: Int


def target_transform_stage(mut prog: EncProgram, r: TargetApply, region: Bool) raises -> Int:
    # Returns the offset of the (n, d * outputs) encoded block.
    var n = r.n
    var d = r.d
    var T = r.outputs
    var codes = cat_codes_stage(prog, n, d, r.x, r.cats, r.device_codes).codes
    var enc = enc_input(prog, r.enc_words, r.enc)
    var meta = enc_input(prog, r.meta_words, r.meta)
    var dst = prog.output(n * d * T, region)
    prep_program_stage(prog.program, TE_OP_TE_APPLY, n * d * T, codes, n, d, T, TE_NONE, enc, r.cmax, meta, 0, dst)
    return dst
