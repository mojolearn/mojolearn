# SPDX-License-Identifier: Apache-2.0
"""RUN of the category builders the encoders share (OrdinalEncoder,
OneHotEncoder, TargetEncoder, the label owner's numeric list fit, the
selectors' discrete features): the typed requests and stage builders of the
category fit, the code lookup, the infrequent grouping, the grouping remap
and the category gather (lane dw-prep-encode, 2026-11-21).

The door (bindings/prep_encode_native_door.mojo `_fit_categories_caps`,
`_fit_category_words`, `_fit_categories_with_codes`, `_codes`,
`_fit_infrequent`, `_given_categories`) is UNPACK (the route of the image,
X and the fitted category block as owners of the door's `EncProg`, the
per-column grouping lists as native lists), then one builder here on the
typed `EncProgram`, then the typed run, then PACK. No builder reads a field,
a mode or an entry table. The same allocations and stages in the same order
as the door's former builders, so the bits do not move. No Python import
(tools/python_free_dirs.sh)."""
from bindings.prep_encode_prog_run import EncProgram, enc_input, enc_label_chunk, enc_floats_of
from core.preprocessing_program import prep_program_stage

comptime CAT_NONE = -1
# x_prep stage codes (_expansion_prep._OPS).
comptime CAT_OP_SORT_COLS = 0
comptime CAT_OP_UNIQUE_COLS = 5
comptime CAT_OP_LOOKUP = 7
comptime CAT_OP_COUNT_NEG = 8
comptime CAT_OP_I2F = 10
comptime CAT_OP_CAT_GATHER = 78
comptime CAT_OP_CODE_COUNTS = 88
comptime CAT_OP_REMAP_CODES = 89
comptime CAT_OP_UNIQ_COUNT = 120
comptime CAT_OP_UNIQ_SCAN = 121
comptime CAT_OP_UNIQ_WRITE = 122
comptime CAT_OP_CAT_ZERO = 157
comptime CAT_OP_CAT_PRESENT = 158
comptime CAT_OP_PRES_COUNT = 159
comptime CAT_OP_PRES_WRITE = 160
comptime CAT_OP_CAT_PACK = 161
comptime CAT_OP_UNIQUE_INVERSE = 177
comptime CAT_OP_INF_M0 = 249
comptime CAT_OP_INF_M1 = 250
comptime CAT_OP_INF_MAP = 251
# `_CAT_CAP`, `_CAT_R`, `_CAT_CH` (x_prep/cat_cls2.mojo): the packed distinct
# values' host region (words), the presence flags per column, flags per chunk.
comptime CAT_CAP = 65536
comptime CAT_R = 4096
comptime CAT_CH = 64
# `CatFit.route`: a sort per column and unique_cols; a sort and one run scan
# per column (the FAST image's x_prep_fast_unique); the distinct values
# packed into a small host region after a sort, or after the presence flags
# (the FAST image's x_prep_cls2_cat bits); the sort with unique_inverse (the
# C08 fit, which also leaves the (n, d) inverse).
comptime CAT_FIT_UNIQUE = 0
comptime CAT_FIT_SCAN = 1
comptime CAT_FIT_PACK_SORT = 2
comptime CAT_FIT_PACK_PRESENT = 3
comptime CAT_FIT_INVERSE = 4


def cat_uniq_scans(mut prog: EncProgram, base: Int, rows: Int, d: Int, stride: Int, counts: Int, dst: Int,
                   presence: Bool):
    """One device run scan per column (uniq_count or pres_count, uniq_scan,
    uniq_write or pres_write) over the column at base + c * stride."""
    var ch = CAT_CH if presence else enc_label_chunk(rows)
    var nch = (rows + ch - 1) // ch
    for c in range(d):  # driver-loop: stages one device scan per column (feature count)
        var cnt = prog.work(nch)
        var off = prog.work(nch)
        prep_program_stage(prog.program, CAT_OP_PRES_COUNT if presence else CAT_OP_UNIQ_COUNT, nch, base + c * stride,
                           rows, ch, cnt)
        prep_program_stage(prog.program, CAT_OP_UNIQ_SCAN, 1, cnt, nch, off, counts + c)
        prep_program_stage(prog.program, CAT_OP_PRES_WRITE if presence else CAT_OP_UNIQ_WRITE, nch, base + c * stride,
                           rows, ch, off, dst + c * stride)


# ---------------------------------------------------------------- the category fit
@fieldwise_init
struct CatFit(Copyable, Movable):
    """The sorted distinct values (-0.0 folded into 0.0) of every column of
    an (n, d) float32 block: owner `x` of the door, or words already in the
    arena at `held` (not negative: program-owned label words)."""
    var n: Int
    var d: Int
    var x: Int
    var held: Int
    var route: Int


@fieldwise_init
struct CatFitPlan(Copyable, Movable):
    """Where the fit leaves its results. `counts`: the d float32 category
    counts. `packed` False: column c's values at `values + c * n`. `packed`
    True: the columns' values one after another at `values`, valid when no
    word of the d int32 `flags` is set (`flags` not negative) and the counts
    sum to at most `cap`. `inverse`: the (n, d) codes of CAT_FIT_INVERSE."""
    var values: Int
    var counts: Int
    var packed: Bool
    var flags: Int
    var cap: Int
    var inverse: Int


def cat_fit_stage(mut prog: EncProgram, r: CatFit) -> CatFitPlan:
    var n = r.n
    var d = r.d
    var xo = r.held
    if xo < 0:
        xo = prog.input(n * d, r.x)
    if r.route == CAT_FIT_PACK_PRESENT or r.route == CAT_FIT_PACK_SORT:
        var co = prog.alloc(d)
        var cap = min(n * d, CAT_CAP)
        var pk = prog.alloc(cap)
        var bad = CAT_NONE
        if r.route == CAT_FIT_PACK_PRESENT:
            bad = prog.alloc(d)
            var fl = prog.work(d * CAT_R)
            var uo = prog.work(d * CAT_R)
            prep_program_stage(prog.program, CAT_OP_CAT_ZERO, d * CAT_R, fl)
            prep_program_stage(prog.program, CAT_OP_CAT_PRESENT, n * d, xo, n, d, CAT_R, fl, bad)
            cat_uniq_scans(prog, fl, CAT_R, d, CAT_R, co, uo, True)
            prep_program_stage(prog.program, CAT_OP_CAT_PACK, d * CAT_R, uo, CAT_R, d, co, cap, pk)
        else:
            var so = prog.work(n * d)
            var uo = prog.work(n * d)
            prep_program_stage(prog.program, CAT_OP_SORT_COLS, d, xo, n, d, so, 1)
            cat_uniq_scans(prog, so, n, d, n, co, uo, False)
            prep_program_stage(prog.program, CAT_OP_CAT_PACK, n * d, uo, n, d, co, cap, pk)
        return CatFitPlan(pk, co, True, bad, cap, 0)
    if r.route == CAT_FIT_INVERSE:
        var ordered = prog.work(n * d)
        var dictionary = prog.alloc(n * d)
        var counts = prog.alloc(d)
        var inverse = prog.alloc(n * d)
        prep_program_stage(prog.program, CAT_OP_SORT_COLS, d, xo, n, d, ordered, 1)
        prep_program_stage(prog.program, CAT_OP_UNIQUE_INVERSE, d, ordered, n, d, dictionary, counts, xo, inverse)
        return CatFitPlan(dictionary, counts, False, CAT_NONE, 0, inverse)
    var so = prog.work(n * d)
    var uo = prog.alloc(n * d)
    var co = prog.alloc(d)
    prep_program_stage(prog.program, CAT_OP_SORT_COLS, d, xo, n, d, so, 1)
    if r.route == CAT_FIT_SCAN:
        cat_uniq_scans(prog, so, n, d, n, co, uo, False)
    else:
        prep_program_stage(prog.program, CAT_OP_UNIQUE_COLS, d, so, n, d, uo, co)
    return CatFitPlan(uo, co, False, CAT_NONE, 0, 0)


# ---------------------------------------------------------------- the fitted category block
@fieldwise_init
struct CatBlock(Copyable, Movable):
    """A fitted category record as owners of the door: the (d, kmax) float32
    category block (owner `block`, `block_words` words) and the d float32
    per-column sizes (owner `sizes`, `sizes_words` words; -1 when the
    program does not read them)."""
    var block: Int
    var block_words: Int
    var kmax: Int
    var sizes: Int
    var sizes_words: Int


@fieldwise_init
struct CatCodes(Copyable, Movable):
    """The (n, d) category indices (-1: not a category) and the d
    per-column unknown counts (-1 when the codes are device only)."""
    var codes: Int
    var unknown: Int


def cat_codes_stage(mut prog: EncProgram, n: Int, d: Int, x: Int, cats: CatBlock, device_only: Bool) -> CatCodes:
    """`_codes`: each element's category index (or -1) of the (n, d) block
    of owner `x` and, unless device_only, each column's unknown count."""
    var xo = prog.input(n * d, x)
    var block = enc_input(prog, cats.block_words, cats.block)
    var co = enc_input(prog, cats.sizes_words, cats.sizes)
    var codes = prog.scratch(n * d) if device_only else prog.alloc(n * d)
    prep_program_stage(prog.program, CAT_OP_LOOKUP, n * d, xo, n, d, block, cats.kmax, co, codes)
    if device_only:
        return CatCodes(codes, CAT_NONE)
    var neg = prog.alloc(d)
    prep_program_stage(prog.program, CAT_OP_COUNT_NEG, d, codes, n, d, neg)
    return CatCodes(codes, neg)


def cat_held_codes_stage(mut prog: EncProgram, n: Int, d: Int, inverse: Int) -> CatCodes:
    """The codes of a C08 fit's (n, d) float32 inverse (owner `inverse`) and
    each column's unknown count."""
    var codes = enc_input(prog, n * d, inverse)
    var neg = prog.alloc(d)
    prep_program_stage(prog.program, CAT_OP_COUNT_NEG, d, codes, n, d, neg)
    return CatCodes(codes, neg)


def cat_gather_stage(mut prog: EncProgram, codes: Int, n: Int, d: Int, cats: CatBlock) -> Int:
    # `_gather_categories`: each code's category value from the (d, kmax) block.
    var block = enc_input(prog, cats.block_words, cats.block)
    var dst = prog.alloc(n * d)
    prep_program_stage(prog.program, CAT_OP_CAT_GATHER, n * d, codes, n, d, block, cats.kmax, dst)
    return dst


def cat_i2f_stage(mut prog: EncProgram, count: Int, codes: Int) -> Int:
    # `_Prog.put_codes`: the int32 codes of owner `codes` as words, then their floats (i2f).
    var bits = enc_input(prog, count, codes)
    var dst = prog.alloc(count)
    prep_program_stage(prog.program, CAT_OP_I2F, count, bits, dst)
    return dst


# ---------------------------------------------------------------- grouping
@fieldwise_init
struct CatTable(Copyable, Movable):
    # (MAP, MSTRIDE, NMAP) of remap_codes.
    var map: Int
    var stride: Int
    var lens: Int


def cat_list_max(values: List[Int]) -> Int:
    var top = values[0] if len(values) > 0 else 0
    for i in range(len(values)):  # driver-loop: over one column grouping
        top = max(top, values[i])
    return top


def cat_grouping_table(mut prog: EncProgram, groups: List[List[Int]], inverse: Bool) -> CatTable:
    """`_grouping_table`: (MAP, MSTRIDE, NMAP) for remap_codes from the
    per-column grouping lists (empty: the column is not grouped): category
    -> grouped code, or (inverse) grouped code -> category index with the
    infrequent code -> -3."""
    var tables = List[List[Int]]()
    for j in range(len(groups)):  # driver-loop: builds the category grouping table (per-column category groupings)
        if len(groups[j]) == 0:
            tables.append(List[Int]())
            continue
        if not inverse:
            tables.append(groups[j].copy())
            continue
        var nf = cat_list_max(groups[j])
        var back = List[Int](length=nf + 1, fill=0)
        for i in range(len(groups[j])):  # driver-loop: inverts one column grouping table
            if groups[j][i] < nf:
                back[groups[j][i]] = i
        back[nf] = -3
        tables.append(back^)
    var stride = 1
    for j in range(len(tables)):  # driver-loop: over the per-column grouping tables
        stride = max(stride, len(tables[j]))
    var flat = List[Float64]()
    var lens = List[Float64]()
    for j in range(len(tables)):  # driver-loop: pads the per-column grouping tables
        for i in range(stride):  # driver-loop: over one padded grouping table
            flat.append(Float64(tables[j][i]) if i < len(tables[j]) else Float64(0.0))
        lens.append(Float64(len(tables[j])))
    var mo = prog.put_floats(flat)
    var no = prog.put_floats(lens)
    return CatTable(mo, stride, no)


def cat_remap_stage(mut prog: EncProgram, codes: Int, n: Int, d: Int, table: CatTable, neg: Int) -> Int:
    var dst = prog.alloc(n * d)
    prep_program_stage(prog.program, CAT_OP_REMAP_CODES, n * d, codes, n * d, d, table.map, table.stride, table.lens,
                       neg, dst)
    return dst


def cat_grouped_sizes(sizes: List[Int], groups: List[List[Int]]) -> List[Int]:
    """The per-column category counts after grouping: a grouped column
    counts its grouped codes (the largest code plus one)."""
    var grouped = sizes.copy()
    for j in range(min(len(grouped), len(groups))):  # driver-loop: per-column grouped category counts
        if len(groups[j]) > 0:
            grouped[j] = cat_list_max(groups[j]) + 1
    return grouped^


# ---------------------------------------------------------------- infrequent categories
@fieldwise_init
struct InfrequentFit(Movable):
    """`_fit_infrequent` over the (n, d) block of owner `x` looked up in
    `cats`, or (`from_inverse`) over the C08 fit's (n, d) inverse (owner
    `inverse`). `ks`: the per-column counts of groupable categories; `kmax`
    the widest column; `threshold` the minimum count; `max_categories` -1
    for none."""
    var n: Int
    var d: Int
    var x: Int
    var cats: CatBlock
    var from_inverse: Bool
    var inverse: Int
    var ks: List[Int]
    var kmax: Int
    var threshold: Int
    var max_categories: Int


@fieldwise_init
struct InfrequentPlan(Copyable, Movable):
    # The (d, kmax) int32 infrequent masks and grouped codes.
    var mask: Int
    var map: Int


def cat_infrequent_stage(mut prog: EncProgram, r: InfrequentFit) -> InfrequentPlan:
    var n = r.n
    var d = r.d
    var kmax = r.kmax
    var codes = 0
    if r.from_inverse:
        codes = enc_input(prog, n * d, r.inverse)
    else:
        codes = cat_codes_stage(prog, n, d, r.x, r.cats, False).codes
    var cnt = prog.alloc(d * kmax)
    var kc = prog.put_floats(enc_floats_of(r.ks))
    var m0 = prog.work(d * kmax)
    var m1 = prog.alloc(d * kmax)
    var mp = prog.alloc(d * kmax)
    prep_program_stage(prog.program, CAT_OP_CODE_COUNTS, d, codes, n, d, kmax, cnt)
    prep_program_stage(prog.program, CAT_OP_INF_M0, d * kmax, cnt, kc, kmax, r.threshold, m0)
    prep_program_stage(prog.program, CAT_OP_INF_M1, d * kmax, cnt, kc, kmax, m0, r.max_categories, m1)
    prep_program_stage(prog.program, CAT_OP_INF_MAP, d * kmax, kc, kmax, m1, mp)
    return InfrequentPlan(m1, mp)
