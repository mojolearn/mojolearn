# SPDX-License-Identifier: Apache-2.0
"""RUN of OneHotEncoder: the typed requests of transform and
inverse_transform and the stage builders of their x_prep programs (lane
dw-prep-encode, 2026-11-21). The fit's programs are the shared category
builders (bindings/prep_encode_cats_run.mojo).

The door (bindings/prep_encode_native_door.mojo `_onehot_transform`,
`_onehot_inverse`) is UNPACK (the parameters and fitted fields read once
into `OneHotApply` / `OneHotInverse`; X, the C08 inverse and the fitted
category block become owners of the door's `EncProg`), then one builder
here on the typed `EncProgram`, then the typed run, then PACK. No builder
reads a field, a mode or an entry table. The same allocations and stages in
the same order as the door's former builders, so the bits do not move. No
Python import (tools/python_free_dirs.sh)."""
from bindings.prep_encode_prog_run import EncProgram, enc_floats_of, enc_bad_rows_stages
from bindings.prep_encode_kbins_run import enc_prefix
from bindings.prep_encode_cats_run import (
    CatBlock, CatCodes, cat_codes_stage, cat_held_codes_stage, cat_gather_stage, cat_grouping_table,
    cat_remap_stage, cat_list_max)
from core.preprocessing_program import prep_program_stage

comptime OH_NONE = -1
# x_prep stage codes (_expansion_prep._OPS).
comptime OH_OP_ONEHOT = 9
comptime OH_OP_BLOCK_ARGMAX = 76


def _total(widths: List[Int]) -> Int:
    var total = 0
    for j in range(len(widths)):  # driver-loop: total one-hot output width
        total += widths[j]
    return total


@fieldwise_init
struct OneHotApply(Movable):
    """OneHotEncoder.transform of the (n, d) block of owner `x` looked up in
    `cats`, or (`from_inverse`) of the C08 fit's (n, d) inverse (owner
    `inverse`). `grouped`: the infrequent grouping `groups` remaps the
    codes; with `unknown_slots` an unknown value takes its column's
    infrequent code. `widths` / `drops`: the per-column block widths and
    dropped codes (-1: none)."""
    var n: Int
    var d: Int
    var x: Int
    var cats: CatBlock
    var from_inverse: Bool
    var inverse: Int
    var grouped: Bool
    var groups: List[List[Int]]
    var unknown_slots: Bool
    var widths: List[Int]
    var drops: List[Int]


@fieldwise_init
struct OneHotApplyPlan(Copyable, Movable):
    # The (n, sum(widths)) one-hot block and the d per-column unknown counts.
    var encoded: Int
    var unknown: Int


def onehot_transform_stage(mut prog: EncProgram, r: OneHotApply, region: Bool) raises -> OneHotApplyPlan:
    """`_transform_encoded`: lookup (or the C08 fit's inverse), the grouping
    remap with the unknown slots, and onehot."""
    var n = r.n
    var d = r.d
    var staged = cat_held_codes_stage(prog, n, d, r.inverse) if r.from_inverse else cat_codes_stage(
        prog, n, d, r.x, r.cats, False)
    var codes = staged.codes
    if r.grouped:
        var table = cat_grouping_table(prog, r.groups, False)
        # `_unknown_to`: the grouped code an unknown value takes
        var unk = OH_NONE
        if r.unknown_slots:
            var slots = List[Int]()
            for j in range(len(r.groups)):  # driver-loop: per-column infrequent slot
                slots.append(-1 if len(r.groups[j]) == 0 else cat_list_max(r.groups[j]))
            unk = prog.put_floats(enc_floats_of(slots))
        codes = cat_remap_stage(prog, codes, n, d, table, unk)
    var total = _total(r.widths)
    var so = prog.put_floats(enc_prefix(r.widths))
    var do = prog.put_floats(enc_floats_of(r.drops))
    var dst = prog.output(n * total, region)
    prep_program_stage(prog.program, OH_OP_ONEHOT, n * d, codes, n, d, so, do, total, dst)
    return OneHotApplyPlan(dst, staged.unknown)


@fieldwise_init
struct OneHotInverse(Movable):
    """OneHotEncoder.inverse_transform of the (n, sum(widths)) block of
    owner `x`. `strict[j]` 1.0: an all-zero block of column j cannot be
    inverted (else it is unknown, NaN)."""
    var n: Int
    var x: Int
    var cats: CatBlock
    var widths: List[Int]
    var drops: List[Int]
    var grouped: Bool
    var groups: List[List[Int]]
    var strict: List[Float64]


@fieldwise_init
struct OneHotInversePlan(Copyable, Movable):
    """The (n, d) category values and the (rows, count) of the rows holding
    an all-zero block of a strict column (`enc_bad_rows_stages`)."""
    var values: Int
    var bad_rows: Int
    var bad_count: Int


def onehot_inverse_stage(mut prog: EncProgram, r: OneHotInverse) -> OneHotInversePlan:
    """block_argmax, the inverse grouping remap, cat_gather and the strict
    bad-row compaction."""
    var n = r.n
    var d = len(r.widths)
    var total = _total(r.widths)
    # `_block_argmax`
    var xo = prog.input(n * total, r.x)
    var so = prog.put_floats(enc_prefix(r.widths))
    var wo = prog.put_floats(enc_floats_of(r.widths))
    var dro = prog.put_floats(enc_floats_of(r.drops))
    var codes = prog.alloc(n * d)
    prep_program_stage(prog.program, OH_OP_BLOCK_ARGMAX, n * d, xo, n, total, d, so, wo, dro, 1, codes)
    var grouped = codes
    if r.grouped:
        var table = cat_grouping_table(prog, r.groups, True)
        codes = cat_remap_stage(prog, codes, n, d, table, OH_NONE)
    var dst = cat_gather_stage(prog, codes, n, d, r.cats)
    var sto = prog.put_floats(r.strict)
    var staged = enc_bad_rows_stages(prog, grouped, n, d, -1, sto)
    return OneHotInversePlan(dst, staged[0], staged[1])


# ---------------------------------------------------------------- the restricted fit's drops
#: OneHotEncoder `drop` as a word (a per-column category array is not a word:
#: the door's `_set_drop_idx` serves it).
comptime OH_DROP_NONE = 0
comptime OH_DROP_FIRST = 1
comptime OH_DROP_IF_BINARY = 2


def onehot_auto_drops(sizes: List[Int], drop: Int) -> List[Int]:
    """`_set_drop_idx` without infrequent grouping for a drop word: column
    j's dropped category index (-1: none), which is also its `_drop_after`
    (no grouping remaps it). 'first' drops category 0 of every column,
    'if_binary' of the two-category columns only. The door and the Prep
    member both call it (one path)."""
    var drops = List[Int](capacity=len(sizes))
    for j in range(len(sizes)):  # driver-loop: over the per-column drop indices (feature metadata)
        if drop == OH_DROP_FIRST or (drop == OH_DROP_IF_BINARY and sizes[j] == 2):
            drops.append(0)
        else:
            drops.append(OH_NONE)
    return drops^


def onehot_widths(sizes: List[Int], drops: List[Int]) -> List[Int]:
    # `_widths`: per-column block widths (a dropped category takes no column).
    var widths = List[Int](capacity=len(sizes))
    for j in range(len(sizes)):  # driver-loop: per-column one-hot block widths
        widths.append(sizes[j] - (0 if drops[j] < 0 else 1))
    return widths^


def onehot_total(widths: List[Int]) -> Int:
    return _total(widths)


# ---------------------------------------------------------------- the general fit's drops (lane mf-enc6)
#: `drop` given as one category per column (an array-like: DATA, read once at UNPACK).
comptime OH_DROP_GIVEN = 3


struct OneHotDrops(Movable):
    """`_set_drop_idx`'s fitted words: `none` for drop=None (both fields
    None), else per column `after` (`_drop_after`, in grouped codes) and
    `idx` (drop_idx_, in category indices), -1 for None."""
    var none: Bool
    var after: List[Int]
    var idx: List[Int]

    def __init__(out self):
        self.none = True
        self.after = List[Int]()
        self.idx = List[Int]()


def onehot_drop_index(word: Int, sizes: List[Int], columns: List[List[Float32]], groups: List[List[Int]],
                      infrequent: List[List[Int]], given_nan: List[Bool], given: List[Float32],
                      given_text: List[String]) raises -> OneHotDrops:
    """`_set_drop_idx` in its order and words (one body: the encode door and
    the Prep member call it). `word`: OH_DROP_*; `sizes`: the per-column
    code counts after the grouping (`cat_grouped_sizes`); `columns`: each
    column's fitted category words; `groups` / `infrequent`: the fitted
    grouping (empty lists: a column not grouped); with OH_DROP_GIVEN, per
    column the user's value (`given_nan`: a Python float NaN, else its
    float32 word in `given`) and its str() text `given_text` for the
    refusal."""
    var out = OneHotDrops()
    if word == OH_DROP_NONE:
        return out^
    out.none = False
    var d = len(sizes)
    if word == OH_DROP_FIRST or word == OH_DROP_IF_BINARY:
        out.after = onehot_auto_drops(sizes, word)
    else:
        if len(given) != d:
            raise Error("mojolearn: `drop` should have length equal to the number of features ("
                        + String(d) + "), got " + String(len(given)))
        var missing = String()
        var any_missing = False
        for j in range(d):  # driver-loop: validates the user drop argument (per-column drop arguments)
            var cl = columns[j].copy()
            var hit = -1
            if given_nan[j]:
                if len(cl) > 0 and cl[len(cl) - 1] != cl[len(cl) - 1]:
                    hit = len(cl) - 1
            else:
                for i in range(len(cl)):  # driver-loop: finds the user drop category in its column
                    if cl[i] == given[j]:
                        hit = i
                        break
            if hit < 0:
                missing += ("\n" if any_missing else "") + "Category: " + given_text[j] + ", Feature: " + String(j)
                any_missing = True
                continue
            var i = hit
            if j < len(groups) and len(groups[j]) > 0:
                for k in range(len(infrequent[j])):  # driver-loop: over one column's infrequent categories
                    if infrequent[j][k] == i:
                        raise Error("mojolearn: Unable to drop category " + String(Float64(cl[i])) + " from feature "
                                    + String(j) + " because it is infrequent")
                i = groups[j][i]
            out.after.append(i)
        if any_missing:
            raise Error("mojolearn: The following categories were supposed to be dropped, but were not "
                        + "found in the training data.\n" + missing)
    for j in range(d):  # driver-loop: maps drop indices through the grouping (per-column drop indices)
        var a = out.after[j]
        if a < 0 or j >= len(groups) or len(groups[j]) == 0:
            out.idx.append(a)
            continue
        var found = -1
        for i in range(len(groups[j])):  # driver-loop: over one column grouping
            if groups[j][i] == a:
                found = i
                break
        if found < 0:
            raise Error(String(a) + " is not in list")
        out.idx.append(found)
    return out^
