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
