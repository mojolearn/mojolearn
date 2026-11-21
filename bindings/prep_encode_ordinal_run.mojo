# SPDX-License-Identifier: Apache-2.0
"""RUN of OrdinalEncoder: the typed requests of transform and
inverse_transform and the stage builders of their x_prep programs (lane
dw-prep-encode, 2026-11-21). The fit's programs are the shared category
builders (bindings/prep_encode_cats_run.mojo).

The door (bindings/prep_encode_native_door.mojo `_ordinal_transform`,
`_ordinal_inverse`) is UNPACK (the parameters and fitted fields read once
into `OrdinalApply` / `OrdinalInverse`; X, the C08 inverse and the fitted
category block become owners of the door's `EncProg`), then one builder
here on the typed `EncProgram`, then the typed run, then PACK. No builder
reads a field, a mode or an entry table. The same allocations and stages in
the same order as the door's former builders, so the bits do not move. No
Python import (tools/python_free_dirs.sh)."""
from bindings.prep_encode_prog_run import EncProgram, enc_floats_of, enc_bad_rows_stages
from bindings.prep_encode_cats_run import (
    CatBlock, CatCodes, CatTable, cat_codes_stage, cat_held_codes_stage, cat_gather_stage, cat_grouping_table,
    cat_remap_stage)
from core.preprocessing_program import prep_program_stage

comptime ORD_NONE = -1
# x_prep stage codes (_expansion_prep._OPS).
comptime ORD_OP_WHERE_NEG = 19
comptime ORD_OP_ORD_INVERSE = 77
comptime ORD_OP_WHERE_CODE = 79


def _one(value: Float64) -> List[Float64]:
    var words = List[Float64]()
    words.append(value)
    return words^


@fieldwise_init
struct OrdinalApply(Movable):
    """OrdinalEncoder.transform of the (n, d) block of owner `x` looked up
    in `cats`, or (`from_inverse`) of the C08 fit's (n, d) inverse (owner
    `inverse`). `grouped`: the infrequent grouping `groups` remaps the
    codes. `use_unknown`: an unknown value becomes `unknown_value`.
    `any_missing`: the code of column j's NaN slot `missing[j]` (-1: none)
    becomes `encoded_missing`."""
    var n: Int
    var d: Int
    var x: Int
    var cats: CatBlock
    var from_inverse: Bool
    var inverse: Int
    var grouped: Bool
    var groups: List[List[Int]]
    var use_unknown: Bool
    var unknown_value: Float64
    var any_missing: Bool
    var missing: List[Int]
    var encoded_missing: Float64


@fieldwise_init
struct OrdinalApplyPlan(Copyable, Movable):
    # The (n, d) encoded block and the d per-column unknown counts.
    var encoded: Int
    var unknown: Int


def ordinal_transform_stage(mut prog: EncProgram, r: OrdinalApply) -> OrdinalApplyPlan:
    """`_transform_encoded`: lookup (or the C08 fit's inverse), the grouping
    remap, where_neg and where_code."""
    var n = r.n
    var d = r.d
    var staged = cat_held_codes_stage(prog, n, d, r.inverse) if r.from_inverse else cat_codes_stage(
        prog, n, d, r.x, r.cats, False)
    var codes = staged.codes
    var dst = codes
    if r.grouped:
        var table = cat_grouping_table(prog, r.groups, False)
        dst = cat_remap_stage(prog, codes, n, d, table, ORD_NONE)
    if r.use_unknown:
        var val = prog.put_floats(_one(r.unknown_value))
        var src = dst
        dst = prog.alloc(n * d)
        prep_program_stage(prog.program, ORD_OP_WHERE_NEG, n * d, src, n * d, val, dst)
    if r.any_missing:
        var src = dst
        dst = prog.alloc(n * d)
        var mo = prog.put_floats(enc_floats_of(r.missing))
        var eo = prog.put_floats(_one(r.encoded_missing))
        prep_program_stage(prog.program, ORD_OP_WHERE_CODE, n * d, codes, n, d, mo, eo, src, dst)
    return OrdinalApplyPlan(dst, staged.unknown)


@fieldwise_init
struct OrdinalInverse(Movable):
    """OrdinalEncoder.inverse_transform of the (n, d) codes of owner `x`.
    `sizes`: the per-column code counts (after grouping). `encoded`:
    handle_unknown is use_encoded_value, with `unknown_value`."""
    var n: Int
    var d: Int
    var x: Int
    var cats: CatBlock
    var grouped: Bool
    var groups: List[List[Int]]
    var sizes: List[Int]
    var missing: List[Int]
    var encoded_missing: Float64
    var encoded: Bool
    var unknown_value: Float64


@fieldwise_init
struct OrdinalInversePlan(Copyable, Movable):
    """The (n, d) category values and the (rows, count) of the rows holding
    a code that names no category (`enc_bad_rows_stages`)."""
    var values: Int
    var bad_rows: Int
    var bad_count: Int


def ordinal_inverse_stage(mut prog: EncProgram, r: OrdinalInverse) -> OrdinalInversePlan:
    """ord_inverse (`_inverse_codes`), the inverse grouping remap,
    cat_gather and the bad-row compaction."""
    var n = r.n
    var d = r.d
    var back = cat_grouping_table(prog, r.groups, True) if r.grouped else CatTable(0, 0, 0)
    var xo = prog.input(n * d, r.x)
    var mo = prog.put_floats(enc_floats_of(r.missing))
    var eo = prog.put_floats(_one(r.encoded_missing))
    var uo = prog.put_floats(_one(r.unknown_value if r.encoded else 0.0))
    var no = prog.put_floats(enc_floats_of(r.sizes))
    var codes = prog.alloc(n * d)
    prep_program_stage(prog.program, ORD_OP_ORD_INVERSE, n * d, xo, n, d, mo, eo, 1 if r.encoded else 0, uo, no, codes)
    if r.grouped:
        codes = cat_remap_stage(prog, codes, n, d, back, ORD_NONE)
    var dst = cat_gather_stage(prog, codes, n, d, r.cats)
    var staged = enc_bad_rows_stages(prog, codes, n, d, -2, ORD_NONE)
    return OrdinalInversePlan(dst, staged[0], staged[1])
