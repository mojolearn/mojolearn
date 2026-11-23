# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""OrdinalEncoder and OneHotEncoder programs against scikit-learn's values (lane gbdt-cpu-rest-enc, 2026-11-23;
ported from python/mojolearn/tests/test_x_prep_encoders.py `test_ordinal`, `test_ordinal_missing_and_inverse`,
`test_onehot`, `test_given_categories`, and test_x_prep_parity.py `test_infrequent`).

The inputs are the original tests' NumPy draws and the references the scikit-learn 1.9.1 calls they made, stored as
words in `fixtures/prep_enc/` by `tools/py2mojo_native_fixtures_b2_enc.py` (a scikit-learn None of an inverse
transform stored as NaN, as the originals compare it). The programs are the typed builders the encode door builds
through (bindings/prep_encode_cats_run.mojo `cat_fit_stage`, `cat_codes_stage`, `cat_infrequent_stage`;
bindings/prep_encode_ordinal_run.mojo `ordinal_transform_stage`, `ordinal_inverse_stage`, `ordinal_auto_clash`;
bindings/prep_encode_onehot_run.mojo `onehot_transform_stage`, `onehot_inverse_stage`, `onehot_auto_drops`) with
the requests bindings/prep_encode_native_door.mojo `_ordinal_fit`, `_ordinal_transform`, `_ordinal_inverse`,
`_onehot_fit`, `_onehot_transform`, `_onehot_inverse` and `_fit_infrequent` make on the identical image (the sorted
category fit with unique_cols, CAT_FIT_UNIQUE; no C08 capture), run on the host column by
`prep_enc_host.enc_host_run`. The category block is the door's record layout: (d, kmax) float32 words, row j's first
sizes[j] words column j's categories, and the d size words.

    test_ordinal                     categories_ and transform exact; use_encoded_value -1 on the unknown 99;
                                     handle_unknown='error' finds the unknown column
    test_ordinal_missing_and_inverse the four keyword sets: categories_ (NaN last), transform and inverse_transform
                                     exact with NaN where NaN, the unknown rows too; encoded_missing_value=1 and
                                     unknown_value=0 refused by `ordinal_auto_clash`
    test_onehot                      the three keyword sets: transform, transform of the unknown 55 under 'ignore',
                                     inverse_transform with NaN for the unknown; an all-zero row refused (row 4)
    test_given_categories            categories=<list>: the ordinal codes and two one-hot blocks exact; a training
                                     value outside its list is found unknown (the fit's refusal)
    test_infrequent                  min_frequency / max_categories: OrdinalEncoder's infrequent_categories_ and
                                     transform; OneHotEncoder's transform and inverse_transform under
                                     infrequent_if_exist, ignore and warn x drop None, 'first', 'if_binary'

Stays in Python (the public boundary): the parameter admission and its refusals (handle_unknown, drop, min_frequency
and max_categories words), the given category lists' validation (unsorted, NaN not last, equal in float32) and their
categories_ arrays, an explicit drop list (`_set_drop_idx` over the user's values) and drop_idx_, the warning
OneHotEncoder 'warn' emits, object and string categories. Every case is an independent unit; the case tables run
across host cores.
"""
from std.math import isnan, ceil, nan
from std.memory import bitcast
from bindings.prep_encode_prog_run import EncProgram
from bindings.prep_encode_cats_run import (
    CatAuto, CatBlock, CatFit, InfrequentFit, CAT_NONE, CAT_FIT_UNIQUE, cat_fit_stage, cat_auto_held, cat_auto_state,
    cat_codes_stage, cat_infrequent_stage, cat_grouped_sizes,
)
from bindings.prep_encode_ordinal_run import (
    OrdinalApply, OrdinalInverse, ordinal_transform_stage, ordinal_inverse_stage, ordinal_auto_clash,
)
from bindings.prep_encode_onehot_run import (
    OneHotApply, OneHotInverse, onehot_transform_stage, onehot_inverse_stage, onehot_auto_drops, onehot_widths,
    onehot_total, OH_DROP_NONE, OH_DROP_FIRST, OH_DROP_IF_BINARY,
)
from checks.native_tests.prep_enc_host import enc_host_run
from checks.native_tests.prep_cls_host import arena_words
from checks.native_tests.py2mojo_fixture_words import fixture_f64
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require, fixture_f32, fixture_i32

comptime PREP_ENC_DIR = "checks/native_tests/fixtures/prep_enc/"

# ---------------------------------------------------------------- comparisons
def same_word(got: Float32, want: Float64, label: String) raises:
    """assert_array_equal for one value: NaN only against NaN (-0.0 equals 0.0)."""
    var a = Float64(got)
    if isnan(want) or isnan(a):
        require(isnan(want) and isnan(a), label + ": " + String(got) + " expected " + String(want))
        return
    require(a == want, label + ": " + String(got) + " expected " + String(want))


def same_words(got: List[Float32], want: List[Float64], label: String) raises:
    require(len(got) == len(want), label + ": " + String(len(got)) + " values, reference " + String(len(want)))
    for i in range(len(want)):  # small-loop(oracle: one encoded block)
        same_word(got[i], want[i], label + " word " + String(i))


def to_f32(values: List[Float64]) -> List[Float32]:
    var words = List[Float32](length=len(values), fill=0)
    for i in range(len(values)):  # small-loop(fixture: reference codes as the float32 words the door admits)
        words[i] = Float32(values[i])
    return words^


def int_word(arena: List[Float32], offset: Int) -> Int:
    return Int(bitcast[DType.int32](arena[offset]))


# ---------------------------------------------------------------- the fitted categories
def size_words(sizes: List[Int]) -> List[Float32]:
    # `_category_build`'s size words: the d sizes as float32 ([0.0] when empty).
    var words = List[Float32](length=max(len(sizes), 1), fill=0)
    for j in range(len(sizes)):  # small-loop(params: the per-column category counts)
        words[j] = Float32(sizes[j])
    return words^


def block_of(c: CatAuto, sizes: Bool) -> CatBlock:
    """The category record as owners 1 (block) and 2 (size words, when `sizes`) of the program (`_cat_block`)."""
    if sizes:
        return CatBlock(1, len(c.block), c.kmax, 2, max(len(c.sizes), 1))
    return CatBlock(1, len(c.block), c.kmax, CAT_NONE, 0)


def owners_of(x: List[Float32], c: CatAuto, sizes: Bool) -> List[List[Float32]]:
    var owners = List[List[Float32]]()
    owners.append(x.copy())
    owners.append(c.block.copy())
    if sizes:
        owners.append(size_words(c.sizes))
    return owners^


def fit_auto(x: List[Float32], n: Int, d: Int) raises -> CatAuto:
    """`_auto_fit` on the identical image: the sorted category fit with unique_cols, its typed state."""
    var prog = EncProgram(False)
    var plan = cat_fit_stage(prog, CatFit(n, d, 0, CAT_NONE, CAT_FIT_UNIQUE))
    var owners = List[List[Float32]]()
    owners.append(x.copy())
    var arena = enc_host_run(prog, owners)
    var address = Int(arena.unsafe_ptr())
    require(cat_auto_held(address, plan, d), "category fit: the unpacked plan did not hold its values")
    var state = cat_auto_state(address, plan, n, d)
    _ = arena^  # the state was read from these words
    return state^


def given_cats(values: List[Float32], sizes: List[Int32]) -> CatAuto:
    """The record of a given category list (`_given_categories`' float32 words, NaN last)."""
    var c = CatAuto()
    c.d = len(sizes)
    for j in range(c.d):  # small-loop(params: the per-column category counts)
        c.kmax = max(c.kmax, Int(sizes[j]))
    c.block = List[Float32](length=max(c.d * c.kmax, 1), fill=Float32(0))
    var at = 0
    for j in range(c.d):  # small-loop(params: four or five columns)
        var size = Int(sizes[j])
        for i in range(size):  # small-loop(params: one column's categories)
            c.block[j * c.kmax + i] = values[at + i]
        at += size
        c.sizes.append(size)
        var last = values[at - 1] if size > 0 else Float32(0)
        c.missing.append(size - 1 if size > 0 and last != last else -1)
    return c^


def check_categories(c: CatAuto, tag: String, label: String) raises:
    var want_sizes = fixture_i32(PREP_ENC_DIR + tag + "-sizes.i32", c.d)
    var total = 0
    for j in range(c.d):  # small-loop(oracle: the reference's category counts)
        require(c.sizes[j] == Int(want_sizes[j]), label + ": categories_[" + String(j) + "] holds " + String(c.sizes[j])
            + " values, scikit-learn " + String(want_sizes[j]))
        total += Int(want_sizes[j])
    var want = fixture_f64(PREP_ENC_DIR + tag + "-cats.f64", total)
    var at = 0
    for j in range(c.d):  # small-loop(oracle: the columns)
        for i in range(c.sizes[j]):  # small-loop(oracle: one column's categories)
            same_word(c.block[j * c.kmax + i], want[at + i], label + ": categories_[" + String(j) + "][" + String(i)
                + "]")
        at += c.sizes[j]


def no_groups(d: Int) -> List[List[Int]]:
    var groups = List[List[Int]]()
    for _ in range(d):  # small-loop(params: the columns)
        groups.append(List[Int]())
    return groups^


# ---------------------------------------------------------------- the programs
@fieldwise_init
struct Encoded(Movable):
    var words: List[Float32]    # the encoded (or inverted) block
    var unknown: List[Float32]  # the per-column unknown counts (transform) or [bad rows, first bad row] (inverse)


def ordinal_transform(c: CatAuto, x: List[Float32], n: Int, grouped: Bool, groups: List[List[Int]],
                      use_unknown: Bool, unknown_value: Float64, encoded_missing: Float64) raises -> Encoded:
    """`_ordinal_transform`'s request: the fitted `_missing` slots, the encoded missing value only with one."""
    var d = c.d
    var any_missing = False
    for j in range(d):  # small-loop(params: the per-column NaN slots)
        any_missing = any_missing or c.missing[j] >= 0
    var prog = EncProgram(False)
    var plan = ordinal_transform_stage(prog, OrdinalApply(n, d, 0, block_of(c, True), False, CAT_NONE, grouped,
        groups.copy(), use_unknown, unknown_value if use_unknown else 0.0, any_missing, c.missing.copy(),
        encoded_missing if any_missing else 0.0))
    var arena = enc_host_run(prog, owners_of(x, c, True))
    return Encoded(arena_words(arena, plan.encoded, n * d), arena_words(arena, plan.unknown, d))


def ordinal_inverse(c: CatAuto, z: List[Float32], n: Int, grouped: Bool, groups: List[List[Int]],
                    encoded_missing: Float64, encoded: Bool, unknown_value: Float64) raises -> Encoded:
    var d = c.d
    var sizes = cat_grouped_sizes(c.sizes, groups) if grouped else c.sizes.copy()
    var prog = EncProgram(False)
    var plan = ordinal_inverse_stage(prog, OrdinalInverse(n, d, 0, block_of(c, False), grouped, groups.copy(),
        sizes^, c.missing.copy(), encoded_missing, encoded, unknown_value if encoded else 0.0))
    var arena = enc_host_run(prog, owners_of(z, c, False))
    var bad = int_word(arena, plan.bad_count)
    var flags: List[Float32] = [Float32(bad), Float32(int_word(arena, plan.bad_rows) if bad > 0 else -1)]
    return Encoded(arena_words(arena, plan.values, n * d), flags^)


def onehot_sizes(c: CatAuto, grouped: Bool, groups: List[List[Int]]) -> List[Int]:
    return cat_grouped_sizes(c.sizes, groups) if grouped else c.sizes.copy()


def onehot_transform(c: CatAuto, x: List[Float32], n: Int, grouped: Bool, groups: List[List[Int]],
                     unknown_slots: Bool, drops: List[Int]) raises -> Encoded:
    var d = c.d
    var widths = onehot_widths(onehot_sizes(c, grouped, groups), drops)
    var total = onehot_total(widths)
    var prog = EncProgram(False)
    var plan = onehot_transform_stage(prog, OneHotApply(n, d, 0, block_of(c, True), False, CAT_NONE, grouped,
        groups.copy(), unknown_slots, widths^, drops.copy()), False)
    var arena = enc_host_run(prog, owners_of(x, c, True))
    return Encoded(arena_words(arena, plan.encoded, n * total), arena_words(arena, plan.unknown, d))


def onehot_inverse(c: CatAuto, z: List[Float32], n: Int, grouped: Bool, groups: List[List[Int]],
                   drops: List[Int], strict: List[Float64]) raises -> Encoded:
    var widths = onehot_widths(onehot_sizes(c, grouped, groups), drops)
    var prog = EncProgram(False)
    var plan = onehot_inverse_stage(prog, OneHotInverse(n, 0, block_of(c, False), widths^, drops.copy(), grouped,
        groups.copy(), strict.copy()))
    var arena = enc_host_run(prog, owners_of(z, c, False))
    var bad = int_word(arena, plan.bad_count)
    var flags: List[Float32] = [Float32(bad), Float32(int_word(arena, plan.bad_rows) if bad > 0 else -1)]
    return Encoded(arena_words(arena, plan.values, n * c.d), flags^)


def unknown_columns(x: List[Float32], n: Int, c: CatAuto) raises -> Int:
    """`_given_categories`' check_unknown (and handle_unknown='error''s): how many columns hold a value outside their
    categories."""
    var prog = EncProgram(False)
    var staged = cat_codes_stage(prog, n, c.d, 0, block_of(c, True), False)
    var arena = enc_host_run(prog, owners_of(x, c, True))
    var count = 0
    for j in range(c.d):  # small-loop(oracle: the per-column unknown counts)
        if arena[staged.unknown + j] > 0:
            count += 1
    return count


def no_strict(d: Int, hard: Bool) -> List[Float64]:
    return List[Float64](length=d, fill=1.0 if hard else 0.0)


# ---------------------------------------------------------------- test_ordinal
comptime ENC_N = 257
comptime ENC_D = 5


def ordinal_unit(unit: Int) raises:
    var n = ENC_N
    var d = ENC_D
    var x = fixture_f32(PREP_ENC_DIR + "ord-x.f32", n * d)
    var c = fit_auto(x, n, d)
    if unit == 0:
        check_categories(c, "ord", "ordinal")
        var got = ordinal_transform(c, x, n, False, no_groups(d), False, 0.0, nan[DType.float64]())
        same_words(got.words, fixture_f64(PREP_ENC_DIR + "ord-t.f64", n * d), "ordinal transform")
        return
    var xh = fixture_f32(PREP_ENC_DIR + "ord-xh.f32", n * d)
    var got = ordinal_transform(c, xh, n, False, no_groups(d), unit == 1, -1.0, nan[DType.float64]())
    if unit == 1:
        same_words(got.words, fixture_f64(PREP_ENC_DIR + "ord-th.f64", n * d), "ordinal use_encoded_value -1")
        return
    # handle_unknown='error': the transform's unknown counts name column 0 (the 99)
    require(got.unknown[0] > 0, "ordinal: the unknown category of column 0 was not found")


def test_ordinal() raises:
    check_cases("ordinal", 3, ordinal_unit)


# ---------------------------------------------------------------- test_ordinal_missing_and_inverse
def ordinal_missing_unit(unit: Int) raises:
    var n = ENC_N
    var d = ENC_D
    var x = fixture_f32(PREP_ENC_DIR + "om-x.f32", n * d)
    var c = fit_auto(x, n, d)
    # the keyword sets: (handle_unknown use_encoded_value, unknown_value, encoded_missing_value)
    var uses: List[Bool] = [False, False, True, True]
    var unknowns: List[Float64] = [0.0, 0.0, -1.0, nan[DType.float64]()]
    var missings: List[Float64] = [nan[DType.float64](), -7.0, nan[DType.float64](), -2.0]
    if unit >= 4:
        # encoded_missing_value=1, and use_encoded_value with unknown_value=0: the fit refuses both
        var with_unknown = unit == 5
        var clash = ordinal_auto_clash(c, with_unknown, 0.0, 1.0 if not with_unknown else nan[DType.float64]())
        require(len(clash) > 0, "ordinal refusal " + String(unit - 4) + ": the clashing code was not refused")
        return
    var tag = "om" + String(unit)
    var label = "ordinal missing " + String(unit)
    var use = uses[unit]
    check_categories(c, tag, label)
    var got = ordinal_transform(c, x, n, False, no_groups(d), use, unknowns[unit], missings[unit])
    var z = fixture_f64(PREP_ENC_DIR + tag + "-t.f64", n * d)
    same_words(got.words, z, label + " transform")
    var back = ordinal_inverse(c, to_f32(z), n, False, no_groups(d), missings[unit], use, unknowns[unit])
    require(back.unknown[0] == 0, label + ": inverse_transform refused " + String(back.unknown[0]) + " rows")
    same_words(back.words, fixture_f64(PREP_ENC_DIR + tag + "-inv.f64", n * d), label + " inverse_transform")
    if not use:
        return
    var xh = fixture_f32(PREP_ENC_DIR + "om-xh.f32", n * d)
    var goth = ordinal_transform(c, xh, n, False, no_groups(d), use, unknowns[unit], missings[unit])
    var zh = fixture_f64(PREP_ENC_DIR + tag + "-th.f64", n * d)
    same_words(goth.words, zh, label + " transform of the unknown rows")
    var backh = ordinal_inverse(c, to_f32(zh), n, False, no_groups(d), missings[unit], use, unknowns[unit])
    require(backh.unknown[0] == 0, label + ": inverse_transform of the unknown rows refused rows")
    same_words(backh.words, fixture_f64(PREP_ENC_DIR + tag + "-invh.f64", n * d),
        label + " inverse_transform of the unknown rows")


def test_ordinal_missing_and_inverse() raises:
    check_cases("ordinal_missing_and_inverse", 6, ordinal_missing_unit)


# ---------------------------------------------------------------- test_onehot
def onehot_unit(unit: Int) raises:
    var n = ENC_N
    var d = ENC_D
    var x = fixture_f32(PREP_ENC_DIR + "oh-x.f32", n * d)
    var c = fit_auto(x, n, d)
    if unit == 3:
        # OneHotEncoder() (drop None, handle_unknown 'error'): row 4 of its own transform zeroed cannot be inverted
        var drops = onehot_auto_drops(c.sizes, OH_DROP_NONE)
        var got = onehot_transform(c, x, n, False, no_groups(d), False, drops)
        var cols = len(got.words) // n
        for k in range(cols):  # small-loop(fixture: the original's Z[4, :] = 0)
            got.words[4 * cols + k] = 0
        var back = onehot_inverse(c, got.words, n, False, no_groups(d), drops, no_strict(d, True))
        require(back.unknown[0] == 1 and back.unknown[1] == 4, "onehot: the all-zero row 4 was not refused ("
            + String(back.unknown[0]) + " rows, first " + String(back.unknown[1]) + ")")
        return
    # the keyword sets: (handle_unknown='ignore', drop None), (error, 'first'), (ignore, 'if_binary')
    var ignores: List[Bool] = [True, False, True]
    var words: List[Int] = [OH_DROP_NONE, OH_DROP_FIRST, OH_DROP_IF_BINARY]
    var tag = "oh" + String(unit)
    var label = "onehot " + String(unit)
    var ignore = ignores[unit]
    var drops = onehot_auto_drops(c.sizes, words[unit])
    var shape = fixture_i32(PREP_ENC_DIR + tag + "-shape.i32", 2)
    var cols = Int(shape[1])
    var got = onehot_transform(c, x, n, False, no_groups(d), False, drops)
    require(len(got.words) == n * cols, label + ": " + String(len(got.words) // n) + " columns, scikit-learn "
        + String(cols))
    var z = fixture_f64(PREP_ENC_DIR + tag + "-t.f64", n * cols)
    same_words(got.words, z, label + " transform")
    var strict = no_strict(d, not ignore)
    var back = onehot_inverse(c, to_f32(z), n, False, no_groups(d), drops, strict)
    require(back.unknown[0] == 0, label + ": inverse_transform refused " + String(back.unknown[0]) + " rows")
    same_words(back.words, fixture_f64(PREP_ENC_DIR + tag + "-inv.f64", n * d), label + " inverse_transform")
    if not ignore:
        return
    var xh = fixture_f32(PREP_ENC_DIR + "oh-xh.f32", n * d)
    var goth = onehot_transform(c, xh, n, False, no_groups(d), False, drops)
    var zh = fixture_f64(PREP_ENC_DIR + tag + "-th.f64", n * cols)
    same_words(goth.words, zh, label + " transform of the unknown 55")
    var backh = onehot_inverse(c, to_f32(zh), n, False, no_groups(d), drops, strict)
    require(backh.unknown[0] == 0, label + ": inverse_transform of the unknown rows refused rows")
    same_words(backh.words, fixture_f64(PREP_ENC_DIR + tag + "-invh.f64", n * d),
        label + " inverse_transform of the unknown 55")


def test_onehot() raises:
    check_cases("onehot", 4, onehot_unit)


# ---------------------------------------------------------------- test_given_categories
def given_unit(unit: Int) raises:
    var n = ENC_N
    var d = ENC_D
    var x = fixture_f32(PREP_ENC_DIR + "gc-x.f32", n * d)
    var sizes = fixture_i32(PREP_ENC_DIR + "gc-sizes.i32", d)
    var total = 0
    for j in range(d):  # small-loop(fixture: the given lists' lengths)
        total += Int(sizes[j])
    var c = given_cats(fixture_f32(PREP_ENC_DIR + "gc-cats.f32", total), sizes)
    if unit == 0:
        # OrdinalEncoder(categories=cats, handle_unknown='use_encoded_value', unknown_value=-1)
        var got = ordinal_transform(c, x, n, False, no_groups(d), True, -1.0, nan[DType.float64]())
        same_words(got.words, fixture_f64(PREP_ENC_DIR + "gc-ord-t.f64", n * d), "given ordinal transform")
        return
    if unit == 3:
        # the default handle_unknown='error': each list leaves out a training value, so the fit refuses
        require(unknown_columns(x, n, c) > 0, "given categories: the training values outside the lists were not found")
        return
    # OneHotEncoder(categories=cats, handle_unknown='ignore'), and with drop='first'
    var k = unit - 1
    var tag = "gc-oh" + String(k)
    var drops = onehot_auto_drops(c.sizes, OH_DROP_NONE if k == 0 else OH_DROP_FIRST)
    var shape = fixture_i32(PREP_ENC_DIR + tag + "-shape.i32", 2)
    var cols = Int(shape[1])
    var got = onehot_transform(c, x, n, False, no_groups(d), False, drops)
    require(len(got.words) == n * cols, "given onehot " + String(k) + ": " + String(len(got.words) // n)
        + " columns, scikit-learn " + String(cols))
    same_words(got.words, fixture_f64(PREP_ENC_DIR + tag + "-t.f64", n * cols), "given onehot " + String(k))


def test_given_categories() raises:
    check_cases("given_categories", 4, given_unit)


# ---------------------------------------------------------------- test_infrequent (test_x_prep_parity.py)
comptime INF_N = 400
comptime INF_D = 4


@fieldwise_init
struct Grouping(Movable):
    var groups: List[List[Int]]      # `_grouping`: per column the grouped code of each category (empty: none)
    var infrequent: List[List[Int]]  # `_infrequent`: per column the infrequent category indices (empty: None)


def fit_infrequent(c: CatAuto, x: List[Float32], n: Int, ignore_missing: Bool, threshold: Int,
                   max_categories: Int) raises -> Grouping:
    """`_fit_infrequent`: the groupable counts (OrdinalEncoder leaves a NaN category out), one program, the masks
    and grouped codes as the per-column lists."""
    var d = c.d
    var ks = List[Int]()
    var kmax = 0
    for j in range(d):  # small-loop(params: one size per column)
        var size = c.sizes[j]
        ks.append(size - 1 if (ignore_missing and c.missing[j] >= 0) else size)
        kmax = max(kmax, size)
    var prog = EncProgram(False)
    var plan = cat_infrequent_stage(prog, InfrequentFit(n, d, 0, block_of(c, True), False, CAT_NONE, ks.copy(), kmax,
        threshold, max_categories))
    var arena = enc_host_run(prog, owners_of(x, c, True))
    var groups = List[List[Int]]()
    var infrequent = List[List[Int]]()
    for j in range(d):  # small-loop(oracle: the device's tables as the fitted per-column lists)
        var inf = List[Int]()
        for i in range(ks[j]):  # small-loop(oracle: one column's categories)
            if int_word(arena, plan.mask + j * kmax + i) != 0:
                inf.append(i)
        var group = List[Int]()
        if len(inf) > 0:
            for i in range(ks[j]):  # small-loop(oracle: one column's categories)
                group.append(int_word(arena, plan.map + j * kmax + i))
        infrequent.append(inf^)
        groups.append(group^)
    return Grouping(groups^, infrequent^)


def inf_params(k: Int) -> Tuple[Int, Int]:
    # (threshold, max_categories) of min_frequency=30; max_categories=3; min_frequency=0.05 with max_categories=4;
    # max_categories=1 (`_infrequent_threshold`: a fraction is ceil(n * f) in binary64)
    if k == 0:
        return (30, -1)
    if k == 1:
        return (0, 3)
    if k == 2:
        return (Int(ceil(Float64(INF_N) * 0.05)), 4)
    return (0, 1)


def infrequent_ordinal_unit(k: Int) raises:
    var n = INF_N
    var d = INF_D
    var label = "infrequent ordinal " + String(k)
    var x = fixture_f32(PREP_ENC_DIR + "inf-x.f32", n * d)
    var c = fit_auto(x, n, d)
    var params = inf_params(k)
    var g = fit_infrequent(c, x, n, True, params[0], params[1])
    var counts = fixture_i32(PREP_ENC_DIR + "inf" + String(k) + "-ord-infcount.i32", d)
    var total = 0
    for j in range(d):  # small-loop(oracle: the reference's infrequent counts)
        var want = Int(counts[j])
        var have = len(g.infrequent[j]) if len(g.infrequent[j]) > 0 else -1
        require(have == want, label + ": infrequent_categories_[" + String(j) + "] holds " + String(have)
            + " values, scikit-learn " + String(want) + " (-1: None)")
        total += max(want, 0)
    if total > 0:
        var vals = fixture_f64(PREP_ENC_DIR + "inf" + String(k) + "-ord-infvals.f64", total)
        var at = 0
        for j in range(d):  # small-loop(oracle: the columns)
            for i in range(len(g.infrequent[j])):  # small-loop(oracle: one column's infrequent categories)
                same_word(c.block[j * c.kmax + g.infrequent[j][i]], vals[at + i], label + ": infrequent_categories_["
                    + String(j) + "][" + String(i) + "]")
            at += len(g.infrequent[j])
    var xh = fixture_f32(PREP_ENC_DIR + "inf-xh.f32", n * d)
    var got = ordinal_transform(c, xh, n, True, g.groups, True, -1.0, nan[DType.float64]())
    same_words(got.words, fixture_f64(PREP_ENC_DIR + "inf" + String(k) + "-ord-th.f64", n * d), label + " transform")


def infrequent_onehot_unit(unit: Int) raises:
    # units 0..26: min_frequency / max_categories keyword sets 0..2 x handle_unknown x drop; 27..29: set 3
    # (max_categories=1) x handle_unknown, drop None only, as the original skips the drops there
    var k = unit // 9 if unit < 27 else 3
    var h = (unit % 9) // 3 if unit < 27 else unit - 27
    var dr = unit % 3 if unit < 27 else 0
    var n = INF_N
    var d = INF_D
    var tag = "inf" + String(k) + "-oh" + String(h) + String(dr)
    var label = "infrequent onehot " + tag
    var x = fixture_f32(PREP_ENC_DIR + "inf-x.f32", n * d)
    var c = fit_auto(x, n, d)
    var params = inf_params(k)
    var g = fit_infrequent(c, x, n, False, params[0], params[1])
    var words: List[Int] = [OH_DROP_NONE, OH_DROP_FIRST, OH_DROP_IF_BINARY]
    # handle_unknown: 0 infrequent_if_exist, 1 ignore, 2 warn (`_unknown_to`: an unknown takes the infrequent code
    # under infrequent_if_exist and warn)
    var ignore = h == 1
    var drops = onehot_auto_drops(cat_grouped_sizes(c.sizes, g.groups), words[dr])
    var strict = List[Float64]()
    for j in range(d):  # small-loop(params: `_onehot_inverse`'s per-column strictness)
        strict.append(1.0 if (not ignore and len(g.infrequent[j]) > 0) else 0.0)
    var shape = fixture_i32(PREP_ENC_DIR + tag + "-shape.i32", 2)
    var cols = Int(shape[1])
    var xh = fixture_f32(PREP_ENC_DIR + "inf-xh.f32", n * d)
    var got = onehot_transform(c, xh, n, True, g.groups, not ignore, drops)
    require(len(got.words) == n * cols, label + ": " + String(len(got.words) // n) + " columns, scikit-learn "
        + String(cols))
    same_words(got.words, fixture_f64(PREP_ENC_DIR + tag + "-th.f64", n * cols), label + " transform")
    var own = onehot_transform(c, x, n, True, g.groups, not ignore, drops)
    var back = onehot_inverse(c, own.words, n, True, g.groups, drops, strict)
    require(back.unknown[0] == 0, label + ": inverse_transform refused " + String(back.unknown[0]) + " rows")
    same_words(back.words, fixture_f64(PREP_ENC_DIR + tag + "-inv.f64", n * d), label + " inverse_transform")


def test_infrequent() raises:
    check_cases("infrequent_ordinal", 4, infrequent_ordinal_unit)
    check_cases("infrequent_onehot", 30, infrequent_onehot_unit)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
