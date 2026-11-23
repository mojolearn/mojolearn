# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""TargetEncoder programs against scikit-learn's values (lane gbdt-cpu-rest-enc, 2026-11-23; ported from
python/mojolearn/tests/test_x_prep_target_encoder.py `test_fit_transform_like_reference`,
`test_cross_fit_uses_fold_encodings`, `test_stratified_folds_unshuffled_are_the_reference`, and
test_x_prep_parity.py `test_target_encoder_cv_categories`).

The inputs are the original tests' NumPy draws and the references the scikit-learn 1.9.1 calls they made, stored as
words in `fixtures/prep_te/` by `tools/py2mojo_native_fixtures_b2_enc.py` (class targets as their codes). The
programs are the typed builders the encode door builds through (bindings/prep_encode_cats_run.mojo `cat_fit_stage`;
bindings/prep_encode_target_run.mojo `target_fit_stage`, `te_folds_stage`, `target_transform_stage`) with the
requests bindings/prep_encode_native_door.mojo `_target_run` and `_target_transform` make on the identical host
image: the categories from the sorted fit with unique_cols (CAT_FIT_UNIQUE; the C08 codes route is opt-in), the codes
in host words (no target scratch), te_enc grouped the host column's way (`host_column`), smooth 'auto' as -1.0, the
seed of `random_state` as its two words. They run on the host column by `prep_enc_host.enc_host_run`.

    test_fit_transform_like_reference   fit, then transform of rows with an unseen 9: continuous (smooth 'auto'
                                        and 2.0), binary, multiclass, rtol 2e-4, atol 2e-5
    test_cross_fit_uses_fold_encodings  the KFold fold stage (seed 7, shuffled) gives the stored folds exactly,
                                        and fit_transform equals scikit-learn fitted on each fold's training rows
                                        applied to its test rows (rtol 2e-4, atol 2e-5)
    test_stratified_folds_unshuffled_are_the_reference
                                        the stratified fold stage unshuffled equals StratifiedKFold(5) (binary and
                                        a reversed multiclass target); fit_transform(shuffle=False) equals
                                        scikit-learn's for both and for the continuous target; shuffled (seed 11)
                                        every class spreads over the folds within one row
    test_target_encoder_cv_categories   categories=<list> with an unseen 9 and user folds (KFold(4),
                                        StratifiedKFold(3), KFold(5) pairs) as the row -> fold words, smooth 2.0:
                                        fit_transform and encodings_ rtol 1e-5, atol 1e-6

Stays in Python (the public boundary): target type detection over the Python target and the label encoding of its
classes, the cv argument (a scikit-learn splitter refused by name, a mojolearn splitter's folds, the (train, test)
pairs' cover and training-row checks), the given category lists' validation, and the parameter refusals. Every case
is an independent unit; the case tables run across host cores.
"""
from bindings.prep_encode_prog_run import EncProgram
from bindings.prep_encode_cats_run import (
    CatAuto, CatBlock, CatFit, CAT_NONE, CAT_FIT_UNIQUE, cat_fit_stage, cat_auto_held, cat_auto_state, cat_i2f_stage,
)
from bindings.prep_encode_target_run import (
    TargetFit, TargetApply, target_fit_stage, target_transform_stage, te_folds_stage, TE_NONE, TE_FOLDS_NONE,
    TE_FOLDS_STAGED, TE_FOLDS_GIVEN,
)
from checks.native_tests.prep_enc_host import enc_host_run
from checks.native_tests.prep_cls_host import arena_words, check_close, codes_words
from checks.native_tests.py2mojo_fixture_words import fixture_f64
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require, fixture_f32, fixture_i32

comptime PREP_TE_DIR = "checks/native_tests/fixtures/prep_te/"
comptime TE_N = 400
comptime TE_D = 3


# ---------------------------------------------------------------- categories
def size_words(sizes: List[Int]) -> List[Float32]:
    var words = List[Float32](length=max(len(sizes), 1), fill=0)
    for j in range(len(sizes)):  # small-loop(params: the per-column category counts)
        words[j] = Float32(sizes[j])
    return words^


def block_of(c: CatAuto) -> CatBlock:
    # The category record as owners 1 (block) and 2 (size words) of the program (`_cat_block`).
    return CatBlock(1, len(c.block), c.kmax, 2, max(len(c.sizes), 1))


def fit_auto(x: List[Float32], n: Int, d: Int) raises -> CatAuto:
    """`_fit_categories_caps` on the identical image: the sorted category fit with unique_cols."""
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


def given_cats(lists: List[List[Float32]]) -> CatAuto:
    # A given numeric category list per column (sorted, no NaN): the record `_given_categories` builds.
    var c = CatAuto()
    c.d = len(lists)
    for j in range(c.d):  # small-loop(params: the columns)
        c.kmax = max(c.kmax, len(lists[j]))
    c.block = List[Float32](length=max(c.d * c.kmax, 1), fill=Float32(0))
    for j in range(c.d):  # small-loop(params: the columns)
        for i in range(len(lists[j])):  # small-loop(params: one column's categories)
            c.block[j * c.kmax + i] = lists[j][i]
        c.sizes.append(len(lists[j]))
        c.missing.append(-1)
    return c^


# ---------------------------------------------------------------- the target and the programs
@fieldwise_init
struct Target(Movable):
    """`TargetInfo`: the target words (float32 y, or the int32 class codes as words), whether it is continuous,
    the class count and T (1, or the class count of a multiclass target)."""
    var words: List[Float32]
    var continuous: Bool
    var n_classes: Int
    var outputs: Int


def continuous_target(y: List[Float32]) -> Target:
    return Target(y.copy(), True, 0, 1)


def class_target(codes: List[Int32], n_classes: Int) -> Target:
    # `_class_target`: binary (T 1) for at most two classes, else multiclass (T = the class count)
    return Target(codes_words(codes), False, n_classes, 1 if n_classes <= 2 else n_classes)


def class_count(codes: List[Int32]) -> Int:
    var top = 0
    for i in range(len(codes)):  # small-loop(fixture: the codes' largest value)
        top = max(top, Int(codes[i]) + 1)
    return top


@fieldwise_init
struct TargetModel(Movable):
    var cats: CatAuto
    var outputs: Int
    var cmax: Int
    var enc: List[Float32]      # `_enc`: the (d, cmax, T) encodings of the full fit
    var meta: List[Float32]     # `_meta`: the full fit's (T, 2) target statistics
    var applied: List[Float32]  # fit_transform's (n, d * T) block (empty without apply)
    var refused: Bool           # the stratified folds' too-few-members flag


def target_fit(x: List[Float32], n: Int, c: CatAuto, t: Target, folds_kind: Int, n_folds: Int, folds: List[Int32],
               seed: Int, shuffle: Bool, smooth: Float64, apply: Bool) raises -> TargetModel:
    """`_target_run`'s request on the host column: owners 0 X, 1 the category block, 2 its sizes, 3 the target,
    4 the given folds."""
    var d = c.d
    var T = t.outputs
    var F = n_folds
    var cmax = 0
    for j in range(d):  # small-loop(params: the per-column category counts)
        cmax = max(cmax, c.sizes[j])
    var given = folds_kind == TE_FOLDS_GIVEN
    var prog = EncProgram(False)
    var request = TargetFit(n, d, T, 0, block_of(c), False, False, CAT_NONE, t.continuous, 3, len(t.words),
        t.n_classes, folds_kind, F, 4 if given else CAT_NONE, len(folds) if given else 0, seed & 0xFFFFFFFF,
        (seed >> 32) & 0xFFFFFFFF, shuffle, c.sizes.copy(), cmax, smooth, True, apply)
    var plan = target_fit_stage(prog, request, False)
    var owners = List[List[Float32]]()
    owners.append(x.copy())
    owners.append(c.block.copy())
    owners.append(size_words(c.sizes))
    owners.append(t.words.copy())
    if given:
        owners.append(codes_words(folds))
    var arena = enc_host_run(prog, owners)
    var refused = plan.flag >= 0 and arena[plan.flag] != 0
    var base = F * d * cmax * T
    var enc = arena_words(arena, plan.enc + base, d * cmax * T)
    var meta = arena_words(arena, plan.meta + 2 * F * T, 2 * T)
    var applied = arena_words(arena, plan.applied, n * d * T) if apply else List[Float32]()
    return TargetModel(c.copy(), T, cmax, enc^, meta^, applied^, refused)


def target_transform(m: TargetModel, x: List[Float32], n: Int) raises -> List[Float32]:
    """`_target_transform`: owners 0 X, 1 the block, 2 its sizes, 3 `_enc`, 4 `_meta`."""
    var d = m.cats.d
    var T = m.outputs
    var prog = EncProgram(False)
    var dst = target_transform_stage(prog, TargetApply(n, d, T, m.cmax, 0, block_of(m.cats), False, 3, len(m.enc),
        4, len(m.meta)), False)
    var owners = List[List[Float32]]()
    owners.append(x.copy())
    owners.append(m.cats.block.copy())
    owners.append(size_words(m.cats.sizes))
    owners.append(m.enc.copy())
    owners.append(m.meta.copy())
    var arena = enc_host_run(prog, owners)
    return arena_words(arena, dst, n * d * T)


def native_folds(n: Int, n_folds: Int, seed: Int, shuffle: Bool, codes: List[Int32],
                 n_classes: Int) raises -> List[Int]:
    """`_native_folds`: the fold stage `_stage_folds` gives the fit (KFold without codes, else StratifiedKFold over
    them), one program; refused when every class has fewer rows than the folds."""
    var prog = EncProgram(False)
    var owners = List[List[Float32]]()
    var co = TE_NONE
    if len(codes) > 0:
        owners.append(codes_words(codes))
        co = cat_i2f_stage(prog, n, 0)
    var staged = te_folds_stage(prog, n, n_folds, seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF, shuffle, co, n_classes)
    var arena = enc_host_run(prog, owners)
    if staged[1] >= 0:
        require(arena[staged[1]] == 0, "folds: n_splits=" + String(n_folds) + " is more than every class's members")
    var folds = List[Int](length=n, fill=0)
    for i in range(n):  # small-loop(oracle: the fold words as ints)
        folds[i] = Int(arena[staged[0] + i])
    return folds^


def same_folds(got: List[Int], want: List[Int32], label: String) raises:
    require(len(got) == len(want), label + ": " + String(len(got)) + " rows, reference " + String(len(want)))
    for i in range(len(want)):  # small-loop(oracle: the row -> fold table)
        require(got[i] == Int(want[i]), label + ": row " + String(i) + " in fold " + String(got[i]) + " expected "
            + String(want[i]))


# ---------------------------------------------------------------- test_fit_transform_like_reference
def like_reference_unit(unit: Int) raises:
    var n = TE_N
    var d = TE_D
    var label = "target like_reference " + String(unit)
    var x = fixture_f32(PREP_TE_DIR + "te-x0.f32", n * d)
    var c = fit_auto(x, n, d)
    # (y, smooth 'auto'), (y, smooth 2.0), (yb, 'auto'), (ym, 'auto')
    var t = continuous_target(fixture_f32(PREP_TE_DIR + "te-y0.f32", n))
    if unit == 2:
        var codes = fixture_i32(PREP_TE_DIR + "te-yb0.i32", n)
        t = class_target(codes, class_count(codes))
    elif unit == 3:
        var codes = fixture_i32(PREP_TE_DIR + "te-ym0.i32", n)
        t = class_target(codes, class_count(codes))
    var smooth = 2.0 if unit == 1 else -1.0
    var T = t.outputs
    var m = target_fit(x, n, c, t, TE_FOLDS_NONE, 0, List[Int32](), 0, False, smooth, False)
    var xh = fixture_f32(PREP_TE_DIR + "te-xh.f32", n * d)
    var got = target_transform(m, xh, n)
    check_close(got, fixture_f64(PREP_TE_DIR + "te-ref" + String(unit) + ".f64", n * d * T), 2e-4, 2e-5,
        label + " transform")


def test_fit_transform_like_reference() raises:
    check_cases("target_like_reference", 4, like_reference_unit)


# ---------------------------------------------------------------- test_cross_fit_uses_fold_encodings
def cross_fit_unit(unit: Int) raises:
    var n = TE_N
    var d = TE_D
    var want = fixture_i32(PREP_TE_DIR + "te-folds3.i32", n)
    if unit == 0:
        same_folds(native_folds(n, 5, 7, True, List[Int32](), 0), want, "target KFold folds, seed 7")
        return
    var x = fixture_f32(PREP_TE_DIR + "te-x3.f32", n * d)
    var c = fit_auto(x, n, d)
    # TargetEncoder(random_state=7).fit_transform: cv 5, shuffled, smooth 'auto', the folds staged in the fit
    var m = target_fit(x, n, c, continuous_target(fixture_f32(PREP_TE_DIR + "te-y3.f32", n)), TE_FOLDS_STAGED, 5,
        List[Int32](), 7, True, -1.0, True)
    require(not m.refused, "target cross fit: the KFold folds were refused")
    check_close(m.applied, fixture_f64(PREP_TE_DIR + "te-cross3.f64", n * d), 2e-4, 2e-5,
        "target cross fit against the fold encodings")


def test_cross_fit_uses_fold_encodings() raises:
    check_cases("target_cross_fit", 2, cross_fit_unit)


# ---------------------------------------------------------------- test_stratified_folds_unshuffled_are_the_reference
def stratified_unit(unit: Int) raises:
    var n = TE_N
    var d = TE_D
    var tags: List[String] = ["b", "m"]
    if unit < 2:
        # the stratified fold stage unshuffled against StratifiedKFold(5)
        var codes = fixture_i32(PREP_TE_DIR + "te-y" + tags[unit] + "4.i32", n)
        same_folds(native_folds(n, 5, 0, False, codes, class_count(codes)),
            fixture_i32(PREP_TE_DIR + "te-strat-" + tags[unit] + ".i32", n), "target stratified folds " + tags[unit])
        return
    if unit == 5:
        # shuffled (seed 11): every class spread over the five folds as evenly as the reference's allocation
        var codes = fixture_i32(PREP_TE_DIR + "te-ym4.i32", n)
        var K = class_count(codes)
        var folds = native_folds(n, 5, 11, True, codes, K)
        for cl in range(K):  # small-loop(oracle: the classes)
            var cnt = List[Int](length=5, fill=0)
            for i in range(n):  # small-loop(oracle: the rows)
                if Int(codes[i]) == cl:
                    cnt[folds[i]] += 1
            var lo = cnt[0]
            var hi = cnt[0]
            for f in range(5):  # small-loop(oracle: the folds)
                lo = min(lo, cnt[f])
                hi = max(hi, cnt[f])
            require(hi - lo <= 1, "target shuffled stratified folds: class " + String(cl) + " spreads " + String(lo)
                + " to " + String(hi) + " rows over the folds")
        return
    # fit_transform(shuffle=False): binary, multiclass, continuous
    var x = fixture_f32(PREP_TE_DIR + "te-x4.f32", n * d)
    var c = fit_auto(x, n, d)
    var t = continuous_target(fixture_f32(PREP_TE_DIR + "te-y4.f32", n))
    var tag = String("c")
    if unit < 4:
        tag = tags[unit - 2]
        var codes = fixture_i32(PREP_TE_DIR + "te-y" + tag + "4.i32", n)
        t = class_target(codes, class_count(codes))
    var T = t.outputs
    var m = target_fit(x, n, c, t, TE_FOLDS_STAGED, 5, List[Int32](), 0, False, -1.0, True)
    require(not m.refused, "target stratified fit_transform " + tag + ": the folds were refused")
    check_close(m.applied, fixture_f64(PREP_TE_DIR + "te-ft" + tag + ".f64", n * d * T), 2e-4, 2e-5,
        "target fit_transform(shuffle=False) " + tag)


def test_stratified_folds_unshuffled_are_the_reference() raises:
    check_cases("target_stratified", 6, stratified_unit)


# ---------------------------------------------------------------- test_target_encoder_cv_categories (parity)
def cv_categories_unit(unit: Int) raises:
    var n = 200
    var d = 2
    var label = "target cv categories " + String(unit)
    var x = fixture_f32(PREP_TE_DIR + "cv-x.f32", n * d)
    var lists = List[List[Float32]]()
    var first: List[Float32] = [0.0, 1.0, 2.0, 3.0, 4.0, 9.0]
    var second: List[Float32] = [0.0, 1.0, 2.0, 3.0, 4.0]
    lists.append(first^)
    lists.append(second^)
    var c = given_cats(lists)
    # (y, KFold(4)), (yb, StratifiedKFold(3)), (y, KFold(5) pairs): the user folds as each row's test fold
    var t = continuous_target(fixture_f32(PREP_TE_DIR + "cv-y.f32", n))
    if unit == 1:
        var codes = fixture_i32(PREP_TE_DIR + "cv-yb.i32", n)
        t = class_target(codes, class_count(codes))
    var tag = "cv" + String(unit)
    var F = Int(fixture_i32(PREP_TE_DIR + tag + "-nfolds.i32", 1)[0])
    var folds = fixture_i32(PREP_TE_DIR + tag + "-folds.i32", n)
    var m = target_fit(x, n, c, t, TE_FOLDS_GIVEN, F, folds, 0, False, 2.0, True)
    check_close(m.applied, fixture_f64(PREP_TE_DIR + tag + "-ft.f64", n * d), 1e-5, 1e-6, label + " fit_transform")
    # encodings_[j]: the first sizes[j] (category, T) words of column j's block
    var total = 0
    for j in range(d):  # small-loop(oracle: the per-column category counts)
        total += m.cats.sizes[j]
    var want = fixture_f64(PREP_TE_DIR + tag + "-enc.f64", total)
    var got = List[Float32]()
    for j in range(d):  # small-loop(oracle: the columns)
        for i in range(m.cats.sizes[j]):  # small-loop(oracle: one column's encodings)
            got.append(m.enc[j * m.cmax + i])
    check_close(got, want, 1e-5, 1e-6, label + " encodings_")


def test_target_encoder_cv_categories() raises:
    check_cases("target_cv_categories", 3, cv_categories_unit)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
