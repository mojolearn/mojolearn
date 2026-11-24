# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""Lane mf-enc6: the OrdinalEncoder / OneHotEncoder member forms this lane added (core/prep_member.mojo kinds 10,
11 over the encode door's typed bodies: bindings/prep_encode_cats_run.mojo `cat_given_column`, `cat_given_state`,
`cat_infrequent_*`, bindings/prep_encode_ordinal_run.mojo `ordinal_cards_clash`, bindings/prep_encode_onehot_run.mojo
`onehot_drop_index`), driven on the host column through `HostMember` (checks/native_tests/py2mojo_host_members.mojo)
with the parameter words providers/python/member_prep_boundary_door.mojo `_ordinal_admit` / `_onehot_admit` write.
Every expected value is scikit-learn's (worked by hand from sklearn/preprocessing/_encoders.py, 1.9: the given
lists, `_identify_infrequent`, `_map_infrequent_categories`, `_set_drop_idx`, `_compute_n_features_outs`); every
refusal is the encode door's words.

 0. given categories (OrdinalEncoder): categories=[[1, 2, 3, 5]] fits X = [1, 3, 2]: categories_ block [1, 2, 3, 5],
    transform [5, 1] -> [3, 0]; under 'error' a training 4 is refused ("found unknown categories in column(s) [0]
    during fit"); [[3, 1]] is refused ("unsorted categories"); two lists for one column are refused ("categories has
    2 lists; X has 1 features"); use_encoded_value / unknown_value=-1 fits [1, 4] and encodes 4 as -1.
 1. infrequent grouping (OrdinalEncoder): X = [0, 0, 0, 1, 1, 2, 3]; min_frequency=2 -> transform [0, 0, 0, 1, 1, 2,
    2], `_grouping` words [4, 0, 1, 2, 2], `_infrequent` words [2, 2, 3]; max_categories=2 -> [0, 0, 0, 1, 1, 1, 1].
 2. per-column drop with the grouping (OneHotEncoder, min_frequency=2): drop=[1.0] -> `_drop_after` [1],
    drop_idx_ [1], transform 7 x 2 [1 0, 1 0, 1 0, 0 0, 0 0, 0 1, 0 1]; drop=[2.0] is refused ("Unable to drop
    category 2.0 from feature 0 because it is infrequent"); drop=[7.0] is refused ("Category: 7.0, Feature: 0").
 3. handle_unknown='warn' (OneHotEncoder): fit [0, 1], transform [5] -> [0, 0] with the notice words [13, 0, 0];
    'ignore' with drop='first' fit [0, 1, 2], transform [9] -> [0, 0] with [13, 0, 1]; 'ignore' without a drop
    keeps no notice; 'error' refuses ("found unknown categories in column(s) [0] during transform").
 4. the FAST image's packed category routes: OrdinalEncoder on CAT_FIT_PACK_SORT (2) and CAT_FIT_PACK_PRESENT (3)
    gives the CAT_FIT_UNIQUE (0) state and codes ([5, 1, 5, 3, 1] -> [2, 0, 2, 1, 0]) on every image with ops
    157-161 (x_prep/cat_cls2.mojo CAT_PACK_OPS: FAST + Apple and every IDENTICAL image, the host column included);
    an image without them refuses 2 and 3 at MEMBER_OP_NEW; route word 4 is refused at MEMBER_OP_NEW ("category
    fit route").

FAIL-PROOF EDITS (each must make the named case fail): in core/prep_member.mojo `_given_state` append `lists[j]`
instead of `cat_given_column(lists[j], j, who)` (case 0: [[3, 1]] is not refused); in `_fit_encoder` drop the
`_infrequent_fit` call (case 1: transform [0, 0, 0, 1, 1, 2, 3]); in bindings/prep_encode_onehot_run.mojo
`onehot_drop_index` drop the infrequent-category loop (case 2: drop=[2.0] is not refused); in core/prep_member.mojo
`_transform_onehot` drop the `elif handle == PREP_ONEHOT_WARN ...` branch (case 3: no notice words); in
`_prep_member` MEMBER_OP_NEW accept `source[1] > 3.0` (case 4: route 4 mints); in x_prep/units.mojo guard op 161
by `CAT_CLS2_PACK` alone (case 4 on the host column: route 2 categories_ [0, 0, 0]).
"""
from std.math import nan
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require
from checks.native_tests.py2mojo_host_members import HostMember, MEMBER_PREP
from x_prep.cat_cls2 import CAT_PACK_OPS
from core.member_abi import MemberCall, MEMBER_OP_EXPORT, MEMBER_BLOCK_NOTICES, member_call

comptime CASES = 5


# ---------------------------------------------------------------- the admit's words
def ordinal_words(route: Float64, use: Bool, unknown: Float64, grouping: Bool, max_categories: Float64,
                  mf_kind: Float64, mf_value: Float64, lists: List[List[Float64]], given: Bool) -> List[Float64]:
    """`_ordinal_admit`'s words (header 15; the shared words at 9): no C08 route, no R3 region,
    encoded_missing_value NaN (scikit-learn's default)."""
    var words = List[Float64](length=15, fill=0.0)
    words[0] = 10.0
    words[1] = route
    words[4] = 1.0 if use else 0.0
    words[5] = unknown if use else 0.0
    words[6] = 1.0
    words[7] = nan[DType.float64]()
    words[9] = 1.0 if grouping else 0.0
    words[10] = max_categories
    words[11] = mf_kind
    words[12] = mf_value
    words[13] = Float64(len(lists)) if given else -1.0
    for j in range(len(lists)):  # small-loop(params: the given category lists)
        words.append(Float64(len(lists[j])))
        for i in range(len(lists[j])):  # small-loop(params: one given list)
            words.append(lists[j][i])
    return words^


def onehot_words(handle: Float64, drop: Float64, grouping: Bool, mf_value: Float64, drops: List[Float64],
                 texts: List[String]) -> List[Float64]:
    """`_onehot_admit`'s words (header 12; the shared words at 6): categories 'auto', min_frequency a count when
    `grouping`, a drop entry per value of `drops` (str() codepoints `texts`)."""
    var words = List[Float64](length=12, fill=0.0)
    words[0] = 11.0
    words[4] = handle
    words[5] = drop
    words[6] = 1.0 if grouping else 0.0
    words[7] = -1.0
    words[8] = 1.0 if grouping else 0.0
    words[9] = mf_value
    words[10] = -1.0
    words[11] = Float64(len(drops))
    for j in range(len(drops)):  # small-loop(params: the given drop entries)
        words.append(0.0)
        words.append(drops[j])
        var points = List[Float64]()
        for cp in texts[j].codepoints():  # small-loop(params: one value's str() text)
            points.append(Float64(Int(cp)))
        words.append(Float64(len(points)))
        for c in range(len(points)):  # small-loop(params: one value's str() text)
            words.append(points[c])
    return words^


def column(values: List[Float64]) -> List[Float32]:
    var x = List[Float32](length=len(values), fill=0)
    for i in range(len(values)):  # small-loop(fixture: one column of X)
        x[i] = Float32(values[i])
    return x^


def same(got: List[Float32], want: List[Float64], label: String) raises:
    require(len(got) == len(want), label + ": " + String(len(got)) + " words, expected " + String(len(want)))
    for i in range(len(want)):  # small-loop(oracle: one block)
        require(Float64(got[i]) == want[i], label + " word " + String(i) + ": " + String(got[i]) + " expected "
            + String(want[i]))


def notices(member: HostMember) raises -> List[Float64]:
    """The state's MEMBER_BLOCK_NOTICES words (by name)."""
    var call = MemberCall(member.route, member.state)
    call.block = MEMBER_BLOCK_NOTICES
    call.flags = 0
    _ = member_call(member.member, MEMBER_OP_EXPORT, call)
    var count = call.dst.rows * call.dst.cols
    var words = List[Float64](length=count, fill=0.0)
    if count > 0:
        var source = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call.dst.data)
        for i in range(count):  # small-loop(fixture: a few notice triples)
            words[i] = source[i]
    return words^


def refused_fit(var words: List[Float64], x: List[Float32], words_wanted: String, label: String) raises:
    var member = HostMember(MEMBER_PREP, words^)
    var raised = String("")
    try:
        member.fit(x, len(x), 1)
    except problem:
        raised = String(problem)
    member.free()
    require(raised.find(words_wanted) >= 0, label + ": expected the refusal '" + words_wanted + "', got '" + raised
        + "'")


def lists_of(a: List[Float64]) -> List[List[Float64]]:
    var lists = List[List[Float64]]()
    lists.append(a.copy())
    return lists^


# ---------------------------------------------------------------- the cases
def check_given() raises:
    var cats: List[Float64] = [1.0, 2.0, 3.0, 5.0]
    var member = HostMember(MEMBER_PREP, ordinal_words(0.0, False, 0.0, False, -1.0, 0.0, 0.0, lists_of(cats), True))
    try:
        var x: List[Float64] = [1.0, 3.0, 2.0]
        member.fit(column(x), 3, 1)
        same(member.block(1), cats, "given categories_ block")
        var probe: List[Float64] = [5.0, 1.0]
        var want: List[Float64] = [3.0, 0.0]
        same(member.transform(column(probe), 2), want, "given transform")
    finally:
        member.free()
    var bad: List[Float64] = [1.0, 4.0]
    refused_fit(ordinal_words(0.0, False, 0.0, False, -1.0, 0.0, 0.0, lists_of(cats), True), column(bad),
                "OrdinalEncoder found unknown categories in column(s) [0] during fit", "given unknown")
    var unsorted: List[Float64] = [3.0, 1.0]
    refused_fit(ordinal_words(0.0, False, 0.0, False, -1.0, 0.0, 0.0, lists_of(unsorted), True), column(bad),
                "unsorted categories are not supported for numerical categories", "given unsorted")
    var two = lists_of(cats)
    two.append(cats.copy())
    refused_fit(ordinal_words(0.0, False, 0.0, False, -1.0, 0.0, 0.0, two, True), column(bad),
                "categories has 2 lists; X has 1 features", "given count")
    var encoded = HostMember(MEMBER_PREP, ordinal_words(0.0, True, -1.0, False, -1.0, 0.0, 0.0, lists_of(cats), True))
    try:
        encoded.fit(column(bad), 2, 1)
        var four: List[Float64] = [4.0]
        var minus: List[Float64] = [-1.0]
        same(encoded.transform(column(four), 1), minus, "given use_encoded_value")
    finally:
        encoded.free()


def check_infrequent() raises:
    var x: List[Float64] = [0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0]
    var none = List[List[Float64]]()
    var member = HostMember(MEMBER_PREP, ordinal_words(0.0, False, 0.0, True, -1.0, 1.0, 2.0, none.copy(), False))
    try:
        member.fit(column(x), 7, 1)
        var want: List[Float64] = [0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0]
        same(member.transform(column(x), 7), want, "min_frequency=2 transform")
        var groups: List[Float64] = [4.0, 0.0, 1.0, 2.0, 2.0]
        same(member.block(6), groups, "min_frequency=2 _grouping")
        var infrequent: List[Float64] = [2.0, 2.0, 3.0]
        same(member.block(7), infrequent, "min_frequency=2 _infrequent")
    finally:
        member.free()
    var capped = HostMember(MEMBER_PREP, ordinal_words(0.0, False, 0.0, True, 2.0, 0.0, 0.0, none.copy(), False))
    try:
        capped.fit(column(x), 7, 1)
        var want: List[Float64] = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        same(capped.transform(column(x), 7), want, "max_categories=2 transform")
    finally:
        capped.free()


def check_drop() raises:
    var x: List[Float64] = [0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0]
    var one: List[Float64] = [1.0]
    var one_text: List[String] = [String("1.0")]
    var member = HostMember(MEMBER_PREP, onehot_words(0.0, 3.0, True, 2.0, one, one_text))
    try:
        member.fit(column(x), 7, 1)
        var after: List[Float64] = [1.0]
        same(member.block(4), after, "drop=[1.0] _drop_after")
        same(member.block(8), after, "drop=[1.0] drop_idx_")
        var want: List[Float64] = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0]
        same(member.transform(column(x), 7), want, "drop=[1.0] transform")
    finally:
        member.free()
    var two: List[Float64] = [2.0]
    var two_text: List[String] = [String("2.0")]
    refused_fit(onehot_words(0.0, 3.0, True, 2.0, two, two_text), column(x),
                "Unable to drop category 2.0 from feature 0 because it is infrequent", "drop infrequent")
    var seven: List[Float64] = [7.0]
    var seven_text: List[String] = [String("7.0")]
    refused_fit(onehot_words(0.0, 3.0, True, 2.0, seven, seven_text), column(x),
                "were not found in the training data.\nCategory: 7.0, Feature: 0", "drop not found")


def check_warn() raises:
    var empty = List[Float64]()
    var no_text = List[String]()
    var fit2: List[Float64] = [0.0, 1.0]
    var fit3: List[Float64] = [0.0, 1.0, 2.0]
    var zeros: List[Float64] = [0.0, 0.0]
    var five: List[Float64] = [5.0]
    var warn = HostMember(MEMBER_PREP, onehot_words(3.0, 0.0, False, 0.0, empty, no_text))
    try:
        warn.fit(column(fit2), 2, 1)
        same(warn.transform(column(five), 1), zeros, "warn transform")
        var want: List[Float64] = [13.0, 0.0, 0.0]
        var got = notices(warn)
        require(len(got) == 3 and got[0] == want[0] and got[1] == want[1] and got[2] == want[2],
            "warn notice words: " + String(len(got)) + " words")
    finally:
        warn.free()
    var ignore_drop = HostMember(MEMBER_PREP, onehot_words(1.0, 1.0, False, 0.0, empty, no_text))
    try:
        ignore_drop.fit(column(fit3), 3, 1)
        same(ignore_drop.transform(column(five), 1), zeros, "ignore + first transform")
        var got = notices(ignore_drop)
        require(len(got) == 3 and got[0] == 13.0 and got[1] == 0.0 and got[2] == 1.0,
            "ignore + first notice words: " + String(len(got)) + " words")
    finally:
        ignore_drop.free()
    var ignore = HostMember(MEMBER_PREP, onehot_words(1.0, 0.0, False, 0.0, empty, no_text))
    try:
        ignore.fit(column(fit2), 2, 1)
        _ = ignore.transform(column(five), 1)
        require(len(notices(ignore)) == 0, "ignore without a drop keeps no notice")
    finally:
        ignore.free()
    var strict = HostMember(MEMBER_PREP, onehot_words(0.0, 0.0, False, 0.0, empty, no_text))
    var raised = String("")
    try:
        strict.fit(column(fit2), 2, 1)
        _ = strict.transform(column(five), 1)
    except problem:
        raised = String(problem)
    strict.free()
    require(raised.find("OneHotEncoder found unknown categories in column(s) [0] during transform") >= 0,
        "error transform refusal: '" + raised + "'")


def check_routes() raises:
    var x: List[Float64] = [5.0, 1.0, 5.0, 3.0, 1.0]
    var codes: List[Float64] = [2.0, 0.0, 2.0, 1.0, 0.0]
    var cats: List[Float64] = [1.0, 3.0, 5.0]
    var none = List[List[Float64]]()
    for route in range(4):  # small-loop(params: the four category fit routes)
        if route == 1:
            continue  # CAT_FIT_SCAN is the round-5 route, checked by prep_encoders
        comptime if not CAT_PACK_OPS:
            if route >= 2:
                # an image without ops 157-161 refuses the packed route words at NEW (lane w3-fix-c)
                var refused = String("")
                try:
                    var none_here = HostMember(MEMBER_PREP, ordinal_words(Float64(route), False, 0.0, False, -1.0,
                                                                          0.0, 0.0, none.copy(), False))
                    none_here.free()
                except problem:
                    refused = String(problem)
                require(refused.find("no packed category fit route") >= 0,
                        "route " + String(route) + " refusal: '" + refused + "'")
                continue
        var member = HostMember(MEMBER_PREP, ordinal_words(Float64(route), False, 0.0, False, -1.0, 0.0, 0.0,
                                                           none.copy(), False))
        try:
            member.fit(column(x), 5, 1)
            same(member.block(1), cats, "route " + String(route) + " categories_")
            same(member.transform(column(x), 5), codes, "route " + String(route) + " transform")
        finally:
            member.free()
    var raised = String("")
    try:
        var bad = HostMember(MEMBER_PREP, ordinal_words(4.0, False, 0.0, False, -1.0, 0.0, 0.0, none.copy(), False))
        bad.free()
    except problem:
        raised = String(problem)
    require(raised.find("category fit route") >= 0, "route 4 refusal: '" + raised + "'")


def test_mf_enc6_encoders() raises:
    def run_case(which: Int) raises:
        if which == 0:
            check_given()
        elif which == 1:
            check_infrequent()
        elif which == 2:
            check_drop()
        elif which == 3:
            check_warn()
        else:
            check_routes()

    check_cases("mf_enc6_encoders", CASES, run_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
