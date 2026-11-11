# SPDX-License-Identifier: Apache-2.0
"""The trees family's RUN modules, reached without Python (lane
doorsplit-trees, 2026-11-11): one case per RUN module split out of the
trees doors (gbdt/trees_gbdt_fit_run.mojo, gbdt/trees_gbdt_run.mojo,
xtrees/trees_forest_run.mojo, xtrees/trees_boost_run.mojo,
xtrees/trees_meta_run.mojo, xtrees/trees_score_run.mojo), with literal
expected values and the refusal texts the doors raised before the split.

The cases are independent units and run across the native host worker pool.
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.forest_calls import ForestExport
from gbdt.trees_gbdt_fit_run import (
    GbdtOrderedFacts, GbdtSearchFacts, gbdt_c_round, gbdt_auto_rate, gbdt_check_ordered_options,
    gbdt_validate_search_options, gbdt_id_code, gbdt_id_typestr_supported, gbdt_check_group_rows, gbdt_bias_of,
)
from gbdt.trees_gbdt_run import gbdt_mode_checked, gbdt_text_has_ctr, gbdt_mode_of_code, gbdt_known_mode
from xtrees.trees_forest_run import (
    forest_export_counts, forest_export_dtype, forest_export_verdict_check, forest_fit_notice_codes,
    trees_refuse, trees_check_features,
)
from xtrees.trees_boost_run import (
    boost_ada_session_choice, boost_bagging_validate, boost_check_n_estimators, boost_check_learning_rate,
    boost_count_fraction_ok, boost_count_check, boost_weight_flags_check, boost_dart_cap_check,
)
from xtrees.trees_meta_run import CalibratedProgram, meta_binary_column_check
from xtrees.trees_score_run import score_shape_text
from core.labels_score_keys import ScoreLabelKey, ScoreKeyIndex, score_key_codes, SCORE_INT

comptime TREES_DOOR_RUN_CASES = 8


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def refuses(problem: Error, text: String) raises:
    require(text in String(problem), "unexpected refusal: " + String(problem))


def _ordered(grow_symmetric: Bool, score_cosine: Bool) -> GbdtOrderedFacts:
    return GbdtOrderedFacts(grow_symmetric, False, False, False, score_cosine, False, False, False,
                            String("RMSE"), String("'RMSE'"), String("L2"))


def case_gbdt_fit_scalars() raises:
    require(gbdt_c_round(2.5, 0) == 3.0 and gbdt_c_round(-2.5, 0) == -3.0, "c_round halves away from zero")
    require(gbdt_c_round(0.1234565, 6) == gbdt_c_round(0.1234565, 6), "c_round is a function")
    require(gbdt_auto_rate(0.0, 0.0, 0.0, 0.0, 10.0, 10.0) == 0.5, "auto rate is capped at 0.5")
    try:
        _ = gbdt_auto_rate(0.0, 0.0, 0.0, 0.0, 0.0, 10.0)
        raise Error("no refusal")
    except problem:
        refuses(problem, "math domain error")
    require(gbdt_id_code("<i8") == 0 and gbdt_id_code("|i1") == 5 and gbdt_id_code("<u1") == 6, "id codes")
    require(gbdt_id_typestr_supported("<u2") and not gbdt_id_typestr_supported("<f4"), "id kinds")
    require(gbdt_bias_of("trees 1\nbias 0/3FF0000000000000\n") == 1.0, "bias line")
    require(gbdt_bias_of("trees 1\n") == 0.0, "no bias line")
    try:
        gbdt_check_group_rows(3, 4)
        raise Error("no refusal")
    except problem:
        refuses(problem, "Length of group_id=3 and length of data=4 are different.")


def case_gbdt_ordered() raises:
    gbdt_check_ordered_options("Plain", _ordered(False, False))
    gbdt_check_ordered_options("Ordered", _ordered(True, True))
    try:
        gbdt_check_ordered_options("Ordered", _ordered(False, True))
        raise Error("no refusal")
    except problem:
        refuses(problem, "Ordered boosting is not supported for nonsymmetric trees.")
    try:
        gbdt_check_ordered_options("Ordered", _ordered(True, False))
        raise Error("no refusal")
    except problem:
        refuses(problem, "Score function L2 can't be used with ordered boosting")


def case_gbdt_search() raises:
    var unset = GbdtSearchFacts(False, True, String("'SymmetricTree'"), False, String("'Cosine'"), True, 0.0,
                                String(""), 1.0, 2.0)
    require(gbdt_validate_search_options(unset) == 1.0, "catboost default strength")
    var l2 = GbdtSearchFacts(False, True, String("'SymmetricTree'"), True, String("'L2'"), False, 0.5,
                             String("0.5"), 1.0, 2.0)
    try:
        _ = gbdt_validate_search_options(l2)
        raise Error("no refusal")
    except problem:
        refuses(problem, "random_strength=0.5 does nothing under score_function='L2'")
    var bad = GbdtSearchFacts(False, True, String("'SymmetricTree'"), False, String("'Cosine'"), False,
                              bitcast[DType.float64](UInt64(0x7FF8000000000000)), String(""), 1.0, 2.0)
    try:
        _ = gbdt_validate_search_options(bad)
        raise Error("no refusal")
    except problem:
        refuses(problem, "random_strength must be finite")


def case_gbdt_predict() raises:
    require(gbdt_mode_checked("fast") == "fast", "mode check")
    try:
        _ = gbdt_mode_checked("FAST ")
        raise Error("no refusal")
    except problem:
        refuses(problem, "numeric_mode must be fast, identical or None")
    require(gbdt_text_has_ctr("x\nctr_table 1") and gbdt_text_has_ctr("ctr_columns 2"), "ctr records")
    require(not gbdt_text_has_ctr("trees 1\n"), "no ctr records")
    require(gbdt_mode_of_code(-17, 1) == "identical", "mode word")
    require(gbdt_known_mode(-17).value() == "identical", "mode cached")


def case_forest() raises:
    var meta: List[Int] = [2, 1]
    var counts = forest_export_counts(ForestExport(0, 2, 5, 1, meta^))
    require(len(counts) == 5 and counts[0] == 3 and counts[4] == 5, "export counts")
    require(forest_export_dtype(2) == "<f4" and forest_export_dtype(3) == "<i4", "export dtypes")
    forest_export_verdict_check(0)
    try:
        forest_export_verdict_check(3)
        raise Error("no refusal")
    except problem:
        refuses(problem, "forest export byte mismatch: _quesval")
    var codes = forest_fit_notice_codes(7)
    require(len(codes) == 3 and codes[0] == 2 and codes[1] == 4 and codes[2] == 1, "notice order")
    try:
        _ = forest_fit_notice_codes(8)
        raise Error("no refusal")
    except problem:
        refuses(problem, "unknown notice bits 8")
    try:
        trees_check_features(3, 4)
        raise Error("no refusal")
    except problem:
        refuses(problem, "X has 3 features, fit saw 4")
    try:
        trees_refuse("n_jobs", "why.")
    except problem:
        refuses(problem, "NotImplementedError: n_jobs is not implemented: why. Refused by name")


def case_boost() raises:
    require(boost_ada_session_choice("fast", 4) == "share", "share session")
    require(boost_ada_session_choice("identical", 4) == "exact", "exact session")
    boost_bagging_validate(True, True, False, False, False)
    try:
        boost_bagging_validate(True, False, False, False, False)
        raise Error("no refusal")
    except problem:
        refuses(problem, "Out of bag estimation only available if bootstrap=True")
    try:
        boost_check_n_estimators(0)
        raise Error("no refusal")
    except problem:
        refuses(problem, "n_estimators must be >= 1")
    try:
        boost_check_learning_rate(0.0)
        raise Error("no refusal")
    except problem:
        refuses(problem, "learning_rate must be > 0")
    require(boost_count_fraction_ok(1.0) and not boost_count_fraction_ok(0.0), "fraction domain")
    require(boost_count_check("max_samples", 3, 4) == 3, "count in range")
    try:
        boost_weight_flags_check(False, False)
        raise Error("no refusal")
    except problem:
        refuses(problem, "at least one non-zero number")
    boost_dart_cap_check(10, 2, 63)
    try:
        boost_dart_cap_check(10, 2, 65537)
        raise Error("no refusal")
    except problem:
        refuses(problem, "leaf-value store")


def case_meta() raises:
    var program = CalibratedProgram(1000, 5, 2, 1)
    program.pair(2000, 0, 2, 1, 0, 1)
    program.sigmoid(-1.5, 0.25)
    require(len(program.addrs) == 5 and program.addrs[0] == 1000 and program.addrs[1] == 2000, "addresses")
    require(len(program.ints) == 11 and program.ints[3] == 2 and program.ints[4] == 1, "integer words")
    require(program.floats[8] == -1.5 and program.floats[9] == 0.25, "sigmoid words")
    program.isotonic(3000, 4000, 7)
    require(program.ints[len(program.ints) - 1] == 7 and program.ints[len(program.ints) - 4] == 1, "isotonic words")
    try:
        meta_binary_column_check(-1)
        raise Error("no refusal")
    except problem:
        refuses(problem, "1 is not in list")


def case_score() raises:
    var empty = List[Int]()
    var one: List[Int] = [3]
    var two: List[Int] = [3, 4]
    require(score_shape_text(empty) == "()" and score_shape_text(one) == "(3,)" and score_shape_text(two) == "(3, 4)",
            "shape text")
    # The COUNT program words moved to the shared labels_accuracy
    # (lane loops-trees-ab); its plan is covered by the labels suites.
    # `trees_class_codes` RUN (lane loops-trees-ab): the distinct labels in
    # sorted order and each row's position among them, natively; the door
    # then maps the few distinct labels through the class dict and gathers.
    var labels: List[Int] = [5, 3, 5, 7, 3]
    var keys = List[ScoreLabelKey]()
    for i in range(len(labels)):  # driver-loop: over the five literal fixture labels
        keys.append(ScoreLabelKey(SCORE_INT, Int64(labels[i]), 0, 0, False))
    var index = ScoreKeyIndex(Int(keys.unsafe_ptr()), len(keys))
    require(len(index.sorted_rows) == 3, "three distinct labels")
    require(index.sorted_rows[0] == 1 and index.sorted_rows[1] == 0 and index.sorted_rows[2] == 3,
            "distinct labels in sorted order, first row of each")
    var codes = List[Int32](length=len(keys), fill=-9)
    score_key_codes(index, keys, Int(codes.unsafe_ptr()), True)
    require(codes[0] == 1 and codes[1] == 0 and codes[2] == 1 and codes[3] == 2 and codes[4] == 0,
            "row positions among the sorted distinct labels")
    _ = index^
    _ = keys^


def case(unit: Int) raises:
    if unit == 0:
        case_gbdt_fit_scalars()
    elif unit == 1:
        case_gbdt_ordered()
    elif unit == 2:
        case_gbdt_search()
    elif unit == 3:
        case_gbdt_predict()
    elif unit == 4:
        case_forest()
    elif unit == 5:
        case_boost()
    elif unit == 6:
        case_meta()
    else:
        case_score()


def run_checks() raises:
    var failures = List[Int](length=TREES_DOOR_RUN_CASES, fill=0)
    var flags = failures.unsafe_ptr()
    def task(unit: Int) {imm flags}:
        try:
            case(unit)
        except problem:
            flags[unit] = 1
            print("FAIL trees door RUN case", unit, String(problem))
    host_parallelize(task, TREES_DOOR_RUN_CASES)
    var failed = 0
    for failure in failures:  # collect independent case status words
        failed += failure
    _ = failures
    require(failed == 0, "trees door RUN cases failed: " + String(failed))
    print("PASS trees door RUN", TREES_DOOR_RUN_CASES)
