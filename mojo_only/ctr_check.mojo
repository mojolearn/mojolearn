"""The CTR option surface and the two calcers, against answers written down
in advance.

    pixi run check-ctr

Five sections, and each one gates a thing that did not exist before this
round:

1. **THE THREE-PRIOR FAN-OUT.** `TCatFeatureParams.default()` must produce
   FOUR configs for one categorical feature -- three `Borders` at priors
   {0,1}, {0.5,1}, {1,1} and one `FeatureFreq` at {0,1} -- and the two ctr
   types must get DIFFERENT binarization grids, Uniform 15 for Borders and
   MinEntropy 15 for FeatureFreq. The count is the assertion that matters:
   a port that emits one column per cat feature is not slightly off, it is
   missing two thirds of what their learner splits on.
2. **TARGET BINARIZATION**, against CatBoost's own borders in
   `bench/ctr_target_oracle.txt`, at budgets 1, 2 and 3. Budget 1 is the
   GPU default and is a different code path: `bins == 2` makes the DP's
   per-level loop body run zero times, so the answer comes entirely from
   the "Last match" scan whose tie-break is the OPPOSITE of the loops'.
3. **FeatureFreq, ANALYTICALLY.** Category counts are PLANTED, so the
   right answer is `(planted_count + prior) / (n + prior_observations)`
   written down before the code runs -- not a second execution of the same
   loop. Compared per cell, never as a total. Cardinalities cross every
   step of `policy_for_fold_count` and every byte boundary: 2, 3, 15, 16,
   17, 31, 32, 255, 256, 1000, 4096, plus k = 1 which must RAISE.
   Both `counter_calc_method` arms run and must agree BIT FOR BIT.
4. **Borders, THE ORDERED TARGET STATISTIC**, against an independent
   O(n^2) tally: for each row, the count of EARLIER rows sharing its
   category, and how many of those had a binarized target above the target
   bin. That tally shares no code with the calcer -- no sort, no segment
   flags, no scan -- which is the point. All three priors are checked, and
   the check requires the three columns to actually DIFFER, because three
   identical columns would mean the prior loop ran once and copied.

WHY NO CatBoost ORACLE FOR SECTIONS 3 AND 4.

`simple_ctr='FeatureFreq'` raises on CatBoost's CPU -- "Ctr type
FeatureFreq is not implemented on CPU yet",
`catboost_options.cpp:509`, verified against the shipped 1.2.10 binary on
this box -- and their GPU arm cannot run on Apple silicon. `Counter` is the
CPU spelling of the same counts under a different normalization: the same
INFORMATION CLASS, not the same bits. **So any claim that our FeatureFreq
is bit-exact to theirs rests on reading their source, and this file says so
rather than implying an oracle it does not have.**

`simple_ctr='Borders'` IS accepted on their CPU and does train, but what it
exposes is not what this section computes. `save_model(format='json')`
gives the FINAL apply-time table (`ctr_data.hash_map`, one bucket per
category hash) and the per-config `prior_numerator` / `prior_denomerator` /
`scale` / `shift`; it does not expose the per-row ORDERED statistic that
the GPU calcer produces during training, and their CPU quantizes online CTR
values on a fixed `scale 15, shift 0` grid over [0,1] where the GPU builds
Uniform-15 borders from the observed column. Verified by inspection of a
1.2.10 CPU fit's JSON. The CPU arm is therefore a quality comparison, not a
value oracle, and the values are gated analytically here instead.
"""

from gbdt.ctrs.ctr import (
    CTR_BORDERS,
    CTR_BUCKETS,
    CTR_COUNTER,
    CTR_FEATURE_FREQ,
    TCtrConfig,
    TPrior,
    ctr_type_name,
)
from gbdt.ctrs.ctr_binarization import (
    BORDER_SELECTION_MIN_ENTROPY,
    BORDER_SELECTION_UNIFORM,
    TBinarizationOptions,
    border_selection_name,
    build_binarized_target,
    build_target_borders,
)
from gbdt.ctrs.ctr_bins_builder import TCtrBinBuilder
from gbdt.ctrs.ctr_calcers import (
    THistoryBasedCtrCalcer,
    compute_simple_ctrs,
)
from gbdt.options.catboost_options import TCatFeatureParams


comptime CTR_TARGET_ORACLE = "bench/ctr_target_oracle.txt"


def _hashed(x: Int, salt: Int) -> Int:
    var v = ((x + 1) * 2654435761 + salt * 40503) % 1000003
    if v < 0:
        v += 1000003
    return v


def _shuffled(var v: List[Int]) -> List[Int]:
    """Deterministic Fisher-Yates. Scattered, not strided: a stride has a
    period a strided bug can line up with."""
    var state = UInt64(0xD1B54A32D192ED03)
    for i in range(len(v) - 1, 0, -1):
        state = state * UInt64(6364136223846793005) + UInt64(
            1442695040888963407
        )
        var j = Int((state >> 33) % UInt64(i + 1))
        var t = v[i]
        v[i] = v[j]
        v[j] = t
    return v^


# ---------------------------------------------------------------------------
# 1. the three-prior fan-out
# ---------------------------------------------------------------------------


def _check_fan_out(mut failures: List[String]) raises:
    var params = TCatFeatureParams.default()
    var configs = params.simple_ctr_configs()

    print("  default simple_ctr configs for ONE categorical feature:")
    for i in range(len(configs)):
        ref c = configs[i]
        var grid = params.ctr_binarization_for(c)
        print(
            "    ",
            ctr_type_name(c.ctr_type),
            " prior {",
            c.prior.numerator,
            ",",
            c.prior.denumerator,
            "} param_id",
            c.param_id,
            " binarization",
            border_selection_name(grid.border_selection_type),
            grid.border_count,
        )

    if len(configs) != 4:
        failures.append(
            String("the GPU simple_ctr default must emit FOUR columns per")
            + String(" categorical feature (three Borders priors + one")
            + String(" FeatureFreq); got ")
            + String(len(configs))
        )
        return

    var want_num = [Float32(0.0), Float32(0.5), Float32(1.0)]
    for i in range(3):
        ref c = configs[i]
        if c.ctr_type != CTR_BORDERS:
            failures.append(
                String("config ") + String(i) + String(" should be Borders")
            )
        if c.prior.numerator != want_num[i] or c.prior.denumerator != (
            Float32(1.0)
        ):
            failures.append(
                String("Borders prior ")
                + String(i)
                + String(" is {")
                + String(c.prior.numerator)
                + String(",")
                + String(c.prior.denumerator)
                + String("}, GetDefaultPriors says {")
                + String(want_num[i])
                + String(",1}")
            )
        if c.param_id != 0:
            failures.append(
                String("at one target border Borders emits ParamId 0 only;")
                + String(" got ")
                + String(c.param_id)
            )
        var grid = params.ctr_binarization_for(c)
        if grid.border_selection_type != BORDER_SELECTION_UNIFORM or (
            grid.border_count != 15
        ):
            failures.append(
                String("a Borders column is quantized with Uniform 15")
                + String(" (the two-argument TCtrDescription constructor,")
                + String(" cat_feature_options.cpp:167-170); got ")
                + border_selection_name(grid.border_selection_type)
                + String(" ")
                + String(grid.border_count)
            )
    ref f = configs[3]
    if f.ctr_type != CTR_FEATURE_FREQ:
        failures.append(String("config 3 should be FeatureFreq"))
    if f.prior.numerator != Float32(0.0) or f.prior.denumerator != (
        Float32(1.0)
    ):
        failures.append(String("FeatureFreq's only prior is {0.0, 1}"))
    var fgrid = params.ctr_binarization_for(f)
    if fgrid.border_selection_type != BORDER_SELECTION_MIN_ENTROPY or (
        fgrid.border_count != 15
    ):
        failures.append(
            String("a FeatureFreq column is quantized with MinEntropy 15")
            + String(" (CreateDefaultCounter, catboost_options.cpp:392-415);")
            + String(" got ")
            + border_selection_name(fgrid.border_selection_type)
            + String(" ")
            + String(fgrid.border_count)
        )

    var freq_only = TCatFeatureParams.feature_freq_only()
    var freq_configs = freq_only.simple_ctr_configs()
    if len(freq_configs) != 1:
        failures.append(
            String("feature_freq_only() must emit exactly one column; got ")
            + String(len(freq_configs))
        )


# ---------------------------------------------------------------------------
# 2. target binarization against CatBoost's own borders
# ---------------------------------------------------------------------------



#: THE FMA ALLOWANCE IS GONE, AND ITS REMOVAL IS THE RESULT.
#:
#: Four (column, budget) pairs used to land on an adjacent, equally-optimal
#: cut rather than CatBoost's, because Mojo CONTRACTED `_penalty_min_entropy`'s
#: multiply-then-add into an FMA across the inlined call while clang, at its
#: default `-ffp-contract=on`, contracts only within one SOURCE expression and
#: `Penalty<type>(...)` is a separate call. One ULP, landing on a symmetric
#: plateau where the last match's `<` tie-break then picked the other arm:
#: measured 30412.210990606218 against 30412.210990606214 at 4001 distinct
#: values and budget 1, where cuts 1999 and 2000 are mathematically equal.
#:
#: `@no_inline` on `_penalty_min_entropy` (PORTING.md 54) fixed it, and this
#: check is what proved it: it was written to FAIL if a known-diverging pair
#: ever matched exactly, so that the allowance could not outlive the defect it
#: described. It did fail, on all four pairs at once, which is why there is
#: nothing left here to allow. All 15 cases are now compared exactly.
def _adjacent_cut(
    values: List[Float32], ours: Float32, theirs: Float32
) raises -> Bool:
    """Are the two borders midpoints of ADJACENT pairs of sorted unique
    values? That is what "the other end of the same plateau" means, and it
    is what separates a tie-break difference from a wrong answer."""
    var sorted_values = values.copy()
    var n = len(sorted_values)
    var gap = n // 2
    while gap > 0:
        for i in range(gap, n):
            var tmp = sorted_values[i]
            var j = i
            while j >= gap and sorted_values[j - gap] > tmp:
                sorted_values[j] = sorted_values[j - gap]
                j -= gap
            sorted_values[j] = tmp
        gap //= 2
    var uniques = List[Float32]()
    for i in range(n):
        if i == 0 or sorted_values[i] != sorted_values[i - 1]:
            uniques.append(sorted_values[i])

    var t_ours = -1
    var t_theirs = -1
    for t in range(len(uniques) - 1):
        var mid = (uniques[t] + uniques[t + 1]) / 2
        if mid == ours:
            t_ours = t
        if mid == theirs:
            t_theirs = t
    if t_ours < 0 or t_theirs < 0:
        return False
    var d = t_ours - t_theirs
    if d < 0:
        d = -d
    return d == 1


def _check_target_binarization(mut failures: List[String]) raises:
    var f = open(CTR_TARGET_ORACLE, "r")
    var text = f.read()
    f.close()

    var lines = List[String]()
    for line in text.splitlines():
        var s = String(String(line).strip())
        if s.byte_length() > 0:
            lines.append(s^)

    var pos = 0
    var head = lines[pos].split(" ")
    if String(head[0]) != "columns":
        raise Error("malformed ctr target oracle header")
    var n_columns = Int(String(head[1]))
    pos += 1

    var columns = List[List[Float32]]()
    for _ in range(n_columns):
        var ch = lines[pos].split(" ")
        if String(ch[0]) != "column":
            raise Error("malformed column header")
        var n_values = Int(String(ch[2]))
        pos += 1
        var col = List[Float32]()
        for i in range(n_values):
            col.append(Float32(Float64(String(lines[pos + i]))))
        pos += n_values
        columns.append(col^)

    var ch2 = lines[pos].split(" ")
    if String(ch2[0]) != "cases":
        raise Error("malformed cases header")
    var n_cases = Int(String(ch2[1]))
    pos += 1

    var exact = 0
    for _ in range(n_cases):
        var hc = lines[pos].split(" ")
        var col_id = Int(String(hc[1]))
        var budget = Int(String(hc[2]))
        var n_borders = Int(String(hc[3]))
        pos += 1
        var want = List[Float32]()
        for i in range(n_borders):
            want.append(Float32(Float64(String(lines[pos + i]))))
        pos += n_borders

        var got = build_target_borders(
            columns[col_id],
            TBinarizationOptions(BORDER_SELECTION_MIN_ENTROPY, budget),
        )
        if len(got) != len(want):
            failures.append(
                String("target borders, column ")
                + String(col_id)
                + String(" budget ")
                + String(budget)
                + String(": ")
                + String(len(got))
                + String(" borders, CatBoost gives ")
                + String(len(want))
            )
            continue
        var wrong = 0
        for i in range(len(want)):
            if got[i] != want[i]:
                wrong += 1
        if wrong != 0:
            var detail = String("")
            var adjacent = wrong == 1
            for i in range(len(want)):
                if got[i] != want[i]:
                    detail += String(" [") + String(i) + String("] ours ")
                    detail += String(Float64(got[i])) + String(" theirs ")
                    detail += String(Float64(want[i]))
                    if not _adjacent_cut(columns[col_id], got[i], want[i]):
                        adjacent = False
            if True:
                failures.append(
                    String("target borders, column ")
                    + String(col_id)
                    + String(" budget ")
                    + String(budget)
                    + String(": ")
                    + String(wrong)
                    + String(" of ")
                    + String(len(want))
                    + String(" differ from CatBoost's")
                    + (
                        String("")
                        if adjacent
                        else String(" and are NOT an adjacent equally-optimal")
                        + String(" cut, so this is not the FMA plateau")
                    )
                    + String(";")
                    + detail
                )
        else:
            exact += 1

        # and the BINARIZATION itself, per row, against a second tally
        if budget == 1 and len(got) == 1:
            var bt = build_binarized_target(columns[col_id], got)
            var mismatched = 0
            var ones = 0
            for r in range(len(columns[col_id])):
                var expect = UInt8(1) if columns[col_id][r] > got[0] else (
                    UInt8(0)
                )
                if bt[r] != expect:
                    mismatched += 1
                if bt[r] == UInt8(1):
                    ones += 1
            if mismatched != 0:
                failures.append(
                    String("binarized target, column ")
                    + String(col_id)
                    + String(": ")
                    + String(mismatched)
                    + String(" rows disagree with `value > border`")
                )
            if ones == 0 or ones == len(columns[col_id]):
                failures.append(
                    String("binarized target, column ")
                    + String(col_id)
                    + String(" is constant (")
                    + String(ones)
                    + String(" ones); the fixture cannot see a border move")
                )
    print(
        "  target binarization:",
        exact,
        "of",
        n_cases,
        "cases match CatBoost's own MinEntropy borders exactly",
    )


# ---------------------------------------------------------------------------
# 3. FeatureFreq, against planted counts
# ---------------------------------------------------------------------------


def _cardinalities() -> List[Int]:
    """Every step of `policy_for_fold_count` plus both byte boundaries plus
    two cardinalities far above anything a one-hot path could take."""
    return [2, 3, 15, 16, 17, 31, 32, 255, 256, 1000, 4096]


def _check_feature_freq(mut failures: List[String]) raises:
    var params = TCatFeatureParams.feature_freq_only()
    var configs = params.simple_ctr_configs()
    var checked = 0

    var cards = _cardinalities()
    for ci in range(len(cards)):
        var k = cards[ci]

        # PLANT the counts. `sizes[c]` is hashed, never a ramp and never
        # uniform: a uniform fixture makes every expected value identical,
        # which verifies the total and nothing about placement.
        var sizes = List[Int]()
        var n = 0
        for c in range(k):
            var s = 1 + _hashed(c, 7) % 23
            sizes.append(s)
            n += s

        var flat = List[Int]()
        for c in range(k):
            for _ in range(sizes[c]):
                flat.append(c)
        var codes_int = _shuffled(flat^)
        var codes = List[UInt32]()
        for r in range(n):
            codes.append(UInt32(codes_int[r]))

        # the answer, written down before the calcer runs
        var want = List[Float32]()
        for r in range(n):
            want.append(
                (Float32(sizes[codes_int[r]]) + Float32(0.0))
                / (Float32(n) + Float32(1.0))
            )

        var by_arm = List[List[Float32]]()
        for arm in range(2):
            var cols = compute_simple_ctrs(
                codes, k, configs, List[UInt8](), arm == 1
            )
            if len(cols) != 1:
                failures.append(
                    String("k=")
                    + String(k)
                    + String(": feature_freq_only produced ")
                    + String(len(cols))
                    + String(" columns")
                )
                return
            by_arm.append(cols[0].copy())

        for arm in range(2):
            var wrong = 0
            var first = -1
            for r in range(n):
                if by_arm[arm][r] != want[r]:
                    wrong += 1
                    if first < 0:
                        first = r
            if wrong != 0:
                failures.append(
                    String("k=")
                    + String(k)
                    + String(" counter_calc_method=")
                    + (String("Full") if arm == 1 else String("SkipTest"))
                    + String(": ")
                    + String(wrong)
                    + String(" of ")
                    + String(n)
                    + String(" cells differ from the planted count/(n+1),")
                    + String(" first at row ")
                    + String(first)
                    + String(" (got ")
                    + String(by_arm[arm][first])
                    + String(", want ")
                    + String(want[first])
                    + String(")")
                )
        # both sides of the switch, and they must agree bit for bit
        var arm_diff = 0
        for r in range(n):
            if by_arm[0][r] != by_arm[1][r]:
                arm_diff += 1
        if arm_diff != 0:
            failures.append(
                String("k=")
                + String(k)
                + String(": SkipTest and Full disagree on ")
                + String(arm_diff)
                + String(" cells; with no test pool they are the same")
                + String(" arithmetic")
            )
        # a fixture whose expected values are all equal proves nothing
        var distinct = 0
        for r in range(1, n):
            if want[r] != want[0]:
                distinct += 1
        if distinct == 0:
            failures.append(
                String("k=")
                + String(k)
                + String(": every expected value is identical; the fixture")
                + String(" cannot see a permutation bug")
            )
        checked += 1

    # k = 1 is their `CB_ENSURE(uniqueValues > 1, "useless catFeature")`
    var one = List[UInt32]()
    for _ in range(64):
        one.append(UInt32(0))
    var refused = False
    try:
        _ = compute_simple_ctrs(one, 1, configs, List[UInt8](), False)
    except:
        refused = True
    if not refused:
        failures.append(
            String("a single-category feature must raise their")
            + String(" 'useless catFeature found'")
        )

    print(
        "  FeatureFreq:",
        checked,
        "cardinalities, both counter_calc_method arms, every cell equal to"
        " the planted count/(n+1)",
    )



# ---------------------------------------------------------------------------
# 5. the option refusals, by name
# ---------------------------------------------------------------------------


def _expect_params_refusal(
    mut failures: List[String], var p: TCatFeatureParams, what: String
) raises:
    var refused = False
    try:
        p.check()
    except:
        refused = True
    if not refused:
        failures.append(what + String(" should have been refused"))
    else:
        print("    refused:", what)


def _check_option_refusals(mut failures: List[String]) raises:
    """`TCatFeatureParams.check()`, both sides.

    The rule this file inherits from `options_check.mojo`: an option present
    and IGNORED is worse than one absent, because absent fails loudly and
    ignored fails silently. A `check()` nobody exercises is the machinery
    this repository keeps finding unwired.
    """
    print("  cat-feature option refusals:")
    var d = TCatFeatureParams.default()
    d.check()
    var f = TCatFeatureParams.feature_freq_only()
    f.check()
    print("    both shipped configurations pass")

    var a = TCatFeatureParams.default()
    a.max_tensor_complexity = 4
    _expect_params_refusal(
        failures, a^, String("max_ctr_complexity=4, CatBoost's own default")
    )

    var b = TCatFeatureParams.default()
    b.counter_calc_method = 7
    _expect_params_refusal(failures, b^, String("counter_calc_method=7"))

    var c = TCatFeatureParams.default()
    c.target_binarization = TBinarizationOptions(BORDER_SELECTION_UNIFORM, 1)
    _expect_params_refusal(
        failures, c^, String("target_binarization border_type=Uniform")
    )

    var e = TCatFeatureParams.default()
    e.target_binarization = TBinarizationOptions(
        BORDER_SELECTION_MIN_ENTROPY, 0
    )
    _expect_params_refusal(
        failures, e^, String("ctr_target_border_count=0")
    )

    # Counter is the CPU spelling and is not a GPU ctr type at all
    var g = TCatFeatureParams.feature_freq_only()
    g.simple_ctrs[0].ctr_type = CTR_COUNTER
    _expect_params_refusal(failures, g^, String("simple_ctr=Counter"))

    # Buckets IS a supported GPU type and still has no calcer here
    var h = TCatFeatureParams.feature_freq_only()
    h.simple_ctrs[0].ctr_type = CTR_BUCKETS
    _expect_params_refusal(failures, h^, String("simple_ctr=Buckets"))

    var i = TCatFeatureParams.feature_freq_only()
    i.simple_ctrs[0].priors = List[TPrior]()
    _expect_params_refusal(failures, i^, String("a CTR with no priors"))


# ---------------------------------------------------------------------------
# 4. Borders: the ordered target statistic
# ---------------------------------------------------------------------------


def _check_borders(mut failures: List[String]) raises:
    var params = TCatFeatureParams.default()
    var configs = params.simple_ctr_configs()

    var checked = 0
    var cards = [2, 3, 17, 256]
    for ci in range(len(cards)):
        var k = cards[ci]
        var n = 2003

        var codes = List[UInt32]()
        var codes_int = List[Int]()
        for r in range(n):
            var c = _hashed(r, 13) % k
            codes_int.append(c)
            codes.append(UInt32(c))

        # a SCATTERED binarized target, so the running sums differ per row
        var target = List[UInt8]()
        for r in range(n):
            target.append(UInt8(1) if _hashed(r, 19) % 3 == 0 else UInt8(0))

        var builder = TCtrBinBuilder(n)
        builder.add_cat_feature_bins(codes, k)
        var calcer = THistoryBasedCtrCalcer(builder)
        calcer.set_binarized_sample(target.copy())

        var borders_configs = List[TCtrConfig]()
        for i in range(len(configs)):
            if configs[i].ctr_type == CTR_BORDERS:
                borders_configs.append(configs[i])
        var got = calcer.visit_cat_feature_ctr(borders_configs)

        if len(got) != 3:
            failures.append(
                String("k=")
                + String(k)
                + String(": three priors must give three columns; got ")
                + String(len(got))
            )
            return

        # THE INDEPENDENT TALLY. O(n^2), shares no code with the calcer:
        # no sort, no segment flags, no scan. For each row, how many
        # EARLIER rows share its category and how many of those had a
        # binarized target above the target bin (ParamId 0, so `> 0`).
        var prior_num = [Float32(0.0), Float32(0.5), Float32(1.0)]
        for p in range(3):
            var wrong = 0
            var first = -1
            for r in range(n):
                var count = Float32(0.0)
                var hits = Float32(0.0)
                for q in range(r):
                    if codes_int[q] == codes_int[r]:
                        count += Float32(1.0)
                        if target[q] > UInt8(0):
                            hits += Float32(1.0)
                var want = (hits + prior_num[p]) / (count + Float32(1.0))
                if got[p][r] != want:
                    wrong += 1
                    if first < 0:
                        first = r
            if wrong != 0:
                failures.append(
                    String("k=")
                    + String(k)
                    + String(" prior ")
                    + String(prior_num[p])
                    + String(": ")
                    + String(wrong)
                    + String(" of ")
                    + String(n)
                    + String(" ordered statistics wrong, first at row ")
                    + String(first)
                    + String(" (got ")
                    + String(got[p][first])
                    + String(")")
                )

        # three priors must give three DIFFERENT columns, or the loop ran
        # once and copied
        for p in range(1, 3):
            var same = True
            for r in range(n):
                if got[p][r] != got[0][r]:
                    same = False
                    break
            if same:
                failures.append(
                    String("k=")
                    + String(k)
                    + String(": prior ")
                    + String(prior_num[p])
                    + String(" produced the same column as prior 0.0")
                )
        checked += 1

    print(
        "  Borders ordered statistic:",
        checked,
        "cardinalities x 3 priors, every row equal to an independent"
        " O(n^2) tally over the preceding same-category rows",
    )



def check_ctrs() raises:
    print("CTR option surface and calcers:")
    var failures = List[String]()
    _check_fan_out(failures)
    _check_target_binarization(failures)
    _check_feature_freq(failures)
    _check_borders(failures)
    _check_option_refusals(failures)
    if len(failures) > 0:
        var msg = String("CTR checks FAILED:")
        for i in range(len(failures)):
            msg += String("\n    ") + failures[i]
        raise Error(msg)
    print("  all five sections green")


def main() raises:
    check_ctrs()
