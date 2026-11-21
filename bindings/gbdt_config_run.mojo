# SPDX-License-Identifier: Apache-2.0
"""The GradientBoosting fit configuration as plain values, and the fit
request words built from it (RUN of bindings/trees_public_gbdt_fit_door.mojo
and of the Gbdt member, core/gbdt_member.mojo; no Python import).

`GbdtConfig` is what one `GradientBoosting.fit` reads of the estimator's
settings and of the gbdt option tables, read ONCE by an UNPACK: the fit
door's (`gbdt_config` in the door, once per fit, where the former `_params`
read them) or the member admit's (once per member; the words travel in
`MemberPlan.params`, `gbdt_config_words` / `gbdt_config_of_words`). The
parts that depend on the training rows (the automatic learning rate, the
leaf-iteration default, the boosting type, the Plain-only refusals) are
resolved here at fit time, from the rows the fit has:

  * `gbdt_fit_words` (the former `_params`): the 36 fixed slots, the class
    weights and the optional min_split_gain / min_child_hessian /
    feature_fraction tail, as the two views and the flag byte of
    `GbdtFitArgs` (gbdt/fit_request.mojo), and the resolved rate.
  * `gbdt_pad_tail` / `gbdt_push_int` (the former `_padded_tail` and the
    appended multi-target / query / pair slots).
  * `gbdt_fit_options` (the former boosting resolution, the Plain-only
    refusals and the option strings).

One conversion of the words, so the door and the member build the same
request: the fitted model text has the same bytes on both routes. Each
function is a per-fit scalar decision over about forty slots; there is no
independent work to split across cores.
"""
from std.memory import bitcast
from gbdt.trees_gbdt_fit_run import GbdtOrderedFacts, gbdt_auto_rate, gbdt_check_ordered_options

#: the request's unset float sentinel (`_params`' `f(None)`)
comptime GBDT_UNSET = -1.0
#: the fixed slots ahead of the class weights; slot 35 is their count
comptime GBDT_FIXED_SLOTS = 36
#: `rate_form`: a fixed rate, or CatBoost's automatic rate from the rows
comptime GBDT_RATE_FIXED = 0
comptime GBDT_RATE_AUTO = 1


struct GbdtConfig(Copyable, Movable):
    """The settings one fit reads (see the module docstring). Int slots are
    the values `int(...)` gave at UNPACK (-1 where the request carries
    "unset"), float slots the values `float(...)` gave (GBDT_UNSET where
    unset). `tail` holds the 0 to 3 tail words already in their order."""
    var loss: String
    var n_estimators: Int
    var border_count: Int
    var max_depth: Int
    var random_state: Int
    var l2: Float64
    var score_code: Int
    var loss_alpha: Float64
    var loss_q: Float64
    var loss_delta: Float64
    var loss_variance_power: Float64
    var loss_border: Float64
    var leaf_iters: Int
    var method_code: Int
    var bagging_temperature: Float64
    var subsample: Float64
    var od_pvalue: Float64
    var od_wait: Int
    var use_best: Int
    var best_model_min_trees: Int
    var strength: Float64
    var pointwise: Bool
    var border_build_max_samples: Int
    var permutation_count: Int
    var ctr_permutation_id: Int
    var boost_from_average: Int
    var grow_code: Int
    var grow_symmetric: Bool
    var max_leaves: Int
    var min_data_in_leaf: Int
    var max_ctr_complexity: Int
    var class_weights: List[Float64]
    var tail: List[Float64]
    # The learning rate: GBDT_RATE_FIXED `rate`, or GBDT_RATE_AUTO from the
    # table row of (loss, use_best_model, boost_from_average) for each
    # use_best_model answer (`auto_have[u]`, `auto_coef[4 u + 0..3]`), else
    # `catboost_rate`.
    var rate_form: Int
    var rate: Float64
    var catboost_rate: Float64
    var auto_have: List[Bool]
    var auto_coef: List[Float64]
    # The boosting type: `explicit` (empty when unset), else Plain when
    # `plain_forced`, else Ordered below `ordered_max_rows` rows with at
    # least `ordered_min_iterations` iterations.
    var explicit: String
    var plain_forced: Bool
    var ordered_max_rows: Int
    var ordered_min_iterations: Int
    # The Plain-only refusals and the Ordered option strings.
    var fold_len: Float64
    var fold_block_set: Bool
    var fold_block: Int
    var permutation_set: Bool
    var has_cat: Bool
    # The option strings: bootstrap and od are empty when unset.
    var bootstrap: String
    var od: String
    var nan_mode: String
    var border: String
    # The Ordered checks: `ordered` with its texts when `ordered_texts`
    # (the member's admit takes them once); the door runs its own check,
    # which builds the texts only for an Ordered fit.
    var ordered_texts: Bool
    var ordered: GbdtOrderedFacts

    def __init__(out self):
        self.loss = String("")
        self.n_estimators = 0
        self.border_count = 0
        self.max_depth = 0
        self.random_state = 0
        self.l2 = 0.0
        self.score_code = 0
        self.loss_alpha = GBDT_UNSET
        self.loss_q = GBDT_UNSET
        self.loss_delta = GBDT_UNSET
        self.loss_variance_power = GBDT_UNSET
        self.loss_border = GBDT_UNSET
        self.leaf_iters = -1
        self.method_code = -1
        self.bagging_temperature = 0.0
        self.subsample = GBDT_UNSET
        self.od_pvalue = GBDT_UNSET
        self.od_wait = -1
        self.use_best = -1
        self.best_model_min_trees = 0
        self.strength = 0.0
        self.pointwise = False
        self.border_build_max_samples = 0
        self.permutation_count = -1
        self.ctr_permutation_id = -1
        self.boost_from_average = -1
        self.grow_code = 0
        self.grow_symmetric = True
        self.max_leaves = -1
        self.min_data_in_leaf = 0
        self.max_ctr_complexity = 1
        self.class_weights = List[Float64]()
        self.tail = List[Float64]()
        self.rate_form = GBDT_RATE_FIXED
        self.rate = 0.0
        self.catboost_rate = 0.0
        self.auto_have = List[Bool](length=2, fill=False)
        self.auto_coef = List[Float64](length=8, fill=0.0)
        self.explicit = String("")
        self.plain_forced = True
        self.ordered_max_rows = 0
        self.ordered_min_iterations = 0
        self.fold_len = 2.0
        self.fold_block_set = False
        self.fold_block = 0
        self.permutation_set = False
        self.has_cat = False
        self.bootstrap = String("")
        self.od = String("")
        self.nan_mode = String("")
        self.border = String("GreedyLogSum")
        self.ordered_texts = False
        self.ordered = GbdtOrderedFacts(True, False, False, False, True, False, False, False, String(""),
                                        String(""), String(""))


struct GbdtWords(Movable):
    """The request's parameter slots: the two views and the flag byte
    `GbdtFitArgs` takes, and the resolved learning rate."""
    var ints: List[Int]
    var floats: List[Float64]
    var bad: List[UInt8]
    var rate: Float64

    def __init__(out self):
        self.ints = List[Int](capacity=48)
        self.floats = List[Float64](capacity=48)
        self.bad = List[UInt8](capacity=48)
        self.rate = 0.0


@always_inline
def gbdt_push_int(mut words: GbdtWords, value: Int):
    """An integer slot (both readings succeed)."""
    words.ints.append(value)
    words.floats.append(Float64(value))
    words.bad.append(UInt8(0))


@always_inline
def gbdt_push_float(mut words: GbdtWords, value: Float64):
    """A float slot: both readings succeed, the integer view is the
    truncated value where it has one (bindings/forest_config_run.mojo
    `_slot_float`, the one rule of the float slots)."""
    var iv = 0
    if value == value and value > -9.0e18 and value < 9.0e18:
        iv = Int(value)
    words.ints.append(iv)
    words.floats.append(value)
    words.bad.append(UInt8(0))


def gbdt_resolved_rate(config: GbdtConfig, n_rows: Int, n_eval_rows: Int, eval_constant: Bool) raises -> Float64:
    """`GradientBoosting._resolved_learning_rate` from the rows."""
    if config.rate_form == GBDT_RATE_FIXED:
        return config.rate
    var use_best = (n_eval_rows > 0 and not eval_constant) if config.use_best < 0 else config.use_best == 1
    var u = 1 if use_best else 0
    if not config.auto_have[u]:
        return config.catboost_rate
    return gbdt_auto_rate(config.auto_coef[4 * u], config.auto_coef[4 * u + 1], config.auto_coef[4 * u + 2],
                          config.auto_coef[4 * u + 3], Float64(n_rows), Float64(config.n_estimators))


def gbdt_resolved_boosting(config: GbdtConfig, n_rows: Int) -> String:
    """`GradientBoosting._resolved_boosting_type` from the rows."""
    if config.explicit != "":
        return config.explicit
    if config.plain_forced:
        return String("Plain")
    if n_rows >= config.ordered_max_rows or config.n_estimators < config.ordered_min_iterations:
        return String("Plain")
    return String("Ordered")


def gbdt_fit_words(config: GbdtConfig, n_rows: Int, n_features: Int, n_flags: Int, n_weights: Int,
                   n_eval_rows: Int, eval_constant: Bool) raises -> GbdtWords:
    """`GradientBoosting._params`: the 36 slots in the binding's order, the
    class weights, then the tail; the resolved rate in `rate`."""
    var w = GbdtWords()
    var iters = config.leaf_iters
    if iters < 0 and config.grow_symmetric and config.n_estimators < 200 and n_features < 20:
        iters = 1
    w.rate = gbdt_resolved_rate(config, n_rows, n_eval_rows, eval_constant)
    gbdt_push_int(w, n_rows)
    gbdt_push_int(w, n_features)
    gbdt_push_int(w, n_weights)
    gbdt_push_int(w, n_flags)
    gbdt_push_int(w, config.border_count)
    gbdt_push_int(w, config.n_estimators)
    gbdt_push_int(w, config.max_depth)
    gbdt_push_float(w, w.rate)
    gbdt_push_float(w, config.l2)
    gbdt_push_int(w, config.random_state)
    gbdt_push_int(w, config.score_code)
    gbdt_push_float(w, config.loss_alpha)
    gbdt_push_float(w, config.loss_q)
    gbdt_push_float(w, config.loss_delta)
    gbdt_push_float(w, config.loss_variance_power)
    gbdt_push_float(w, config.loss_border)
    gbdt_push_int(w, iters)
    gbdt_push_int(w, config.method_code)
    gbdt_push_float(w, config.bagging_temperature)
    gbdt_push_float(w, config.subsample)
    gbdt_push_int(w, n_eval_rows)
    gbdt_push_float(w, config.od_pvalue)
    gbdt_push_int(w, config.od_wait)
    gbdt_push_int(w, config.use_best)
    gbdt_push_int(w, config.best_model_min_trees)
    gbdt_push_float(w, config.strength)
    gbdt_push_int(w, 1 if config.pointwise else 0)
    gbdt_push_int(w, config.border_build_max_samples)
    gbdt_push_int(w, config.permutation_count)
    gbdt_push_int(w, config.ctr_permutation_id)
    gbdt_push_int(w, config.boost_from_average)
    gbdt_push_int(w, config.grow_code)
    gbdt_push_int(w, config.max_leaves)
    gbdt_push_int(w, config.min_data_in_leaf)
    gbdt_push_int(w, config.max_ctr_complexity)
    gbdt_push_int(w, len(config.class_weights))
    for k in range(len(config.class_weights)):  # small-loop(class weights): one per class, once per fit
        gbdt_push_float(w, config.class_weights[k])
    for k in range(len(config.tail)):  # small-loop(tail): at most three words
        gbdt_push_float(w, config.tail[k])
    return w^


#: The float slots of the fixed layout (the rest are integers): what
#: `gbdt_inspect` hands back as Python floats.
def gbdt_slot_is_float(index: Int) -> Bool:
    if index >= GBDT_FIXED_SLOTS:
        return True
    return (index == 7 or index == 8 or (index >= 11 and index <= 15) or index == 18 or index == 19
            or index == 21 or index == 25)


def gbdt_pad_tail(mut words: GbdtWords):
    """The slots past the class weights padded to their three defaults
    (`[-1.0, -1.0, 1.0][len(tail):]`), so a later slot lands in place."""
    var fixed = GBDT_FIXED_SLOTS + words.ints[GBDT_FIXED_SLOTS - 1]
    var have = len(words.ints) - fixed
    for i in range(have, 3):  # driver-loop: over fixed call structure, small-loop(defaults): at most three tail defaults
        gbdt_push_float(words, 1.0 if i == 2 else -1.0)


struct GbdtOptions(Movable):
    """The resolved boosting type and the option strings of the request."""
    var boosting: String
    var strs: List[String]

    def __init__(out self, var boosting: String, var strs: List[String]):
        self.boosting = boosting^
        self.strs = strs^


def gbdt_fit_options(config: GbdtConfig, n_rows: Int) raises -> GbdtOptions:
    """The boosting type from the rows, the Ordered checks (when the config
    carries their texts), the Plain-only refusals, and the option strings
    `[loss, bootstrap, od, nan_mode]` + border (when not GreedyLogSum or
    Ordered) + the three Ordered fold strings."""
    var boosting = gbdt_resolved_boosting(config, n_rows)
    if boosting == "Ordered" and config.ordered_texts:
        gbdt_check_ordered_options(boosting, config.ordered)
    if boosting == "Plain":
        if config.fold_len != 2.0 or config.fold_block_set:
            raise Error("mojolearn: fold_len_multiplier and fold_permutation_block "
                        + "are read only by Ordered boosting, and this fit resolved "
                        + "to Plain; they would be accepted and ignored")
        if config.permutation_set and not config.has_cat:
            raise Error("mojolearn: permutation_count is read only by the "
                        + "categorical path and by Ordered boosting, and this fit "
                        + "resolved to Plain with no cat_features; it would be "
                        + "accepted and ignored")
    var strs: List[String] = [config.loss, config.bootstrap, config.od, config.nan_mode]
    if config.border != "GreedyLogSum" or boosting == "Ordered":
        strs.append(config.border)
    if boosting == "Ordered":
        strs.append(String("Ordered"))
        strs.append(String(bitcast[DType.uint64](config.fold_len)))
        strs.append(String(config.fold_block if config.fold_block_set else 0))
    return GbdtOptions(boosting^, strs^)


# ------------------------------------------------- the member's words


@always_inline
def _put_int(mut words: List[Float64], value: Int):
    # An int travels as the Float64 bit pattern of its Int64 (seeds exact).
    words.append(bitcast[DType.float64](Int64(value)))


@always_inline
def _put_bool(mut words: List[Float64], value: Bool):
    words.append(1.0 if value else 0.0)


def _put_str(mut words: List[Float64], text: String):
    var bytes = text.as_bytes()
    words.append(Float64(len(bytes)))
    for k in range(len(bytes)):  # small-loop(option text): a short setting name, once per member
        words.append(Float64(Int(bytes[k])))


def _put_list(mut words: List[Float64], values: List[Float64]):
    words.append(Float64(len(values)))
    for k in range(len(values)):  # small-loop(class weights / tail): a few words, once per member
        words.append(values[k])


struct _Reader:
    var words: List[Float64]
    var at: Int

    def __init__(out self, var words: List[Float64]):
        self.words = words^
        self.at = 0

    def take(mut self) raises -> Float64:
        if self.at >= len(self.words):
            raise Error("member ABI: the Gbdt member's parameter words end early")
        var v = self.words[self.at]
        self.at += 1
        return v

    def int(mut self) raises -> Int:
        return Int(bitcast[DType.int64](self.take()))

    def bool(mut self) raises -> Bool:
        return self.take() != 0.0

    def count(mut self) raises -> Int:
        var n = Int(self.take())
        if n < 0 or n > len(self.words) - self.at:
            raise Error("member ABI: a Gbdt member's parameter list is out of range")
        return n

    def text(mut self) raises -> String:
        var n = self.count()
        var out = String(capacity=n)
        for _ in range(n):  # small-loop(option text): a short setting name, once per state
            out += chr(Int(self.take()))
        return out^

    def list(mut self) raises -> List[Float64]:
        var n = self.count()
        var out = List[Float64](capacity=n)
        for _ in range(n):  # small-loop(class weights / tail): a few words, once per state
            out.append(self.take())
        return out^


def gbdt_config_words(config: GbdtConfig) -> List[Float64]:
    """The config as MemberPlan.params words (core/gbdt_member.mojo's
    docstring lists them in this order)."""
    var w = List[Float64](capacity=96)
    _put_str(w, config.loss)
    _put_int(w, config.n_estimators)
    _put_int(w, config.border_count)
    _put_int(w, config.max_depth)
    _put_int(w, config.random_state)
    w.append(config.l2)
    _put_int(w, config.score_code)
    w.append(config.loss_alpha)
    w.append(config.loss_q)
    w.append(config.loss_delta)
    w.append(config.loss_variance_power)
    w.append(config.loss_border)
    _put_int(w, config.leaf_iters)
    _put_int(w, config.method_code)
    w.append(config.bagging_temperature)
    w.append(config.subsample)
    w.append(config.od_pvalue)
    _put_int(w, config.od_wait)
    _put_int(w, config.use_best)
    _put_int(w, config.best_model_min_trees)
    w.append(config.strength)
    _put_bool(w, config.pointwise)
    _put_int(w, config.border_build_max_samples)
    _put_int(w, config.permutation_count)
    _put_int(w, config.ctr_permutation_id)
    _put_int(w, config.boost_from_average)
    _put_int(w, config.grow_code)
    _put_bool(w, config.grow_symmetric)
    _put_int(w, config.max_leaves)
    _put_int(w, config.min_data_in_leaf)
    _put_int(w, config.max_ctr_complexity)
    _put_list(w, config.class_weights)
    _put_list(w, config.tail)
    _put_int(w, config.rate_form)
    w.append(config.rate)
    w.append(config.catboost_rate)
    _put_bool(w, config.auto_have[0])
    _put_bool(w, config.auto_have[1])
    for k in range(8):  # small-loop(two table rows of four coefficients)
        w.append(config.auto_coef[k])
    _put_str(w, config.explicit)
    _put_bool(w, config.plain_forced)
    _put_int(w, config.ordered_max_rows)
    _put_int(w, config.ordered_min_iterations)
    w.append(config.fold_len)
    _put_bool(w, config.fold_block_set)
    _put_int(w, config.fold_block)
    _put_bool(w, config.permutation_set)
    _put_bool(w, config.has_cat)
    _put_str(w, config.bootstrap)
    _put_str(w, config.od)
    _put_str(w, config.nan_mode)
    _put_str(w, config.border)
    _put_bool(w, config.ordered_texts)
    _put_bool(w, config.ordered.grow_symmetric)
    _put_bool(w, config.ordered.loss_multi_output)
    _put_bool(w, config.ordered.loss_multi_regression)
    _put_bool(w, config.ordered.loss_querywise)
    _put_bool(w, config.ordered.score_cosine)
    _put_bool(w, config.ordered.leaf_exact)
    _put_bool(w, config.ordered.pointwise)
    _put_bool(w, config.ordered.fraction_below_one)
    _put_str(w, config.ordered.loss_str)
    _put_str(w, config.ordered.loss_repr)
    _put_str(w, config.ordered.score_str)
    return w^


def gbdt_config_of_words(var words: List[Float64], start: Int) raises -> GbdtConfig:
    """The config from MemberPlan.params words, from index `start`; every
    word must be read (a count that does not match is refused)."""
    var r = _Reader(words^)
    r.at = start
    var c = GbdtConfig()
    c.loss = r.text()
    c.n_estimators = r.int()
    c.border_count = r.int()
    c.max_depth = r.int()
    c.random_state = r.int()
    c.l2 = r.take()
    c.score_code = r.int()
    c.loss_alpha = r.take()
    c.loss_q = r.take()
    c.loss_delta = r.take()
    c.loss_variance_power = r.take()
    c.loss_border = r.take()
    c.leaf_iters = r.int()
    c.method_code = r.int()
    c.bagging_temperature = r.take()
    c.subsample = r.take()
    c.od_pvalue = r.take()
    c.od_wait = r.int()
    c.use_best = r.int()
    c.best_model_min_trees = r.int()
    c.strength = r.take()
    c.pointwise = r.bool()
    c.border_build_max_samples = r.int()
    c.permutation_count = r.int()
    c.ctr_permutation_id = r.int()
    c.boost_from_average = r.int()
    c.grow_code = r.int()
    c.grow_symmetric = r.bool()
    c.max_leaves = r.int()
    c.min_data_in_leaf = r.int()
    c.max_ctr_complexity = r.int()
    c.class_weights = r.list()
    c.tail = r.list()
    c.rate_form = r.int()
    c.rate = r.take()
    c.catboost_rate = r.take()
    c.auto_have[0] = r.bool()
    c.auto_have[1] = r.bool()
    for k in range(8):  # small-loop(two table rows of four coefficients)
        c.auto_coef[k] = r.take()
    c.explicit = r.text()
    c.plain_forced = r.bool()
    c.ordered_max_rows = r.int()
    c.ordered_min_iterations = r.int()
    c.fold_len = r.take()
    c.fold_block_set = r.bool()
    c.fold_block = r.int()
    c.permutation_set = r.bool()
    c.has_cat = r.bool()
    c.bootstrap = r.text()
    c.od = r.text()
    c.nan_mode = r.text()
    c.border = r.text()
    c.ordered_texts = r.bool()
    c.ordered.grow_symmetric = r.bool()
    c.ordered.loss_multi_output = r.bool()
    c.ordered.loss_multi_regression = r.bool()
    c.ordered.loss_querywise = r.bool()
    c.ordered.score_cosine = r.bool()
    c.ordered.leaf_exact = r.bool()
    c.ordered.pointwise = r.bool()
    c.ordered.fraction_below_one = r.bool()
    c.ordered.loss_str = r.text()
    c.ordered.loss_repr = r.text()
    c.ordered.score_str = r.text()
    if r.at != len(r.words):
        raise Error("member ABI: the Gbdt member's parameter words do not match its layout")
    return c^
