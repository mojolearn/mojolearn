"""When to stop boosting, decided from a HELD-OUT error curve.

PORT OF `catboost/libs/overfitting_detector/overfitting_detector.{h,cpp}`
at CatBoost `54a8143a`. Transliterated. Do not improve.

THREE OF THEIR FOUR TYPES ARE HERE and the fourth is not:

    None      never stops (`TNoOverfittingDetector`, `:11-33`)
    IncToDec  their DEFAULT (`overfitting_detector_options.cpp:9`), a
              decaying estimate of how much the error could still improve
    Iter      `IncToDec` AT THRESHOLD 1.0 -- their own factory says so
              (`:194-197`), and it is not a separate class
    Wilcoxon  NOT PORTED: `NStatistics::Wilcoxon` is a rank-sum test in
              `library/cpp/statistics`, a dependency outside
              `catboost/`, and nothing this port runs selects it.
              Listed in `gbdt/UNPORTED.tsv`.

## WHY `Iter` IS THE ONE PEOPLE MEAN, and why it is not its own code

`Iter` is what everyone calls "early stopping with `od_wait`": stop after
`od_wait` iterations without a new best. Their factory builds it as
`TOverfittingDetectorIncToDec(maxIsOptimal, 1.0, iterationsWait, hasTest)`
-- the SAME class at threshold 1.0. And that works because
`IsNeedStop()` is `CurrentPValue < Threshold` and `UpdatePValue` returns
exactly 1.0 while `IterationsFromLocalMax < IterationsWait`. So at
threshold 1.0 the detector fires on the first iteration after the wait
expires and never before. Porting it as a second class would have been
inventing a distinction their code does not draw.

## THE SIGN CONVENTION, which is a real trap

`AddError` negates when `!MaxIsOptimal` (`:130-131`), so everything below
is written as if LARGER IS BETTER. Every loss this port trains is
minimized, so `max_is_optimal` is False and the errors handed in are
negated on entry. `LocalMax` is therefore the best (smallest) loss seen,
stored negated. A reader who forgets this will read `LocalMax` as a
maximum of the loss and conclude the detector is upside down.

## NO TEST SET MEANS NO DETECTION, AND THEY ENFORCE IT

Their constructors take `hasTest` and pass `hasTest ? threshold : 0`
(`:122-124`), and `IsActive()` is `Threshold > 0`. So a detector built
without a test set is inert whatever the user asked for. That is not
politeness: stopping on the LEARN error would stop on a curve that falls
almost by construction. This port keeps the same gate and
`gbdt/train.mojo` refuses `od_type != None` without held-out rows rather
than silently never firing.
"""

from std.math import exp

comptime OD_NONE = 0
comptime OD_INC_TO_DEC = 1
comptime OD_ITER = 2

#: `TOverfittingDetectorIncToDec`'s four constants (`:160-163`).
comptime OD_LAMBDA_FORGET = Float64(0.99)
comptime OD_ITERATION_FORGET = 2000
comptime OD_LAMBDA_SCALE = Float64(0.5)
comptime OD_EPS = Float64(1e-10)


def od_type_from_name(name: String) raises -> Int:
    """Their `EOverfittingDetectorType` spellings."""
    if name == "None" or name == "":
        return OD_NONE
    if name == "IncToDec":
        return OD_INC_TO_DEC
    if name == "Iter":
        return OD_ITER
    if name == "Wilcoxon":
        raise Error(
            "od_type=Wilcoxon is NOT PORTED: it needs"
            " NStatistics::Wilcoxon from library/cpp/statistics, outside"
            " catboost/. Use IncToDec (their default) or Iter."
        )
    raise Error(
        "unknown od_type '" + name + "': None, IncToDec, Iter"
    )


def od_type_name(od_type: Int) raises -> String:
    """The inverse, their `ToString(EOverfittingDetectorType)`.

    It exists because the option layer resolves a type and `train` takes a
    spelling, and a resolved type that could not be spelled would have to
    be passed as a bare integer through a signature whose whole point is
    to be readable in a stack trace.
    """
    if od_type == OD_NONE:
        return String("None")
    if od_type == OD_INC_TO_DEC:
        return String("IncToDec")
    if od_type == OD_ITER:
        return String("Iter")
    raise Error("unknown od_type code " + String(od_type))


@fieldwise_init
struct OverfittingDetector(Copyable, Movable):
    """`TOverfittingDetectorIncToDec` (`overfitting_detector.cpp:120-183`),
    which is also `Iter` at threshold 1.0 and `None` at threshold 0.

    Build with `make_overfitting_detector`, which is their factory
    (`:186-207`).
    """

    var threshold: Float64
    var max_is_optimal: Bool
    var iterations_wait: Int
    var is_empty: Bool
    var current_p_value: Float64

    var errors: List[Float64]
    var local_max: Float64
    var expected_inc: Float64
    var last_error: Float64
    var iterations_from_local_max: Int

    #: NO CATBOOST COUNTERPART. Theirs tracks the best iteration in the
    #: training loop rather than in the detector (`train.cpp`'s
    #: `TLearnProgress`); ours keeps it here because the detector is the
    #: only object that already knows when a new best arrived, and a
    #: caller that wants `use_best_model` needs the index.
    var best_iteration: Int
    var best_error: Float64
    var seen: Int

    def is_active(self) -> Bool:
        """`IsActive()` (`:66-68`)."""
        return self.threshold > 0.0

    def is_need_stop(self) -> Bool:
        """`IsNeedStop()` (`:62-64`)."""
        return (not self.is_empty) and (
            self.current_p_value < self.threshold
        )

    def add_error(mut self, raw_err: Float64):
        """`AddError` (`:127-157`), copied.

        `raw_err` is the loss as the caller measures it -- POSITIVE and
        falling. The negation for `!MaxIsOptimal` is theirs and happens
        here, on entry, exactly as at `:130-131`.
        """
        # their `best iteration` bookkeeping, which is ours
        if self.seen == 0 or raw_err < self.best_error:
            self.best_error = raw_err
            self.best_iteration = self.seen
        self.seen += 1

        if self.threshold <= 0.0:
            return

        var err = raw_err
        if not self.max_is_optimal:
            err = -err

        if self.is_empty or err > self.local_max:
            if self.is_empty:
                self.is_empty = False
                self.expected_inc = 0.0
            self.local_max = err
            self.iterations_from_local_max = 0
        else:
            self.iterations_from_local_max += 1

        # `Errors.push_front(err)`, capped at ITERATION_FORGET (`:145-149`)
        self.errors.insert(0, err)
        if len(self.errors) > OD_ITERATION_FORGET:
            _ = self.errors.pop()

        # `:151-156`: the decaying maximum of `err - Errors[i]`, which is
        # how much the error could still be expected to improve.
        self.expected_inc *= OD_LAMBDA_FORGET
        var cur_mult = Float64(1.0)
        for i in range(len(self.errors)):
            var cand = cur_mult * (err - self.errors[i])
            if cand > self.expected_inc:
                self.expected_inc = cand
            cur_mult *= OD_LAMBDA_FORGET

        self.last_error = err
        self._update_p_value()

    def _update_p_value(mut self):
        """`UpdatePValue` (`:166-173`), term for term:

            if (IterationsFromLocalMax >= IterationsWait) {
                CurrentPValue = ExpectedInc / Max(LocalMax - LastError, EPS);
                CurrentPValue = exp(-LAMBDA_SCALE / Max(CurrentPValue, EPS));
            } else {
                CurrentPValue = 1.0;
            }

        NOTE THE 1.0 IN THE ELSE ARM: it is what makes `Iter` work as
        `IncToDec` at threshold 1.0, since `IsNeedStop` is a STRICT `<`.
        """
        if self.iterations_from_local_max >= self.iterations_wait:
            var denom = self.local_max - self.last_error
            if denom < OD_EPS:
                denom = OD_EPS
            var p = self.expected_inc / denom
            if p < OD_EPS:
                p = OD_EPS
            self.current_p_value = exp(-OD_LAMBDA_SCALE / p)
        else:
            self.current_p_value = 1.0


def make_overfitting_detector(
    od_type: Int,
    max_is_optimal: Bool,
    threshold: Float64,
    iterations_wait: Int,
    has_test: Bool,
) raises -> OverfittingDetector:
    """`CreateOverfittingDetector` (`overfitting_detector.cpp:186-207`).

        None      -> TNoOverfittingDetector          (threshold 0)
        IncToDec  -> IncToDec at the caller's threshold
        Iter      -> IncToDec AT THRESHOLD 1.0, their `:194-197`

    and every arm passes `hasTest ? threshold : 0` (`:122-124`), so a
    detector with no test set is inert.
    """
    var t = threshold
    if od_type == OD_NONE:
        t = 0.0
    elif od_type == OD_ITER:
        # `MakeHolder<TOverfittingDetectorIncToDec>(maxIsOptimal, 1.0, ...)`
        t = 1.0
    elif od_type != OD_INC_TO_DEC:
        raise Error("unknown od_type " + String(od_type))

    if not has_test:
        t = 0.0
    if t < 0.0:
        raise Error(
            "Auto-stop PValue in OD-detector should be >= 0"
        )

    return OverfittingDetector(
        t, max_is_optimal, iterations_wait,
        True, 1.0,
        List[Float64](), 0.0, 0.0, 0.0, 0,
        0, 0.0, 0,
    )
