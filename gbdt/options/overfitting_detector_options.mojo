"""The overfitting detector's options, and the dispatch that picks its type.

PORT OF `catboost/private/libs/options/overfitting_detector_options.{h,cpp}`
at CatBoost `54a8143a`. Transliterated. Do not improve.

## Why a file for three numbers

Because the TYPE IS INFERRED FROM WHICH OF THE OTHER TWO THE CALLER SET, and
nothing about the three fields says so. Their `Load` (`:24-32`) reads:

    no "type" given:
        "stop_pvalue" present    ->  IncToDec
        "wait_iterations" present -> Iter
        neither                  ->  None

so `od_wait=20` alone is early stopping and `od_pvalue=0.01` alone is a
DIFFERENT detector, while passing neither trains to `n_estimators` no matter
what the held-out curve does. A port that defaulted the type to their
constructor's `IncToDec` (`:10`) and stopped there would stop early on every
fit that named an eval set -- the constructor default exists to be overwritten
by `Load`, and reading it as the effective default is reading half the file.

`stop_pvalue`'s constructor default is 0 (`:9`), and 0 is what makes IncToDec
inert: `IsActive()` is `threshold > 0` (`overfitting_detector.h:66-68`). So
their two defaults do not conflict; they compose into "off".

The `Iter` + `stop_pvalue` refusal (`:35-40`) is theirs, and their comment
explains the escape hatch: the default value is SERIALIZED, so a round-tripped
config carries `stop_pvalue: 0` it never asked for, and only a NON-ZERO one is
an error.
"""

from gbdt.overfitting_detector.overfitting_detector import (
    OD_INC_TO_DEC,
    OD_ITER,
    OD_NONE,
    od_type_from_name,
)

#: `TOverfittingDetectorOptions` (`:8-12`): `stop_pvalue` 0, `type`
#: IncToDec, `wait_iterations` 20. NOTE that the default p-value is ZERO,
#: so their DEFAULT detector is INACTIVE -- `IsActive()` is
#: `Threshold > 0`. CatBoost does not early-stop unless asked, and neither
#: does this.
#:
#: These two lived in `gbdt/overfitting_detector/overfitting_detector.mojo`
#: until 2026-08-21 and cited THIS file's line numbers while doing so,
#: which is a mirror address that names its own defect.
comptime OD_DEFAULT_STOP_PVALUE = Float64(0.0)
comptime OD_DEFAULT_WAIT_ITERATIONS = 20


@fieldwise_init
struct TOverfittingDetectorOptions(Copyable, ImplicitlyCopyable, Movable):
    """Their `TOverfittingDetectorOptions`, already resolved.

    `od_type` is one of the `OD_*` codes, never a string: the string
    spellings are theirs and `od_type_from_name` is where they are read.
    """

    var auto_stop_p_value: Float64
    var od_type: Int
    var iterations_wait: Int

    def validate(self) raises:
        """`Validate()` (`:47-50`), both `CB_ENSURE`s."""
        if self.iterations_wait <= 0:
            raise Error(
                "Wait iterations in OD-detector should be > 0, got "
                + String(self.iterations_wait)
            )
        if self.auto_stop_p_value < 0.0:
            raise Error(
                "Auto-stop PValue in OD-detector should be >= 0, got "
                + String(self.auto_stop_p_value)
            )


def load_overfitting_detector_options(
    type_name: String,
    p_value: Float64,
    wait_iterations: Int,
) raises -> TOverfittingDetectorOptions:
    """`Load` (`:24-40`), where an EMPTY / negative argument is their
    `options.Has(...)` returning false.

    The three "was it given" flags their JSON carries have to be encoded
    somehow on this side, and this is the encoding: an empty `type_name`,
    a negative `p_value` and a negative `wait_iterations` each mean the
    caller said nothing. Zero is a REAL p-value (it is their own default
    and it means "inert"), which is why the sentinel is negative rather
    than zero.
    """
    var has_type = type_name.byte_length() > 0
    var has_p_value = p_value >= 0.0
    var has_wait = wait_iterations >= 0

    var resolved_type: Int
    if has_type:
        resolved_type = od_type_from_name(type_name)
    elif has_p_value:
        resolved_type = OD_INC_TO_DEC
    elif has_wait:
        resolved_type = OD_ITER
    else:
        resolved_type = OD_NONE

    var opts = TOverfittingDetectorOptions(
        p_value if has_p_value else OD_DEFAULT_STOP_PVALUE,
        resolved_type,
        wait_iterations if has_wait else OD_DEFAULT_WAIT_ITERATIONS,
    )

    # `:35-40`. Only a NON-ZERO p-value is the error, for the reason their
    # comment gives: the default is serialized, so a config that round
    # tripped carries a `stop_pvalue` of 0 nobody asked for.
    if opts.od_type == OD_ITER and has_p_value and p_value != 0.0:
        raise Error(
            "Auto-stop PValue is not a valid parameter for Iter"
            " overfitting detector."
        )

    opts.validate()
    return opts
