"""`nan_mode`, and what it means for a feature that has no NaNs.

PORT OF the `NanMode` corner of
`catboost/private/libs/options/data_processing_options.{h,cpp}` at CatBoost
`54a8143a`, plus `ENanMode` (`enums.h:107-111`). Transliterated. Do not
improve.

## The option is a REQUEST, not a fact

`ENanMode` has three values and only two of them can be chosen:

    Min        NaN goes to the LOWEST bin, below every real value
    Max        NaN goes to the HIGHEST bin, above every real value
    Forbidden  there are no NaNs, and one in a test row is an error

`Forbidden` is not something a caller asks for -- it is what a column
RESOLVES to when it contains no NaN, whatever the option said.
`ComputeNanMode` (`libs/data/quantized_features_info.cpp:202-239`) is that
resolution:

    if (option == Forbidden) return Forbidden;      // asked for, and honoured
    hasNans = <scan the column>;
    if (hasNans) return option;
    return Forbidden;                                // nothing to treat

So `nan_mode` is PER FEATURE and DATA DEPENDENT, and two columns of one fit
routinely end up in different modes. A port that carried one mode for the
whole dataset would spend a border on a NaN bin in every column, including
the ones with no NaN at all.

**Their default is `Min`** (`data_processing_options.cpp:17`), which is why
a CatBoost user who never heard of `nan_mode` still gets NaNs handled: they
land in a bin of their own, below everything, and the tree learns what to do
with them.
"""

#: `ENanMode` (`enums.h:107-111`)
comptime NAN_MODE_MIN = 0
comptime NAN_MODE_MAX = 1
comptime NAN_MODE_FORBIDDEN = 2

#: `NanMode(ENanMode::Min)` (`data_processing_options.cpp:15-17`)
comptime DEFAULT_NAN_MODE = NAN_MODE_MIN


def nan_mode_from_name(name: String) raises -> Int:
    """Their `ENanMode` spellings."""
    if name == "Min" or name == "":
        return NAN_MODE_MIN
    if name == "Max":
        return NAN_MODE_MAX
    if name == "Forbidden":
        return NAN_MODE_FORBIDDEN
    raise Error(
        "unknown nan_mode '" + name + "': Min, Max, Forbidden"
    )


def nan_mode_name(nan_mode: Int) raises -> String:
    if nan_mode == NAN_MODE_MIN:
        return String("Min")
    if nan_mode == NAN_MODE_MAX:
        return String("Max")
    if nan_mode == NAN_MODE_FORBIDDEN:
        return String("Forbidden")
    raise Error("unknown nan_mode code " + String(nan_mode))
