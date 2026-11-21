# SPDX-License-Identifier: Apache-2.0
"""PACK of a RandomForest / DecisionTree fit's notices (the third export
metadata word, an `ensemble/rf_notices.mojo` bit mask) as Python
UserWarnings, in the order the fit reached them. One function, read by the
estimator's own fit door (`trees_public_forest_door._rf_fit_notices`), the
forest member PACK (`forest_member_pack_common`) and the model-selection
member units (`msel_member_units_door`), so every route raises the same
words in the same category (lane fx-audit, audit-trees F1).

SERIAL BY NATURE: at most three warnings per fit, GIL held."""
from std.python import PythonObject
from bindings.classical_warnings_unpack import classical_warning
from bindings.metrics_native_support_door import metric_warning_location
from bindings.trees_public_inputs_door import trees_builtins
from ensemble.rf_notices import rf_notice_text
from xtrees.trees_forest_run import forest_fit_notice_codes


def rf_fit_notices_word(word: Int, context: PythonObject) raises:
    """Raise the notices of `word` (0: none, nothing is read)."""
    if word == 0:
        return
    var codes = forest_fit_notice_codes(word)
    for i in range(len(codes)):  # small-loop(codes): at most the three fit notice bits (PACK)
        classical_warning(rf_notice_text(codes[i]), trees_builtins().UserWarning, context,
                          metric_warning_location(2), "UserWarning")
