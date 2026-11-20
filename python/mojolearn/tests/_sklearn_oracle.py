# SPDX-License-Identifier: Apache-2.0
"""The scikit-learn version a reference comparison runs against.

The prep and metrics comparisons follow the scikit-learn that Python 3.11 and
later resolve (1.9). Python 3.10's newest is 1.7.2, which lacks parameters and
functions added since (QuadraticDiscriminantAnalysis ``solver``,
SplineTransformer ``handle_missing``, ``d2_brier_score``, a splitter object as
TargetEncoder ``cv``) and has different results for others (LDA/QDA covariance
scaling, SimpleImputer ``constant`` on an all-missing column, the
KBinsDiscretizer quantile default, OneHotEncoder ``handle_unknown='warn'``).
Mojolearn's results on Python 3.10 equal 1.9.1's for every one of these; a
comparison that depends on one skips on an older reference and says why.
"""
import pytest
import sklearn

SKLEARN_VERSION = tuple(int(part) for part in sklearn.__version__.split(".")[:2] if part.isdigit())
# The version the comparisons are written against (see the module docstring).
REFERENCE = (1, 9)


def older_reference():
    """True when the installed scikit-learn is older than the comparisons' reference."""
    return SKLEARN_VERSION < REFERENCE


def needs_reference(what):
    """A skipif mark for a comparison that depends on scikit-learn 1.9 behaviour."""
    return pytest.mark.skipif(
        older_reference(),
        reason=f"reference scikit-learn {sklearn.__version__} predates 1.9: {what}")
