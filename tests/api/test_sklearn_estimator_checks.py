# SPDX-License-Identifier: Apache-2.0
"""scikit-learn's own estimator checks over every Mojolearn class that claims a
scikit-learn twin: a test of the Python boundary (construction, parameters,
fitted state, input admission, output shapes and errors), not of numerics.

The classes are the twins of the generated table docs/api/sklearn_twins.json
(`python3 tools/api_twin_table.py`), so the table and the checks cover the
same names; `test_the_table_and_the_checked_classes_agree` fails when they
drift. The instances, the expected failures (check name and reason, every
entry explicit) and the per-check instances are the ones
python/mojolearn/tests/test_sklearn_compat.py defines, imported from there so
that there is one list: `_make` (scikit-learn's own small test parameters),
`_expected_failed_checks` and `_instance_for_check`. The classes this file
cannot check are listed in NOT_CHECKED with their reason.

Fitting needs a device or a CPU training binding, so the checks run only with
`MOJOLEARN_RUN_SKLEARN_CHECKS=1` (as in test_sklearn_compat.py). One pytest
process per class, all cores: tools/sklearn_checks_per_estimator.sh.
`MOJOLEARN_SKLEARN_CHECKS_ONLY=Name[,Name]` restricts the run to those classes.
"""
import json
import os
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from mojolearn.tests import test_sklearn_compat as C  # noqa: E402

TABLE = Path(__file__).resolve().parents[2] / "docs" / "api" / "sklearn_twins.json"
TWINS = sorted(json.loads(TABLE.read_text(encoding="utf-8"))["classes"])

#: Twins whose fitting checks are not run here, each with its reason.
NOT_CHECKED = {name: ("float32-only input contract: a float64 array raises TypeError, and "
                      "scikit-learn's checks generate float64 data; parameters, tags and clone are "
                      "checked by test_sklearn_compat.py")
               for name in C.FLOAT32_ONLY}
NOT_CHECKED.update({name: "not a scikit-learn-style estimator here (test_sklearn_compat.NOT_SKLEARN_STYLE)"
                    for name in C.NOT_SKLEARN_STYLE})
NOT_CHECKED.update({name: "not verified (test_sklearn_compat.UNVERIFIED)" for name in C.UNVERIFIED})


def test_the_table_and_the_checked_classes_agree():
    """Every twin in the table is a public class scikit-learn's checks can see, and back."""
    shared = set(C.SHARED_NAMES)
    in_table_only = sorted(set(TWINS) - shared - set(NOT_CHECKED))
    in_package_only = sorted(shared - set(TWINS))
    assert not in_table_only and not in_package_only, (
        f"regenerate docs/api (python3 tools/api_twin_table.py): table only {in_table_only}, "
        f"package only {in_package_only}")


def test_every_expected_failure_names_a_reason():
    for name in TWINS:
        if name in NOT_CHECKED:
            continue
        for check, reason in C._expected_failed_checks(C._make(name)).items():
            assert check.startswith("check_") and isinstance(reason, str) and reason.strip(), (name, check)


def _checked():
    only = os.environ.get("MOJOLEARN_SKLEARN_CHECKS_ONLY")
    only = set(only.split(",")) if only else None
    # The instances are built while collecting (one per class): a constructor that
    # raises is a failure of test_sklearn_compat.test_params_clone_and_tags, so it is
    # left out here rather than hiding the other classes' checks.
    out = []
    for name in TWINS:
        if name in NOT_CHECKED or (only is not None and name not in only):
            continue
        try:
            out.append(C._make(name))
        except Exception:
            continue
    return out


if C.RUN_FITS:
    from sklearn.utils.estimator_checks import parametrize_with_checks

    @parametrize_with_checks(_checked(), expected_failed_checks=C._expected_failed_checks)
    def test_scikit_learn_estimator_checks(estimator, check):
        check(C._instance_for_check(estimator, check))
else:
    @pytest.mark.skip(reason=C.FIT_REASON)
    def test_scikit_learn_estimator_checks():
        pass
