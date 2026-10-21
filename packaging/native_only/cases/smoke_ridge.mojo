# SPDX-License-Identifier: Apache-2.0
# Native ridge fit at three regularization strengths against independent float64 closed form.
from glm.checks.ridge_check import check_ridge_matches_closed_form


def main() raises:
    check_ridge_matches_closed_form()
    print("NATIVE_ONLY_PASS ridge")
