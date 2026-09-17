# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Lane kmeans-linear-speed (2026-09-17): the two checks of DEVIATIONS 3080 and
# 3081 alone, for a box where the whole of `cluster/kmeans_main.mojo` is too
# long to wait on. `cluster/kmeans_main.mojo` runs them too.
#
#   tools/with_identical_mode.sh pixi run mojo run -I . cluster/tools/kmeans_linear_checks_main.mojo
from cluster.checks.estimator_check import check_plan_sum_scale_certified
from cluster.checks.kmeans_check import (
    check_blocked_accumulate,
    check_privatized_accumulate,
)


def main() raises:
    check_privatized_accumulate()
    check_blocked_accumulate()
    check_plan_sum_scale_certified()
