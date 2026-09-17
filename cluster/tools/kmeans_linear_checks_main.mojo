# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Lane kmeans-linear-speed (2026-09-17): the checks of DEVIATIONS 3080 and
# 3081, ONE PER PROCESS. `cluster/kmeans_main.mojo` runs them too, but on the
# RunPod RTX 4090 box of 2026-09-17 (driver 580.159.04) the SECOND check of
# any process that opens a fresh DeviceContext per check never returns from
# its first device allocation, whichever check it is (seen with
# `check_privatized_accumulate` then `check_blocked_accumulate` and with the
# two swapped), so that file cannot finish there.
#
#   mojo build -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 cluster/tools/kmeans_linear_checks_main.mojo -o kls_checks
#   ./kls_checks blocked ; ./kls_checks scale ; ./kls_checks privatized
from std.sys import argv

from cluster.checks.estimator_check import check_plan_sum_scale_certified
from cluster.checks.kmeans_check import (
    check_blocked_accumulate,
    check_privatized_accumulate,
)


def main() raises:
    var args = argv()
    if len(args) != 2:
        raise Error("usage: kls_checks <blocked|scale|privatized>")
    var which = String(args[1])
    if which == "blocked":
        check_blocked_accumulate()
    elif which == "scale":
        check_plan_sum_scale_certified()
    elif which == "privatized":
        check_privatized_accumulate()
    else:
        raise Error("unknown check " + which)
