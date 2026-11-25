# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# The composed-model verify lane's runner arms (lane session-ops-verify, 2026-11-25).
#
# The lane ops-composed-cases (tools/identity_lanes/composed_cases.py, cases python/mojolearn/verify_reference/
# composed-*.json) runs data-driven composed models on both sides. This file adds no operation: it holds only the
# sabotage switches of the runner's arms (tools/identity_lanes/coverage_sabotage.tsv
# coverage-autograd-session_ops_verify_drop_accumulate_*). Off in every build; each arm's patch switches one on, so
# the product build is unchanged. The plain name faults the device executor only, _HOST the host executor only.
#   DROP_ACCUMULATE: Tape._backward (autograd/tape.mojo, group session-ops-verify) skips the accumulate of every
#   gradient contribution after a tensor's first, so a tensor used on two paths keeps only one path's gradient.
#   The case tape-two-paths REFUSES (its gradient of x differs from central differences); the other cases move.
from std.sys.compile import is_defined

comptime SESSION_OPS_VERIFY_SABOTAGE_DROP_ACCUMULATE = is_defined["MOJOLEARN_SESSION_OPS_VERIFY_SABOTAGE_DROP_ACCUMULATE"]()
comptime SESSION_OPS_VERIFY_SABOTAGE_DROP_ACCUMULATE_HOST = is_defined[
    "MOJOLEARN_SESSION_OPS_VERIFY_SABOTAGE_DROP_ACCUMULATE_HOST"
]()
