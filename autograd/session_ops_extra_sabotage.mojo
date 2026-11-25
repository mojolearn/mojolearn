# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# The sabotage switches of sub-lane session-ops-extra's arms (2026-11-25; tools/identity_lanes/coverage_sabotage.tsv
# coverage-autograd-session_ops_extra_*). Off in every build; each arm's patch switches one on, so the product build is
# unchanged. The plain name faults the device executor only, _HOST the host executor only. Lane ops-composed-cases,
# case file python/mojolearn/verify_reference/composed-extra.json, catches each:
#   SEQ_MASK: sequence_mask compares t <= lengths[b] instead of t < lengths[b] (one position too many per row).
#   TOPK_ZERO: topk keeps -0.0 below +0.0 instead of equal (a -0.0 / 0.0 tie no longer goes to the lower position).
#   GATHER_GRAD: gather's backward drops flat cell 0 of x's gradient (the fold treats it as padding).
#   LSE_SETTLE: logsumexp does not settle an infinite maximum (a row of only -inf gives NaN).
#   SOFT_CE: the soft-target cross-entropy loses its minus sign.
from std.sys.compile import is_defined

comptime SOX_SABOTAGE_SEQ_MASK = is_defined["MOJOLEARN_SESSION_OPS_EXTRA_SABOTAGE_SEQ_MASK"]()
comptime SOX_SABOTAGE_SEQ_MASK_HOST = is_defined["MOJOLEARN_SESSION_OPS_EXTRA_SABOTAGE_SEQ_MASK_HOST"]()
comptime SOX_SABOTAGE_TOPK_ZERO = is_defined["MOJOLEARN_SESSION_OPS_EXTRA_SABOTAGE_TOPK_ZERO"]()
comptime SOX_SABOTAGE_TOPK_ZERO_HOST = is_defined["MOJOLEARN_SESSION_OPS_EXTRA_SABOTAGE_TOPK_ZERO_HOST"]()
comptime SOX_SABOTAGE_GATHER_GRAD = is_defined["MOJOLEARN_SESSION_OPS_EXTRA_SABOTAGE_GATHER_GRAD"]()
comptime SOX_SABOTAGE_GATHER_GRAD_HOST = is_defined["MOJOLEARN_SESSION_OPS_EXTRA_SABOTAGE_GATHER_GRAD_HOST"]()
comptime SOX_SABOTAGE_LSE_SETTLE = is_defined["MOJOLEARN_SESSION_OPS_EXTRA_SABOTAGE_LSE_SETTLE"]()
comptime SOX_SABOTAGE_LSE_SETTLE_HOST = is_defined["MOJOLEARN_SESSION_OPS_EXTRA_SABOTAGE_LSE_SETTLE_HOST"]()
comptime SOX_SABOTAGE_SOFT_CE = is_defined["MOJOLEARN_SESSION_OPS_EXTRA_SABOTAGE_SOFT_CE"]()
comptime SOX_SABOTAGE_SOFT_CE_HOST = is_defined["MOJOLEARN_SESSION_OPS_EXTRA_SABOTAGE_SOFT_CE_HOST"]()
# Added by lane fable-finalize (2026-11-25), caught by the same lane and case file (case logsumexp-offset):
#   LSE_GRAD: logsumexp's backward takes exp(x - rounded result) instead of the forward's shifted exponential over its
#     sum (the defect fixed 2026-11-25: scores of 1e8 give a gradient of 1 a column instead of 1 / columns).
comptime SOX_SABOTAGE_LSE_GRAD = is_defined["MOJOLEARN_SESSION_OPS_EXTRA_SABOTAGE_LSE_GRAD"]()
comptime SOX_SABOTAGE_LSE_GRAD_HOST = is_defined["MOJOLEARN_SESSION_OPS_EXTRA_SABOTAGE_LSE_GRAD_HOST"]()
