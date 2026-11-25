# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# The sabotage switches of the session-ops layer lanes (lane fable-review-2, 2026-11-25; arms
# coverage-autograd-session_ops_rest_* and coverage-autograd-session_ops_resize_pixel_shuffle_* in
# tools/identity_lanes/coverage_sabotage.tsv). Off in every build; each arm's patch switches one on, so the product
# build is unchanged. The plain name faults the device executor only, _HOST the host executor only. Every fault sits
# in the SESSION op's own glue (autograd/session_ops_rest.mojo, session_ops_resize.mojo), not in a body Sequential
# shares, so an arm moves its own lane's digest and no other lane's (they replace the two generic one-ulp
# arithmetic arms, which moved every lane of the side at once and so showed nothing about any one of them):
#   NORM_EPS (ops-layer-norms): batch_norm normalizes with eps + 1 (every batch_norm output, its gradient and the Program part move).
#   MISH_SCALE (ops-layer-activations): mish returns twice its value (the mish part and the gradient of x move).
#   MHA_CAUSAL (ops-layer-attention): multi_head_attention drops is_causal (a query attends to later keys: the self_causal part, the gradients and the Program part move).
#   SMOOTH_L1_BETA (ops-layer-losses): smooth_l1_loss uses beta + 1 (its value and gradient move).
#   CTC_REDUCTION (ops-layer-ctc): ctc_loss swaps 'mean' and 'sum' (the loss, the gradient and the Program part move).
#   SSM_SCALE (ops-layer-ssm): selective_scan returns twice its output (the selective_scan part and its gradients move).
#   PIXEL_SHUFFLE (ops-resize): pixel_shuffle / pixel_unshuffle return twice the rearranged values (the shuffle part moves).
from std.sys.compile import is_defined

comptime SOL_SABOTAGE_NORM_EPS = is_defined["MOJOLEARN_SESSION_OPS_LAYER_SABOTAGE_NORM_EPS"]()
comptime SOL_SABOTAGE_NORM_EPS_HOST = is_defined["MOJOLEARN_SESSION_OPS_LAYER_SABOTAGE_NORM_EPS_HOST"]()
comptime SOL_SABOTAGE_MISH_SCALE = is_defined["MOJOLEARN_SESSION_OPS_LAYER_SABOTAGE_MISH_SCALE"]()
comptime SOL_SABOTAGE_MISH_SCALE_HOST = is_defined["MOJOLEARN_SESSION_OPS_LAYER_SABOTAGE_MISH_SCALE_HOST"]()
comptime SOL_SABOTAGE_MHA_CAUSAL = is_defined["MOJOLEARN_SESSION_OPS_LAYER_SABOTAGE_MHA_CAUSAL"]()
comptime SOL_SABOTAGE_MHA_CAUSAL_HOST = is_defined["MOJOLEARN_SESSION_OPS_LAYER_SABOTAGE_MHA_CAUSAL_HOST"]()
comptime SOL_SABOTAGE_SMOOTH_L1_BETA = is_defined["MOJOLEARN_SESSION_OPS_LAYER_SABOTAGE_SMOOTH_L1_BETA"]()
comptime SOL_SABOTAGE_SMOOTH_L1_BETA_HOST = is_defined["MOJOLEARN_SESSION_OPS_LAYER_SABOTAGE_SMOOTH_L1_BETA_HOST"]()
comptime SOL_SABOTAGE_CTC_REDUCTION = is_defined["MOJOLEARN_SESSION_OPS_LAYER_SABOTAGE_CTC_REDUCTION"]()
comptime SOL_SABOTAGE_CTC_REDUCTION_HOST = is_defined["MOJOLEARN_SESSION_OPS_LAYER_SABOTAGE_CTC_REDUCTION_HOST"]()
comptime SOL_SABOTAGE_SSM_SCALE = is_defined["MOJOLEARN_SESSION_OPS_LAYER_SABOTAGE_SSM_SCALE"]()
comptime SOL_SABOTAGE_SSM_SCALE_HOST = is_defined["MOJOLEARN_SESSION_OPS_LAYER_SABOTAGE_SSM_SCALE_HOST"]()
comptime SOL_SABOTAGE_PIXEL_SHUFFLE = is_defined["MOJOLEARN_SESSION_OPS_LAYER_SABOTAGE_PIXEL_SHUFFLE"]()
comptime SOL_SABOTAGE_PIXEL_SHUFFLE_HOST = is_defined["MOJOLEARN_SESSION_OPS_LAYER_SABOTAGE_PIXEL_SHUFFLE_HOST"]()
