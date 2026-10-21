# SPDX-License-Identifier: Apache-2.0
"""Shared device/host dispatch choice for decomposition variance means."""

# The launch helper's tiled fold is the variance means' only order, on the
# device and in the host column, in every mode (lane debt-block-per-n,
# 2026-10-21). The direct one-block-per-column arm that
# `-D MOJOLEARN_IDN_DECOMP_MEAN_LAUNCH_OFF` selected walked all n rows per
# block; the define is retired and refused in
# core/six_lane_experiment_guards.mojo. The name stays for the host oracle's
# `comptime if`.
comptime IDN_DECOMP_MEAN_LAUNCH = True
