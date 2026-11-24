# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The operation words of `metric_slots`, the device-input form of the
`_mojolearn_metrics` kernel entries (metrics/slot_entry_device.mojo has the
table of their inputs and words). A module of constants only, so the doors
(bindings/mcls_*_door.mojo, in the family host owner) name an operation
without importing the device bodies."""

comptime MS_ACCURACY = 0
comptime MS_ACCURACY_W = 1
comptime MS_RAND = 2
comptime MS_ARI = 3
comptime MS_MI = 4
comptime MS_FM = 5
comptime MS_HOMOG = 6
comptime MS_COMPL = 7
comptime MS_V_MEASURE = 8
comptime MS_HCV = 9
comptime MS_ENTROPY = 10
comptime MS_R2 = 11
comptime MS_R2_W = 12
comptime MS_MSE = 13
comptime MS_MAE = 14
comptime MS_RMSE = 15
comptime MS_KL = 16
comptime MS_ROC_AUC = 17
comptime MS_PR_CURVE = 18
comptime MS_LOG_LOSS = 19
comptime MS_CONFUSION = 20
comptime MS_PRFS = 21
comptime MS_SILHOUETTE = 22
comptime MS_TRUST = 23
comptime MS_ENCODE = 24
#: lane vdf-xm-wire: the device union step of a label pair (both vectors' codes over
#: the sorted union of their values) and the device form of a door's class-order map
comptime MS_ENCODE_PAIR = 25
comptime MS_GATHER = 26
comptime MS_OP_COUNT = 27
