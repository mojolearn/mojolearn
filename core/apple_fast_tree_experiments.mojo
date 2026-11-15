# SPDX-License-Identifier: Apache-2.0
"""P01–P12, experiments/apple_fast_trees/IDEAS.md: the programme is deleted.

Every card was never run (no A/B, no measurement). The names other files
still import stay here as the folded constants of the deleted cards (each
card off), so those files compile until their own lanes fold them; the
integrator deletes this file when nothing imports it any more.
"""
# TOMBSTONE: MOJOLEARN_AFT_P01 MOJOLEARN_AFT_P02 MOJOLEARN_AFT_P03 MOJOLEARN_AFT_P04 MOJOLEARN_AFT_P05 MOJOLEARN_AFT_P06
# MOJOLEARN_AFT_P07 MOJOLEARN_AFT_P08 MOJOLEARN_AFT_P09 MOJOLEARN_AFT_P10 MOJOLEARN_AFT_P11 MOJOLEARN_AFT_P12
# (the never-run Apple FAST trees programme P01-P12, experiments/apple_fast_trees/P.json) deleted 2026-11-15 by
# lane/expdel2-03-base; code recoverable at 5bc2865240.
# Restore: git apply experiments/removed/MOJOLEARN_AFT_P<nn>.patch (one per card); record in docs/TOMBSTONES.md.

# P05 (gbdt/models/kernel/add_bin_values.mojo, gbdt/resident_model.mojo),
# P06 (gbdt/models/kernel/resident_link.mojo), P07-P09 and P10's fold width
# (xtrees/shap_device.mojo), P11/P12's row tiles (xtrees/dart_device.mojo).
comptime AFT_P05 = False
comptime AFT_P06 = False
comptime AFT_P07 = False
comptime AFT_P08 = False
comptime AFT_P09 = False
comptime AFT_SHAP_FOLD_ITEMS = 1
comptime AFT_DART_ROWS = 1
comptime AFT_DART_ADD_ROWS = 1
