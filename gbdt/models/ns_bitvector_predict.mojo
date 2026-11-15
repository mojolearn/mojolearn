# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Depthwise / Lossguide bitvector apply switch, kept as a constant.

The bitvector (QuickScorer style) apply of a non-symmetric ensemble was an
opt-in arm that was never tested on a device or measured; it was deleted on
2026-11-15 (docs/TOMBSTONES.md). `GBDT_NS_BITVEC` stays False for the
modules that still import it.
"""

# TOMBSTONE: MOJOLEARN_GBDT_NS_PREDICT_BITVEC (DELETE-ARM, opt-in arm, never tested or measured, rule a) deleted 2026-11-15 by lane/expdel2-44-trees; code recoverable at 4cdd684efa.
# Restore: git apply experiments/removed/MOJOLEARN_GBDT_NS_PREDICT_BITVEC.patch; record in docs/TOMBSTONES.md.
#: The switch, folded: the incumbent packed apply is the only path.
comptime GBDT_NS_BITVEC = False
