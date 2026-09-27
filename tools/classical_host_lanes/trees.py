# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# THE TREES LANE'S CLASSICAL HOST GATE PROBES (docs/lanes/ALGORITHM_EXPANSION_BRIEFS.md).
#
# Owned by the `trees` expansion lane; merged by tools/classical_host_gate.py
# (`_merge_gate_fragments`). Only needed for a lane the manifest declares as an
# INFERENCE lane (a family's `inference_lanes` in python/mojolearn/_surface_trees.py):
# then it needs an entry here too, in the shapes of the gate's own tables:
#
#     LANES = {"trees-example": ("Example", lambda e, X: (e.transform(X),), {})}
#     PROBE_NAMES = {"trees-example": "transform"}
#     LANE_PROBE_ROWS = {}      # lane -> rows, only when the probe is not PROBE_ROWS
#     KIND_PROBES = ()          # lanes whose probe also takes the fixture kind
LANES = {}
PROBE_NAMES = {}
LANE_PROBE_ROWS = {}
KIND_PROBES = ()
