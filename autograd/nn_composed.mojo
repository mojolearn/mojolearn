# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# The composed tagger's fault-injection switches (lane nn-blocks-composed, 2026-11-25).
#
# The composed tagger (tools/identity_lanes/nn_composed.py, lane ops-composed-tagger; native check
# checks/native_tests/nn_composed.mojo) puts the nn-blocks pieces together in one model: an embedding concatenated
# with per-step float features, a captured cell with a dropout inside it run by Session.scan, a weighted per-step
# masked cross-entropy, and an optimizer whose parameter groups hold the embedding and the head apart; it trains,
# saves, resumes in a fresh session and checks that the next update equals uninterrupted training bit for bit.
# This file adds no operation: it holds only the sabotage switches its arms force on
# (tools/identity_lanes/coverage_sabotage.tsv coverage-autograd-nn_composed_*). Off in every build; each arm's
# patch switches one on. The plain name faults the device executor only, _HOST the host executor only.
#   SCAN_KEY: a dropout inside a scanned cell ignores the scan's key (Session.scan counter=, advanced every minibatch
#   by Program.fit) and keeps the capture-time counter 0, so every training step redraws the first step's masks
#   (autograd/ops_api.mojo OpsSession._scan, group nn-blocks-composed);
#   GROUP_STATE: restoring a grouped optimizer's state (load_optimizer_state / load_optimizer_state_dict) drops the
#   state of every group after the first, so the second group's moment buffers and step count start over
#   (autograd/ops_api.mojo OpsSession.load_optimizer_state, group nn-blocks-composed).
from std.sys.compile import is_defined

comptime COMPOSED_SABOTAGE_SCAN_KEY = is_defined["MOJOLEARN_NN_COMPOSED_SABOTAGE_SCAN_KEY"]()
comptime COMPOSED_SABOTAGE_SCAN_KEY_HOST = is_defined["MOJOLEARN_NN_COMPOSED_SABOTAGE_SCAN_KEY_HOST"]()
comptime COMPOSED_SABOTAGE_GROUP_STATE = is_defined["MOJOLEARN_NN_COMPOSED_SABOTAGE_GROUP_STATE"]()
comptime COMPOSED_SABOTAGE_GROUP_STATE_HOST = is_defined["MOJOLEARN_NN_COMPOSED_SABOTAGE_GROUP_STATE_HOST"]()
