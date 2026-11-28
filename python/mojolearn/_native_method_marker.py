# SPDX-License-Identifier: Apache-2.0
"""The marker of a public method whose compiled owner does the whole call."""
# Moved here unchanged from `_native_robust_bridge.py` (2026-11-28, lane
# lane-loads-2). The k-NN, IVF, ANN, agglomerative and kernel doors import this
# decorator; while it lived in the RobustScaler bridge, which names the x_prep
# binding, every one of those doors counted as loading the preprocessing
# family. This module names no binding: keep it that way. The robust bridge
# re-exports the name, the same object.


def native_method(method):
    # glue: only these complete methods own output/fitted/input handling;
    # application subclass overrides retain the ordinary compatibility wrappers.
    method._mojolearn_public_out = True
    method._mojolearn_tracks_fit = True
    method._mojolearn_needs_fit = True
    method._mojolearn_validates = True
    return method
