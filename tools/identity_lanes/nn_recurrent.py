# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# THE RECURRENT OPTIONS' IDENTITY LANES (lane nn-recurrent, 2026-11-23).
#
# Loaded by tools/identity_break.py `_load_lane_fragments` like tools/identity_lanes/sequence.py, into the same
# namespace; it reuses that fragment's seeded-step helpers (`_sequence_blocks_data`, `_sequence_blocks_step`),
# which every fragment can call once all are loaded. Registered in python/mojolearn/_surface_sequence.py
# (training_lanes, names, unwatched) and the routine profile (python/mojolearn/_verification_profiles.py).


@lane("sequence-blocks-bigru-lengths")
def _(ml, X, yc, yr, Xh=None):
    """A two-layer bidirectional GRU (hidden 6, dropout 0.25 between the layers, the last layer's final state of
    both directions) over 32 sequences of 8 steps with lengths 1 .. 8: each direction starts at each row's own
    last valid step. One seeded training step: the loss, gradient, parameters and probabilities."""
    x, y, xh = _sequence_blocks_data(X, yc, Xh, False)
    lengths = (np.arange(32) % 8 + 1).astype(np.int64)
    m = (ml.Sequential(input_shape=(8, 4), numeric_mode="identical")
         .gru(6, num_layers=2, bidirectional=True, dropout=0.25, output="final").flatten().linear(2))
    return _sequence_blocks_step(ml, m, x, y, xh, lengths)


@lane("sequence-blocks-bilstm")
def _(ml, X, yc, yr, Xh=None):
    """A bidirectional LSTM (hidden 6) over 32 sequences of 8 steps without lengths, every step's state of both
    directions into a linear head. One seeded training step."""
    x, y, xh = _sequence_blocks_data(X, yc, Xh, False)
    m = (ml.Sequential(input_shape=(8, 4), numeric_mode="identical")
         .lstm(6, bidirectional=True).flatten().linear(2))
    return _sequence_blocks_step(ml, m, x, y, xh)


_batch_decl("n/a:training-step (one seeded training step through the tape; its forward outputs are train "
            "parts, digested whole)", "sequence-blocks-bigru-lengths", "sequence-blocks-bilstm")
