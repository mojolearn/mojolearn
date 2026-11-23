# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# THE CLASSIC ATTENTION IDENTITY LANES (lane nn-attention, 2026-11-23).
#
# A fragment of tools/identity_break.py (loaded by `_load_lane_fragments` into its namespace; the
# helpers `_sequence_tape_*` and `_fit` are that namespace's, read when a lane runs). Each lane fits a
# Sequential built from multihead_attention, positional_encoding and the Transformer encoder and decoder
# layers one seeded step and digests the loss, the parameters and the predictions.

_NN_ATTENTION_ROWS = 16


def _nn_attention_run(ml, shape, build, stream):
    B = _sequence_tape_binding(ml)
    width = int(np.prod(shape))
    x = _sequence_tape_seeded(B, (_NN_ATTENTION_ROWS, width), 31, stream)
    xh = _sequence_tape_seeded(B, (_NN_ATTENTION_ROWS, width), 37, stream)
    y = _sequence_tape_seeded(B, (_NN_ATTENTION_ROWS, 1), 41, stream)
    m = build(ml.Sequential(numeric_mode="identical", input_shape=shape))
    m = m.compile(loss="mse", optimizer="adam", lr=0.01)
    m = m.fit(x, y, epochs=1, batch_size=_NN_ATTENTION_ROWS, shuffle=False, seed=0)
    d = lambda a: _sequence_tape_digest(B, a)
    return _fit(dict(loss=d(np.asarray(m.loss_curve_, dtype=np.float32)), params=d(m.params_),
                     pred=d(m.predict(x)), held_pred=d(m.predict(xh))))


@lane("sequence-attention-encoder")
def _(ml, X, yc, yr, Xh=None):
    """(6, 4) samples: linear(8), positional_encoding, a post-norm encoder layer (2 heads, causal) and a
    pre-norm one (gelu), flatten, linear(1); dropout 0.1 in training."""
    return _nn_attention_run(ml, (6, 4), lambda m: m.linear(8).positional_encoding(8)
                             .transformer_encoder_layer(8, 2, 16, is_causal=True)
                             .transformer_encoder_layer(8, 2, 16, activation="gelu", norm_first=True)
                             .flatten().linear(1), 6)


@lane("sequence-attention-cross")
def _(ml, X, yc, yr, Xh=None):
    """(6, 4) samples: linear(8) (layer 0, the memory), then a decoder layer (2 heads, causal self-attention,
    attention to layer 0) and a multihead_attention to the model input (kdim 4), flatten, linear(1)."""
    return _nn_attention_run(ml, (6, 4), lambda m: m.linear(8)
                             .transformer_decoder_layer(8, 2, 16, dropout=0.0, memory=0, tgt_is_causal=True)
                             .multihead_attention(8, 2, kdim=4, vdim=4, source=-1)
                             .flatten().linear(1), 7)


_batch_decl("n/a:training-step (one seeded training step of a composed model through the tape; its forward "
            "outputs are train parts, digested whole)", "sequence-attention-encoder", "sequence-attention-cross")
