# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# THE CONVOLUTION AND SHAPE LAYERS' IDENTITY LANES (lane nn-conv, 2026-11-23).
#
# Loaded by tools/identity_break.py `_load_lane_fragments` in the harness namespace, like
# tools/identity_lanes/sequence.py, whose seeded-buffer helpers (`_sequence_tape_*`) these lanes call
# at run time. One lane per module family: each builds a small Sequential from the public layers, fits
# one epoch of one full unshuffled batch (one forward, backward and SGD step through the tape) on seeded
# inputs, and digests the loss, the updated parameters and the predictions on the training rows and on
# seeded held-out rows.

_NN_CONV_ROWS = 8


def _nn_conv_composed(ml, shape, build, stream):
    B = _sequence_tape_binding(ml)
    width = int(np.prod(shape))
    x = _sequence_tape_seeded(B, (_NN_CONV_ROWS, width), 43, stream)
    xh = _sequence_tape_seeded(B, (_NN_CONV_ROWS, width), 47, stream)
    y = _sequence_tape_seeded(B, (_NN_CONV_ROWS, 1), 53, stream)
    m = build(ml.Sequential(numeric_mode="identical", input_shape=shape))
    m = m.compile(loss="mse", optimizer="sgd", lr=0.01)
    m = m.fit(x, y, epochs=1, batch_size=_NN_CONV_ROWS, shuffle=False, seed=0)
    d = lambda a: _sequence_tape_digest(B, a)
    return _fit(dict(loss=d(np.asarray(m.loss_curve_, dtype=np.float32)), params=d(m.params_),
                     pred=d(m.predict(x)), held_pred=d(m.predict(xh))))


@lane("sequence-nnconv-transpose")
def _(ml, X, yc, yr, Xh=None):
    """Transposed convolutions over (2, 8, 8) samples: conv_transpose2d(3, 3, stride 2, padding 1,
    output_padding 1), tanh, conv2d(2, 3, stride 2, padding 1), flatten, linear(1); and over the
    (16,) flat row viewed as (2, 8): conv_transpose1d(4, 4, stride 2, padding 1, groups 2) ahead."""
    return _nn_conv_composed(ml, (2, 8, 8), lambda m: m.conv_transpose2d(3, 3, stride=2, padding=1,
                             output_padding=1).tanh().conv2d(2, 3, stride=2, padding=1).reshape((2, 64))
                             .conv_transpose1d(4, 4, stride=2, padding=1, groups=2).flatten().linear(1), 10)


@lane("sequence-nnconv-pools")
def _(ml, X, yc, yr, Xh=None):
    """Adaptive and global pools over (2, 7, 5) samples: conv2d(3, 3, padding 1), adaptive_avg_pool2d
    (3, 4), adaptive_max_pool2d(2), reshape (3, 4), adaptive_max_pool1d(3), global_max_pool, linear(1)."""
    return _nn_conv_composed(ml, (2, 7, 5), lambda m: m.conv2d(3, 3, padding=1).adaptive_avg_pool2d((3, 4))
                             .adaptive_max_pool2d(2).reshape((3, 4)).adaptive_max_pool1d(3).global_max_pool()
                             .linear(1), 11)


@lane("sequence-nnconv-pads")
def _(ml, X, yc, yr, Xh=None):
    """Every pad over (2, 4, 5) samples: reflection_pad2d(1), replication_pad2d((1, 0, 2, 1)),
    constant_pad2d(1, 0.5), zero_pad2d((-1, 0, 0, -1)), conv2d(2, 3), reshape, reflection_pad1d(2),
    replication_pad1d(1), constant_pad1d((0, 2), -0.25), zero_pad1d(1), flatten, linear(1)."""
    def build(m):
        m = (m.reflection_pad2d(1).replication_pad2d((1, 0, 2, 1)).constant_pad2d(1, 0.5)
             .zero_pad2d((-1, 0, 0, -1)).conv2d(2, 3))
        return (m.reshape((2, -1)).reflection_pad1d(2).replication_pad1d(1).constant_pad1d((0, 2), -0.25)
                .zero_pad1d(1).flatten().linear(1))
    return _nn_conv_composed(ml, (2, 4, 5), build, 12)


@lane("sequence-nnconv-upsample")
def _(ml, X, yc, yr, Xh=None):
    """Upsampling over (2, 3, 4) samples: nearest x2, conv2d(2, 3, padding 1), bilinear to (9, 11),
    bilinear x1.5 with align_corners, reshape (2, -1), linear upsample to length 40, flatten,
    linear(1)."""
    return _nn_conv_composed(ml, (2, 3, 4), lambda m: m.upsample(scale_factor=2).conv2d(2, 3, padding=1)
                             .upsample(size=(9, 11), mode="bilinear")
                             .upsample(scale_factor=1.5, mode="bilinear", align_corners=True)
                             .reshape((2, -1)).upsample(size=40, mode="linear").flatten().linear(1), 13)


@lane("sequence-nnconv-shuffle")
def _(ml, X, yc, yr, Xh=None):
    """Pixel shuffle over (8, 3, 4) samples: conv2d(8, 3, padding 1), pixel_shuffle(2), tanh,
    pixel_unshuffle(2), conv2d(2, 1), flatten, linear(1)."""
    return _nn_conv_composed(ml, (8, 3, 4), lambda m: m.conv2d(8, 3, padding=1).pixel_shuffle(2).tanh()
                             .pixel_unshuffle(2).conv2d(2, 1).flatten().linear(1), 14)


_NN_CONV_LANES = ("sequence-nnconv-transpose", "sequence-nnconv-pools", "sequence-nnconv-pads",
                  "sequence-nnconv-upsample", "sequence-nnconv-shuffle")
_batch_decl("n/a:training-step (one seeded training step of a composed model through the tape; its forward "
            "outputs are train parts, digested whole)", *_NN_CONV_LANES)
