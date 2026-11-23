# SPDX-License-Identifier: Apache-2.0
# Lane nn-conv3d (2026-11-23): models composed from Sequential's 3-D layers (conv3d, max_pool3d, avg_pool3d,
# adaptive_avg_pool3d), one seeded training step each through the tape (tools/identity_lanes/sequence.py
# `_composed`: 16 seeded rows, mse, sgd, one full unshuffled batch): the loss, the updated parameters and the
# predictions on the training and held-out rows are digested. Sizes are small correctness fixtures.


@lane("sequence-composed-conv3d")
def _(ml, X, yc, yr, Xh=None):
    """A 3-D CNN from the public layers over (1, 4, 4, 4) volumes: conv3d(4, 3, padding 1), relu,
    max_pool3d(2), conv3d(4, (1, 2, 2), padding (0, 1, 1)), relu, adaptive_avg_pool3d(1), flatten,
    linear(1)."""
    return _composed(ml, (1, 4, 4, 4), lambda m: m.conv3d(4, 3, padding=1).relu().max_pool3d(2)
                     .conv3d(4, (1, 2, 2), padding=(0, 1, 1)).relu().adaptive_avg_pool3d(1).flatten().linear(1), 6)


@lane("sequence-composed-pool3d")
def _(ml, X, yc, yr, Xh=None):
    """The 3-D pools' options from the public layers over (2, 5, 4, 4) volumes: a grouped anisotropic
    conv3d(4, (3, 3, 1), padding (1, 1, 0), groups 2), avg_pool3d(3, stride 2, padding 1, ceil_mode,
    count_include_pad False), max_pool3d(2, padding 1, ceil_mode), flatten, linear(1)."""
    return _composed(ml, (2, 5, 4, 4), lambda m: m.conv3d(4, (3, 3, 1), padding=(1, 1, 0), groups=2).silu()
                     .avg_pool3d(3, stride=2, padding=1, ceil_mode=True, count_include_pad=False)
                     .max_pool3d(2, padding=1, ceil_mode=True).flatten().linear(1), 7)


_NN_CONV3D_LANES = ("sequence-composed-conv3d", "sequence-composed-pool3d")
_batch_decl("n/a:training-step (one seeded training step of a composed model through the tape; its forward "
            "outputs are train parts, digested whole)", *_NN_CONV3D_LANES)
