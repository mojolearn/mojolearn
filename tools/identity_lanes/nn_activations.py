# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# LANE nn-activations' IDENTITY LANE (2026-11-23): a model composed from the activation layers and the identity
# layer (Sequential.leaky_relu, elu, prelu, mish, identity, hardswish, hardsigmoid), one seeded training step of
# one full unshuffled batch through the tape (sequence.py's `_composed`, resolved when the lane runs), digesting
# the loss, the updated parameters (PReLU's slopes included) and the predictions on the training and held-out
# rows. Executed by tools/identity_break.py in its own namespace like every fragment (see sequence.py's header).


@lane("sequence-composed-activations")
def _(ml, X, yc, yr, Xh=None):
    """Every activation layer in one model over (4, 8) samples: linear(8), leaky_relu(0.1), elu, prelu per
    channel (4), mish, identity, hardswish, linear(8), hardsigmoid, shared prelu, flatten, linear(1)."""
    return _composed(ml, (4, 8), lambda m: m.linear(8).leaky_relu(0.1).elu().prelu(4).mish().identity()
                     .hardswish().linear(8).hardsigmoid().prelu().flatten().linear(1), 6)


_batch_decl("n/a:training-step (one seeded training step of a composed model through the tape; its forward "
            "outputs are train parts, digested whole)", "sequence-composed-activations")
