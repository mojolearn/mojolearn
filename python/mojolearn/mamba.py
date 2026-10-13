# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Mamba-1, Mamba-2 and Mamba-3 blocks in float32.

The block and state classes are also exported from ``mojolearn``. Each block takes
caller-supplied weights and provides ``forward(x, state=None)``, ``step(x, state)``,
``allocate_state(...)`` and ``backward(x, grad_output)``. The recurrent state is
explicit and owned by the caller, so prefill, continuation and token-by-token decode
give the same bits. Forward runs under ``numeric_mode`` 'identical' (the default) or
'fast' (Apple GPUs only). ``backward`` returns the gradients of a zero-state prefill with respect to
``x`` and every constructor weight (11 arrays for Mamba-1, 10 each for Mamba-2 and
Mamba-3), recomputing the forward internally; gradients through a carried state or a
decode step are not available. Inputs are read through the buffer protocol, so PyTorch
and NumPy are not required.

`Mamba1DecodeSession` (from `Mamba1Block.decode_session`) keeps a Mamba-1 block and
its state resident on the device for fast decoding.

Examples
--------
With float32 ``weights``, ``x`` and ``dy`` of the right shapes::

    from mojolearn.mamba import Mamba2Block
    block = Mamba2Block(weights, numeric_mode="identical")
    y = block.forward(x)
    gradients = block.backward(x, dy)
    state = block.allocate_state(batch_size=x.shape[0])
    y0 = block.step(x[:, :1], state)
"""

from ._mamba_impl import (
    Mamba1Block,
    Mamba1State,
    Mamba2Block,
    Mamba2State,
    Mamba3Block,
    Mamba3State,
    Mamba1DecodeSession,
)

__all__ = [
    "Mamba1Block",
    "Mamba1DecodeSession",
    "Mamba1State",
    "Mamba2Block",
    "Mamba2State",
    "Mamba3Block",
    "Mamba3State",
]
