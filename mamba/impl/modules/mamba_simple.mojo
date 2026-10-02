# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`mamba_ssm/modules/mamba_simple.py::Mamba.step` (:208-253) and
`::allocate_inference_cache` (:255-266), state-spaces/mamba `e9594ce`.

IMPLEMENTED. The DECODE half of profile `mojolearn.identical.mamba1.fp32.v1`
(`mamba/IDENTICAL_MAMBA_CONTRACT.md`, section 5). One token at a time,
carrying the two pieces of recurrent state their `step` carries: the conv
WINDOW (`conv_state`, the last `d_conv = 4` conv inputs, oldest first) and
the SSM state h (`ssm_state`), both zeros before the first token
(`allocate_inference_cache`).

## What this file is NOT

It is not a second copy of the block's arithmetic. Contract section 5 says
prefill and decode are bit-identical BY CONSTRUCTION and not by luck: ONE
spelling serves both paths, because two spellings that agree today are two
spellings that can drift tomorrow. So `mamba_step` reads as their `step`
reads -- the same order of the same operations, cited line by line below --
but every seam it reaches is the block spelling with `l = 1`, and the only
arithmetic written out longhand in this file is the SABOTAGE arm, which
exists to be falsified.

That is why the reference's own `step` has two arms at :215 and :238 (the torch
fallback and the fused CUDA kernel) and this file has one. Those two arms
do NOT agree bitwise -- the CUDA `selective_state_update` rounds
`B * (delta * u)` where the torch reference rounds `(delta * B) * u`
(contract seam S8, `selective_scan_fwd_kernel.cuh:162,222`) -- so an implementation
that followed the branch would follow a bitwise fork. DEVIATION 732.

## The four departures from their spelling, numbered

DEVIATION 721 -- THE BIAS SEED. Their step (:218-220) sums the conv taps
first and adds `conv1d.bias` AFTER; the prefill kernels seed the
accumulator WITH the bias (MAX `causal_conv1d.mojo:190-205`, and the CUDA
`causal_conv1d` kernel likewise). Those are two different roundings of the
same conv, and the reference's own two paths therefore disagree with each
other. The profile takes the prefill kernels' bias SEED on BOTH paths,
because keeping both spellings would make contract gate D (decode ==
prefill, bitwise, per token) false by construction -- a gate that can only
be met by an accident of rounding is not a gate. This is the one place in
this lane where "do what they do" is not available, because there is no
single thing they do. Adopted at the narrowest possible width: only where
the bias enters the accumulator moves; the tap order (k ascending, oldest
first), the fusion (one fma per tap, contract S13), and the flush at every
seam are all unchanged. `conv_step_upstream_bias_last` below is their
:218-220 order, kept as the sabotage arm, and `check_decode_equals_prefill`
runs it and must FAIL: that is the proof that 721 is load bearing and not
cosmetic.

DEVIATION 732 -- ONE ARM, NOT TWO. Their :215 and :238 select between a
torch fallback and a fused CUDA kernel at import time. The two arms are not
bitwise equal (S8 above; also the CUDA scan's `D * u` seeding, S11). The
profile is the reference's arm, and this file has no kernel-present branch
to take. The name `causal_conv1d_update` appears nowhere here for the same
reason.

DEVIATION 733 -- THE ROLL, OUT OF PLACE. Their :216-217 update the window
in place: `roll(conv_state, -1)` then `conv_state[:, :, -1] = x`. The block
spelling rebuilds the window from the sequence and the incoming window
(`mamba_oracle.mojo`'s `new_win` loop: position `l - d_conv + j`, read from
the sequence when it is nonnegative and from the incoming window
otherwise). At `l = 1` the two are the same four values in the same order
-- `[w1, w2, w3, x]` -- because the window carries PRE-conv values, which
is what makes one spelling able to serve both paths at all. A copy is not
an arithmetic seam (contract section 4), so this moves no bits; it is
recorded because it is a visible difference in the implementation and because the
identity between them is a claim the gate checks (`conv.window` after every
step, compared against the prefill card's).

DEVIATION 734 -- THE CACHE'S SIGNATURE. `allocate_inference_cache(self,
batch_size, max_seqlen, dtype=None, **kwargs)` becomes
`allocate_inference_cache(batch_size, dims)` for the host reference and
`allocate_inference_cache(ctx, batch_size, dims)` for device state.
`max_seqlen` is dropped
because Mamba's cache does not depend on it (their own body ignores it too:
the shapes at :258-265 are `(B, d_inner, d_conv)` and `(B, d_inner,
d_state)`, no sequence length in either). The device overload takes the
caller's `DeviceContext`; dtype remains fixed to Float32 by the profile. The ZEROS -- the whole content of their function -- are
unchanged, and are what makes the first token's conv read zero padding and
its scan start from h = 0.

DEVIATION 735 -- A BLOCK STEP, NOT A MIXER STEP. Their `Mamba.step` is the
MIXER only; the norm and the residual live in `mamba_ssm/modules/block.py`,
whose order is `Add -> LN -> Mixer` with the residual threaded between
blocks. The profile's block order is HuggingFace's instead (contract
section 2 and section 1's block-order pin: `MambaBlock.forward` MM:505-530,
`residual = hidden; hidden = norm(hidden); hidden = mixer(hidden); hidden =
residual + hidden`), so the decode step here covers the whole block, norm
and residual included. The two references genuinely disagree about where
those two operations sit, the contract already chose, and a decode step
that stopped at `out_proj` could not be compared against a prefill card
that does not (gate D is per STAGE, and `norm.sumsq` and `residual.out` are
stages).

## Where the arithmetic comes from, today and tomorrow

The device overload of `mamba_step` delegates to the certified
`modeling_mamba.mamba_block_forward` at L=1. The public Python decode
binding and the device decode/prefill gate both call that overload.
The host-list overload and `_one_block_call`, which keep the independent
oracle for the reference gate and its arithmetic negative controls, live in
`mamba/checks/mamba_simple_reference.mojo`.

Run the gate:

    pixi run check-mamba-decode                  gate D, several corpus cases
    pixi run check-mamba-decode probe            + the per-stage reach probe
    pixi run check-mamba-decode sabotage         DEVIATION 721 undone, must FAIL
    pixi run check-mamba-decode sabotage-window  the state carry cut, must FAIL
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from mamba.impl.modeling.modeling_mamba import (
    mamba_block_forward,
    MAMBA_GUARD,
    MambaDeviceStages, MambaDeviceState, MambaDeviceWeights,
)

from core.identity_trace import IdentityTrace
from mamba.checks.mamba_fixture import MambaDims


comptime DECODE_TOKENS = 1
"""Their :210 assert, as a constant: "Only support decoding with 1 token at
a time for now"."""


# ===========================================================================
# allocate_inference_cache (mamba_simple.py:255-266) and Mamba.step
# (mamba_simple.py:208-253), on the device. The host-list overloads over the
# oracle are in mamba/checks/mamba_simple_reference.mojo.
# ===========================================================================


def allocate_inference_cache(
    ctx: DeviceContext, batch_size: Int, dims: MambaDims,
) raises -> MambaDeviceState:
    """Allocate the zero conv window and SSM state on the caller's device."""
    if batch_size <= 0:
        raise Error("allocate_inference_cache: batch_size must be positive")
    return MambaDeviceState(ctx, batch_size, dims)


def mamba_step(
    ctx: DeviceContext,
    mut stages: MambaDeviceStages,
    mut state: MambaDeviceState,
    mut w: MambaDeviceWeights,
    mut hidden_states: DeviceBuffer[DType.float32],
    b: Int,
    mut trace: IdentityTrace,
    prefix: String,
) raises:
    """Decode one token per row on the GPU, updating the caller's state.

    The same block kernels serve prefill and decode. The block validates
    that stages and state match B and L=1 before launching any work.
    """
    # the poison build carries MAMBA_GUARD band elements after the logical
    # length (DEVIATION 2712); in production the band is 0 and this is exact
    if len(hidden_states) != b * w.dims.d_model + MAMBA_GUARD:
        raise Error("mamba_step: expected exactly one token per batch row")
    mamba_block_forward(
        ctx, stages, state, w, hidden_states, b, DECODE_TOKENS, trace, prefix,
    )

