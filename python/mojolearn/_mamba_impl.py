# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Mamba-1, Mamba-2 and Mamba-3 blocks on the GPU, for a Python
caller.

PRIVATE MODULE. `Mamba1Block`, `Mamba2Block`, `Mamba3Block` and their
state classes are re-exported from `mojolearn/__init__.py` and from
`mojolearn.mamba`.

The Python surface was introduced on 2026-09-01. The scope authority is
`archive/evidence/mamba/FEATURE_PARITY.md` and the versioned contracts;
its SHIP LATER rows (knobs for d_state/d_conv/expand/dt_rank, varlen,
the multi-block backbone, generation) and REFUSE rows (fused-path knobs,
bitwise torch-RNG initializers, tensor parallel, CUDA graphs) are not
here, each for the reason written there.

THE BINDING THIS FILE CALLS IS `_mojolearn_mamba` (the FOURTEENTH), and
its ABI is the two-list fold (DEVIATION 791): each entry point takes
`addrs` (every buffer address, order written in the binding docstring and
at the call site below in the same words) and `params` (the scalars),
because `def_function` stops elaborating above roughly nine arguments
(measured 2026-09-01 on the gp binding's first spelling) and
`mamba1_forward` carries fourteen addresses. EVERY ARRAY WHOSE ADDRESS
GOES INTO `addrs` IS BOUND TO A LOCAL for the duration of the call: an
address inside a list keeps nothing alive (`_buffer.py`).

WHAT A BLOCK IS HERE. ONE Mamba block -- norm, mixer, residual -- taking
`(B, L, d_model)` float32 in and handing `(B, L, d_model)` float32 back,
with the recurrent state EXPLICIT and CALLER-OWNED (DEVIATION 792): a
`Mamba1State` is the contract's two pieces (conv window, h), a
`Mamba2State` its three (conv window, boundary h, the open-chunk buffer
pair plus `buffered_tokens`), all plain float32 buffers (`mojolearn.Array`
as allocated, or any writable buffer you allocate) a consumer can
serialize, inspect and round-trip byte for byte -- which is the
exact-state-handoff requirement, and why there is no hidden cache object.
Prefill, chunked-prefill continuation, `initial_states` (a nonzero h in a
fresh `Mamba2State`) and single-token decode are ALL the same certified
entry point on the Mojo side; `step` exists as a name because the reference's
`step` is the name a reader expects, and it forwards to the same spelling
at L = 1 (mamba1 contract section 5; mamba2 DEVIATION 786, decode is
prefill resumption).

FLOAT32 ONLY, AND LOUDLY (DEVIATION 793). This surface REFUSES any
non-float32 dtype BY NAME -- bfloat16 and float16 because the reference
implementations' reduced-precision runs are a MIXTURE of cast boundaries
that is not this profile (a certified bf16 profile is SHIP LATER under
its own version name; `archive/evidence/mamba/FEATURE_PARITY.md` section 7's paragraph),
and float64 because a silent downcast would make the bits that ran bits
you did not make. This DIFFERS from the classical estimators' `as_f32_c`
convenience conversion, deliberately: those surfaces summarize data,
this one certifies bits. Non-finite VALUES are not judged here -- they
are refused BY NAME and by flat index on the device path
(`mamba_refuse_bad_inputs` / `mamba2_refuse_bad_inputs` /
`mamba3_refuse_bad_inputs`, contract section 6), and a Python-side copy
would make those refusals unreachable.

THE ARRAYS ARE BUFFERS, NOT NUMPY (numpy-free-0.7, DEVIATIONS 2407-2410).
`x`, every weight and every state piece is read through the buffer
protocol (`_buffer.view`): a NumPy array, an `array.array('f')`, a
`mojolearn.Array` or any other float32 exporter is accepted and NumPy is
not imported. What this module ALLOCATES -- `allocate_state`'s pieces,
the block output, the backward's gradients, the report stages -- is a
`mojolearn.Array`, on which `numpy.asarray` is zero-copy. The state
pieces are updated IN PLACE through their own addresses, exactly as
before, whatever object the caller allocated them as.

FAST AND IDENTICAL (lane neural, 2026-09-27). `numeric_mode=` accepts
'identical' (the default) or 'fast'; FAST runs the same kernels with the
pins on the free schedule and promises quality, never bits. The history:
from 2026-09-10 to 2026-09-27 this lane was IDENTICAL-only. It used to build all three
tiers. It never should have: the fused kernels in `mamba/impl/ops/` and
`mamba/impl/modules/` were gated on `GLOBAL_NUMERIC_MODE ==
NUMERIC_IDENTICAL`, so a FAST build fell back to the
unfused arms and the legacy host copies and ran SLOWER than the default
while promising less. DEVIATION 2300 is what that cost: a `k_last`
failure against ref64 that existed ONLY in a lower tier, found
in a shipped 0.7.0 qualification. `_extension()` checks
the binary's compile-time mode against the requested tier.

EVIDENCE SNAPSHOT. Native certificates, public API checks and installed-wheel
qualification are separate. Historical Apple/NVIDIA/AMD native comparisons
covered five cases and 54 gradient tensors; see
`bench/results/e1g/2026-09-05_042552-amd-mamba/cross-device.json` and
`bench/results/e1/2026-09-05_111524-mojolearn-e2-amd/comparisons.json`.
The retained September 6 NVIDIA run3 reports 102 forward/API checks passing
in each of FAST and IDENTICAL, and five Mamba2/3 backward surface tests
passing; see `bench/results/resume/2026-09-06-root-feature-nvidia/run3/remote/feature-finish/`.
That source-run evidence does not certify an unbuilt alpha wheel or every
vendor, shape and mode. All three classes expose zero-state IDENTICAL
prefill backward. Current scope and packaging boundaries are documented in
SUPPORT_MATRIX.md; older comments are not release certificates.
"""

from ._array import Array as _Array
from . import _portable_math as math

from . import _backend
from . import lowbit as _lowbit
from ._mode import NumericModeMixin

#: `checks/numerics.mojo` codes, duplicated from `_backend._MODE_CODE` on
#: purpose, for `_arima_impl.py`'s reason: the read-back must not share a
#: table with the thing it checks.
_MODE_CODE = {"fast": 0, "identical": 1}

# Profile constants, mirrored from the contracts (mamba1 section 3 /
# mamba2 section 3 and `mamba/checks/*_fixture.mojo`). NOT parameters:
# changing any is a v2 profile (FEATURE_PARITY.md's SHIP LATER rows name
# the triggers), so none of them appears in a constructor signature.
_M1_D_STATE = 16
_M1_D_CONV = 4
_M1_EXPAND = 2
_M2_D_STATE = 128
_M2_D_CONV = 4
_M2_EXPAND = 2
_M2_HEADDIM = 64
_M2_NGROUPS = 1
_M2_CHUNK_SIZE = 256
# Mamba-3 (mamba3 contract section 3 / `mamba/checks/mamba3_fixture.mojo`).
# NOTE the DIFFERENT chunk size: 64, PART OF THE ARITHMETIC (DEVIATIONS
# 827/783), and NUM_ROPE_ANGLES 32 (rope_fraction 0.5). There is NO conv
# and NO dt clamp in Mamba-3.
_M3_D_STATE = 128
_M3_EXPAND = 2
_M3_HEADDIM = 64
_M3_NGROUPS = 1
_M3_CHUNK_SIZE = 64
_M3_NUM_ROPE_ANGLES = 32


# TOMBSTONE: _take (dead code: no caller; the native constructor takes the weight dict) deleted 2026-11-09 by 78dffe4a08; code recoverable at 78dffe4a08^.
# Restore: git show 78dffe4a08^:python/mojolearn/_mamba_impl.py (the commit also moved the retained block drivers to tests/_mamba_block_reference.py)


class Mamba1State:
    """Recurrent state of a `Mamba1Block`, owned by the caller.

    Make one with `Mamba1Block.allocate_state`; it is all zeros before the first
    token. `forward` and `step` update the arrays in place. The arrays are
    `mojolearn.Array` when `allocate_state` made them, or any writable float32
    buffer (a NumPy array included) you supply; their bytes can be saved and
    restored exactly.

    Attributes
    ----------
    conv_window : float32 buffer, shape (B, d_inner, 4)
        The last four inputs to the convolution, oldest first.
    h : float32 buffer, shape (B, d_inner, 16)
        SSM state.
    """

    def __init__(self, conv_window, h, *, profile_id=1, absolute_position=0,
                 affine_boundary=None, affine_a=None, affine_b=None):
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "state_init", (self, "mamba1", (conv_window, h, profile_id,
                absolute_position, affine_boundary, affine_a, affine_b)),
            _native_neural_bridge.CONTEXT)


class Mamba2State:
    """Recurrent state of a `Mamba2Block`, owned by the caller.

    With H = d_model / 32 heads and CD = 2 * d_model + 256 convolution channels.
    Make one with `Mamba2Block.allocate_state`; it is all zeros (and
    ``buffered_tokens == 0``) before the first token. `forward` and `step` update it
    in place. Setting `h` nonzero on a fresh state supplies the reference's
    `initial_states`.

    Attributes
    ----------
    conv_window : float32 buffer, shape (B, CD, 4)
        The last four pre-convolution xBC inputs, oldest first.
    h : float32 buffer, shape (B, H, 64, 128)
        SSM state as of the last completed 256-token chunk.
    buffer_xbc : float32 buffer, shape (B, 256, CD)
        Post-convolution, post-SiLU xBC rows of the open chunk.
    buffer_dtraw : float32 buffer, shape (B, 256, H)
        Raw dt rows of the open chunk.
    buffered_tokens : int
        Number of valid rows in the open-chunk buffers, in [0, 256). Decoding
        resumes the open chunk from these rows.
    """

    def __init__(self, conv_window, h, buffer_xbc, buffer_dtraw,
                 buffered_tokens=0):
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "state_init", (self, "mamba2", (conv_window, h, buffer_xbc,
                buffer_dtraw, buffered_tokens)), _native_neural_bridge.CONTEXT)


class _MambaBase(NumericModeMixin):
    _BINDING = "_mojolearn_mamba"

    def set_params(self, **params):
        """Update constructor parameters and rebuild the block.

        Parameters
        ----------
        **params
            Constructor parameter names and their new values.

        Returns
        -------
        self
            The updated block.

        Examples
        --------
        >>> block.set_params(weights=weights)
        """
        return _native_neural_bridge.RUNTIME.neural_block_set_params(self, params, _native_neural_bridge.CONTEXT)

    def _extension(self):
        """The `_mojolearn_mamba` binding for THIS block's tier, with the
        binary's own compile-time answer cross-checked against it --
        `_gp_impl.py`'s pattern, for its reason: a wrong-arm measurement
        that is correctly labelled by accident is the failure the
        three-tier design exists to prevent."""
        mod = self._bind()
        want = getattr(self, "numeric_mode", None) or _backend.default_mode()
        fn = getattr(mod, "mamba_numeric_mode", None)
        if fn is not None:
            got = int(fn())
            if got != _MODE_CODE.get(want):
                raise RuntimeError(
                    f"mojolearn {type(self).__name__}: numeric_mode="
                    f"{want!r} was requested but {mod.__name__} reports "
                    f"compile-time mode code {got}; the binary and the "
                    "directory it sits in disagree, rebuild it with "
                    "bash bindings/build_mamba.sh"
                )
        return mod


# Contract: profile mojolearn.identical.mamba1.fp32.v1 (mamba/IDENTICAL_MAMBA_CONTRACT.md); reference
# transformers modeling_mamba.py:505-530; weight names per mamba/corpus/README.md. Parity table:
# archive/evidence/mamba/FEATURE_PARITY.md section 1 (d_state/d_conv/expand/dt_rank are profile
# constants, a change is a v2; use_fast_path refused, DEVIATION 732; no initializer by design;
# float32 only). d_model gated at 8 and 16 in the lane checks. State explicit (DEVIATION 792);
# non-finite x/state refused in Mojo per call, weights once per instance (DEVIATION 1886).
# decode_session: DEVIATION 2941 (lane/infer-speed-neural); the host route uses
# mamba1_decode_step (bindings/_mojolearn_mamba_host.mojo) per token as an ownership wrapper only,
# no speed claim.
class Mamba1Block(_MambaBase, _native_constructor=True):
    """One Mamba-1 block: RMSNorm, selective-scan mixer and residual.

    Matches Hugging Face's `MambaBlock.forward`. Takes (B, L, d_model) float32 and
    returns (B, L, d_model) float32, with the recurrent state explicit and
    caller-owned (`Mamba1State`).

    Constructor keyword ``numeric_mode`` selects 'identical' (the default) or 'fast' (Apple GPUs only).
    Inputs, weights and state must be float32: bfloat16, float16 and float64 are
    refused by name rather than converted, and non-finite values are refused by name.
    Arrays are read through the buffer protocol, so NumPy is not required; arrays the
    block allocates are `mojolearn.Array`, zero-copy under `numpy.asarray`. There is
    no initializer: weights are passed in, so a cross-check can give both sides the
    same values. Projection weights may also be packed with `mojolearn.lowbit`; they
    are expanded exactly to float32 (see `weight_format`).

    Fixed by the block: d_state 16, d_conv 4, expand 2, dt_rank ``ceil(d_model / 16)``,
    convolution bias on, projection bias off, SiLU activation.

    Parameters
    ----------
    weights : dict of str to float32 buffer
        Weights under the reference parameter names, with d_inner = 2 * d_model and
        dt_rank = ceil(d_model / 16):

        - ``norm.weight`` (d_model,)
        - ``in_proj.weight`` (2 * d_inner, d_model)
        - ``conv1d.weight`` (d_inner, 1, 4) or (d_inner, 4)
        - ``conv1d.bias`` (d_inner,)
        - ``x_proj.weight`` (dt_rank + 32, d_inner)
        - ``dt_proj.weight`` (d_inner, dt_rank)
        - ``dt_proj.bias`` (d_inner,)
        - ``A_log`` (d_inner, 16)
        - ``D`` (d_inner,)
        - ``out_proj.weight`` (d_model, d_inner)

    Attributes
    ----------
    d_model : int
        Model dimension, read from ``norm.weight``.
    d_inner : int
        Inner dimension, 2 * d_model.
    dt_rank : int
        Rank of the dt projection.
    weight_format : str
        Format the weights were given in.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> block = mojolearn.Mamba1Block(weights)   # dict of float32 arrays, as above
    >>> x = np.random.default_rng(0).random((2, 16, block.d_model), dtype=np.float32)
    >>> y = block.forward(x)
    >>> state = block.allocate_state(2)
    >>> y = block.forward(x, state)
    >>> y_next = block.step(x[:, -1], state)
    """

    # Final slot from class creation; native public operations own field admission.
    __getattribute__ = object.__getattribute__

    _W_NAMES = (
        "norm.weight", "in_proj.weight", "conv1d.weight", "conv1d.bias",
        "x_proj.weight", "dt_proj.weight", "dt_proj.bias", "A_log", "D",
        "out_proj.weight",
    )

    def __init__(self, weights, *, numeric_mode=None):
        return _native_neural_bridge.RUNTIME.neural_block_init(
            self, weights, "mamba1", (None, numeric_mode), _native_neural_bridge.CONTEXT)

    def allocate_state(self, batch_size):
        """Return a zero `Mamba1State` for `batch_size` sequences.

        Parameters
        ----------
        batch_size : int
            Number of sequences B.

        Returns
        -------
        Mamba1State
        """
        return _native_neural_bridge.RUNTIME.neural_block_allocate(
            self, batch_size, None, _native_neural_bridge.CONTEXT)

    def forward(self, x, state=None, *, lengths=None):
        """Run the block over a sequence.

        Parameters
        ----------
        x : float32 buffer, shape (B, L, d_model)
            Input activations: a NumPy array, `array.array('f')`, `mojolearn.Array` or any
            float32 buffer. Other dtypes raise `TypeError`.
        state : Mamba1State or None, default None
            None runs a self-contained prefill from a zero state and discards the final
            state. Pass a state from `allocate_state` to carry it: it is read at entry
            and updated in place, so a later `forward` or `step` continues the sequence
            exactly as if the whole sequence had run in one call.
        lengths : sequence of int or None, default None
            Makes the batch ragged: B integers in [1, L]. Row i is real at positions
            [0, lengths[i]) and padding after. Each real position's output equals the row
            run alone at its own length, and every padding position's output is exactly
            +0.0. Not allowed together with `state`.

        Returns
        -------
        y : mojolearn.Array, shape (B, L, d_model)
            Block output, residual included. ``numpy.asarray(y)`` is zero-copy.
        """
        return _native_neural_bridge.RUNTIME.neural_block_forward(
            self, x, state, False, lengths, _native_neural_bridge.CONTEXT)

    def backward(self, x, grad_output):
        """Gradients of a zero-state prefill (a vector-Jacobian product).

        Recomputes the forward with the current weights on every call; no earlier
        forward or cached state is used or changed. Runs under ``numeric_mode``
        'identical' or 'fast' (Apple GPUs only); other modes raise `NotImplementedError`. Gradients with
        respect to a carried state, decode backward and final-state cotangents are not
        available.

        Parameters
        ----------
        x : float32 buffer, shape (B, L, d_model)
            Forward input.
        grad_output : float32 buffer, shape (B, L, d_model)
            Gradient of the loss with respect to the block output. Must be finite.

        Returns
        -------
        grads : dict of str to mojolearn.Array
            ``'x'`` plus one entry per constructor weight, under the same names.
        """
        return _native_neural_bridge.RUNTIME.neural_block_backward(
            self, x, grad_output, _native_neural_bridge.CONTEXT)

    def step(self, x, state):
        """Run one decode token.

        The same computation as `forward` at L = 1 with the state carried, so decoding
        token by token gives the same bits as prefilling the whole sequence.

        Parameters
        ----------
        x : float32 buffer, shape (B, 1, d_model) or (B, d_model)
            One token per row.
        state : Mamba1State
            Required; updated in place.

        Returns
        -------
        y : mojolearn.Array, shape (B, 1, d_model)
        """
        return _native_neural_bridge.RUNTIME.neural_block_forward(
            self, x, state, True, None, _native_neural_bridge.CONTEXT)

    def decode_session(self, state):
        """Open a `Mamba1DecodeSession` for fast token-by-token decoding.

        On a GPU the session keeps the weights and the state on the device across
        steps; its `step` gives the same bytes as `Mamba1Block.step`. The session copies
        the weights and the state when it opens:

        - until `close()` or `sync_state()`, the state's arrays are stale, and
          `forward`/`step` on this block refuse that state by name;
        - weight edits after opening are not seen; close and open a new session;
        - edits to the state arrays are not seen until `load_state()`.

        Parameters
        ----------
        state : Mamba1State
            State to decode from.

        Returns
        -------
        Mamba1DecodeSession
        """
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "session_open", (self, state), _native_neural_bridge.CONTEXT)

    __call__ = forward


# Two arms, one set of bytes (DEVIATION 2941): a binding exporting mamba1_session_create keeps
# weights/state/L=1 stages resident; the host binding has no session entry and each step calls
# mamba1_decode_step on the session's own copies (same fourteen addresses, same order). The
# mamba1-decode-session lane hashes the byte-for-byte claim on both arms. The host arm makes no
# speed claim.
class Mamba1DecodeSession:
    """Token-by-token decoding on one `Mamba1Block` and one `Mamba1State`.

    Made by `Mamba1Block.decode_session(state)`. Every output is byte for byte what
    `Mamba1Block.step` gives on the same block and state. On a GPU the weights, the
    state and the decode buffers stay on the device across steps; on a CPU-only
    install the session holds its own copies and steps on the host. A session is
    not thread-safe. Use it as a context manager to close it automatically.

    Attributes
    ----------
    state : Mamba1State
        The state this session decodes from.
    is_open : bool
        Whether the session is open.

    Examples
    --------
    >>> with block.decode_session(state) as session:
    ...     for x_t in tokens:
    ...         y_t = session.step(x_t)
    """

    # Every method is one native call (bindings/mamba_public_native.mojo); the
    # session's weights, state copies, position and open flag live in its
    # native handle.
    def __init__(self, block, state):
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "session_init", (self, block, state), _native_neural_bridge.CONTEXT)

    @property
    def state(self):
        """The `Mamba1State` this session decodes from."""
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "session_state", (self,), _native_neural_bridge.CONTEXT)

    @property
    def is_open(self):
        """Whether the session is open."""
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "session_is_open", (self,), _native_neural_bridge.CONTEXT)

    @property
    def _native(self):
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "session_device", (self,), _native_neural_bridge.CONTEXT)

    def step(self, x):
        """Decode one token per row.

        Parameters
        ----------
        x : float32 buffer, shape (B, 1, d_model) or (B, d_model)
            B must equal the session's batch size.

        Returns
        -------
        y : mojolearn.Array, shape (B, 1, d_model)
        """
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "session_step", (self, x), _native_neural_bridge.CONTEXT)

    def decode(self, x):
        """Decode `N` tokens per row in one call.

        The output is byte for byte what `N` calls to `step` give, token by token,
        and the state ends where they leave it. On a GPU the tokens run on the
        device with one upload and one download.

        Parameters
        ----------
        x : float32 buffer, shape (B, N, d_model)
            B must equal the session's batch size.

        Returns
        -------
        y : mojolearn.Array, shape (B, N, d_model)

        Examples
        --------
        >>> with block.decode_session(state) as session:
        ...     y = session.decode(x)
        """
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "session_decode_public", (self, x), _native_neural_bridge.CONTEXT)

    def sync_state(self):
        """Copy the session's current state into the `Mamba1State` arrays and return the state."""
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "session_sync", (self,), _native_neural_bridge.CONTEXT)

    def load_state(self):
        """Reload the session from the `Mamba1State` arrays (after you edit them) and return the state."""
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "session_load", (self,), _native_neural_bridge.CONTEXT)

    def close(self):
        """Copy the state back, release the session's buffers and close it. Closing twice is a no-op."""
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "session_close", (self,), _native_neural_bridge.CONTEXT)

    def __enter__(self):
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "session_enter", (self,), _native_neural_bridge.CONTEXT)

    def __exit__(self, *exc):
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "session_exit", (self,), _native_neural_bridge.CONTEXT)

    def __del__(self):
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "session_drop", (self,), _native_neural_bridge.CONTEXT)

    def __repr__(self):
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "session_repr", (self,), _native_neural_bridge.CONTEXT)


# Contract: profile mojolearn.identical.mamba2.fp32.v1 (mamba/IDENTICAL_MAMBA2_CONTRACT.md); reference
# HF modeling_mamba2.py:617-631, mixer mamba2.py:209-276 (non-mem-eff arm). Parity table:
# archive/evidence/mamba/FEATURE_PARITY.md section 2. d_model % 32 rule duplicated from Mamba2Dims.of
# to size the state. dt_limit is the S9 clamp (DEVIATION 788), pinned whether or not limits bind.
# Gate before norm (DEVIATION 787). CHUNK_SIZE is part of the arithmetic (DEVIATION 783). Decode is
# prefill resumption (DEVIATION 786); the reference per-token recurrence rounds differently and is
# the lane's required-RED sabotage arm. use_mem_eff_path refused (DEVIATION 732 principle).
# h_last_ is the S17 report state (contract section 5, gate D2).
class Mamba2Block(_MambaBase, _native_constructor=True):
    """One Mamba-2 (SSD) block: RMSNorm, mixer and residual.

    Matches Hugging Face's `Mamba2Block.forward` (the non-memory-efficient mixer).
    Takes (B, L, d_model) float32 and returns (B, L, d_model) float32, with the
    recurrent state explicit and caller-owned (`Mamba2State`).

    Constructor keyword ``numeric_mode`` selects 'identical' (the default) or 'fast' (Apple GPUs only).
    Inputs, weights and state must be float32: bfloat16, float16 and float64 are
    refused by name rather than converted, and non-finite values are refused by name.
    Arrays are read through the buffer protocol, so NumPy is not required; arrays the
    block allocates are `mojolearn.Array`, zero-copy under `numpy.asarray`. There is
    no initializer: weights are passed in, so a cross-check can give both sides the
    same values. Projection weights may also be packed with `mojolearn.lowbit`; they
    are expanded exactly to float32 (see `weight_format`).

    `d_model` must be a multiple of 32. Fixed by the block: d_state 128, head
    dimension 64, one group, chunk size 256, expand 2, RMSNorm on with the gate
    applied before the norm. Decoding resumes the open chunk, so `step` and a
    `forward` with a carried state continue a prefill bit for bit.

    After every call, `h_last_` holds the SSM state after the final padded chunk.
    This is a reporting value, not the resumption state; feeding it back as
    `initial_states` matches resumption only on a chunk boundary.

    Parameters
    ----------
    weights : dict of str to float32 buffer
        With d_inner = 2 * d_model, H = d_inner / 64 heads and CD = d_inner + 256:

        - ``block_norm.weight`` (d_model,), the block norm
        - ``in_proj.weight`` (2 * d_inner + 256 + H, d_model), column order z | xBC | dt
        - ``conv1d.weight`` (CD, 1, 4) or (CD, 4)
        - ``conv1d.bias`` (CD,)
        - ``dt_bias`` (H,)
        - ``A_log`` (H,), one value per head
        - ``D`` (H,)
        - ``norm.weight`` (d_inner,), the gated norm
        - ``out_proj.weight`` (d_model, d_inner)
    dt_limit : tuple of float, default (0.0, inf)
        Clamp ``(lo, hi)`` applied to dt.

    Attributes
    ----------
    d_model : int
        Model dimension, read from ``block_norm.weight``.
    weight_format : str
        Format the weights were given in.

    Examples
    --------
    >>> block = mojolearn.Mamba2Block(weights, numeric_mode="identical")
    >>> y = block.forward(x)
    >>> grads = block.backward(x, dy)
    >>> state = block.allocate_state(batch_size=x.shape[0])
    >>> y0 = block.step(x[:, :1], state)
    """

    # Final slot from class creation; native public operations own field admission.
    __getattribute__ = object.__getattribute__

    _W_NAMES = (
        "block_norm.weight", "in_proj.weight", "conv1d.weight",
        "conv1d.bias", "dt_bias", "A_log", "D", "norm.weight",
        "out_proj.weight",
    )

    def __init__(self, weights, *, dt_limit=(0.0, float("inf")), numeric_mode=None):
        return _native_neural_bridge.RUNTIME.neural_block_init(
            self, weights, "mamba2", (None, numeric_mode, dt_limit), _native_neural_bridge.CONTEXT)

    def allocate_state(self, batch_size):
        """Return a zero `Mamba2State` for `batch_size` sequences.

        Set its `h` nonzero before the first call to supply `initial_states`.

        Parameters
        ----------
        batch_size : int
            Number of sequences B.

        Returns
        -------
        Mamba2State
        """
        return _native_neural_bridge.RUNTIME.neural_block_allocate(
            self, batch_size, None, _native_neural_bridge.CONTEXT)

    def forward(self, x, state=None, *, lengths=None):
        """Run the block over a sequence.

        Parameters
        ----------
        x : float32 buffer, shape (B, L, d_model)
            Input activations: a NumPy array, `array.array('f')`, `mojolearn.Array` or any
            float32 buffer. Other dtypes raise `TypeError`.
        state : Mamba2State or None, default None
            None runs a self-contained prefill from a zero state and discards the final
            state. Pass a state from `allocate_state` to carry it: it is read at entry
            and updated in place, so a later `forward` or `step` continues the sequence
            exactly as if the whole sequence had run in one call.
        lengths : sequence of int or None, default None
            Makes the batch ragged: B integers in [1, L]. Row i is real at positions
            [0, lengths[i]) and padding after. Each real position's output equals the row
            run alone at its own length, and every padding position's output is exactly
            +0.0. Not allowed together with `state`.

        Returns
        -------
        y : mojolearn.Array, shape (B, L, d_model)
            Block output, residual included. ``numpy.asarray(y)`` is zero-copy.
        """
        return _native_neural_bridge.RUNTIME.neural_block_forward(
            self, x, state, False, lengths, _native_neural_bridge.CONTEXT)

    def backward(self, x, grad_output):
        """Gradients of a zero-state prefill (a vector-Jacobian product).

        Recomputes the forward with the current weights on every call; no earlier
        forward or cached state is used or changed. Runs under ``numeric_mode``
        'identical' or 'fast' (Apple GPUs only); other modes raise `NotImplementedError`. Gradients with
        respect to a carried state, decode backward and final-state cotangents are not
        available.

        Parameters
        ----------
        x : float32 buffer, shape (B, L, d_model)
            Forward input.
        grad_output : float32 buffer, shape (B, L, d_model)
            Gradient of the loss with respect to the block output. Must be finite.

        Returns
        -------
        grads : dict of str to mojolearn.Array
            ``'x'`` plus one entry per constructor weight, under the same names.
        """
        return _native_neural_bridge.RUNTIME.neural_block_backward(
            self, x, grad_output, _native_neural_bridge.CONTEXT)

    def step(self, x, state):
        """Run one decode token.

        The same computation as `forward` at L = 1 with the state carried, so decoding
        token by token gives the same bits as prefilling the whole sequence.

        Parameters
        ----------
        x : float32 buffer, shape (B, 1, d_model) or (B, d_model)
            One token per row.
        state : Mamba2State
            Required; updated in place.

        Returns
        -------
        y : mojolearn.Array, shape (B, 1, d_model)
        """
        return _native_neural_bridge.RUNTIME.neural_block_forward(
            self, x, state, True, None, _native_neural_bridge.CONTEXT)

    def decode_session(self, state):
        """Open a `Mamba2DecodeSession` for fast token-by-token decoding.

        On a GPU the session keeps the weights and the state on the device across
        steps; its `step` gives the same bytes as `Mamba2Block.step`. The session copies
        the weights and the state when it opens:

        - until `close()` or `sync_state()`, the state's arrays are stale, and
          `forward`/`step` on this block refuse that state by name;
        - weight edits after opening are not seen; close and open a new session;
        - edits to the state arrays are not seen until `load_state()`.

        Parameters
        ----------
        state : Mamba2State
            State to decode from.

        Returns
        -------
        Mamba2DecodeSession
        """
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "session_open", (self, state), _native_neural_bridge.CONTEXT)

    __call__ = forward


# The Mamba-2 and Mamba-3 resident sessions were underscore-private from
# 2026-09-22 (fix/mamba-decode-surface-sep22) until their identity lanes
# existed; `mamba2-decode-session` and `mamba3-decode-session`
# (tools/identity_lanes/coverage.py) hold steps, sync_state and load_state to
# the per-call step in the cell, and the one pre-release verify records them
# (lane fix-d1-decode-sessions, 2026-11-08: decode on the session is the
# public and board path; mamba3-decode was 2.17x on the per-call door).


class Mamba2DecodeSession:
    """Token-by-token decoding on one `Mamba2Block` and one `Mamba2State`.

    Made by `Mamba2Block.decode_session(state)`. Every output is byte for byte what
    `Mamba2Block.step` gives on the same block and state, and the step sets the
    block's report attributes as `step` does. On a GPU the weights, the state and the
    decode buffers stay on the device across steps; on a CPU-only install the session
    holds its own copies and steps on the host. A session is not thread-safe. Use it
    as a context manager to close it automatically.

    Attributes
    ----------
    state : Mamba2State
        The state this session decodes from.
    is_open : bool
        Whether the session is open.

    Examples
    --------
    >>> with block.decode_session(state) as session:
    ...     for x_t in tokens:
    ...         y_t = session.step(x_t)
    """

    # One native call per method; the native boundary retains the step's
    # finished report buffers on the block before returning the output.
    def __init__(self, block, state):
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "session_init", (self, block, state), _native_neural_bridge.CONTEXT)

    @property
    def state(self):
        """The `Mamba2State` this session decodes from."""
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "session_state", (self,), _native_neural_bridge.CONTEXT)

    @property
    def is_open(self):
        """Whether the session is open."""
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "session_is_open", (self,), _native_neural_bridge.CONTEXT)

    @property
    def _native(self):
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "session_device", (self,), _native_neural_bridge.CONTEXT)

    def step(self, x):
        """Decode one token per row.

        Parameters
        ----------
        x : float32 buffer, shape (B, 1, d_model) or (B, d_model)
            B must equal the session's batch size.

        Returns
        -------
        y : mojolearn.Array, shape (B, 1, d_model)
        """
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "session_step_public", (self, x), _native_neural_bridge.CONTEXT)

    def decode(self, x):
        """Decode `N` tokens per row in one call.

        The output is byte for byte what `N` calls to `step` give, token by token,
        and the state ends where they leave it. On a GPU the tokens run on the
        device with one upload and one download and sets the block's report attributes to the last token's.

        Parameters
        ----------
        x : float32 buffer, shape (B, N, d_model)
            B must equal the session's batch size.

        Returns
        -------
        y : mojolearn.Array, shape (B, N, d_model)

        Examples
        --------
        >>> with block.decode_session(state) as session:
        ...     y = session.decode(x)
        """
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "session_decode_public", (self, x), _native_neural_bridge.CONTEXT)

    def sync_state(self):
        """Copy the session's current state into the `Mamba2State` arrays and return the state."""
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "session_sync", (self,), _native_neural_bridge.CONTEXT)

    def load_state(self):
        """Reload the session from the `Mamba2State` arrays (after you edit them) and return the state."""
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "session_load", (self,), _native_neural_bridge.CONTEXT)

    def close(self):
        """Copy the state back, release the session's buffers and close it. Closing twice is a no-op."""
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "session_close", (self,), _native_neural_bridge.CONTEXT)

    def __enter__(self):
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "session_enter", (self,), _native_neural_bridge.CONTEXT)

    def __exit__(self, *exc):
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "session_exit", (self,), _native_neural_bridge.CONTEXT)

    def __del__(self):
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "session_drop", (self,), _native_neural_bridge.CONTEXT)

    def __repr__(self):
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "session_repr", (self,), _native_neural_bridge.CONTEXT)


# Pieces per DEVIATION 832 (buffer never empties after the first call, 832(i)); Input_States
# pending flag per DEVIATION 794 (the fresh-state rule is judged in Mojo, not respelled here).
class Mamba3State:
    """Recurrent state of a `Mamba3Block`, owned by the caller.

    With H = d_model / 32 heads, N = 128 (QK head dimension), P = 64 (V head
    dimension), R = 32 rotary angles and Q = 64 (the Mamba-3 chunk size). Make one
    with `Mamba3Block.allocate_state`; it is all zeros (``buffered_tokens == 0``,
    ``pending == False``) before the first token. `forward` and `step` update it in
    place.

    Attributes
    ----------
    theta : float32 buffer, shape (B, H, R)
        Rotary angles, each in [0, 2*pi).
    h : float32 buffer, shape (B, H, P, N)
        SSM state entering the last working chunk.
    buffer_qrot, buffer_krot : float32 buffer, shape (B, Q, H, N)
        Rotated, unscaled q and k rows of the working chunk.
    buffer_v : float32 buffer, shape (B, Q, H, P)
        Raw v rows.
    buffer_dt, buffer_sig, buffer_adt : float32 buffer, shape (B, Q, H)
        dt, sigmoid(trap) and A*dt rows.
    pending_k : float32 buffer, shape (B, H, N)
        Input-state k, used only when `pending`.
    pending_v : float32 buffer, shape (B, H, P)
        Input-state v, used only when `pending`.
    buffered_tokens : int
        Valid buffer rows, in [0, Q]. After the first call it is always in [1, Q].
    pending : bool
        True when `set_input_states` supplied the state; consumed by the next call.
    """

    def __init__(self, theta, h, buffer_qrot, buffer_krot, buffer_v,
                 buffer_dt, buffer_sig, buffer_adt, pending_k, pending_v,
                 buffered_tokens=0, pending=False):
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "state_init", (self, "mamba3", (theta, h, buffer_qrot, buffer_krot,
                buffer_v, buffer_dt, buffer_sig, buffer_adt, pending_k,
                pending_v, buffered_tokens, pending)), _native_neural_bridge.CONTEXT)

    def set_input_states(self, theta, h, k, v):
        """Start from a reference `Input_States` continuation.

        Copies the given values into `theta`, `h`, `pending_k` and `pending_v` and marks
        them pending for the next call. The state must be fresh (no tokens yet); that is
        checked at the next call. Arrays must already be float32 with the shapes above;
        nothing is converted.

        Parameters
        ----------
        theta : float32 buffer, shape (B, H, R)
        h : float32 buffer, shape (B, H, P, N)
        k : float32 buffer, shape (B, H, N)
        v : float32 buffer, shape (B, H, P)
        """
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "set_input_states", (self, theta, h, k, v), _native_neural_bridge.CONTEXT)


# Contract: profile mojolearn.identical.mamba3.siso.fp32.v1 (mamba/IDENTICAL_MAMBA3_CONTRACT.md);
# reference mamba_ssm mamba3.py:160-278 inside block.py:51-53; weight names from
# mamba/checks/mamba3_fixture.mojo ids 42-50; in_proj column order mamba3.py:106-107. Parity:
# archive/evidence/mamba/FEATURE_PARITY.md "Mamba-3 SURFACE KNOBS". CHUNK_SIZE 64 is part of the
# arithmetic (DEVIATIONS 827/783). Decode is prefill resumption (DEVIATION 831); the reference
# per-token recurrence is the required-RED STEP_UPSTREAM_RECURRENCE arm. Report stages vs sealed
# boundary: DEVIATION 832. Owned weights are the NN40 IDENTICAL arm.
class Mamba3Block(_MambaBase, _native_constructor=True):
    """One Mamba-3 (SISO) block: RMSNorm, mixer and residual.

    Matches mamba_ssm's `Mamba3.forward` (SISO) inside the non-fused block. Takes
    (B, L, d_model) float32 and returns (B, L, d_model) float32, with the recurrent
    state explicit and caller-owned (`Mamba3State`).

    Compared with `Mamba2Block`: no convolution (v is the raw x split), no dt clamp
    (so no `dt_limit`), A depends on the token and head with a floor clamp, the gate
    is applied inside the core (no gated output norm), and B and C have their own
    RMSNorms with per-head biases added after the norm.

    Constructor keyword ``numeric_mode`` selects 'identical' (the default) or 'fast' (Apple GPUs only).
    Inputs, weights and state must be float32: bfloat16, float16 and float64 are
    refused by name rather than converted, and non-finite values are refused by name.
    Arrays are read through the buffer protocol, so NumPy is not required; arrays the
    block allocates are `mojolearn.Array`, zero-copy under `numpy.asarray`. There is
    no initializer: weights are passed in, so a cross-check can give both sides the
    same values. Projection weights may also be packed with `mojolearn.lowbit`; they
    are expanded exactly to float32 (see `weight_format`).

    `d_model` must be a multiple of 32. Fixed by the block: d_state 128, head
    dimension 64, one group, rope fraction 0.5 (32 angles), A floor 1e-4, chunk size
    64, expand 2. MIMO, output-projection norm, fused pre-gate options, more than one
    group and variable-length batches are not available. Decoding resumes the
    working chunk, so `step` and a `forward` with a carried state continue a prefill
    bit for bit.

    After every call, `h_last_`, `k_last_`, `v_last_` and `theta_last_` hold
    reporting values, not the resumption state.

    Parameters
    ----------
    weights : dict of str to float32 buffer
        With d_inner = 2 * d_model, H = d_inner / 64 heads, N = 128 and
        d_in_proj = 2 * d_inner + 256 + 3 * H + 32:

        - ``block_norm.weight`` (d_model,), the block norm
        - ``in_proj.weight`` (d_in_proj, d_model), column order
          z | x | B | C | dd_dt | dd_A | trap | angle
        - ``dt_bias`` (H,)
        - ``B_norm.weight`` (N,), RMSNorm with eps 1e-5
        - ``C_norm.weight`` (N,)
        - ``B_bias`` (H, N)
        - ``C_bias`` (H, N)
        - ``D`` (H,)
        - ``out_proj.weight`` (d_model, d_inner)

    Attributes
    ----------
    d_model : int
        Model dimension, read from ``block_norm.weight``.
    weight_format : str
        Format the weights were given in.

    Examples
    --------
    >>> block = mojolearn.Mamba3Block(weights)
    >>> y = block.forward(x)
    >>> state = block.allocate_state(x.shape[0])
    >>> y = block.forward(x, state)
    >>> y_next = block.step(x[:, -1], state)
    """

    # Final slot from class creation; native public operations own field admission.
    __getattribute__ = object.__getattribute__

    _W_NAMES = (
        "block_norm.weight", "in_proj.weight", "dt_bias",
        "B_norm.weight", "C_norm.weight", "B_bias", "C_bias", "D",
        "out_proj.weight",
    )

    def __init__(self, weights, *, numeric_mode=None):
        return _native_neural_bridge.RUNTIME.neural_block_init(
            self, weights, "mamba3", (None, numeric_mode), _native_neural_bridge.CONTEXT)

    def allocate_state(self, batch_size):
        """Return a zero `Mamba3State` for `batch_size` sequences.

        For a reference `Input_States` continuation, call
        `Mamba3State.set_input_states` on the fresh state.

        Parameters
        ----------
        batch_size : int
            Number of sequences B.

        Returns
        -------
        Mamba3State
        """
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "allocate", (self, batch_size), _native_neural_bridge.CONTEXT)

    def install_owned_weights(self):
        """Snapshot the current weights for later stateless `forward` and `backward` calls.

        Returns the snapshot's generation number. Later edits to your weight arrays take
        effect only after another call. Calls with an explicit state and decoding are
        unaffected. Needs a binding that provides owned weights; otherwise
        `RuntimeError`.
        """
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "install_owned", (self,), _native_neural_bridge.CONTEXT)

    def release_owned_weights(self):
        """Drop the weight snapshot; later calls read your weight arrays again."""
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "release_owned", (self,), _native_neural_bridge.CONTEXT)

    def session_info(self):
        """Counters of the prefill session as a dict, or None when no session is open.

        Reports weight uploads, re-copies and reuses, backward stage reuses, forward
        recomputes and the number of stages held.
        """
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "session_info", (self,), _native_neural_bridge.CONTEXT)

    def export_owned_weights(self):
        """Copy the installed weight snapshot into new host arrays."""
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "export_owned", (self,), _native_neural_bridge.CONTEXT)

    def __getstate__(self):
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "getstate", (self,), _native_neural_bridge.CONTEXT)

    def __setstate__(self, state):
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "setstate", (self, state), _native_neural_bridge.CONTEXT)

    def forward(self, x, state=None, *, lengths=None):
        """Run the block over a sequence.

        Parameters
        ----------
        x : float32 buffer, shape (B, L, d_model)
            Input activations: a NumPy array, `array.array('f')`, `mojolearn.Array` or any
            float32 buffer. Other dtypes raise `TypeError`.
        state : Mamba3State or None, default None
            None runs a self-contained prefill from a zero state and discards the final
            state. Pass a state from `allocate_state` to carry it: it is read at entry
            and updated in place, so a later `forward` or `step` continues the sequence
            exactly as if the whole sequence had run in one call.
        lengths : sequence of int or None, default None
            Makes the batch ragged: B integers in [1, L]. Row i is real at positions
            [0, lengths[i]) and padding after. Each real position's output equals the row
            run alone at its own length, and every padding position's output is exactly
            +0.0. Not allowed together with `state`.

        Returns
        -------
        y : mojolearn.Array, shape (B, L, d_model)
            Block output, residual included. ``numpy.asarray(y)`` is zero-copy.
        """
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "forward_public", (self, x, state, lengths), _native_neural_bridge.CONTEXT)

    def backward(self, x, grad_output):
        """Gradients of a zero-state prefill (a vector-Jacobian product).

        Recomputes the forward with the current weights on every call; no earlier
        forward or cached state is used or changed. Runs under ``numeric_mode``
        'identical' or 'fast' (Apple GPUs only); other modes raise `NotImplementedError`. Gradients with
        respect to a carried state, decode backward and final-state cotangents are not
        available.

        Parameters
        ----------
        x : float32 buffer, shape (B, L, d_model)
            Forward input.
        grad_output : float32 buffer, shape (B, L, d_model)
            Gradient of the loss with respect to the block output. Must be finite.

        Returns
        -------
        grads : dict of str to mojolearn.Array
            ``'x'`` plus one entry per constructor weight, under the same names.
        """
        return _native_neural_bridge.RUNTIME.neural_block_backward(
            self, x, grad_output, _native_neural_bridge.CONTEXT)

    def step(self, x, state):
        """Run one decode token.

        The same computation as `forward` at L = 1 with the state carried, so decoding
        token by token gives the same bits as prefilling the whole sequence.

        Parameters
        ----------
        x : float32 buffer, shape (B, 1, d_model) or (B, d_model)
            One token per row.
        state : Mamba3State
            Required; updated in place.

        Returns
        -------
        y : mojolearn.Array, shape (B, 1, d_model)
        """
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "step_public", (self, x, state), _native_neural_bridge.CONTEXT)

    def decode_session(self, state):
        """Open a `Mamba3DecodeSession` for fast token-by-token decoding.

        On a GPU the session keeps the weights and the state on the device across
        steps; its `step` gives the same bytes as `Mamba3Block.step`. The session copies
        the weights and the state when it opens:

        - until `close()` or `sync_state()`, the state's arrays are stale, and
          `forward`/`step` on this block refuse that state by name;
        - weight edits after opening are not seen; close and open a new session;
        - edits to the state arrays are not seen until `load_state()`.

        Parameters
        ----------
        state : Mamba3State
            State to decode from.

        Returns
        -------
        Mamba3DecodeSession
        """
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "session_open", (self, state), _native_neural_bridge.CONTEXT)

    __call__ = forward


class Mamba3DecodeSession:
    """Token-by-token decoding on one `Mamba3Block` and one `Mamba3State`.

    Made by `Mamba3Block.decode_session(state)`. Every output is byte for byte what
    `Mamba3Block.step` gives on the same block and state, and the step sets the
    block's report attributes as `step` does. On a GPU the weights, the state and the
    decode buffers stay on the device across steps; on a CPU-only install the session
    holds its own copies and steps on the host. A session is not thread-safe. Use it
    as a context manager to close it automatically.

    Attributes
    ----------
    state : Mamba3State
        The state this session decodes from.
    is_open : bool
        Whether the session is open.

    Examples
    --------
    >>> with block.decode_session(state) as session:
    ...     for x_t in tokens:
    ...         y_t = session.step(x_t)
    """

    _STATE_NAMES = (
        "theta", "h", "buffer_qrot", "buffer_krot", "buffer_v",
        "buffer_dt", "buffer_sig", "buffer_adt", "pending_k", "pending_v",
    )

    # One native call per method; the native boundary retains the step's
    # finished report buffers on the block before returning the output.
    def __init__(self, block, state):
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "session_init", (self, block, state), _native_neural_bridge.CONTEXT)

    @property
    def state(self):
        """The `Mamba3State` this session decodes from."""
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "session_state", (self,), _native_neural_bridge.CONTEXT)

    @property
    def is_open(self):
        """Whether the session is open."""
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "session_is_open", (self,), _native_neural_bridge.CONTEXT)

    @property
    def _native(self):
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "session_device", (self,), _native_neural_bridge.CONTEXT)

    def step(self, x):
        """Decode one token per row.

        Parameters
        ----------
        x : float32 buffer, shape (B, 1, d_model) or (B, d_model)
            B must equal the session's batch size.

        Returns
        -------
        y : mojolearn.Array, shape (B, 1, d_model)
        """
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "session_step_public", (self, x), _native_neural_bridge.CONTEXT)

    def decode(self, x):
        """Decode `N` tokens per row in one call.

        The output is byte for byte what `N` calls to `step` give, token by token,
        and the state ends where they leave it. On a GPU the tokens run on the
        device with one upload and one download and sets the block's report attributes to the last token's.

        Parameters
        ----------
        x : float32 buffer, shape (B, N, d_model)
            B must equal the session's batch size.

        Returns
        -------
        y : mojolearn.Array, shape (B, N, d_model)

        Examples
        --------
        >>> with block.decode_session(state) as session:
        ...     y = session.decode(x)
        """
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "session_decode_public", (self, x), _native_neural_bridge.CONTEXT)

    def sync_state(self):
        """Copy the session's current state into the `Mamba3State` arrays and return the state."""
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "session_sync", (self,), _native_neural_bridge.CONTEXT)

    def load_state(self):
        """Reload the session from the `Mamba3State` arrays (after you edit them) and return the state."""
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "session_load", (self,), _native_neural_bridge.CONTEXT)

    def close(self):
        """Copy the state back, release the session's buffers and close it. Closing twice is a no-op."""
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "session_close", (self,), _native_neural_bridge.CONTEXT)

    def __enter__(self):
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "session_enter", (self,), _native_neural_bridge.CONTEXT)

    def __exit__(self, *exc):
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "session_exit", (self,), _native_neural_bridge.CONTEXT)

    def __del__(self):
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "session_drop", (self,), _native_neural_bridge.CONTEXT)

    def __repr__(self):
        return _native_neural_bridge.RUNTIME.mamba_native_call(
            "session_repr", (self,), _native_neural_bridge.CONTEXT)


from . import _native_neural_bridge
from .models import causal_lm as _causal_native_connection
