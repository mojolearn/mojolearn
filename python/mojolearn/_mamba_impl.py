# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Mamba-1 and Mamba-2 blocks on the GPU, for a Python caller.

PRIVATE MODULE. `Mamba1Block`, `Mamba2Block`, `Mamba1State` and
`Mamba2State` are re-exported from `mojolearn/__init__.py` and from
`mojolearn.mamba`.

WRITTEN 2026-09-01, closing `mamba/FEATURE_PARITY.md`'s consumer table
row "PyPI surface: NONE EXISTS" -- until this file, the certified Mamba-1
block (profile `mojolearn.identical.mamba1.fp32.v1`, three-vendor
identity card, IDENTITY_PATHS row 55) and the Mamba-2 block (profile
`mojolearn.identical.mamba2.fp32.v1`, gated on the Apple M4, cross-vendor
legs OWED) exported no Python symbol at all. The parity document's SHIP
NOW dispositions are the scope authority for everything on this surface;
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
address inside a list keeps nothing alive (`_arrays.py`).

WHAT A BLOCK IS HERE. ONE Mamba block -- norm, mixer, residual -- taking
`(B, L, d_model)` float32 in and handing `(B, L, d_model)` float32 back,
with the recurrent state EXPLICIT and CALLER-OWNED (DEVIATION 792): a
`Mamba1State` is the contract's two pieces (conv window, h), a
`Mamba2State` its three (conv window, boundary h, the open-chunk buffer
pair plus `buffered_tokens`), all plain NumPy arrays a consumer can
serialize, inspect and round-trip byte for byte -- which is the
exact-state-handoff requirement, and why there is no hidden cache object.
Prefill, chunked-prefill continuation, `initial_states` (a nonzero h in a
fresh `Mamba2State`) and single-token decode are ALL the same certified
entry point on the Mojo side; `step` exists as a name because upstream's
`step` is the name a reader expects, and it forwards to the same spelling
at L = 1 (mamba1 contract section 5; mamba2 DEVIATION 786, decode is
prefill resumption).

FLOAT32 ONLY, AND LOUDLY (DEVIATION 793). This surface REFUSES any
non-float32 dtype BY NAME -- bfloat16 and float16 because the reference
implementations' reduced-precision runs are a MIXTURE of cast boundaries
that is not this profile (a certified bf16 profile is SHIP LATER under
its own version name; `mamba/FEATURE_PARITY.md` section 7's paragraph),
and float64 because a silent downcast would make the bits that ran bits
you did not make. This DIFFERS from the classical estimators' `as_f32_c`
convenience conversion, deliberately: those surfaces summarize data,
this one certifies bits. Non-finite VALUES are not judged here -- they
are refused BY NAME and by flat index on the device path
(`mamba_refuse_bad_inputs` / `mamba2_refuse_bad_inputs`, contract
section 6), and a Python-side copy would make those refusals
unreachable.

THE NUMERIC TIER IS THE LOADED BINARY'S, SELECTED AT BUILD TIME OF THE
.so. `numeric_mode=` on either class (or the process default) picks
which compiled set answers -- fast (no promise), deterministic (same
bits run to run on one device), identical (also the same bits across
vendors, where the lane's cards say so: THREE vendors for Mamba-1, ONE
gated vendor for Mamba-2 today, cross-vendor legs owed) -- and
`_extension()` cross-checks the binary's own compile-time answer against
the tier the package resolved, `_gp_impl.py`'s pattern, so a wrong-arm
measurement cannot be correctly labelled by accident.

EVERYTHING RUNNABLE HERE IS UNVERIFIED, RUN OWED: this file and its
binding have never been built or imported. The gate is
`python/mojolearn/tests/test_mamba_surface.py`; the build is
`bash bindings/build_mamba.sh` per tier.
"""

import math

import numpy as np

from . import _backend
from ._arrays import _addr, _addr_ro
from ._mode import NumericModeMixin

#: `checks/numerics.mojo` codes, duplicated from `_backend._MODE_CODE` on
#: purpose, for `_arima_impl.py`'s reason: the read-back must not share a
#: table with the thing it checks.
_MODE_CODE = {"fast": 0, "identical": 1, "deterministic": 2}

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


def _f32_strict(a, what, name):
    """`a` as a C-contiguous float32 ndarray, REFUSING every other dtype
    BY NAME (module header: this surface certifies bits, so there is no
    convenience downcast). Layout may be fixed silently -- a copy to C
    order moves bytes untouched, so no output bit can move -- but the
    dtype may not."""
    a = np.asarray(a)
    if a.dtype != np.float32:
        raise TypeError(
            f"mojolearn {what}: {name} has dtype {a.dtype.name}; this "
            "surface is float32 ONLY (the fp32 profiles, "
            "mamba/IDENTICAL_MAMBA_CONTRACT.md section 3 / "
            "IDENTICAL_MAMBA2_CONTRACT.md section 3). bfloat16 and "
            "float16 are refused BY NAME: the references' "
            "reduced-precision runs are a mixture of cast boundaries "
            "that is not this profile, and a certified bf16 profile is "
            "SHIP LATER under its own version name "
            "(mamba/FEATURE_PARITY.md section 7). float64 is refused "
            "rather than downcast so the bits that ran are bits you "
            "made. Convert yourself and pass float32."
        )
    return np.ascontiguousarray(a)


def _want_shape(a, what, name, shape, alt=None):
    """Exact-shape check with the expected spelling in the message.
    `alt` admits one alternative spelling (the conv weight's upstream
    `[C, 1, 4]` beside the squeezed `[C, 4]` -- same bytes)."""
    if a.shape == shape or (alt is not None and a.shape == alt):
        return a
    raise ValueError(
        f"mojolearn {what}: {name} has shape {a.shape}, want {shape}"
        + (f" (or {alt})" if alt is not None else "")
    )


def _state_buf(a, what, name, shape):
    """A state buffer: float32, C-contiguous, WRITABLE, exactly `shape`.
    No silent fixups at all -- the state is read AND written in place
    (DEVIATION 792), so a copy here would update the copy and the
    caller's next call would carry a stale state, which is a wrong
    answer with no diagnostic."""
    if not isinstance(a, np.ndarray):
        raise TypeError(
            f"mojolearn {what}: state buffer {name} must be a numpy "
            f"ndarray (it is read and written in place), got {type(a)!r}"
        )
    if a.dtype != np.float32:
        raise TypeError(
            f"mojolearn {what}: state buffer {name} has dtype "
            f"{a.dtype.name}, want float32 (the state is part of the "
            "fp32 profile and round-trips byte for byte)"
        )
    if a.shape != shape:
        raise ValueError(
            f"mojolearn {what}: state buffer {name} has shape {a.shape},"
            f" want {shape}"
        )
    if not a.flags["C_CONTIGUOUS"] or not a.flags["WRITEABLE"]:
        raise ValueError(
            f"mojolearn {what}: state buffer {name} must be C-contiguous"
            " and writable; it is updated IN PLACE so the caller's array"
            " always holds the post-call state (allocate_state() makes"
            " conforming buffers)"
        )
    return a


def _batch_tokens(x, what, d_model, step):
    """`x` as (B, L, d_model) float32. A step call also admits
    (B, d_model) -- their `hidden_states.squeeze(1)`, same bytes."""
    x = _f32_strict(x, what, "x")
    if step and x.ndim == 2:
        x = x.reshape(x.shape[0], 1, x.shape[1])
    if x.ndim != 3:
        raise ValueError(
            f"mojolearn {what}: x must be (B, L, d_model)"
            + (" -- or (B, d_model) for a step --" if step else "")
            + f", got {x.ndim}-D shape {x.shape}"
        )
    if x.shape[2] != d_model:
        raise ValueError(
            f"mojolearn {what}: x has d_model {x.shape[2]}, the weights "
            f"have {d_model}"
        )
    if step and x.shape[1] != 1:
        raise ValueError(
            f"mojolearn {what}: a decode step takes exactly one token "
            f"per batch row (mamba_simple.py:210), got L = {x.shape[1]};"
            " use forward() for prefill"
        )
    return x


def _take(weights, what, names):
    """The weight dict's arrays, in a fixed order, refusing missing and
    unknown names so a typo cannot become a silently-untrained weight."""
    if not hasattr(weights, "keys"):
        raise TypeError(
            f"mojolearn {what}: weights must be a dict of numpy arrays "
            f"keyed by the upstream parameter names {names}"
        )
    missing = [n for n in names if n not in weights]
    extra = [n for n in weights if n not in names]
    if missing or extra:
        raise ValueError(
            f"mojolearn {what}: weight dict mismatch"
            + (f"; missing {missing}" if missing else "")
            + (f"; unknown {extra}" if extra else "")
            + f". The exact key set is {list(names)} -- the corpus/HF "
            "parameter names (mamba/corpus/README.md)"
        )
    return [weights[n] for n in names]


class Mamba1State:
    """The Mamba-1 recurrent state, contract section 5's TWO pieces,
    caller-owned (DEVIATION 792):

        conv_window : (B, d_inner, 4) float32 -- the last d_conv
                      PRE-conv inputs, oldest first
        h           : (B, d_inner, 16) float32 -- the SSM state

    Zeros before the first token (`allocate_inference_cache`). Both
    arrays are updated IN PLACE by `forward` and `step`; serialize them
    however you like, the bytes round-trip exactly."""

    def __init__(self, conv_window, h):
        self.conv_window = conv_window
        self.h = h


class Mamba2State:
    """The Mamba-2 recurrent state, its contract section 5's THREE
    pieces, caller-owned (DEVIATION 792). With H = d_model/32 heads and
    CD = 2*d_model + 256 conv channels:

        conv_window     : (B, CD, 4) float32 -- last d_conv PRE-conv xBC
                          inputs, oldest first
        h               : (B, H, 64, 128) float32 -- the chunk-BOUNDARY
                          SSM state (the S17 value as of the last
                          COMPLETED chunk). Setting this nonzero on a
                          FRESH state (buffered_tokens 0) IS upstream's
                          `initial_states` (ssd_minimal.py:64-66)
        buffer_xbc      : (B, 256, CD) float32 -- the open chunk's
                          post-conv/post-SiLU xBC rows
        buffer_dtraw    : (B, 256, H) float32 -- the open chunk's raw
                          in_proj dt rows
        buffered_tokens : int in [0, 256) -- how many buffer rows are
                          valid (decode is PREFILL RESUMPTION, DEVIATION
                          786: the open chunk is recomputed from these)

    All zeros (and 0) before the first token. Updated IN PLACE
    (`buffered_tokens` reassigned) by `forward` and `step`."""

    def __init__(self, conv_window, h, buffer_xbc, buffer_dtraw,
                 buffered_tokens=0):
        self.conv_window = conv_window
        self.h = h
        self.buffer_xbc = buffer_xbc
        self.buffer_dtraw = buffer_dtraw
        self.buffered_tokens = int(buffered_tokens)


class _MambaBase(NumericModeMixin):
    _BINDING = "_mojolearn_mamba"

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


class Mamba1Block(_MambaBase):
    """One Mamba-1 block on the GPU -- norm, mixer, residual
    (`MambaBlock.forward`, transformers modeling_mamba.py:505-530) --
    under profile `mojolearn.identical.mamba1.fp32.v1`
    (`mamba/IDENTICAL_MAMBA_CONTRACT.md`; three-vendor identity card at
    IDENTITY_PATHS row 55 for the LANE -- this Python path itself is
    UNVERIFIED, RUN OWED until `test_mamba_surface.py` prints).

    WEIGHTS IN, AS GIVEN BITS. The constructor takes a dict keyed by the
    upstream parameter names (the corpus's names,
    `mamba/corpus/README.md`); there is no initializer, deliberately --
    bit-reproducing torch's RNG is a refused validation target
    (FEATURE_PARITY.md section 1's initializer row), and a cross-check
    hands both sides the SAME weights.

        norm.weight      (d_model,)
        in_proj.weight   (2*d_inner, d_model)      d_inner = 2*d_model
        conv1d.weight    (d_inner, 1, 4) or (d_inner, 4) -- same bytes
        conv1d.bias      (d_inner,)
        x_proj.weight    (dt_rank + 32, d_inner)   dt_rank = ceil(dm/16)
        dt_proj.weight   (d_inner, dt_rank)
        dt_proj.bias     (d_inner,)
        A_log            (d_inner, 16)
        D                (d_inner,)
        out_proj.weight  (d_model, d_inner)

    WHAT IS HONORED, WHAT IS FIXED, WHAT IS REFUSED -- the parity table
    (`mamba/FEATURE_PARITY.md` section 1) is normative; one line each:

        d_model         honored   free (the arithmetic reads no shape);
                                  gated at 8 and 16, wider is UNVERIFIED
        d_state/d_conv/ FIXED     16 / 4 / 2 / ceil(dm/16): profile
          expand/dt_rank          constants, a different value is a v2
                                  (SHIP LATER, triggers named there)
        conv_bias/bias  FIXED     True / False (the upstream defaults);
                                  the shapes above encode them
        use_fast_path   refused   no such knob: upstream's two arms round
                                  differently (DEVIATION 732), the
                                  reference rounding IS the profile
        initializers    refused   weights arrive as given bits (above)
        dtype           refused   float32 ONLY; bf16/fp16/float64 by name
        activation      silu only, as upstream itself asserts

    STATE IS EXPLICIT (DEVIATION 792): `allocate_state(B)` makes the
    zero state, `forward`/`step` update it in place, and prefill ==
    decode is the contract's construction, not a coincidence (one
    spelling serves both paths; gate D verifies it per stage).

    Non-finite inputs and weights are refused BY NAME IN MOJO, on every
    call for x and the state, once per instance for the ten weight
    names (DEVIATION 1886's cache), before any stage runs."""

    _W_NAMES = (
        "norm.weight", "in_proj.weight", "conv1d.weight", "conv1d.bias",
        "x_proj.weight", "dt_proj.weight", "dt_proj.bias", "A_log", "D",
        "out_proj.weight",
    )

    def __init__(self, weights):
        what = "Mamba1Block"
        arrs = _take(weights, what, self._W_NAMES)
        norm_w = _f32_strict(arrs[0], what, "norm.weight")
        if norm_w.ndim != 1 or norm_w.shape[0] < 1:
            raise ValueError(
                f"mojolearn {what}: norm.weight must be 1-D (d_model,), "
                f"got shape {norm_w.shape}"
            )
        dm = int(norm_w.shape[0])
        di = _M1_EXPAND * dm
        r = int(math.ceil(dm / 16.0))
        xr = r + 2 * _M1_D_STATE
        self.d_model = dm
        self.d_inner = di
        self.dt_rank = r
        # Every weight to float32 C-order ONCE, at construction, shapes
        # checked against the profile's derivation rules so a transposed
        # projection cannot cross as a plausible buffer.
        self._w = [
            norm_w,
            _want_shape(_f32_strict(arrs[1], what, "in_proj.weight"),
                        what, "in_proj.weight", (2 * di, dm)),
            _want_shape(_f32_strict(arrs[2], what, "conv1d.weight"),
                        what, "conv1d.weight", (di, 1, _M1_D_CONV),
                        alt=(di, _M1_D_CONV)),
            _want_shape(_f32_strict(arrs[3], what, "conv1d.bias"),
                        what, "conv1d.bias", (di,)),
            _want_shape(_f32_strict(arrs[4], what, "x_proj.weight"),
                        what, "x_proj.weight", (xr, di)),
            _want_shape(_f32_strict(arrs[5], what, "dt_proj.weight"),
                        what, "dt_proj.weight", (di, r)),
            _want_shape(_f32_strict(arrs[6], what, "dt_proj.bias"),
                        what, "dt_proj.bias", (di,)),
            _want_shape(_f32_strict(arrs[7], what, "A_log"),
                        what, "A_log", (di, _M1_D_STATE)),
            _want_shape(_f32_strict(arrs[8], what, "D"), what, "D", (di,)),
            _want_shape(_f32_strict(arrs[9], what, "out_proj.weight"),
                        what, "out_proj.weight", (dm, di)),
        ]

    def allocate_state(self, batch_size):
        """`allocate_inference_cache(batch_size)`: the zero state whose
        zero window IS prefill's zero padding and whose zero h is the
        scan's initial state (mamba1 DEVIATION 734 for the dropped
        max_seqlen/dtype/device arguments)."""
        b = int(batch_size)
        if b < 1:
            raise ValueError(
                f"mojolearn Mamba1Block: batch_size must be positive, "
                f"got {batch_size!r}"
            )
        return Mamba1State(
            np.zeros((b, self.d_inner, _M1_D_CONV), dtype=np.float32),
            np.zeros((b, self.d_inner, _M1_D_STATE), dtype=np.float32),
        )

    def _call(self, x, state, step):
        what = "Mamba1Block.step" if step else "Mamba1Block.forward"
        x = _batch_tokens(x, what, self.d_model, step)
        b, l = int(x.shape[0]), int(x.shape[1])
        if state is None:
            state = self.allocate_state(b)
        win = _state_buf(state.conv_window, what, "conv_window",
                         (b, self.d_inner, _M1_D_CONV))
        h = _state_buf(state.h, what, "h",
                       (b, self.d_inner, _M1_D_STATE))
        y = np.empty((b, l, self.d_model), dtype=np.float32)
        # TWO LISTS, NOT FOURTEEN ARGUMENTS (DEVIATION 791). Every array
        # addressed below is bound in this frame -- x, the ten entries of
        # self._w (alive on self), win, h, y -- which is what keeps the
        # addresses alive (_arrays.py).
        w = self._w
        ext = self._extension()
        addrs = (
            # ORDER MATCHES bindings/_mojolearn_mamba.mojo::
            # mamba1_forward_binding: x, norm.weight, in_proj.weight,
            # conv1d.weight, conv1d.bias, x_proj.weight, dt_proj.weight,
            # dt_proj.bias, A_log, D, out_proj.weight, conv_window, h,
            # y_out
            [_addr_ro(x)]
            + [_addr_ro(a) for a in w]
            + [_addr(win), _addr(h), _addr(y)]
        )
        if step:
            # B, d_model -- the L = 1 shape is the entry's own contract.
            ext.mamba1_decode_step(addrs, [b, self.d_model])
        else:
            # B, L, d_model.
            ext.mamba1_forward(addrs, [b, l, self.d_model])
        return y

    def forward(self, x, state=None):
        """One block call: `(B, L, d_model)` float32 in, the block
        output (residual add included) back, any B and L.

        `state=None` runs a self-contained prefill from zeros and
        DISCARDS the final state. Pass a `Mamba1State` to carry it: the
        state is read at entry and updated IN PLACE, so a second call
        continues the sequence exactly (the decode gate's per-token
        claim, at any L)."""
        return self._call(x, state, step=False)

    def step(self, x, state):
        """One decode token: `Mamba.step`'s semantics, the profile's
        spelling -- the SAME entry as `forward` at L = 1 with the state
        carried, no arithmetic of its own (contract section 5: two
        spellings that agree today are two spellings that can drift
        tomorrow). `x` is `(B, 1, d_model)` or `(B, d_model)`; `state`
        is REQUIRED, because a stateless decode step has no meaning."""
        if state is None:
            raise ValueError(
                "mojolearn Mamba1Block.step: state is required (a decode "
                "step continues a sequence; allocate_state(B) makes the "
                "fresh one)"
            )
        return self._call(x, state, step=True)

    __call__ = forward


class Mamba2Block(_MambaBase):
    """One Mamba-2 (SSD) block on the GPU -- norm, mixer, residual
    (`Mamba2Block.forward`, HF modeling_mamba2.py:617-631; mixer the
    non-mem-eff arm of mamba2.py:209-276) -- under profile
    `mojolearn.identical.mamba2.fp32.v1`
    (`mamba/IDENTICAL_MAMBA2_CONTRACT.md`). The LANE is gated on the
    Apple M4 (identical tier, the contract's RUN RECORD); there is NO
    cross-vendor card yet and this class claims none -- and this Python
    path itself is UNVERIFIED, RUN OWED until `test_mamba_surface.py`
    prints.

    WEIGHTS IN, AS GIVEN BITS, keyed by the corpus's names
    (`mamba/corpus/README.md`, Mamba-2 section). With d_inner =
    2*d_model, H = d_inner/64 heads, CD = d_inner + 256:

        block_norm.weight  (d_model,)       the BLOCK norm
        in_proj.weight     (2*d_inner + 256 + H, d_model)
                                            column order z | xBC | dt
        conv1d.weight      (CD, 1, 4) or (CD, 4) -- same bytes
        conv1d.bias        (CD,)
        dt_bias            (H,)             a bare parameter, not a
                                            Linear bias (mamba2.py:117)
        A_log              (H,)             PER HEAD, not per state
        D                  (H,)             D_has_hdim False
        norm.weight        (d_inner,)       the GATED norm (RMSNormGated,
                                            gate BEFORE norm --
                                            DEVIATION 787)
        out_proj.weight    (d_model, d_inner)

    WHAT IS HONORED, WHAT IS FIXED, WHAT IS REFUSED -- the parity table
    (`mamba/FEATURE_PARITY.md` section 2) and the contract's section 3
    are normative:

        d_model         honored   any MULTIPLE OF 32 (headdim 64 with
                                  expand 2; refused by name otherwise,
                                  the rule `Mamba2Dims.of` carries --
                                  duplicated here only because this side
                                  cannot even size the state without it)
        dt_limit        honored   a runtime INPUT `(lo, hi)`, default
                                  (0.0, inf) -- the S9 clamp (DEVIATION
                                  788) is pinned whether or not the
                                  limits bind; values cross UNJUDGED
        initial_states  honored   set `state.h` nonzero on a fresh state
                                  (buffered_tokens 0); no extra argument
                                  exists because no extra buffer exists
        d_state/headdim/ FIXED    128 / 64 / 1 / 256: profile constants;
          ngroups/chunk           CHUNK_SIZE is PART OF THE ARITHMETIC
                                  (DEVIATION 783), never a tuning knob
        rmsnorm=False,  refused   by absence: the shapes above encode
          norm_before_            rmsnorm=True, norm_before_gate=False,
          gate=True,              D_has_hdim=False, d_mlp=0 -- each
          D_has_hdim,             SHIP LATER row names its checkpoint
          d_ssm                   trigger in the parity doc
        use_mem_eff_path refused  no such knob (the DEVIATION-732
                                  principle; the fused arm is a sabotage
                                  target, not a mode)
        varlen/TP/CUDA  absent    parity doc sections 2/3/6 carry each
          graphs                  disposition
        dtype           refused   float32 ONLY; bf16/fp16/float64 by name

    STATE IS EXPLICIT AND THREE-PIECE (`Mamba2State`; DEVIATION 792).
    Decode is PREFILL RESUMPTION (DEVIATION 786): `step` is the same
    entry at L = 1, and `forward` with a carried state is chunked-prefill
    continuation. After every call `h_last_` on this instance holds the
    REPORT state (S17 after the final PADDED chunk) -- NOT the resumption
    state; feeding it back as `initial_states` matches resumption only on
    a chunk boundary (contract section 5, gate D2's case)."""

    _W_NAMES = (
        "block_norm.weight", "in_proj.weight", "conv1d.weight",
        "conv1d.bias", "dt_bias", "A_log", "D", "norm.weight",
        "out_proj.weight",
    )

    def __init__(self, weights, *, dt_limit=(0.0, float("inf"))):
        what = "Mamba2Block"
        arrs = _take(weights, what, self._W_NAMES)
        norm_w = _f32_strict(arrs[0], what, "block_norm.weight")
        if norm_w.ndim != 1 or norm_w.shape[0] < 1:
            raise ValueError(
                f"mojolearn {what}: block_norm.weight must be 1-D "
                f"(d_model,), got shape {norm_w.shape}"
            )
        dm = int(norm_w.shape[0])
        if dm % (_M2_HEADDIM // _M2_EXPAND) != 0:
            # Mamba2Dims.of's rule, quoted; the Mojo constructor remains
            # the authority (raw-binding callers hit it), this copy
            # exists because the state buffers below cannot be sized
            # without a whole nheads.
            raise ValueError(
                f"mojolearn {what}: d_model must be a multiple of "
                f"{_M2_HEADDIM // _M2_EXPAND} so that nheads = "
                f"2*d_model/{_M2_HEADDIM} is whole (profile constants "
                "headdim 64, expand 2 -- Mamba2Dims.of carries the same "
                f"refusal); got {dm}"
            )
        di = _M2_EXPAND * dm
        nh = di // _M2_HEADDIM
        cd = di + 2 * _M2_NGROUPS * _M2_D_STATE
        dip = 2 * di + 2 * _M2_NGROUPS * _M2_D_STATE + nh
        self.d_model = dm
        self.d_inner = di
        self.nheads = nh
        self.conv_dim = cd
        # CONVERSION, NOT POLICY: the pair goes down unjudged (DEVIATION
        # 788's clamp defines its own NaN/zero-sign behavior by
        # construction), a non-number still raises here.
        lo, hi = dt_limit
        self.dt_limit = (float(lo), float(hi))
        self._w = [
            norm_w,
            _want_shape(_f32_strict(arrs[1], what, "in_proj.weight"),
                        what, "in_proj.weight", (dip, dm)),
            _want_shape(_f32_strict(arrs[2], what, "conv1d.weight"),
                        what, "conv1d.weight", (cd, 1, _M2_D_CONV),
                        alt=(cd, _M2_D_CONV)),
            _want_shape(_f32_strict(arrs[3], what, "conv1d.bias"),
                        what, "conv1d.bias", (cd,)),
            _want_shape(_f32_strict(arrs[4], what, "dt_bias"),
                        what, "dt_bias", (nh,)),
            _want_shape(_f32_strict(arrs[5], what, "A_log"),
                        what, "A_log", (nh,)),
            _want_shape(_f32_strict(arrs[6], what, "D"), what, "D", (nh,)),
            _want_shape(_f32_strict(arrs[7], what, "norm.weight"),
                        what, "norm.weight", (di,)),
            _want_shape(_f32_strict(arrs[8], what, "out_proj.weight"),
                        what, "out_proj.weight", (dm, di)),
        ]

    def allocate_state(self, batch_size):
        """The zero three-piece state (`allocate_inference_cache`,
        mamba2.py:345-355 plus DEVIATION 786's buffer piece). Set `.h`
        nonzero before the first call for `initial_states`."""
        b = int(batch_size)
        if b < 1:
            raise ValueError(
                f"mojolearn Mamba2Block: batch_size must be positive, "
                f"got {batch_size!r}"
            )
        return Mamba2State(
            np.zeros((b, self.conv_dim, _M2_D_CONV), dtype=np.float32),
            np.zeros((b, self.nheads, _M2_HEADDIM, _M2_D_STATE),
                     dtype=np.float32),
            np.zeros((b, _M2_CHUNK_SIZE, self.conv_dim), dtype=np.float32),
            np.zeros((b, _M2_CHUNK_SIZE, self.nheads), dtype=np.float32),
            0,
        )

    def _call(self, x, state, step):
        what = "Mamba2Block.step" if step else "Mamba2Block.forward"
        x = _batch_tokens(x, what, self.d_model, step)
        b, l = int(x.shape[0]), int(x.shape[1])
        if state is None:
            state = self.allocate_state(b)
        win = _state_buf(state.conv_window, what, "conv_window",
                         (b, self.conv_dim, _M2_D_CONV))
        h = _state_buf(state.h, what, "h",
                       (b, self.nheads, _M2_HEADDIM, _M2_D_STATE))
        bx = _state_buf(state.buffer_xbc, what, "buffer_xbc",
                        (b, _M2_CHUNK_SIZE, self.conv_dim))
        bd = _state_buf(state.buffer_dtraw, what, "buffer_dtraw",
                        (b, _M2_CHUNK_SIZE, self.nheads))
        q0 = int(state.buffered_tokens)
        # buffered_tokens outside [0, 256) is a boundary disagreement the
        # binding refuses by name; it goes down unjudged.
        y = np.empty((b, l, self.d_model), dtype=np.float32)
        h_last = np.empty(
            (b, self.nheads, _M2_HEADDIM, _M2_D_STATE), dtype=np.float32
        )
        # TWO LISTS, NOT SIXTEEN ARGUMENTS (DEVIATION 791). Every array
        # addressed below is bound in this frame -- x, the nine entries
        # of self._w (alive on self), win, h, bx, bd, y, h_last.
        w = self._w
        ext = self._extension()
        addrs = (
            # ORDER MATCHES bindings/_mojolearn_mamba.mojo::
            # mamba2_forward_binding: x, block norm.weight,
            # in_proj.weight, conv1d.weight, conv1d.bias, dt_bias, A_log,
            # D, gated norm.weight, out_proj.weight, conv_window, h,
            # buffer_xbc, buffer_dtraw, y_out, h_last_out
            [_addr_ro(x)]
            + [_addr_ro(a) for a in w]
            + [_addr(win), _addr(h), _addr(bx), _addr(bd),
               _addr(y), _addr(h_last)]
        )
        lo, hi = self.dt_limit
        if step:
            # B, d_model, buf_len, dt_lo, dt_hi.
            new_len = ext.mamba2_decode_step(
                addrs, [b, self.d_model, q0, lo, hi]
            )
        else:
            # B, L, d_model, buf_len, dt_lo, dt_hi.
            new_len = ext.mamba2_forward(
                addrs, [b, l, self.d_model, q0, lo, hi]
            )
        state.buffered_tokens = int(new_len)
        #: The REPORT state (S17 after the final PADDED chunk), NOT the
        #: resumption state -- contract section 5's distinction.
        self.h_last_ = h_last
        return y

    def forward(self, x, state=None):
        """One block call: `(B, L, d_model)` float32 in, the block
        output back, any B and L. `state=None` runs a self-contained
        prefill from zeros and DISCARDS the final state; pass a
        `Mamba2State` to carry it -- a later `forward` or `step` on that
        state is chunked-prefill continuation / decode, bit-for-bit the
        prefill that ran the whole sequence at once (DEVIATION 786's
        construction; the identical tier's gates verify it)."""
        return self._call(x, state, step=False)

    def step(self, x, state):
        """One decode token: `Mamba2.step`'s semantics, the profile's
        spelling -- PREFILL RESUMPTION at L = 1 through the same entry as
        `forward` (DEVIATION 786; upstream's own per-token recurrence
        rounds differently BY CONSTRUCTION and is the lane's required-RED
        sabotage arm, never a mode here). `state` is REQUIRED."""
        if state is None:
            raise ValueError(
                "mojolearn Mamba2Block.step: state is required (a decode "
                "step continues a sequence; allocate_state(B) makes the "
                "fresh one)"
            )
        return self._call(x, state, step=True)

    __call__ = forward
