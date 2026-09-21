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

THIS LANE IS IDENTICAL-ONLY (2026-09-10). It used to build all three
tiers. It never should have: the fused kernels in `mamba/impl/ops/` and
`mamba/impl/modules/` were gated on `GLOBAL_NUMERIC_MODE ==
NUMERIC_IDENTICAL`, so a FAST or DETERMINISTIC build fell back to the
unfused arms and the legacy host copies and ran SLOWER than the default
while promising less. DEVIATION 2300 is what that cost: a `k_last`
failure against ref64 that existed ONLY in the deterministic tier, found
in a shipped 0.7.0 qualification. `numeric_mode=` now accepts
'identical' or nothing; anything else raises. `_extension()` still checks
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
`mamba/PUBLIC_ALPHA_SURFACE.md`; older comments are not release certificates.
"""

from . import _buffer as _buffers, _bufcheck as _checks
from ._array import Array as _Array
from ._arrays import _addr, _addr_ro
from . import _portable_math as math

from . import _backend
from . import lowbit as _lowbit
from ._buffer import addr, addr_ro, all_finite, as_f32_c, empty, zeros
from ._bufcheck import dtype_name, is_native_f32, memcopy, probe
from ._mode import NumericModeMixin
from . import _ragged

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


def _f32_strict(a, what, name):
    """`a` as a C-contiguous float32 `Array`, REFUSING every other dtype
    BY NAME (module header: this surface certifies bits, so there is no
    convenience downcast). Layout may be fixed silently -- a copy to C
    order moves bytes untouched, so no output bit can move -- but the
    dtype may not. DEVIATION 2407: the dtype is the buffer's FORMAT,
    read through `_buffer.view`; `as_f32_c` is reached only by a float32
    buffer, where its one job is the layout (a zero-copy borrow when the
    buffer is already C-contiguous)."""
    try:
        pb = probe(a)
    except TypeError:
        raise TypeError(
            f"mojolearn {what}: {name} is a {type(a).__name__}, which does "
            "not support the buffer protocol; this surface is float32 ONLY "
            "and takes a float32 array (a numpy array, an array.array('f') "
            "or a mojolearn.Array). Convert yourself and pass float32."
        ) from None
    if not is_native_f32(pb.format):
        raise TypeError(
            f"mojolearn {what}: {name} has dtype {dtype_name(a, pb)}; this "
            "surface is float32 ONLY (the fp32 profiles, "
            "mamba/IDENTICAL_MAMBA_CONTRACT.md section 3 / "
            "IDENTICAL_MAMBA2_CONTRACT.md section 3). bfloat16 and "
            "float16 are refused BY NAME: the references' "
            "reduced-precision runs are a mixture of cast boundaries "
            "that is not this profile, and a certified bf16 profile is "
            "SHIP LATER under its own version name "
            "(archive/evidence/mamba/FEATURE_PARITY.md section 7). float64 is refused "
            "rather than downcast so the bits that ran are bits you "
            "made. Convert yourself and pass float32."
        )
    arr, _copied = as_f32_c(a, ndim=pb.ndim, name=name)
    return arr


def _want_shape(a, what, name, shape, alt=None):
    """Exact-shape check with the expected spelling in the message.
    `alt` admits one alternative spelling (the conv weight's reference
    `[C, 1, 4]` beside the squeezed `[C, 4]` -- same bytes)."""
    if a.shape == shape or (alt is not None and a.shape == alt):
        return a
    raise ValueError(
        f"mojolearn {what}: {name} has shape {a.shape}, want {shape}"
        + (f" (or {alt})" if alt is not None else "")
    )


def _exports(ext, name):
    """Whether the loaded binding exports `name`. On a CPU-only install the
    stand-in for a GPU binding RAISES ImportError by name from `__getattr__`
    (`_backend.py::_HostBinding`), and `hasattr` swallows only
    AttributeError, so a bare `hasattr` probe would take the whole CPU host
    route down. The same guard `_transformer_impl._exports` carries, for the
    same reason; a probe is not a use."""
    try:
        return hasattr(ext, name)
    except ImportError:
        return False


def _private_copy(a):
    """A fresh float32 C-order buffer holding `a`'s bytes.

    The resident decode session's OWNERSHIP clause: the session COPIES the
    weights and the state at open, so a caller's later edit to either is not
    observed. On the GPU arm the copy lands in device memory
    (`MambaDeviceWeights`, `MambaDeviceState`); on the host arm it lands
    here, in a buffer of this process, which is the same promise with the
    same visibility and the same refresh door (`load_state`)."""
    pb = probe(a)
    out = empty(tuple(int(d) for d in pb.shape), "<f4")
    memcopy(addr(out, name="copy"), addr_ro(a, name="source"), 4 * int(out.size))
    return out


def _state_buf(a, what, name, shape):
    """A state buffer: float32, C-contiguous, WRITABLE, exactly `shape`.
    No silent fixups at all -- the state is read AND written in place
    (DEVIATION 792), so a copy here would update the copy and the
    caller's next call would carry a stale state, which is a wrong
    answer with no diagnostic. DEVIATION 2408: "a buffer" is ANY object
    supporting the buffer protocol -- the `mojolearn.Array` that
    `allocate_state` hands out, a NumPy array, an `array.array('f')` --
    judged through `_buffer.view`; the object itself is returned and its
    address taken with `_buffer.addr` (writable required), so the
    caller's own memory is what the kernel updates."""
    try:
        pb = probe(a)
    except TypeError:
        raise TypeError(
            f"mojolearn {what}: state buffer {name} must be a numpy "
            "ndarray or a mojolearn Array -- any writable float32 buffer "
            f"-- (it is read and written in place), got {type(a)!r}"
        ) from None
    if not is_native_f32(pb.format):
        raise TypeError(
            f"mojolearn {what}: state buffer {name} has dtype "
            f"{dtype_name(a, pb)}, want float32 (the state is part of the "
            "fp32 profile and round-trips byte for byte)"
        )
    if pb.shape != shape:
        raise ValueError(
            f"mojolearn {what}: state buffer {name} has shape {pb.shape},"
            f" want {shape}"
        )
    if not pb.c_contiguous or pb.readonly:
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
        x = x.reshape((x.shape[0], 1, x.shape[1]))
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
            f"mojolearn {what}: weights must be a dict of float32 arrays "
            "(numpy arrays or mojolearn Arrays) keyed by the upstream "
            f"parameter names {names}"
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
    however you like, the bytes round-trip exactly. They are
    `mojolearn.Array` when `allocate_state` made them and may be ANY
    writable float32 buffer (a NumPy array included) when you did."""

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
                          FRESH state (buffered_tokens 0) IS the reference's
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

    def _prefill_backward(self, x, grad_output, entry):
        what = type(self).__name__ + ".backward"
        mode = getattr(self, "numeric_mode", None) or _backend.default_mode()
        if mode != "identical":
            raise NotImplementedError(
                f"mojolearn {what}: only IDENTICAL zero-state prefill is "
                f"implemented; got numeric_mode={mode!r}"
            )
        x = _batch_tokens(x, what, self.d_model, False)
        b, l, _ = x.shape
        if b < 1 or l < 1:
            raise ValueError(f"mojolearn {what}: B and L must be positive")
        dy = _want_shape(_f32_strict(grad_output, what, "grad_output"),
                         what, "grad_output", x.shape)
        if not all_finite(dy):
            raise ValueError(f"mojolearn {what}: grad_output must be finite")
        kwargs = {"dt_limit": self.dt_limit} if entry == "mamba2_backward" else {}
        checked = type(self)(dict(zip(self._W_NAMES, self._w)),
                             numeric_mode="identical", **kwargs)
        if checked.d_model != self.d_model:
            raise ValueError(f"mojolearn {what}: current weights changed d_model")
        weights = checked._w
        # Resolve before allocating gradients so an older binary fails clearly.
        extension = self._extension()
        native = getattr(extension, entry, None)
        if native is None:
            raise RuntimeError(
                f"mojolearn {what}: loaded Mamba extension lacks {entry}; "
                "rebuild bindings/build_mamba.sh in IDENTICAL mode"
            )
        # DEVIATION 2409: the gradients are fresh `mojolearn.Array`s.
        gradients = ([empty(x.shape, "<f4")]
                     + [empty(w.shape, "<f4") for w in weights])
        addresses = ([addr_ro(x, name="x")]
                     + [addr_ro(w, name="weight") for w in weights]
                     + [addr_ro(dy, name="grad_output")]
                     + [addr(g, name="gradient") for g in gradients])
        params = [b, l, self.d_model]
        if entry == "mamba2_backward":
            params.extend(checked.dt_limit)
        native(addresses, params)
        return dict(zip(("x",) + self._W_NAMES, gradients))


class Mamba1Block(_MambaBase):
    """One Mamba-1 block on the GPU -- norm, mixer, residual
    (`MambaBlock.forward`, transformers modeling_mamba.py:505-530) --
    under profile `mojolearn.identical.mamba1.fp32.v1`
    (`mamba/IDENTICAL_MAMBA_CONTRACT.md`). The module evidence snapshot
    distinguishes native cross-vendor certificates from Python API checks;
    the Python surface exposes forward, state continuation and a zero-state
    IDENTICAL prefill backward operation.

    WEIGHTS IN, AS GIVEN BITS. The constructor takes a dict keyed by the
    reference parameter names (the corpus's names,
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
    (`archive/evidence/mamba/FEATURE_PARITY.md` section 1) is normative; one line each:

        d_model         honored   free (the arithmetic reads no shape);
                                  gated at 8 and 16, wider is UNVERIFIED
        d_state/d_conv/ FIXED     16 / 4 / 2 / ceil(dm/16): profile
          expand/dt_rank          constants, a different value is a v2
                                  (SHIP LATER, triggers named there)
        conv_bias/bias  FIXED     True / False (the reference defaults);
                                  the shapes above encode them
        use_fast_path   refused   no such knob: the reference's two arms round
                                  differently (DEVIATION 732), the
                                  reference rounding IS the profile
        initializers    refused   weights arrive as given bits (above)
        dtype           refused   float32 ONLY; bf16/fp16/float64 by name
        activation      silu only, as the reference itself asserts

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
        # lane/identical-lowbit-inference (2026-09-17): packed projection
        # weights (mojolearn.lowbit) materialized exactly, fp32 path after.
        weights, self.weight_format = _lowbit.unpack(weights, what)
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
            zeros((b, self.d_inner, _M1_D_CONV), "<f4"),
            zeros((b, self.d_inner, _M1_D_STATE), "<f4"),
        )

    def _call(self, x, state, step):
        what = "Mamba1Block.step" if step else "Mamba1Block.forward"
        x = _batch_tokens(x, what, self.d_model, step)
        b, l = int(x.shape[0]), int(x.shape[1])
        if state is None:
            state = self.allocate_state(b)
        _refuse_resident(state, what)
        win = _state_buf(state.conv_window, what, "conv_window",
                         (b, self.d_inner, _M1_D_CONV))
        h = _state_buf(state.h, what, "h",
                       (b, self.d_inner, _M1_D_STATE))
        y = empty((b, l, self.d_model), "<f4")
        # TWO LISTS, NOT FOURTEEN ARGUMENTS (DEVIATION 791). Every array
        # addressed below is bound in this frame -- x, the ten entries of
        # self._w (alive on self), win, h, y -- which is what keeps the
        # addresses alive (_buffer.py). `addr` (writable) for the state
        # pieces and the output, `addr_ro` for x and the weights.
        w = self._w
        ext = self._extension()
        addrs = (
            # ORDER MATCHES bindings/_mojolearn_mamba.mojo::
            # mamba1_forward_binding: x, norm.weight, in_proj.weight,
            # conv1d.weight, conv1d.bias, x_proj.weight, dt_proj.weight,
            # dt_proj.bias, A_log, D, out_proj.weight, conv_window, h,
            # y_out
            [addr_ro(x, name="x")]
            + [addr_ro(a, name="weight") for a in w]
            + [addr(win, name="conv_window"), addr(h, name="h"),
               addr(y, name="y")]
        )
        if step:
            # B, d_model -- the L = 1 shape is the entry's own contract.
            ext.mamba1_decode_step(addrs, [b, self.d_model])
        else:
            # B, L, d_model.
            ext.mamba1_forward(addrs, [b, l, self.d_model])
        return y

    def forward(self, x, state=None, *, lengths=None):
        """One block call: `(B, L, d_model)` float32 in, the block
        output (residual add included) back, any B and L.

        `state=None` runs a self-contained prefill from zeros and
        DISCARDS the final state. Pass a `Mamba1State` to carry it: the
        state is read at entry and updated IN PLACE, so a second call
        continues the sequence exactly (the decode gate's per-token
        claim, at any L).

        `lengths` (2026-09-15) makes the batch RAGGED: `B` integers in
        `[1, L]`, row `i` real at positions `[0, lengths[i])` and padding
        after. Every real position's output is byte for byte the row run
        alone at its own length (the scan is causal and the contract's
        clause (c) makes a row independent of its batch; no arithmetic
        changes, `_ragged.py` says why) and every padding position's
        output is exactly `+0.0`, whatever the input held there. Refused
        with a carried `state`."""
        if lengths is not None:
            what = type(self).__name__ + ".forward"
            x = _batch_tokens(x, what, self.d_model, False)
            return _ragged.ragged_forward(lambda xp: self._call(xp, None, step=False),
                                          x, state, lengths, "<f4", what)[0]
        return self._call(x, state, step=False)

    def backward(self, x, grad_output):
        """Return a zero-state prefill VJP as independent named float32 arrays.

        The result contains ``x`` and the ten weight names accepted by the
        constructor. ``grad_output`` has the same (B, L, d_model) shape as x.
        Forward is recomputed with the current weights on every call; no
        preceding forward call or cached activations are used. This operation
        is synchronous and supports IDENTICAL only. Stateful/decode backward
        and gradients of final recurrent state are not exposed.
        """
        what = "Mamba1Block.backward"
        mode = getattr(self, "numeric_mode", None) or _backend.default_mode()
        if mode != "identical":
            raise NotImplementedError(
                f"mojolearn {what}: only IDENTICAL zero-state prefill is "
                f"implemented; got numeric_mode={mode!r}"
            )
        x = _batch_tokens(x, what, self.d_model, False)
        b, l, _ = x.shape
        if b < 1 or l < 1:
            raise ValueError(f"mojolearn {what}: B and L must be positive")
        dy = _want_shape(_f32_strict(grad_output, what, "grad_output"),
                         what, "grad_output", x.shape)
        if not all_finite(dy):
            raise ValueError(f"mojolearn {what}: grad_output must be finite")
        # Revalidate current weight layouts before exposing raw addresses.
        # In-place value changes are intentional; dtype/shape mutation is not.
        checked = Mamba1Block(dict(zip(self._W_NAMES, self._w)),
                              numeric_mode="identical")
        if checked.d_model != self.d_model:
            raise ValueError(f"mojolearn {what}: current weights changed d_model")
        weights = checked._w
        extension = self._extension()
        native = getattr(extension, "mamba1_backward", None)
        if native is None:
            raise RuntimeError(
                f"mojolearn {what}: loaded Mamba extension lacks mamba1_backward; "
                "install a current alpha wheel with IDENTICAL Mamba backward support "
                "or rebuild bindings/build_mamba.sh in IDENTICAL mode"
            )
        # DEVIATION 2409: the gradients are fresh `mojolearn.Array`s.
        gradients = ([empty(x.shape, "<f4")]
                     + [empty(w.shape, "<f4") for w in weights])
        addresses = ([addr_ro(x, name="x")]
                     + [addr_ro(w, name="weight") for w in weights]
                     + [addr_ro(dy, name="grad_output")]
                     + [addr(g, name="gradient") for g in gradients])
        native(addresses, [b, l, self.d_model])
        return dict(zip(("x",) + self._W_NAMES, gradients))

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

    def decode_session(self, state):
        """A `Mamba1DecodeSession` on this block and `state`: the ten
        weights, the two state pieces and the L = 1 stages RESIDENT on the
        device across decode steps (DEVIATION 2941, lane/infer-speed-neural,
        2026-09-17). `step` there is `Mamba1Block.step`'s bytes: the same
        certified decode entry (`mamba_step`, the block at L = 1) on the
        same structs, built once instead of per token.

        OWNERSHIP. The session COPIES the weights and the state at open.
        Until `close()` (or `sync_state()`) the state's `conv_window` and
        `h` are STALE and `forward`/`step` on this block refuse the state
        by name. Edits to the weights after open are NOT observed: close
        and open a new session. Edits to the state arrays are not observed
        either: `load_state()` re-uploads them.

        THE CPU HOST ROUTE TAKES THIS DOOR TOO (lane/cpu-routes-gpu-only-four,
        2026-09-20). A host binding exports no `mamba1_session_*` entry and
        never will -- there is no device context to hold anything resident in
        -- but the session's ARITHMETIC is not the residency: `step` is
        `mamba_step`, the block at L = 1 with the state carried, which the
        host binding exports under the per-call name `mamba1_decode_step`
        (`bindings/_mojolearn_mamba_host.mojo`, `mamba_block_oracle` at L = 1,
        the same entry `Mamba1Block.step` takes). So on the host route this
        object holds its own copies of the ten weights and the two state
        pieces and calls that entry per token, in the same order and with the
        same addresses `Mamba1Block._call` builds. It is the ownership
        wrapper, not new arithmetic, and it earns no speed claim of any kind:
        on the host every step re-reads the weights exactly as the per-call
        step does."""
        return Mamba1DecodeSession(self, state)

    __call__ = forward


def _refuse_resident(state, what):
    owner = getattr(state, "_resident_session", None)
    if owner is not None:
        raise ValueError(
            f"mojolearn {what}: the state is owned by an open resident "
            "decode session (its pieces live in the session, on the "
            "device or on the host, and the caller's buffers are stale); "
            "call sync_state() or close() on the session first"
        )


class Mamba1DecodeSession:
    """Resident decode on one `Mamba1Block` and one `Mamba1State`
    (DEVIATION 2941). Made by `Mamba1Block.decode_session(state)`.

    `step(x)`      one decode token per row, `(B, 1, d_model)` or
                   `(B, d_model)` float32 in, `(B, 1, d_model)` out
    `sync_state()` copy the resident state into the state's buffers
    `load_state()` re-upload the state's buffers (the explicit refresh)
    `close()`      sync, release every device buffer and hand the state
                   back; also the context manager exit

    Every output is BYTE FOR BYTE the per-call `step` on the same block
    and state. On a GPU binding the session holds its own device context;
    it is not thread-safe and refuses re-entrant use.

    TWO ARMS, ONE CLASS AND ONE SET OF BYTES. On a binding that exports
    `mamba1_session_create` the weights, the state and the L = 1 stages are
    RESIDENT on the device across steps. On the host route
    (`bindings/_mojolearn_mamba_host.mojo`, which exports no session entry)
    the session owns its copies HERE and each `step` is the host binding's
    `mamba1_decode_step` on them: the same certified entry, the same
    fourteen addresses in the same order, so the byte-for-byte claim above
    holds on both arms and is what the `mamba1-decode-session` lane hashes.
    The host arm is the OWNERSHIP wrapper only and makes NO speed claim: it
    re-reads the weights on every step exactly as the per-call step does."""

    def __init__(self, block, state):
        what = "Mamba1DecodeSession"
        ext = block._extension()
        try:
            # The CPU-only stand-in raises ImportError BY NAME from
            # __getattr__ (_backend.py::_HostBinding); a probe is not a use.
            create = getattr(ext, "mamba1_session_create", None)
        except ImportError:
            create = None
        # THE HOST ARM. No device session entry, but the decode entry the
        # session would have run is right there under its per-call name.
        host = create is None and _exports(ext, "mamba1_decode_step")
        if create is None and not host:
            raise NotImplementedError(
                f"mojolearn {what}: the loaded {type(block).__name__} binding "
                "exports neither a resident decode session nor the per-call "
                "mamba1_decode_step entry the host arm runs; rebuild "
                "bindings/build_mamba.sh (or build_mamba_host.sh) in "
                "IDENTICAL mode"
            )
        if state is None:
            raise ValueError(f"mojolearn {what}: state is required (allocate_state)")
        _refuse_resident(state, what)
        pb = probe(state.h)
        if len(pb.shape) != 3 or pb.shape[0] < 1:
            raise ValueError(f"mojolearn {what}: state.h must be (B, d_inner, 16)")
        b = int(pb.shape[0])
        win = _state_buf(state.conv_window, what, "conv_window", (b, block.d_inner, _M1_D_CONV))
        h = _state_buf(state.h, what, "h", (b, block.d_inner, _M1_D_STATE))
        self._block = block
        self._state = state
        self._ext = ext
        self._open = False
        self._b = b
        w = block._w
        if host:
            # The ten weights and the two state pieces COPIED, which is the
            # ownership clause the device arm satisfies with an upload.
            self._native = None
            self._hw = [_private_copy(a) for a in w]
            self._win = _private_copy(win)
            self._h = _private_copy(h)
        else:
            self._native = create()
            addrs = ([addr_ro(a, name="weight") for a in w]
                     + [addr(win, name="conv_window"), addr(h, name="h")])
            ext.mamba1_session_open(self._native, addrs, [b, block.d_model])
        self._open = True
        state._resident_session = self

    @property
    def state(self):
        return self._state

    @property
    def is_open(self):
        return self._open

    def _require_open(self, what):
        if not self._open:
            raise ValueError(f"mojolearn {what}: the session is closed")

    def step(self, x):
        what = "Mamba1DecodeSession.step"
        self._require_open(what)
        blk = self._block
        x = _batch_tokens(x, what, blk.d_model, True)
        b = int(x.shape[0])
        if b != self._b:
            raise ValueError(f"mojolearn {what}: the session holds {self._b} rows, x has B = {b}")
        y = empty((b, 1, blk.d_model), "<f4")
        if self._native is None:
            # ORDER MATCHES `Mamba1Block._call`, which is the order
            # `bindings/_mojolearn_mamba_host.mojo::mamba1_decode_step`
            # documents: x, the ten weights, conv_window, h, y_out. The
            # session's own buffers stand where the caller's would.
            hw = self._hw
            self._ext.mamba1_decode_step(
                [addr_ro(x, name="x")]
                + [addr_ro(a, name="weight") for a in hw]
                + [addr(self._win, name="conv_window"), addr(self._h, name="h"),
                   addr(y, name="y")],
                [b, blk.d_model])
        else:
            self._ext.mamba1_session_step(self._native, [addr_ro(x, name="x"), addr(y, name="y")])
        return y

    def _state_addrs(self, what):
        st, blk = self._state, self._block
        win = _state_buf(st.conv_window, what, "conv_window", (self._b, blk.d_inner, _M1_D_CONV))
        h = _state_buf(st.h, what, "h", (self._b, blk.d_inner, _M1_D_STATE))
        return win, h

    def sync_state(self):
        what = "Mamba1DecodeSession.sync_state"
        self._require_open(what)
        win, h = self._state_addrs(what)
        if self._native is None:
            memcopy(addr(win, name="conv_window"), addr_ro(self._win, name="resident"),
                    4 * int(self._win.size))
            memcopy(addr(h, name="h"), addr_ro(self._h, name="resident"), 4 * int(self._h.size))
        else:
            self._ext.mamba1_session_export_state(
                self._native, [addr(win, name="conv_window"), addr(h, name="h")])
        return self._state

    def load_state(self):
        what = "Mamba1DecodeSession.load_state"
        self._require_open(what)
        win, h = self._state_addrs(what)
        if self._native is None:
            memcopy(addr(self._win, name="resident"), addr_ro(win, name="conv_window"),
                    4 * int(self._win.size))
            memcopy(addr(self._h, name="resident"), addr_ro(h, name="h"), 4 * int(self._h.size))
        else:
            self._ext.mamba1_session_load_state(
                self._native, [addr(win, name="conv_window"), addr(h, name="h")])
        return self._state

    def _release(self):
        if self._native is None:
            self._hw = self._win = self._h = None
        else:
            self._ext.mamba1_session_close(self._native)

    def close(self):
        if not self._open:
            return
        try:
            self.sync_state()
        finally:
            self._open = False
            self._state._resident_session = None
            self._release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __del__(self):
        try:
            if getattr(self, "_open", False):
                self._open = False
                self._state._resident_session = None
                self._release()
        except Exception:
            pass

    def __repr__(self):
        return "Mamba1DecodeSession(open=%r, rows=%d)" % (self._open, self._b)


class Mamba2Block(_MambaBase):
    """One Mamba-2 (SSD) block on the GPU -- norm, mixer, residual
    (`Mamba2Block.forward`, HF modeling_mamba2.py:617-631; mixer the
    non-mem-eff arm of mamba2.py:209-276) -- under profile
    `mojolearn.identical.mamba2.fp32.v1`
    (`mamba/IDENTICAL_MAMBA2_CONTRACT.md`). Native certificates include
    the five-case Mamba-1/2/3 backward evidence described in the module
    snapshot. Python forward/state qualification of the state-allocation
    fix is retained on Apple; NVIDIA and AMD requalification is pending.
    All three blocks expose zero-state IDENTICAL prefill VJPs. Qualification
    of these new Python entries is separate from historical forward-only results.

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
    (`archive/evidence/mamba/FEATURE_PARITY.md` section 2) and the contract's section 3
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
        # lane/identical-lowbit-inference (2026-09-17): packed projection
        # weights (mojolearn.lowbit) materialized exactly, fp32 path after.
        weights, self.weight_format = _lowbit.unpack(weights, what)
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
            zeros((b, self.conv_dim, _M2_D_CONV), "<f4"),
            zeros((b, self.nheads, _M2_HEADDIM, _M2_D_STATE), "<f4"),
            zeros((b, _M2_CHUNK_SIZE, self.conv_dim), "<f4"),
            zeros((b, _M2_CHUNK_SIZE, self.nheads), "<f4"),
            0,
        )

    def _call(self, x, state, step):
        what = "Mamba2Block.step" if step else "Mamba2Block.forward"
        x = _batch_tokens(x, what, self.d_model, step)
        b, l = int(x.shape[0]), int(x.shape[1])
        if state is None:
            state = self.allocate_state(b)
        _refuse_resident(state, what)
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
        y = empty((b, l, self.d_model), "<f4")
        h_last = empty((b, self.nheads, _M2_HEADDIM, _M2_D_STATE), "<f4")
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
            [addr_ro(x, name="x")]
            + [addr_ro(a, name="weight") for a in w]
            + [addr(win, name="conv_window"), addr(h, name="h"),
               addr(bx, name="buffer_xbc"), addr(bd, name="buffer_dtraw"),
               addr(y, name="y"), addr(h_last, name="h_last")]
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

    def forward(self, x, state=None, *, lengths=None):
        """One block call: `(B, L, d_model)` float32 in, the block
        output back, any B and L. `state=None` runs a self-contained
        prefill from zeros and DISCARDS the final state; pass a
        `Mamba2State` to carry it -- a later `forward` or `step` on that
        state is chunked-prefill continuation / decode, bit-for-bit the
        prefill that ran the whole sequence at once (DEVIATION 786's
        construction; the identical tier's gates verify it).

        `lengths` (2026-09-15) makes the batch RAGGED: `B` integers in
        `[1, L]`, row `i` real at positions `[0, lengths[i])` and padding
        after. Every real position's output is byte for byte the row run
        alone at its own length (the scan is causal and the contract's
        clause (c) makes a row independent of its batch; no arithmetic
        changes, `_ragged.py` says why) and every padding position's
        output is exactly `+0.0`, whatever the input held there. Refused
        with a carried `state`."""
        if lengths is not None:
            what = type(self).__name__ + ".forward"
            x = _batch_tokens(x, what, self.d_model, False)
            return _ragged.ragged_forward(lambda xp: self._call(xp, None, step=False),
                                          x, state, lengths, "<f4", what)[0]
        return self._call(x, state, step=False)

    def backward(self, x, grad_output):
        """Return the zero-state prefill VJP for x and all nine weights.

        IDENTICAL float32 only. Forward is recomputed using current weights;
        no previous forward or cached state is consumed or modified. Returns
        independent arrays under the constructor's exact weight names plus
        ``x``. Incoming-cache and final-state cotangents are not supported.
        Retained NVIDIA fixture checks and outstanding wheel/vendor scopes are
        listed in mamba/PUBLIC_ALPHA_SURFACE.md.
        """
        return self._prefill_backward(x, grad_output, "mamba2_backward")

    def step(self, x, state):
        """One decode token: `Mamba2.step`'s semantics, the profile's
        spelling -- PREFILL RESUMPTION at L = 1 through the same entry as
        `forward` (DEVIATION 786; the reference's own per-token recurrence
        rounds differently BY CONSTRUCTION and is the lane's required-RED
        sabotage arm, never a mode here). `state` is REQUIRED."""
        if state is None:
            raise ValueError(
                "mojolearn Mamba2Block.step: state is required (a decode "
                "step continues a sequence; allocate_state(B) makes the "
                "fresh one)"
            )
        return self._call(x, state, step=True)

    def decode_session(self, state):
        """Keep Mamba-2 weights and recurrent state resident for repeated
        single-token decode.  The session runs the same L=1 block entry as
        :meth:`step`; close or sync it before using ``state`` elsewhere."""
        return Mamba2DecodeSession(self, state)

    __call__ = forward


class Mamba2DecodeSession:
    """Resident repeated decode for one ``Mamba2Block`` and state."""

    def __init__(self, block, state):
        what = "Mamba2DecodeSession"
        ext = block._extension()
        try:
            create = getattr(ext, "mamba2_session_create", None)
        except ImportError:
            create = None
        host = create is None and _exports(ext, "mamba2_decode_step")
        if create is None and not host:
            raise NotImplementedError(f"mojolearn {what}: binding lacks decode entries")
        if state is None:
            raise ValueError(f"mojolearn {what}: state is required")
        _refuse_resident(state, what)
        pb = probe(state.h)
        if len(pb.shape) != 4 or pb.shape[0] < 1:
            raise ValueError(f"mojolearn {what}: state.h has invalid shape")
        b = int(pb.shape[0])
        win = _state_buf(state.conv_window, what, "conv_window", (b, block.conv_dim, _M2_D_CONV))
        h = _state_buf(state.h, what, "h", (b, block.nheads, _M2_HEADDIM, _M2_D_STATE))
        bx = _state_buf(state.buffer_xbc, what, "buffer_xbc", (b, _M2_CHUNK_SIZE, block.conv_dim))
        bd = _state_buf(state.buffer_dtraw, what, "buffer_dtraw", (b, _M2_CHUNK_SIZE, block.nheads))
        self._block, self._state, self._ext = block, state, ext
        self._b, self._open = b, False
        if host:
            self._native = None
            self._hw = [_private_copy(a) for a in block._w]
            self._parts = [_private_copy(a) for a in (win, h, bx, bd)]
            self._q = int(state.buffered_tokens)
        else:
            self._native = create()
            lo, hi = block.dt_limit
            ext.mamba2_session_open(
                self._native,
                [addr_ro(a, name="weight") for a in block._w]
                + [addr(win, name="conv_window"), addr(h, name="h"),
                   addr(bx, name="buffer_xbc"), addr(bd, name="buffer_dtraw")],
                [b, block.d_model, int(state.buffered_tokens), lo, hi])
        self._open = True
        state._resident_session = self

    @property
    def state(self): return self._state

    @property
    def is_open(self): return self._open

    def _require_open(self, what):
        if not self._open:
            raise ValueError(f"mojolearn {what}: the session is closed")

    def _state_parts(self, what):
        st, b, blk = self._state, self._b, self._block
        return (
            _state_buf(st.conv_window, what, "conv_window", (b, blk.conv_dim, _M2_D_CONV)),
            _state_buf(st.h, what, "h", (b, blk.nheads, _M2_HEADDIM, _M2_D_STATE)),
            _state_buf(st.buffer_xbc, what, "buffer_xbc", (b, _M2_CHUNK_SIZE, blk.conv_dim)),
            _state_buf(st.buffer_dtraw, what, "buffer_dtraw", (b, _M2_CHUNK_SIZE, blk.nheads)),
        )

    def step(self, x):
        what = "Mamba2DecodeSession.step"
        self._require_open(what)
        blk = self._block
        x = _batch_tokens(x, what, blk.d_model, True)
        if int(x.shape[0]) != self._b:
            raise ValueError(f"mojolearn {what}: batch size changed")
        y = empty((self._b, 1, blk.d_model), "<f4")
        report = empty((self._b, blk.nheads, _M2_HEADDIM, _M2_D_STATE), "<f4")
        if self._native is None:
            addrs = ([addr_ro(x, name="x")]
                     + [addr_ro(a, name="weight") for a in self._hw]
                     + [addr(a, name="state") for a in self._parts]
                     + [addr(y, name="y"), addr(report, name="h_last")])
            lo, hi = blk.dt_limit
            self._q = int(self._ext.mamba2_decode_step(
                addrs, [self._b, blk.d_model, self._q, lo, hi]))
        else:
            self._q = int(self._ext.mamba2_session_step(
                self._native, [addr_ro(x, name="x"), addr(y, name="y"),
                               addr(report, name="h_last")]))
        blk.h_last_ = report
        return y

    def sync_state(self):
        what = "Mamba2DecodeSession.sync_state"
        self._require_open(what)
        parts = self._state_parts(what)
        if self._native is None:
            for dst, src in zip(parts, self._parts):
                memcopy(addr(dst, name="state"), addr_ro(src, name="resident"), 4 * int(src.size))
            q = self._q
        else:
            q = int(self._ext.mamba2_session_export_state(
                self._native, [addr(a, name="state") for a in parts]))
        self._state.buffered_tokens = q
        return self._state

    def load_state(self):
        what = "Mamba2DecodeSession.load_state"
        self._require_open(what)
        parts = self._state_parts(what)
        q = int(self._state.buffered_tokens)
        if self._native is None:
            for dst, src in zip(self._parts, parts):
                memcopy(addr(dst, name="resident"), addr_ro(src, name="state"), 4 * int(dst.size))
            self._q = q
        else:
            self._ext.mamba2_session_load_state(
                self._native, [addr(a, name="state") for a in parts], [q])
        return self._state

    def _release(self):
        if self._native is None:
            self._hw = self._parts = None
        else:
            self._ext.mamba2_session_close(self._native)

    def close(self):
        if not self._open: return
        try: self.sync_state()
        finally:
            self._open = False
            self._state._resident_session = None
            self._release()

    def __enter__(self): return self
    def __exit__(self, *exc): self.close(); return False
    def __del__(self):
        try:
            if getattr(self, "_open", False):
                self._open = False
                self._state._resident_session = None
                self._release()
        except Exception: pass

    def __repr__(self):
        return "Mamba2DecodeSession(open=%r, rows=%d)" % (self._open, self._b)


class Mamba3State:
    """The Mamba-3 recurrent state, its contract's DEVIATION-832 pieces,
    caller-owned (DEVIATIONS 792/794). With H = d_model/32 heads,
    N = 128 (QK head dim), P = 64 (V head dim), R = 32 rope angles and
    Q = 64 (the Mamba-3 chunk size -- NOT mamba2's 256):

        theta           : (B, H, R) float32 -- the serial rotary angles,
                          each in [0, 2pi) by the S10 mod's invariant
        h               : (B, H, P, N) float32 -- the SEALED
                          chunk-boundary SSM state (the state ENTERING
                          the last working chunk)
        buffer_qrot     : (B, Q, H, N) float32 -- rotated-UNSCALED q
        buffer_krot     : (B, Q, H, N) float32 -- rotated-UNSCALED k
        buffer_v        : (B, Q, H, P) float32 -- raw v (the x split;
                          there is NO conv in Mamba-3)
        buffer_dt       : (B, Q, H) float32 -- dt rows
        buffer_sig      : (B, Q, H) float32 -- sigma(trap) rows
        buffer_adt      : (B, Q, H) float32 -- ADT rows
        pending_k       : (B, H, N) float32 -- the Input_States pair,
        pending_v       : (B, H, P) float32    live only under `pending`
        buffered_tokens : int in [0, Q] INCLUSIVE -- valid buffer rows.
                          UNLIKE mamba2 the buffer NEVER EMPTIES
                          (DEVIATION 832(i): in [1, Q] after every
                          call); 0 only before the first token
        pending         : bool -- True marks theta/h/pending_k/pending_v
                          as a reference Input_States continuation
                          (contract section 5 claim 2), CONSUMED by the
                          next call (DEVIATION 794)

    All zeros (and 0, False) before the first token
    (`allocate_inference_cache`). Updated IN PLACE (`buffered_tokens`
    and `pending` reassigned) by `forward` and `step`."""

    def __init__(self, theta, h, buffer_qrot, buffer_krot, buffer_v,
                 buffer_dt, buffer_sig, buffer_adt, pending_k, pending_v,
                 buffered_tokens=0, pending=False):
        self.theta = theta
        self.h = h
        self.buffer_qrot = buffer_qrot
        self.buffer_krot = buffer_krot
        self.buffer_v = buffer_v
        self.buffer_dt = buffer_dt
        self.buffer_sig = buffer_sig
        self.buffer_adt = buffer_adt
        self.pending_k = pending_k
        self.pending_v = pending_v
        self.buffered_tokens = int(buffered_tokens)
        self.pending = bool(pending)

    def set_input_states(self, theta, h, k, v):
        """The reference four-piece `Input_States` continuation (contract
        section 5 claim 2): copy the given bits into theta/h/pending_k/
        pending_v and mark them pending for the next call. dtype and
        shape are refused here (bits must arrive as made -- a silent
        cast would break the certification); whether the state is FRESH
        is judged IN MOJO at the next call, by the lane's own
        set_input_states refusal (Input_States only has reference meaning
        at buffered_tokens 0), which this method deliberately does not
        respell (DEVIATION 794)."""
        what = "Mamba3State.set_input_states"
        for name, dst, src in (("theta", self.theta, theta),
                               ("h", self.h, h),
                               ("k", self.pending_k, k),
                               ("v", self.pending_v, v)):
            arr = _f32_strict(src, what, name)
            dp = probe(dst)
            if arr.shape != dp.shape:
                raise ValueError(
                    f"mojolearn {what}: {name} has shape {arr.shape}, "
                    f"want {dp.shape}"
                )
            # The destination is judged like every other state piece
            # (float32, C-contiguous, writable), then DEVIATION 2410:
            # `np.copyto(dst, arr, casting="no")` is one `memmove` of the
            # same bytes into the state piece's own memory.
            _state_buf(dst, what, name, arr.shape)
            memcopy(addr(dst, name=name), addr_ro(arr, name=name),
                    arr.nbytes)
        self.pending = True


class Mamba3Block(_MambaBase):
    """One Mamba-3 (SISO) block on the GPU -- norm, mixer, residual
    (`Mamba3.forward`'s SISO arm, mamba_ssm mamba3.py:160-278, inside
    `Block.forward`'s non-fused arm, block.py:51-53) -- under profile
    `mojolearn.identical.mamba3.siso.fp32.v1`
    (`mamba/IDENTICAL_MAMBA3_CONTRACT.md`). The module evidence snapshot
    includes native three-vendor backward certification and qualified Apple
    Python forward/state checks. NVIDIA and AMD API requalification of
    the state-allocation fix is pending. All three blocks expose zero-state
    IDENTICAL prefill VJPs; qualification of these new Python entries is
    separate from historical forward-only results.

    DELTAS FROM Mamba2Block, in one breath (contract section 0): NO conv
    (in_proj feeds the core directly; v is the RAW x split); NO dt clamp
    and therefore NO `dt_limit` argument (S6 is bias -> softplus and
    nothing else); A is DATA-DEPENDENT per (token, head) with the
    A_floor clamp; the gate is applied RAW inside the core (no gated
    output norm); the B/C RMSNorms are new, with per-head biases added
    AFTER the norm.

    WEIGHTS IN, AS GIVEN BITS, keyed by the fixture's tensor names
    (`mamba/checks/mamba3_fixture.mojo`, ids 42-50). With d_inner =
    2*d_model, H = d_inner/64 heads, N = 128, d_in_proj = 2*d_inner +
    256 + 3*H + 32:

        block_norm.weight  (d_model,)         the BLOCK norm
        in_proj.weight     (d_in_proj, d_model)  column order z | x | B
                                              | C | dd_dt | dd_A | trap
                                              | angle (mamba3.py:106-107)
        dt_bias            (H,)
        B_norm.weight      (N,)               the S21 B RMSNorm, eps 1e-5
        C_norm.weight      (N,)               the S21 C RMSNorm
        B_bias             (H, N)             reference ones-init; arrives
                                              as given bits like all the
                                              rest (mimo_rank 1 squeezed)
        C_bias             (H, N)
        D                  (H,)
        out_proj.weight    (d_model, d_inner)

    WHAT IS HONORED, WHAT IS FIXED, WHAT IS REFUSED -- the parity
    addendum row (`archive/evidence/mamba/FEATURE_PARITY.md`, "Mamba-3 SURFACE KNOBS")
    and the contract's section 3 are normative:

        d_model         honored   any MULTIPLE OF 32 (headdim 64 with
                                  expand 2; refused by name otherwise --
                                  Mamba3Dims.of carries the rule, this
                                  side repeats it only to size the state)
        Input_States    honored   `state.set_input_states(theta, h, k,
                                  v)` on a FRESH state; the fresh-state
                                  rule is judged in Mojo (DEVIATION 794)
        d_state/headdim/ FIXED    128 / 64 / 1 / 0.5 (32 angles) / 1e-4
          ngroups/rope_           / 64: profile constants; CHUNK_SIZE
          fraction/A_floor        is PART OF THE ARITHMETIC (DEVIATIONS
          /chunk                  827/783), never a tuning knob
        is_mimo/mimo_rank, refused  by absence, structurally: no such
          is_outproj_norm,          knob exists on this surface (the
          fuse_pregate_*,           parity addendum row names each)
          ngroups>1, varlen
        dt_min/dt_max/  absent    INITIALIZATION facts (mamba3.py:111-
          dt_init_floor           115); weights arrive as given bits
        dtype           refused   float32 ONLY; bf16/fp16/float64 by
                                  name (the shipped bf16 casts are
                                  REFUSED, not reproduced)

    STATE IS EXPLICIT AND TEN-PIECE (`Mamba3State`; DEVIATIONS 792/794).
    Decode is PREFILL RESUMPTION (DEVIATION 831): `step` is the same
    entry at L = 1, and `forward` with a carried state is chunked-prefill
    continuation. After every call `h_last_`, `k_last_`, `v_last_` and
    `theta_last_` on this instance hold the REPORT stages -- NOT the
    resumption state (the sealed-boundary distinction is DEVIATION
    832's)."""

    _W_NAMES = (
        "block_norm.weight", "in_proj.weight", "dt_bias",
        "B_norm.weight", "C_norm.weight", "B_bias", "C_bias", "D",
        "out_proj.weight",
    )

    def __init__(self, weights):
        what = "Mamba3Block"
        # lane/identical-lowbit-inference (2026-09-17): packed projection
        # weights (mojolearn.lowbit) materialized exactly, fp32 path after.
        weights, self.weight_format = _lowbit.unpack(weights, what)
        arrs = _take(weights, what, self._W_NAMES)
        norm_w = _f32_strict(arrs[0], what, "block_norm.weight")
        if norm_w.ndim != 1 or norm_w.shape[0] < 1:
            raise ValueError(
                f"mojolearn {what}: block_norm.weight must be 1-D "
                f"(d_model,), got shape {norm_w.shape}"
            )
        dm = int(norm_w.shape[0])
        if dm % (_M3_HEADDIM // _M3_EXPAND) != 0:
            # Mamba3Dims.of's rule, quoted; the Mojo constructor remains
            # the authority (raw-binding callers hit it), this copy
            # exists because the state buffers below cannot be sized
            # without a whole nheads (DEVIATION 793's accepted
            # duplication, same as Mamba2Block's).
            raise ValueError(
                f"mojolearn {what}: d_model must be a multiple of "
                f"{_M3_HEADDIM // _M3_EXPAND} so that nheads = "
                f"2*d_model/{_M3_HEADDIM} is whole (profile constants "
                "headdim 64, expand 2 -- Mamba3Dims.of carries the same "
                f"refusal); got {dm}"
            )
        di = _M3_EXPAND * dm
        nh = di // _M3_HEADDIM
        dip = (2 * di + 2 * _M3_NGROUPS * _M3_D_STATE + 3 * nh
               + _M3_NUM_ROPE_ANGLES)
        self.d_model = dm
        self.d_inner = di
        self.nheads = nh
        self.d_in_proj = dip
        self._w = [
            norm_w,
            _want_shape(_f32_strict(arrs[1], what, "in_proj.weight"),
                        what, "in_proj.weight", (dip, dm)),
            _want_shape(_f32_strict(arrs[2], what, "dt_bias"),
                        what, "dt_bias", (nh,)),
            _want_shape(_f32_strict(arrs[3], what, "B_norm.weight"),
                        what, "B_norm.weight", (_M3_D_STATE,)),
            _want_shape(_f32_strict(arrs[4], what, "C_norm.weight"),
                        what, "C_norm.weight", (_M3_D_STATE,)),
            _want_shape(_f32_strict(arrs[5], what, "B_bias"),
                        what, "B_bias", (nh, _M3_D_STATE)),
            _want_shape(_f32_strict(arrs[6], what, "C_bias"),
                        what, "C_bias", (nh, _M3_D_STATE)),
            _want_shape(_f32_strict(arrs[7], what, "D"), what, "D", (nh,)),
            _want_shape(_f32_strict(arrs[8], what, "out_proj.weight"),
                        what, "out_proj.weight", (dm, di)),
        ]

    def allocate_state(self, batch_size):
        """The zero ten-piece state (`allocate_inference_cache`,
        mamba3.py:442-482, plus DEVIATION 832's buffer pieces). For a
        reference Input_States continuation call
        `set_input_states(theta, h, k, v)` on the fresh state."""
        b = int(batch_size)
        if b < 1:
            raise ValueError(
                f"mojolearn Mamba3Block: batch_size must be positive, "
                f"got {batch_size!r}"
            )
        nh, q = self.nheads, _M3_CHUNK_SIZE
        return Mamba3State(
            zeros((b, nh, _M3_NUM_ROPE_ANGLES), "<f4"),
            zeros((b, nh, _M3_HEADDIM, _M3_D_STATE), "<f4"),
            zeros((b, q, nh, _M3_D_STATE), "<f4"),
            zeros((b, q, nh, _M3_D_STATE), "<f4"),
            zeros((b, q, nh, _M3_HEADDIM), "<f4"),
            zeros((b, q, nh), "<f4"),
            zeros((b, q, nh), "<f4"),
            zeros((b, q, nh), "<f4"),
            zeros((b, nh, _M3_D_STATE), "<f4"),
            zeros((b, nh, _M3_HEADDIM), "<f4"),
            0,
            False,
        )

    def _call_fresh(self, x, ext):
        """Discard-only prefill: return all reports without a host cache."""
        b, l = int(x.shape[0]), int(x.shape[1])
        nh = self.nheads
        y = _buffers.empty((b, l, self.d_model), '<f4')
        h_last = _buffers.empty((b, nh, _M3_HEADDIM, _M3_D_STATE), '<f4')
        k_last = _buffers.empty((b, nh, _M3_D_STATE), '<f4')
        v_last = _buffers.empty((b, nh, _M3_HEADDIM), '<f4')
        theta_last = _buffers.empty((b, nh, _M3_NUM_ROPE_ANGLES), '<f4')
        addrs = ([_addr_ro(x)] + [_addr_ro(w) for w in self._w]
                 + [_addr(y), _addr(h_last), _addr(k_last), _addr(v_last),
                    _addr(theta_last)])
        ext.mamba3_forward_fresh(addrs, [b, l, self.d_model])
        self.h_last_ = h_last
        self.k_last_ = k_last
        self.v_last_ = v_last
        self.theta_last_ = theta_last
        return y

    def _call(self, x, state, step):
        what = "Mamba3Block.step" if step else "Mamba3Block.forward"
        x = _batch_tokens(x, what, self.d_model, step)
        b, l = int(x.shape[0]), int(x.shape[1])
        fresh_ext = None
        if state is None and not step:
            fresh_ext = self._extension()
            if hasattr(fresh_ext, "mamba3_forward_fresh"):
                return self._call_fresh(x, fresh_ext)
        if state is None:
            state = self.allocate_state(b)
        _refuse_resident(state, what)
        nh, q = self.nheads, _M3_CHUNK_SIZE
        theta = _state_buf(state.theta, what, "theta",
                           (b, nh, _M3_NUM_ROPE_ANGLES))
        h = _state_buf(state.h, what, "h",
                       (b, nh, _M3_HEADDIM, _M3_D_STATE))
        bq = _state_buf(state.buffer_qrot, what, "buffer_qrot",
                        (b, q, nh, _M3_D_STATE))
        bk = _state_buf(state.buffer_krot, what, "buffer_krot",
                        (b, q, nh, _M3_D_STATE))
        bv = _state_buf(state.buffer_v, what, "buffer_v",
                        (b, q, nh, _M3_HEADDIM))
        bd = _state_buf(state.buffer_dt, what, "buffer_dt", (b, q, nh))
        bs = _state_buf(state.buffer_sig, what, "buffer_sig", (b, q, nh))
        ba = _state_buf(state.buffer_adt, what, "buffer_adt", (b, q, nh))
        pk = _state_buf(state.pending_k, what, "pending_k",
                        (b, nh, _M3_D_STATE))
        pv = _state_buf(state.pending_v, what, "pending_v",
                        (b, nh, _M3_HEADDIM))
        q0 = int(state.buffered_tokens)
        pend = 1 if state.pending else 0
        # buffered_tokens outside [0, 64] is a boundary disagreement the
        # binding refuses by name (DEVIATION 794); it goes down unjudged.
        y = empty((b, l, self.d_model), "<f4")
        h_last = empty((b, nh, _M3_HEADDIM, _M3_D_STATE), "<f4")
        k_last = empty((b, nh, _M3_D_STATE), "<f4")
        v_last = empty((b, nh, _M3_HEADDIM), "<f4")
        theta_last = empty((b, nh, _M3_NUM_ROPE_ANGLES), "<f4")
        # TWO LISTS, NOT TWENTY-FIVE ARGUMENTS (DEVIATION 791). Every
        # array addressed below is bound in this frame -- x, the nine
        # entries of self._w (alive on self), the ten state pieces, y and
        # the four reports.
        w = self._w
        ext = fresh_ext if fresh_ext is not None else self._extension()
        addrs = (
            # ORDER MATCHES bindings/_mojolearn_mamba.mojo::
            # mamba3_forward_binding: x, block norm.weight,
            # in_proj.weight, dt_bias, B_norm.weight, C_norm.weight,
            # B_bias, C_bias, D, out_proj.weight, theta, h, buffer_qrot,
            # buffer_krot, buffer_v, buffer_dt, buffer_sig, buffer_adt,
            # pending_k, pending_v, y_out, h_last_out, k_last_out,
            # v_last_out, theta_last_out
            [addr_ro(x, name="x")]
            + [addr_ro(a, name="weight") for a in w]
            + [addr(theta, name="theta"), addr(h, name="h"),
               addr(bq, name="buffer_qrot"), addr(bk, name="buffer_krot"),
               addr(bv, name="buffer_v"), addr(bd, name="buffer_dt"),
               addr(bs, name="buffer_sig"), addr(ba, name="buffer_adt"),
               addr(pk, name="pending_k"), addr(pv, name="pending_v"),
               addr(y, name="y"), addr(h_last, name="h_last"),
               addr(k_last, name="k_last"), addr(v_last, name="v_last"),
               addr(theta_last, name="theta_last")]
        )
        if step:
            # B, d_model, buf_len, pending.
            new_len = ext.mamba3_decode_step(
                addrs, [b, self.d_model, q0, pend]
            )
        else:
            # B, L, d_model, buf_len, pending.
            new_len = ext.mamba3_forward(
                addrs, [b, l, self.d_model, q0, pend]
            )
        state.buffered_tokens = int(new_len)
        # ALWAYS False after a call: the unarmed core consumes a pending
        # continuation and an armed build aborts at PyInit, so no return
        # slot exists for it (DEVIATION 794).
        state.pending = False
        #: The REPORT stages (h/k/v after the final working chunk, theta
        #: after the final token), NOT the resumption state -- DEVIATION
        #: 832's sealed-boundary distinction.
        self.h_last_ = h_last
        self.k_last_ = k_last
        self.v_last_ = v_last
        self.theta_last_ = theta_last
        return y

    def forward(self, x, state=None, *, lengths=None):
        """One block call: `(B, L, d_model)` float32 in, the block
        output back, any B and L. `state=None` runs a self-contained
        prefill from zeros and DISCARDS the final state; pass a
        `Mamba3State` to carry it -- a later `forward` or `step` on that
        state is chunked-prefill continuation / decode, bit-for-bit the
        prefill that ran the whole sequence at once (DEVIATION 831's
        construction; the identical tier's gates verify it).

        `lengths` (2026-09-15) makes the batch RAGGED: `B` integers in
        `[1, L]`, row `i` real at positions `[0, lengths[i])` and padding
        after. Every real position's output is byte for byte the row run
        alone at its own length (the scan is causal and the contract's
        clause (c) makes a row independent of its batch; no arithmetic
        changes, `_ragged.py` says why) and every padding position's
        output is exactly `+0.0`, whatever the input held there. Refused
        with a carried `state`."""
        if lengths is not None:
            what = type(self).__name__ + ".forward"
            x = _batch_tokens(x, what, self.d_model, False)
            return _ragged.ragged_forward(lambda xp: self._call(xp, None, step=False),
                                          x, state, lengths, "<f4", what)[0]
        return self._call(x, state, step=False)

    def backward(self, x, grad_output):
        """Return the zero-state prefill VJP for x and all nine weights.

        IDENTICAL float32 only. Forward is recomputed using current weights;
        no previous forward or cached state is consumed or modified. Returns
        independent arrays under the constructor's exact weight names plus
        ``x``. Incoming-cache and final-state cotangents are not supported.
        Retained NVIDIA fixture checks and outstanding wheel/vendor scopes are
        listed in mamba/PUBLIC_ALPHA_SURFACE.md.
        """
        return self._prefill_backward(x, grad_output, "mamba3_backward")

    def step(self, x, state):
        """One decode token: `Mamba3.step`'s semantics, the profile's
        spelling -- PREFILL RESUMPTION at L = 1 through the same entry
        as `forward` (DEVIATION 831; the reference's own per-token recurrence
        rounds differently BY CONSTRUCTION and is the lane's
        required-RED STEP_UPSTREAM_RECURRENCE arm, never a mode here).
        `state` is REQUIRED."""
        if state is None:
            raise ValueError(
                "mojolearn Mamba3Block.step: state is required (a decode "
                "step continues a sequence; allocate_state(B) makes the "
                "fresh one)"
            )
        return self._call(x, state, step=True)

    def decode_session(self, state):
        """Keep Mamba-3 weights and recurrent state resident for repeated
        single-token decode while executing the existing L=1 block entry."""
        return Mamba3DecodeSession(self, state)

    __call__ = forward


class Mamba3DecodeSession:
    """Resident repeated decode for one ``Mamba3Block`` and state."""

    _STATE_NAMES = (
        "theta", "h", "buffer_qrot", "buffer_krot", "buffer_v",
        "buffer_dt", "buffer_sig", "buffer_adt", "pending_k", "pending_v",
    )

    def __init__(self, block, state):
        what = "Mamba3DecodeSession"
        ext = block._extension()
        try:
            create = getattr(ext, "mamba3_session_create", None)
        except ImportError:
            create = None
        host = create is None and _exports(ext, "mamba3_decode_step")
        if create is None and not host:
            raise NotImplementedError(f"mojolearn {what}: binding lacks decode entries")
        if state is None:
            raise ValueError(f"mojolearn {what}: state is required")
        _refuse_resident(state, what)
        pb = probe(state.h)
        if len(pb.shape) != 4 or pb.shape[0] < 1:
            raise ValueError(f"mojolearn {what}: state.h has invalid shape")
        self._block, self._state, self._ext = block, state, ext
        self._b, self._open = int(pb.shape[0]), False
        parts = self._state_parts(what)
        if host:
            self._native = None
            self._hw = [_private_copy(a) for a in block._w]
            self._parts = [_private_copy(a) for a in parts]
            self._q = int(state.buffered_tokens)
            self._pending = bool(state.pending)
        else:
            self._native = create()
            ext.mamba3_session_open(
                self._native,
                [addr_ro(a, name="weight") for a in block._w]
                + [addr(a, name=name) for a, name in zip(parts, self._STATE_NAMES)],
                [self._b, block.d_model, int(state.buffered_tokens),
                 1 if state.pending else 0])
        self._open = True
        state._resident_session = self

    @property
    def state(self): return self._state

    @property
    def is_open(self): return self._open

    def _require_open(self, what):
        if not self._open:
            raise ValueError(f"mojolearn {what}: the session is closed")

    def _state_parts(self, what):
        st, b, blk = self._state, self._b, self._block
        nh, q = blk.nheads, _M3_CHUNK_SIZE
        shapes = (
            (b, nh, _M3_NUM_ROPE_ANGLES),
            (b, nh, _M3_HEADDIM, _M3_D_STATE),
            (b, q, nh, _M3_D_STATE), (b, q, nh, _M3_D_STATE),
            (b, q, nh, _M3_HEADDIM), (b, q, nh), (b, q, nh), (b, q, nh),
            (b, nh, _M3_D_STATE), (b, nh, _M3_HEADDIM),
        )
        return tuple(_state_buf(getattr(st, name), what, name, shape)
                     for name, shape in zip(self._STATE_NAMES, shapes))

    def step(self, x):
        what = "Mamba3DecodeSession.step"
        self._require_open(what)
        blk = self._block
        x = _batch_tokens(x, what, blk.d_model, True)
        if int(x.shape[0]) != self._b:
            raise ValueError(f"mojolearn {what}: batch size changed")
        b, nh = self._b, blk.nheads
        y = empty((b, 1, blk.d_model), "<f4")
        reports = (
            empty((b, nh, _M3_HEADDIM, _M3_D_STATE), "<f4"),
            empty((b, nh, _M3_D_STATE), "<f4"),
            empty((b, nh, _M3_HEADDIM), "<f4"),
            empty((b, nh, _M3_NUM_ROPE_ANGLES), "<f4"),
        )
        if self._native is None:
            addrs = ([addr_ro(x, name="x")]
                     + [addr_ro(a, name="weight") for a in self._hw]
                     + [addr(a, name="state") for a in self._parts]
                     + [addr(y, name="y")]
                     + [addr(a, name="report") for a in reports])
            self._q = int(self._ext.mamba3_decode_step(
                addrs, [b, blk.d_model, self._q, 1 if self._pending else 0]))
            self._pending = False
        else:
            self._q = int(self._ext.mamba3_session_step(
                self._native, [addr_ro(x, name="x"), addr(y, name="y")]
                + [addr(a, name="report") for a in reports]))
        blk.h_last_, blk.k_last_, blk.v_last_, blk.theta_last_ = reports
        return y

    def sync_state(self):
        what = "Mamba3DecodeSession.sync_state"
        self._require_open(what)
        parts = self._state_parts(what)
        if self._native is None:
            for dst, src in zip(parts, self._parts):
                memcopy(addr(dst, name="state"), addr_ro(src, name="resident"), 4 * int(src.size))
            q, pending = self._q, self._pending
        else:
            q, pending = self._ext.mamba3_session_export_state(
                self._native, [addr(a, name="state") for a in parts])
            q, pending = int(q), bool(pending)
        self._state.buffered_tokens = q
        self._state.pending = pending
        return self._state

    def load_state(self):
        what = "Mamba3DecodeSession.load_state"
        self._require_open(what)
        parts = self._state_parts(what)
        q, pending = int(self._state.buffered_tokens), bool(self._state.pending)
        if self._native is None:
            for dst, src in zip(self._parts, parts):
                memcopy(addr(dst, name="resident"), addr_ro(src, name="state"), 4 * int(dst.size))
            self._q, self._pending = q, pending
        else:
            self._ext.mamba3_session_load_state(
                self._native, [addr(a, name="state") for a in parts],
                [q, 1 if pending else 0])
        return self._state

    def _release(self):
        if self._native is None:
            self._hw = self._parts = None
        else:
            self._ext.mamba3_session_close(self._native)

    def close(self):
        if not self._open: return
        try: self.sync_state()
        finally:
            self._open = False
            self._state._resident_session = None
            self._release()

    def __enter__(self): return self
    def __exit__(self, *exc): self.close(); return False
    def __del__(self):
        try:
            if getattr(self, "_open", False):
                self._open = False
                self._state._resident_session = None
                self._release()
        except Exception: pass

    def __repr__(self):
        return "Mamba3DecodeSession(open=%r, rows=%d)" % (self._open, self._b)
