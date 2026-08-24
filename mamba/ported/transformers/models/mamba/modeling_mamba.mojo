"""`transformers/models/mamba/modeling_mamba.py`: ONE Mamba-1 block, on the
device, under profile `mojolearn.identical.mamba1.fp32.v1`. **COPY, DO NOT
IMPROVE.**

PORT OF huggingface/transformers at `d56c55b`,
`src/transformers/models/mamba/modeling_mamba.py`. Partial, inference only.
What is mirrored here, symbol by symbol:

| theirs | lines | here |
|---|---|---|
| `MambaRMSNorm.forward` | :495-499 | `mamba_rms_norm_kernel`, `mamba_rms_norm` |
| `causal_conv1d_fn` (the torch fallback) | :81-100 | `causal_conv1d_fn_kernel`, `causal_conv1d_fn` |
| `mamba_selective_scan` (the torch fallback) | :175-280 | `mamba_selective_scan` |
| `MambaMixer.forward` | :359-483 | `mamba_mixer_forward` |
| `MambaBlock.forward` | :505-530 | `mamba_block_forward` |

The recurrence itself is NOT here. `mamba_selective_scan` is upstream's
FALLBACK for `selective_scan_fn`, and this file keeps that split: the
recurrent core lives in
`mamba/ported/mamba_ssm/ops/selective_scan_interface.mojo` (state-spaces/mamba
`e9594ce`, `selective_scan_ref` :127-193), which this file CALLS. The four
projections are not here either: `nn.Linear` is cuBLAS upstream and
`gemm.fp32.v1` here (`gemm/mojo_only/gemm_identical.mojo::identical_gemm`,
IDENTITY_PATHS row 40).

THE CONTRACT IS `mamba/IDENTICAL_MAMBA_CONTRACT.md` AND IT IS FROZEN. Section
4's seam table decides every rounding below; section 7's stage list decides
every tag. The host oracle `mamba/mojo_only/mamba_oracle.mojo` is the ANSWER,
bit for bit -- this file is an independent transcription of the same order
into kernels, and the two share only the seam functions themselves
(`ftz`, `identical_mul_add`, `identical_exp`, `identical_div`,
`identical_rsqrt`, `identical_silu`, `identical_softplus`), by design.

THE SEAM SPLIT ACROSS THE TWO PORTED FILES
-------------------------------------------
This file owns S1-S4 (RMSNorm), S12 (the z gate), S13 (the conv tap chain),
S14 (the softplus), S15 (`A = -exp(A_log)`), S16 (the residual add) and the
four S17 calls. `selective_scan_interface.mojo` owns S5-S11, which includes
the `D` skip -- its DEVIATION 723 refuses `z`, `delta_bias` and
`delta_softplus` for exactly the reason this file refuses to compute S11:
a seam with two spellings is a seam with two places to drift. It also EMITS
the three stages `scan.y`, `skip.out` and `scan.h`, so nothing here records
them (`core/identity_trace.mojo`'s tag-uniqueness invariant raises on a
duplicate, which is the loud failure that keeps the split honest).

WHY NO FLOAT CROSSES A THREAD BOUNDARY IN THIS FILE
----------------------------------------------------
Every kernel below owns its output cell entirely. The RMSNorm fold is one
thread per token row (contract S1: "one fold per row, no block fold"); the
conv is one thread per (batch, channel) walking `l` ascending, which is MAX's
`causal_conv1d.mojo:188-205` shape and the CUDA `causal_conv1d` kernel's;
every other seam is one thread per output cell. There is no shared memory, no
warp primitive, no atomic and no cross-block reduction anywhere in this file.
So clause (b) of contract section 8 (the same bits on 8 repeated launches) and
clause (c) (batch-composition invariance) are properties of the SHAPE of these
kernels, not of a check that happens to pass -- the gemm lane's argument at
its own `gemm_identical.mojo` header, made here for the same reason.

`[[mojo-buffer-freed-at-last-use]]`: every buffer this file hands to a kernel
is a FIELD of `MambaDeviceWeights`, `MambaDeviceState` or `MambaDeviceStages`,
whose lifetime is the struct's. No launcher below allocates a scratch buffer
and returns without waiting.

DEVIATIONS
----------
This file owns 725 through 731. 720 (`pinned_mul`) and 721 (the conv's bias
SEED in both the prefill and the decode path) are the oracle's and the
contract's; 722 and 723 are the scan file's. They are cited, not renumbered.

**DEVIATION 725 -- the execution plan.** Upstream is torch ops over whole
tensors; there is no upstream kernel decomposition to mirror for the
elementwise seams. The plan here is one thread per output cell for every seam
except S1 (one thread per token row, because the fold is per row) and S13
(one thread per (batch, channel), because the conv chain is serial in `l` and
that is MAX's and the CUDA kernel's shape). This is an EXECUTION plan quantity
in `IDENTICAL_GEMM_PLAN.md`'s sense: it decides which thread computes a cell,
never the sequence of values accumulated into it.

**DEVIATION 726 -- the conv window is computed OUT OF PLACE.** The window
after the call is written into `MambaDeviceStages.conv_win` and copied into
`MambaDeviceState.conv_win` afterwards. At `L < d_conv` a thread's four new
window slots overlap the four old ones it still has to read, so an in-place
roll reads its own writes. Upstream is out of place too (`causal_conv1d.py`'s
`conv_state.copy_(hidden_states_new[:, :, -state_len:])` copies from a
CONCATENATION, not from `conv_state`), so this mirrors them; the note exists
only because the in-place spelling is the tempting one and `L >= 4` hides it.

**DEVIATION 727 -- the fallback's bundled steps are SPLIT.**
`mamba_selective_scan` (:202-205, :268, :271) applies `delta_bias`, the
softplus, the `D` skip and the `z` gate around the recurrence, and returns one
tensor. Contract section 7 requires `softplus.out`, `scan.y`, `skip.out` and
`gate.out` as four separate recorded stages. S14 and S12 therefore stay in
THIS file's kernels; S11 goes down with the recurrence because the scan file
owns that seam (its DEVIATION 723); and the recurrent core is called with `z`
and `delta_bias` NOT PRESENT and `delta_softplus` False. Same arithmetic, same
order, four stages instead of one. Upstream's own kernel path splits the same
way when `mamba_selective_state_update` is used, so the split is theirs.

**DEVIATION 728 -- `torch.split` is MATERIALIZED.** `torch.split(...)` (:437)
returns views and `gemm.fp32.v1` accepts only contiguous row-major operands
(gemm contract section 2), so `dt`, `B` and `C` are copied into three
contiguous buffers. A copy is not a seam (contract section 4's last sentence)
and the copy kernel touches no arithmetic.

**DEVIATION 729 -- `refuse_nonfinite` reads the device buffers back.**
Contract section 6 refuses a NaN or an infinity BY NAME before any recorded
stage, because NaN payloads are vendor-shaped (row 39 measured three payloads
for one IEEE answer) and a certified stage may not contain one. A kernel
cannot raise, so the entry copies each named input to the host and tests it
there, BY BITS and not by compares (Metal flushes compare operands, row 49).
It costs a drain per input; this profile publishes no timing number (contract
section 9), so the cost buys the clause outright. The host twin of this test
is `mamba_oracle.mojo::refuse_nonfinite` and the two must say the same thing;
it is not imported, because the oracle's header reserves cross-imports to the
SEAM functions and a refusal policy is not one.

**DEVIATION 730 -- `chunk(2, dim=1)` is a COLUMN OFFSET, not a copy.**
`projected_states.chunk(2, dim=1)` (:396) splits `in_proj.out` into the conv
input and the gate. Both halves are read in place from the `[M, 2*d_inner]`
stage buffer at column offsets `0` and `d_inner`. No copy, no seam, and the
recorded `in_proj.out` stage is the whole thing exactly as section 7 lists it.

**DEVIATION 731 -- `ftz` on EVERY buffer load.** Contract section 4 requires
every operand loaded from a buffer to pass `ftz`. The host oracle omits the
flush at two loads whose values were already flushed when they were stored
(`skip_out` into S12, `out_proj` into S16). This file spells the flush at
both: it cannot move a bit (the stored value is flushed) and the contract's
rule is per LOAD, not per value.

THE `OP_NT` TRAP, WRITTEN DOWN SO NOBODY PAYS FOR IT TWICE
-----------------------------------------------------------
Two files in this repository number the GEMM orientations DIFFERENTLY.
`bench/gemm_shapes.mojo` is `OP_NT = 0, OP_TN = 1, OP_NN = 2`.
`gemm/mojo_only/gemm_oracle.mojo` is `OP_NN = 0, OP_NT = 1, OP_TN = 2`, and
that is the numbering `identical_gemm` reads. Passing the bench table's codes
to the GEMM once produced a whole card of plausible, in-bounds, WRONG products
that no assertion caught, because `in_proj.weight` at `[2*d_inner, d_model]`
has exactly as many elements read as `[d_model, 2*d_inner]`. **Every call
below goes through `_gemm_op_nt`, which imports `OP_NT` from `gemm_oracle`
and says which numbering it is in.** The sabotage arm `S17_OP_NUMBERING` is
that mistake, planted, so a gate proves it is caught.

WHAT `NUMERIC_FAST` DOES HERE
------------------------------
Everything, with the pins compiled away, on the same code and the same
launches. FAST makes no identity claim (contract section 8's last sentence).
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from gemm.mojo_only.gemm_identical import identical_gemm

# ORIENTATION NUMBERING: this `OP_NT` is `gemm_oracle`'s, where
# `OP_NN = 0, OP_NT = 1, OP_TN = 2`. It is NOT `bench/gemm_shapes.mojo`'s
# `OP_NT = 0`. `identical_gemm` reads this one. See the header's trap note.
from gemm.mojo_only.gemm_oracle import OP_NN, OP_NT

# `MambaConfig`'s constants and its derived shape
# (`configuration_mamba.py`: `state_size` 16, `conv_kernel` 4,
# `layer_norm_epsilon` 1e-5, `intermediate_size = expand * hidden_size`,
# `time_step_rank = ceil(hidden_size / 16)`). This lane has no ported
# `configuration_mamba.mojo`; the values live once, in the fixture, and are
# read from there rather than copied, because a config constant with two
# homes is a config constant that drifts.
from mamba.mojo_only.mamba_fixture import (
    D_CONV,
    D_STATE,
    MambaDims,
    MambaWeights,
    RMS_EPS,
)
from mamba.ported.mamba_ssm.ops.selective_scan_interface import (
    selective_scan_fn,
)
from mojo_only.numerics import (
    ftz,
    identical_div,
    identical_exp,
    identical_log1p,
    identical_mul_add,
    identical_rsqrt,
    identical_silu,
    identical_softplus,
)


def pinned_mul(a: Float32, b: Float32) -> Float32:
    """DEVIATION 720, the oracle's construction, spelled here so this file
    shares only `mojo_only/numerics.mojo` with the host side (the scan file
    does the same, for the same reason).

    A MULTIPLY no codegen may contract into a neighboring add:
    `identical_mul_add(a, b, -0.0)` is bit-equal to the correctly rounded
    product at every input INCLUDING both zero signs (a `+0.0` addend would
    launder a `-0.0` product -- the gemm lane's F6a lesson) and presents no
    syntactic multiply for a compiler to contract. Used at every seam the
    reference rounds as its own multiply: here that is S3 (`x * rstd`),
    S4 (`weight * hidden`) and S12 (`out * silu(z)`).
    """
    return identical_mul_add(a, b, Float32(-0.0))


# ===========================================================================
# THE SABOTAGE ARMS (rung 4: a clause nobody can falsify is not a clause)
#
# Each one is a named, compile-time alternative spelling of ONE seam decision
# in contract section 4, reachable by a plausible implementer. OFF in every
# build that does not name them:
#
#     tools/with_identical_mode.sh pixi run mojo run \
#         -D MOJOLEARN_MAMBA_SABOTAGE_S14_THRESHOLD_10=1 \
#         -I . mamba/mojo_only/mamba_check.mojo
#
# The names are disjoint from `selective_scan_interface.mojo`'s five
# (S5, S8, S9, S10, S11), so one driver can arm any of the eleven.
# ===========================================================================

comptime SAB_S14_THRESHOLD_10 = is_defined[
    "MOJOLEARN_MAMBA_SABOTAGE_S14_THRESHOLD_10"
]()
"""S14: move the softplus guard from `x <= 20` to `x <= 10`. Contract
section 4's softplus note is the thing to read before believing a gate that
passes this: in FP32 the boundary at 20 cannot itself move a bit, so a
fixture that only straddles 20 passes this arm VACUOUSLY. The distinguishing
range is delta in about [8, 14]."""

comptime SAB_S13_BIAS_LAST = is_defined[
    "MOJOLEARN_MAMBA_SABOTAGE_S13_BIAS_LAST"
]()
"""S13: sum the four taps from `+0.0` and add the bias AFTER, which is
`Mamba.step`'s spelling (mamba_simple.py:218-220) and NOT the profile's.
DEVIATION 721 is the clause this falsifies, and contract section 5's gate D
(decode == prefill) is what it would break."""

comptime SAB_S13_TAPS_REVERSED = is_defined[
    "MOJOLEARN_MAMBA_SABOTAGE_S13_TAPS_REVERSED"
]()
"""S13: walk the taps k = 3..0 (newest first) instead of ascending oldest
first. The window and the weight index move together, so the answer stays a
convolution of the same four numbers -- only the fold order changes."""

comptime SAB_S1_FOLD_DESCENDING = is_defined[
    "MOJOLEARN_MAMBA_SABOTAGE_S1_FOLD_DESCENDING"
]()
"""S1: fold the sum of squares descending instead of ascending. `mean(-1)`
has no documented fold order, so the order is a PROFILE decision and this is
the arm that falsifies it."""

comptime SAB_S12_MUL_SIGMOID = is_defined[
    "MOJOLEARN_MAMBA_SABOTAGE_S12_MUL_SIGMOID"
]()
"""S12: spell silu as `z * sigmoid(z)` (two roundings) instead of the
reference's single quotient `z / (1 + exp(-z))` -- ATen's spelling, also
`selective_scan_fwd_kernel.cuh:298`, DEVIATION 744."""

comptime SAB_S17_OP_NUMBERING = is_defined[
    "MOJOLEARN_MAMBA_SABOTAGE_S17_OP_NUMBERING"
]()
"""S17: pass `bench/gemm_shapes.mojo`'s `OP_NT = 0` to `identical_gemm`,
which reads `gemm_oracle`'s numbering and sees `OP_NN`. Every buffer is still
exactly the right size, so nothing raises and every product is wrong. THE
TRAP, PLANTED."""

comptime BLOCK_ANY_SABOTAGE = (
    SAB_S14_THRESHOLD_10
    or SAB_S13_BIAS_LAST
    or SAB_S13_TAPS_REVERSED
    or SAB_S1_FOLD_DESCENDING
    or SAB_S12_MUL_SIGMOID
    or SAB_S17_OP_NUMBERING
)


def mamba_block_sabotage_name() -> String:
    """The armed sabotage, for a driver that must refuse to certify a
    sabotaged build -- and, more importantly, must refuse to report a
    sabotaged build as CLEAN when the `-D` was misspelled and silently
    ignored (`tools/gemm_ladder.sh:71`'s scar)."""
    comptime if SAB_S14_THRESHOLD_10:
        return String("S14_THRESHOLD_10")
    comptime if SAB_S13_BIAS_LAST:
        return String("S13_BIAS_LAST")
    comptime if SAB_S13_TAPS_REVERSED:
        return String("S13_TAPS_REVERSED")
    comptime if SAB_S1_FOLD_DESCENDING:
        return String("S1_FOLD_DESCENDING")
    comptime if SAB_S12_MUL_SIGMOID:
        return String("S12_MUL_SIGMOID")
    comptime if SAB_S17_OP_NUMBERING:
        return String("S17_OP_NUMBERING")
    return String("none")


# ===========================================================================
# LAUNCH GEOMETRY. An EXECUTION plan quantity (DEVIATION 725): it decides
# which thread owns a cell and nothing else. No kernel below reads
# `block_dim` or `block_idx` in any expression that reaches a fold boundary,
# a tap index or an accumulator seed.
# ===========================================================================

comptime MAMBA_TPB = 128


def _grid(n: Int) -> Int:
    var g = (n + MAMBA_TPB - 1) // MAMBA_TPB
    if g < 1:
        return 1
    return g


# ===========================================================================
# DEVICE-SIDE PARAMETERS, STATE AND STAGES
# ===========================================================================


def mamba_upload(
    ctx: DeviceContext, values: List[Float32]
) raises -> DeviceBuffer[DType.float32]:
    """One host list onto the device, contiguous, in index order."""
    var n = len(values)
    var n_buf = n
    if n_buf < 1:
        n_buf = 1
    var dev = ctx.enqueue_create_buffer[DType.float32](n_buf)
    var host = ctx.enqueue_create_host_buffer[DType.float32](n_buf)
    ctx.synchronize()
    for i in range(n):
        host.unsafe_ptr().unsafe_store(i, values[i])
    for i in range(n, n_buf):
        host.unsafe_ptr().unsafe_store(i, Float32(0.0))
    ctx.enqueue_copy(dst_buf=dev, src_ptr=host.unsafe_ptr())
    ctx.synchronize()
    _ = host^
    return dev^


def mamba_download(
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int
) raises -> List[Float32]:
    """The first `n` elements of a device buffer, as a host list. The gates
    read stages with this, and DEVIATION 729's refusal reads inputs with it."""
    var host = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.synchronize()
    if n == len(buf):
        ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=buf)
    else:
        var view = buf.create_sub_buffer[DType.float32](0, n)
        ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=view)
    ctx.synchronize()
    var out = List[Float32]()
    for i in range(n):
        out.append(host.unsafe_ptr().unsafe_load(i))
    _ = host^
    return out^


def mamba_zeros(
    ctx: DeviceContext, n: Int
) raises -> DeviceBuffer[DType.float32]:
    var n_buf = n
    if n_buf < 1:
        n_buf = 1
    var dev = ctx.enqueue_create_buffer[DType.float32](n_buf)
    dev.enqueue_fill(Float32(0.0))
    ctx.synchronize()
    return dev^


struct MambaDeviceWeights(Movable):
    """One block's parameters on the device, in the upstream shapes
    (`MambaWeights`'s table, which is `mamba/corpus/gen_corpus.py`'s
    `shapes_for`, which is theirs). Row-major, contiguous, no padding.

    `use_bias` is False and `use_conv_bias` is True (the `MambaConfig`
    defaults), so `in_proj`, `x_proj` and `out_proj` carry NO bias and only
    `conv1d` and `dt_proj` do.
    """

    var dims: MambaDims
    var norm_w: DeviceBuffer[DType.float32]  # [d_model]
    var w_in: DeviceBuffer[DType.float32]  # [2*d_inner, d_model]
    var conv_w: DeviceBuffer[DType.float32]  # [d_inner, D_CONV]
    var conv_b: DeviceBuffer[DType.float32]  # [d_inner]
    var w_x: DeviceBuffer[DType.float32]  # [dt_rank+2*D_STATE, d_inner]
    var w_dt: DeviceBuffer[DType.float32]  # [d_inner, dt_rank]
    var b_dt: DeviceBuffer[DType.float32]  # [d_inner]
    var a_log: DeviceBuffer[DType.float32]  # [d_inner, D_STATE]
    var d_skip: DeviceBuffer[DType.float32]  # [d_inner]
    var w_out: DeviceBuffer[DType.float32]  # [d_model, d_inner]

    def __init__(out self, ctx: DeviceContext, w: MambaWeights) raises:
        self.dims = w.dims.copy()
        self.norm_w = mamba_upload(ctx, w.norm_w)
        self.w_in = mamba_upload(ctx, w.w_in)
        self.conv_w = mamba_upload(ctx, w.conv_w)
        self.conv_b = mamba_upload(ctx, w.conv_b)
        self.w_x = mamba_upload(ctx, w.w_x)
        self.w_dt = mamba_upload(ctx, w.w_dt)
        self.b_dt = mamba_upload(ctx, w.b_dt)
        self.a_log = mamba_upload(ctx, w.a_log)
        self.d_skip = mamba_upload(ctx, w.d_skip)
        self.w_out = mamba_upload(ctx, w.w_out)


struct MambaDeviceState(Movable):
    """The recurrent state between calls (contract section 5): the conv
    WINDOW `[B, d_inner, D_CONV]` (the last `d_conv` conv INPUTS, oldest
    first -- `mamba_simple.py:216-217`'s `conv_state` after the roll) and the
    SSM state `h` `[B, d_inner, D_STATE]` (`selective_scan_ref:160`'s `x`).

    Zeros before the first token, which is `allocate_inference_cache`
    (`mamba_simple.py:258-266`). On prefill the zero window IS the
    `padding = d_conv - 1` of `F.conv1d`, which is why ONE spelling serves
    both paths and gate D (decode == prefill) is structural.
    """

    var b: Int
    var d_inner: Int
    var conv_win: DeviceBuffer[DType.float32]
    var h: DeviceBuffer[DType.float32]

    def __init__(out self, ctx: DeviceContext, b: Int, dims: MambaDims) raises:
        self.b = b
        self.d_inner = dims.d_inner
        self.conv_win = mamba_zeros(ctx, b * dims.d_inner * D_CONV)
        self.h = mamba_zeros(ctx, b * dims.d_inner * D_STATE)


struct MambaDeviceStages(Movable):
    """Every recorded stage of one block call, contract section 7, in card
    order and in TOKEN-MAJOR layouts (`M = B * L` rows). The corpus's
    channel-major `[B, d_inner, L]` view of the same values is the corpus
    gate's reindexing, not a second computation.

    Three fields carry no tag of their own: `dt_low`, `b_mat` and `c_mat` are
    `torch.split`'s three views MATERIALIZED (DEVIATION 728). `scan_h` is a
    post-call snapshot of the state; the STAGE `scan.h` is recorded by
    `selective_scan_fn` off `MambaDeviceState.h`, and this copy exists only so
    a gate can read the card after the state moves on to the next token.
    """

    var b: Int
    var l: Int
    var dims: MambaDims
    var norm_sumsq: DeviceBuffer[DType.float32]  # [M]
    var norm_out: DeviceBuffer[DType.float32]  # [M, d_model]
    var in_proj: DeviceBuffer[DType.float32]  # [M, 2*d_inner]
    var a_out: DeviceBuffer[DType.float32]  # [d_inner, D_STATE]
    var conv_out: DeviceBuffer[DType.float32]  # [M, d_inner]
    var silu_out: DeviceBuffer[DType.float32]  # [M, d_inner]
    var conv_win: DeviceBuffer[DType.float32]  # [B, d_inner, D_CONV]
    var x_proj: DeviceBuffer[DType.float32]  # [M, dt_rank+2*D_STATE]
    var dt_proj: DeviceBuffer[DType.float32]  # [M, d_inner] (bias NOT added)
    var softplus_out: DeviceBuffer[DType.float32]  # [M, d_inner]
    var scan_y: DeviceBuffer[DType.float32]  # [M, d_inner] BEFORE D
    var scan_h: DeviceBuffer[DType.float32]  # [B, d_inner, D_STATE] snapshot
    var skip_out: DeviceBuffer[DType.float32]  # [M, d_inner]
    var gate_out: DeviceBuffer[DType.float32]  # [M, d_inner]
    var out_proj: DeviceBuffer[DType.float32]  # [M, d_model]
    var residual_out: DeviceBuffer[DType.float32]  # [M, d_model]
    var dt_low: DeviceBuffer[DType.float32]  # [M, dt_rank]   (split, no tag)
    var b_mat: DeviceBuffer[DType.float32]  # [M, D_STATE]    (split, no tag)
    var c_mat: DeviceBuffer[DType.float32]  # [M, D_STATE]    (split, no tag)

    def __init__(
        out self, ctx: DeviceContext, b: Int, l: Int, dims: MambaDims
    ) raises:
        self.b = b
        self.l = l
        self.dims = dims.copy()
        var m = b * l
        var dm = dims.d_model
        var di = dims.d_inner
        var r = dims.dt_rank
        var xr = dims.x_proj_rows()
        self.norm_sumsq = mamba_zeros(ctx, m)
        self.norm_out = mamba_zeros(ctx, m * dm)
        self.in_proj = mamba_zeros(ctx, m * 2 * di)
        self.a_out = mamba_zeros(ctx, di * D_STATE)
        self.conv_out = mamba_zeros(ctx, m * di)
        self.silu_out = mamba_zeros(ctx, m * di)
        self.conv_win = mamba_zeros(ctx, b * di * D_CONV)
        self.x_proj = mamba_zeros(ctx, m * xr)
        self.dt_proj = mamba_zeros(ctx, m * di)
        self.softplus_out = mamba_zeros(ctx, m * di)
        self.scan_y = mamba_zeros(ctx, m * di)
        self.scan_h = mamba_zeros(ctx, b * di * D_STATE)
        self.skip_out = mamba_zeros(ctx, m * di)
        self.gate_out = mamba_zeros(ctx, m * di)
        self.out_proj = mamba_zeros(ctx, m * dm)
        self.residual_out = mamba_zeros(ctx, m * dm)
        self.dt_low = mamba_zeros(ctx, m * r)
        self.b_mat = mamba_zeros(ctx, m * D_STATE)
        self.c_mat = mamba_zeros(ctx, m * D_STATE)


# ===========================================================================
# `MambaRMSNorm.forward` (:495-499). Seams S1-S4.
#
#     variance = hidden_states.pow(2).mean(-1, keepdim=True)          :497
#     hidden_states = hidden_states * torch.rsqrt(variance + eps)     :498
#     return self.weight * hidden_states                              :499
#
# eps is `config.layer_norm_epsilon`, 1e-5, bits 0x3727C5AC -- contract
# section 3.
# ===========================================================================


def mamba_rms_norm_kernel(
    sumsq: MutPointer[Float32, MutAnyOrigin],
    out_buf: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    weight: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    dm_in: Int32,
):
    """One thread per TOKEN ROW. Contract S1: "serial ascending j from +0.0,
    one fold per row, no block fold" -- so the row's fold never leaves this
    thread's registers and no launch geometry can reorder it."""
    var m = Int(m_in)
    var dm = Int(dm_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= m:
        return

    # S1, FUSED: `acc = ftz(fma(x_j, x_j, acc))`, ascending from +0.0. The
    # +0.0 seed is contract section 6's clause -- an all-zero row sums to
    # +0.0 on every vendor because IEEE says (+0) + (-0) = +0.
    var acc = Float32(0.0)
    comptime if SAB_S1_FOLD_DESCENDING:
        for jj in range(dm):
            var jd = dm - 1 - jj
            var xd = ftz(x.unsafe_load(t * dm + jd))
            acc = ftz(identical_mul_add(xd, xd, acc))
    else:
        for j in range(dm):
            var xj = ftz(x.unsafe_load(t * dm + j))
            acc = ftz(identical_mul_add(xj, xj, acc))
    sumsq.unsafe_store(t, acc)

    # S2: the mean through `identical_div` (row 49) and the reciprocal square
    # root through `identical_rsqrt`, which is the reference's `1 / sqrt`
    # (DEVIATION 741) and NEVER the hardware rsqrt intrinsic (DEVIATION 746
    # RECORDED Metal's intrinsic as correctly rounded where the reference's
    # spelling is not -- being righter than the reference is not the goal).
    var mean = ftz(identical_div(acc, Float32(dm)))
    var rstd = ftz(identical_rsqrt(ftz(mean + RMS_EPS)))

    # S3 and S4, both PRODUCT: `hidden * rstd` then `weight * hidden`, each
    # its own rounding, neither contractible into a neighboring add.
    for j in range(dm):
        var inner = ftz(pinned_mul(ftz(x.unsafe_load(t * dm + j)), rstd))
        out_buf.unsafe_store(
            t * dm + j, ftz(pinned_mul(ftz(weight.unsafe_load(j)), inner))
        )


def mamba_rms_norm(
    ctx: DeviceContext,
    mut sumsq: DeviceBuffer[DType.float32],
    mut out_buf: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut weight: DeviceBuffer[DType.float32],
    m: Int,
    d_model: Int,
) raises:
    """`MambaRMSNorm.forward(hidden_states)` (:495-499) over `M = B * L`
    token rows. ASYNCHRONOUS: the caller synchronizes."""
    ctx.enqueue_function[mamba_rms_norm_kernel](
        sumsq.unsafe_ptr(),
        out_buf.unsafe_ptr(),
        x.unsafe_ptr(),
        weight.unsafe_ptr(),
        Int32(m),
        Int32(d_model),
        grid_dim=(_grid(m), 1, 1),
        block_dim=(MAMBA_TPB, 1, 1),
    )


# ===========================================================================
# `A = -torch.exp(self.A_log.float())` (MM:373). Seam S15.
# ===========================================================================


def mamba_a_from_a_log_kernel(
    a_out: MutPointer[Float32, MutAnyOrigin],
    a_log: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """S15: `-ftz(identical_exp(ftz(A_log)))`. The negation is EXACT (it
    flips one bit), so it is not a seam and needs no flush of its own."""
    var n = Int(n_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= n:
        return
    a_out.unsafe_store(i, -ftz(identical_exp(ftz(a_log.unsafe_load(i)))))


# ===========================================================================
# `causal_conv1d_fn` (:81-100), the torch fallback, with `activation="silu"`.
#
#     padding = weight.shape[-1] - 1                                  :89
#     out = F.conv1d(x, weight.unsqueeze(1), bias, padding, groups)[:, :, :L]
#     out = ACT2FN[activation](out)                                   :98
#
# Seam S13, then `identical_silu`. ONE kernel, because upstream applies the
# activation INSIDE this function and its output is the scan's `u`.
#
# The spelling is the bias-SEEDED accumulator with taps k = 0..3 ascending
# (oldest first), which is MAX `causal_conv1d.mojo:190-205` and the CUDA
# `causal_conv1d` kernel; DEVIATION 721 records that `Mamba.step`'s
# "sum then + bias" order (mamba_simple.py:218-220) is NOT mirrored, because
# two spellings would make gate D false by construction.
# ===========================================================================


def causal_conv1d_fn_kernel(
    conv_out: MutPointer[Float32, MutAnyOrigin],
    silu_out: MutPointer[Float32, MutAnyOrigin],
    in_proj: MutPointer[Float32, MutAnyOrigin],
    conv_w: MutPointer[Float32, MutAnyOrigin],
    conv_b: MutPointer[Float32, MutAnyOrigin],
    win: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32,
    l_in: Int32,
    di_in: Int32,
):
    """One thread per (batch, channel), walking `l` ascending. MAX's shape
    (`causal_conv1d.mojo:188`) and the CUDA kernel's.

    The conv's input is column `d` of the HIDDEN half of `in_proj.out`
    (`chunk(2, dim=1)`, MM:396 -- DEVIATION 730 reads it at column offset 0
    rather than copying it out). A position before the sequence reads the
    state WINDOW, which on prefill is zeros and IS `F.conv1d`'s `padding = 3`.
    """
    var b = Int(b_in)
    var l = Int(l_in)
    var di = Int(di_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= b * di:
        return
    var bb = cell // di
    var d = cell - bb * di

    for li in range(l):
        # S13, FUSED, BIAS-SEEDED.
        var acc = ftz(conv_b.unsafe_load(d))
        comptime if SAB_S13_BIAS_LAST:
            acc = Float32(0.0)
        for kk in range(D_CONV):
            var k = kk
            comptime if SAB_S13_TAPS_REVERSED:
                k = D_CONV - 1 - kk
            var p = li - (D_CONV - 1) + k
            var xv: Float32
            if p >= 0:
                xv = in_proj.unsafe_load((bb * l + p) * 2 * di + d)
            else:
                xv = win.unsafe_load((bb * di + d) * D_CONV + (D_CONV + p))
            acc = ftz(
                identical_mul_add(
                    ftz(conv_w.unsafe_load(d * D_CONV + k)), ftz(xv), acc
                )
            )
        comptime if SAB_S13_BIAS_LAST:
            acc = ftz(acc + ftz(conv_b.unsafe_load(d)))
        # `conv.out` and `silu.out` are token-major `[M, d_inner]`; this loop
        # runs (bb, d, li), so both stores are by index.
        conv_out.unsafe_store((bb * l + li) * di + d, acc)
        silu_out.unsafe_store((bb * l + li) * di + d, ftz(identical_silu(acc)))


def causal_conv1d_window_kernel(
    new_win: MutPointer[Float32, MutAnyOrigin],
    in_proj: MutPointer[Float32, MutAnyOrigin],
    old_win: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32,
    l_in: Int32,
    di_in: Int32,
):
    """The window AFTER the call: the last `D_CONV` conv INPUTS, oldest first
    (`mamba_simple.py:216-217`'s roll; `causal_conv1d.py`'s
    `conv_state.copy_(cat(conv_state, x)[:, :, -state_len:])`).

    COPIES, NOT A SEAM (contract section 4's last sentence). Written into a
    SECOND buffer -- DEVIATION 726: at `L < D_CONV` the slots a thread writes
    overlap the slots it still has to read.
    """
    var b = Int(b_in)
    var l = Int(l_in)
    var di = Int(di_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= b * di:
        return
    var bb = cell // di
    var d = cell - bb * di
    for j in range(D_CONV):
        var p = l - D_CONV + j
        var v: Float32
        if p >= 0:
            v = in_proj.unsafe_load((bb * l + p) * 2 * di + d)
        else:
            v = old_win.unsafe_load((bb * di + d) * D_CONV + (D_CONV + p))
        new_win.unsafe_store((bb * di + d) * D_CONV + j, v)


def causal_conv1d_fn(
    ctx: DeviceContext,
    mut conv_out: DeviceBuffer[DType.float32],
    mut silu_out: DeviceBuffer[DType.float32],
    mut new_win: DeviceBuffer[DType.float32],
    mut in_proj: DeviceBuffer[DType.float32],
    mut conv_w: DeviceBuffer[DType.float32],
    mut conv_b: DeviceBuffer[DType.float32],
    mut old_win: DeviceBuffer[DType.float32],
    b: Int,
    l: Int,
    d_inner: Int,
) raises:
    """`causal_conv1d_fn(hidden_states, weight, bias, activation="silu")`
    (:81-100) plus the cache's window update (`update_conv_state`, MM:415).
    ASYNCHRONOUS."""
    ctx.enqueue_function[causal_conv1d_fn_kernel](
        conv_out.unsafe_ptr(),
        silu_out.unsafe_ptr(),
        in_proj.unsafe_ptr(),
        conv_w.unsafe_ptr(),
        conv_b.unsafe_ptr(),
        old_win.unsafe_ptr(),
        Int32(b),
        Int32(l),
        Int32(d_inner),
        grid_dim=(_grid(b * d_inner), 1, 1),
        block_dim=(MAMBA_TPB, 1, 1),
    )
    ctx.enqueue_function[causal_conv1d_window_kernel](
        new_win.unsafe_ptr(),
        in_proj.unsafe_ptr(),
        old_win.unsafe_ptr(),
        Int32(b),
        Int32(l),
        Int32(d_inner),
        grid_dim=(_grid(b * d_inner), 1, 1),
        block_dim=(MAMBA_TPB, 1, 1),
    )


# ===========================================================================
# `torch.split(self.x_proj(hidden), [dt_rank, d_state, d_state], dim=-1)`
# (MM:437-441). DEVIATION 728: materialized, because gemm v1 takes only
# contiguous row-major operands. Copies, not seams.
# ===========================================================================


def split_x_proj_kernel(
    dt_low: MutPointer[Float32, MutAnyOrigin],
    b_mat: MutPointer[Float32, MutAnyOrigin],
    c_mat: MutPointer[Float32, MutAnyOrigin],
    x_proj: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    r_in: Int32,
    xr_in: Int32,
):
    var m = Int(m_in)
    var r = Int(r_in)
    var xr = Int(xr_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= m:
        return
    for j in range(r):
        dt_low.unsafe_store(t * r + j, x_proj.unsafe_load(t * xr + j))
    for n in range(D_STATE):
        b_mat.unsafe_store(t * D_STATE + n, x_proj.unsafe_load(t * xr + r + n))
        c_mat.unsafe_store(
            t * D_STATE + n, x_proj.unsafe_load(t * xr + r + D_STATE + n)
        )


# ===========================================================================
# `mamba_selective_scan`'s two steps that this file owns (:202-205 and :271),
# each a kernel -- DEVIATION 727.
# ===========================================================================


def softplus_delta_kernel(
    softplus_out: MutPointer[Float32, MutAnyOrigin],
    dt_proj: MutPointer[Float32, MutAnyOrigin],
    b_dt: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    di_in: Int32,
):
    """S14: `dt = dt + delta_bias` (:203) then `dt = F.softplus(dt)` (:205),
    which is `selective_scan_ref:145-148`.

    `dt_proj`'s bias is NOT in the matmul (MM:429 is
    `self.dt_proj.weight @ time_step`, the weight alone); it enters HERE.
    `identical_softplus` is `x <= 20 ? log1p(exp(x)) : x`, the guard verbatim
    from `selective_scan_fwd_kernel.cuh:160` and `F.softplus`'s threshold
    (DEVIATION 745).
    """
    var m = Int(m_in)
    var di = Int(di_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= m * di:
        return
    var d = i % di
    var biased = ftz(ftz(dt_proj.unsafe_load(i)) + ftz(b_dt.unsafe_load(d)))
    comptime if SAB_S14_THRESHOLD_10:
        # SABOTAGE: the guard at 10 instead of 20, otherwise the same two
        # portable transcendentals `identical_softplus` calls.
        if biased <= Float32(10.0):
            softplus_out.unsafe_store(
                i, ftz(identical_log1p(identical_exp(biased)))
            )
        else:
            softplus_out.unsafe_store(i, biased)
    else:
        softplus_out.unsafe_store(i, ftz(identical_softplus(biased)))


def z_gate_kernel(
    gate_out: MutPointer[Float32, MutAnyOrigin],
    skip_out: MutPointer[Float32, MutAnyOrigin],
    in_proj: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    di_in: Int32,
):
    """S12: `scan_output = scan_output * F.silu(z)` (:271), which is
    `selective_scan_ref:190-191`. `z` is the SECOND half of `in_proj.out`
    (`chunk(2, dim=1)`, MM:396), read at column offset `d_inner`
    (DEVIATION 730).

    PRODUCT. `identical_silu` is the SINGLE-QUOTIENT spelling
    `z / (1 + exp(-z))` -- ATen's `F.silu` and
    `selective_scan_fwd_kernel.cuh:298` -- and NOT `z * sigmoid(z)`, which
    rounds twice (DEVIATION 744).
    """
    var m = Int(m_in)
    var di = Int(di_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= m * di:
        return
    var t = i // di
    var d = i - t * di
    var z = ftz(in_proj.unsafe_load(t * 2 * di + di + d))
    var g: Float32
    comptime if SAB_S12_MUL_SIGMOID:
        var s = ftz(
            identical_div(
                Float32(1.0), ftz(Float32(1.0) + ftz(identical_exp(-z)))
            )
        )
        g = ftz(pinned_mul(z, s))
    else:
        g = ftz(identical_silu(z))
    gate_out.unsafe_store(i, ftz(pinned_mul(ftz(skip_out.unsafe_load(i)), g)))


def mamba_selective_scan(
    ctx: DeviceContext,
    mut stages: MambaDeviceStages,
    mut state: MambaDeviceState,
    mut w: MambaDeviceWeights,
    b: Int,
    l: Int,
    mut trace: IdentityTrace,
    prefix: String,
) raises:
    """`mamba_selective_scan(hidden_states, dt, A, B, C, D, z, delta_bias,
    delta_softplus, return_last_state)` (:175-280), the torch fallback, in the
    recurrent-iteration arm (`use_mambapy` False, `use_associative_scan`
    False -- contract section 9 claims no chunked or tree scan).

    THE BUNDLE IS SPLIT, DEVIATION 727. Upstream this one function applies
    `delta_bias` and the softplus (:202-205), runs the recurrence (:246-258),
    then `D` (:268) and `z` (:271), and returns one tensor. Contract section 7
    needs four recorded stages out of that, so:

      * S14 (`softplus.out`) is `softplus_delta_kernel`, here;
      * S5-S10 (`scan.y`) and S11 (`skip.out`) are `selective_scan_fn`, which
        also carries `D` -- that seam is the scan file's (its DEVIATION 723);
      * S12 (`gate.out`) is `z_gate_kernel`, here.

    So the scan is called with `z` and `delta_bias` NOT PRESENT and
    `delta_softplus` False, all three of which it REFUSES if passed True, and
    with `return_last_state` True, which it requires. The arithmetic and its
    order are unchanged.

    STAGE ORDER. `softplus.out` is recorded before the call and `gate.out`
    after it, so the trace reads `softplus.out, scan.y, skip.out, scan.h,
    gate.out`. Contract section 7 lists `scan.h` before `skip.out`; the scan
    file emits those two the other way round. The differ aligns tag
    SEQUENCES, so what matters is that the order is the same on every vendor,
    which it is -- but the card's listed order and the emitted order differ by
    that one transposition and the contract, being frozen, is the thing that
    would have to change.
    """
    var di = stages.dims.d_inner
    var m = b * l

    # dt = dt + delta_bias  (:203);  dt = F.softplus(dt)  (:205).  S14.
    ctx.enqueue_function[softplus_delta_kernel](
        stages.softplus_out.unsafe_ptr(),
        stages.dt_proj.unsafe_ptr(),
        w.b_dt.unsafe_ptr(),
        Int32(m),
        Int32(di),
        grid_dim=(_grid(m * di), 1, 1),
        block_dim=(MAMBA_TPB, 1, 1),
    )
    ctx.synchronize()
    trace.record_device[DType.float32](
        ctx, prefix + ".softplus.out", stages.softplus_out, m * di
    )

    # The recurrence (:246-258) == `selective_scan_ref:160-189`, seams S5-S11.
    # ARGUMENT ORDER AND NAMES ARE UPSTREAM'S, over the oracle's buffer
    # conventions; the file's own header carries the signature verbatim.
    # `z`, `delta_bias` False = upstream's `None`; `delta_softplus` False
    # because `delta` arrives post-softplus; `return_last_state` True because
    # `h_state` is in-and-out on every call (contract section 5).
    # It emits `scan.y`, `skip.out` and `scan.h` itself.
    selective_scan_fn(
        ctx,
        stages.skip_out,
        stages.scan_y,
        state.h,
        stages.silu_out,
        stages.softplus_out,
        stages.a_out,
        stages.b_mat,
        stages.c_mat,
        w.d_skip,
        b,
        l,
        di,
        False,
        False,
        False,
        True,
        trace,
        prefix,
    )
    ctx.synchronize()

    # `scan_output = scan_output * F.silu(z)` (:271). S12.
    ctx.enqueue_function[z_gate_kernel](
        stages.gate_out.unsafe_ptr(),
        stages.skip_out.unsafe_ptr(),
        stages.in_proj.unsafe_ptr(),
        Int32(m),
        Int32(di),
        grid_dim=(_grid(m * di), 1, 1),
        block_dim=(MAMBA_TPB, 1, 1),
    )
    ctx.synchronize()
    trace.record_device[DType.float32](
        ctx, prefix + ".gate.out", stages.gate_out, m * di
    )


# ===========================================================================
# `MambaBlock.forward`'s residual (:521-527). Seam S16.
# ===========================================================================


def residual_add_kernel(
    residual_out: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    mixer_out: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """S16: `hidden_states = residual + hidden_states` (:527), one add.
    `residual` is the block INPUT as given (:521 `residual = hidden_states`,
    taken before the norm), and `residual_in_fp32` is moot in an FP32
    profile."""
    var n = Int(n_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= n:
        return
    residual_out.unsafe_store(
        i, ftz(ftz(x.unsafe_load(i)) + ftz(mixer_out.unsafe_load(i)))
    )


# ===========================================================================
# `MambaMixer.forward` (:359-483) and `MambaBlock.forward` (:505-530)
# ===========================================================================


def _gemm_op_nt() -> Int:
    """`OP_NT` in `gemm_oracle`'s numbering (`OP_NN = 0, OP_NT = 1,
    OP_TN = 2`), which is the numbering `identical_gemm` reads. NOT
    `bench/gemm_shapes.mojo`'s `OP_NT = 0`. See the header's trap note."""
    comptime if SAB_S17_OP_NUMBERING:
        # SABOTAGE: the bench table's `OP_NT = 0`, which `identical_gemm`
        # reads as `OP_NN`. Every operand is still exactly the right SIZE --
        # `[2*d_inner, d_model]` read as `[d_model, 2*d_inner]` has the same
        # element count -- so nothing raises, no bound is exceeded, and every
        # product is wrong.
        return OP_NN
    return OP_NT


def _refuse_nonfinite_named(name: String, values: List[Float32]) raises:
    """Contract section 6 / row 39, the device path's copy of the oracle's
    `refuse_nonfinite`. Tested BY BITS, not by compares: Metal flushes COMPARE
    operands (row 49), so a bit test is the only spelling with one meaning on
    every column. DEVIATION 729 explains why this is not the oracle's function
    imported."""
    for i in range(len(values)):
        var au = bitcast[DType.uint32](values[i]) & UInt32(0x7FFFFFFF)
        if au > UInt32(0x7F800000):
            raise Error(
                String("mamba: NaN in ")
                + name
                + " at flat index "
                + String(i)
                + " REFUSED (row 39: NaN payloads are vendor-shaped; no"
                + " stage may record one)"
            )
        if au == UInt32(0x7F800000):
            raise Error(
                String("mamba: infinity in ")
                + name
                + " at flat index "
                + String(i)
                + " REFUSED (row 39)"
            )


def mamba_refuse_bad_inputs(
    ctx: DeviceContext,
    mut w: MambaDeviceWeights,
    mut x: DeviceBuffer[DType.float32],
    mut state: MambaDeviceState,
    b: Int,
    l: Int,
) raises:
    """Every named input and parameter of one block call, refused if it holds
    a NaN or an infinity. The names are the oracle's `refuse_bad_inputs`
    names, in its order, so the two sides fail identically."""
    var dims = w.dims.copy()
    var dm = dims.d_model
    var di = dims.d_inner
    var r = dims.dt_rank
    var xr = dims.x_proj_rows()
    _refuse_nonfinite_named("x", mamba_download(ctx, x, b * l * dm))
    _refuse_nonfinite_named("norm.weight", mamba_download(ctx, w.norm_w, dm))
    _refuse_nonfinite_named(
        "in_proj.weight", mamba_download(ctx, w.w_in, 2 * di * dm)
    )
    _refuse_nonfinite_named(
        "conv1d.weight", mamba_download(ctx, w.conv_w, di * D_CONV)
    )
    _refuse_nonfinite_named("conv1d.bias", mamba_download(ctx, w.conv_b, di))
    _refuse_nonfinite_named(
        "x_proj.weight", mamba_download(ctx, w.w_x, xr * di)
    )
    _refuse_nonfinite_named(
        "dt_proj.weight", mamba_download(ctx, w.w_dt, di * r)
    )
    _refuse_nonfinite_named("dt_proj.bias", mamba_download(ctx, w.b_dt, di))
    _refuse_nonfinite_named(
        "A_log", mamba_download(ctx, w.a_log, di * D_STATE)
    )
    _refuse_nonfinite_named("D", mamba_download(ctx, w.d_skip, di))
    _refuse_nonfinite_named(
        "out_proj.weight", mamba_download(ctx, w.w_out, dm * di)
    )
    _refuse_nonfinite_named(
        "state.conv_win",
        mamba_download(ctx, state.conv_win, b * di * D_CONV),
    )
    _refuse_nonfinite_named(
        "state.h", mamba_download(ctx, state.h, b * di * D_STATE)
    )


def mamba_mixer_forward(
    ctx: DeviceContext,
    mut stages: MambaDeviceStages,
    mut state: MambaDeviceState,
    mut w: MambaDeviceWeights,
    b: Int,
    l: Int,
    mut trace: IdentityTrace,
    prefix: String,
) raises:
    """`MambaMixer.forward(hidden_states, cache_params)` (:359-483), the torch
    fallback path, inference only.

    Their order, and this function's:

        projected_states = self.in_proj(hidden_states).transpose(1, 2)  :371
        A = -torch.exp(self.A_log.float())                              :373
        hidden_states_B_C, gate = projected_states.chunk(2, dim=1)      :396
        hidden_states_B_C = causal_conv1d_fn(..., activation="silu")    :410
        time_step, B, C = torch.split(self.x_proj(...), [...])          :437
        time_step = self.dt_proj.weight @ time_step.transpose(1, 2)     :443
        scan_output = mamba_selective_scan(...)                         :461
        contextualized_states = self.out_proj(scan_output...)           :481

    `norm.out` arrives as this function's input: `MambaBlock.forward`
    normalizes first (:522) and calls the mixer with the normalized rows, so
    the mixer reads `stages.norm_out` rather than taking a tensor argument.

    Every stage of contract section 7 that this function computes is recorded
    here, in the section's order, under `prefix`. Tags must be UNIQUE within a
    trace (`core/identity_trace.mojo`'s tag invariant), so the DRIVER supplies
    the prefix -- one per block call -- and it carries no machine property
    (that file's rule 2).
    """
    var dims = stages.dims.copy()
    var dm = dims.d_model
    var di = dims.d_inner
    var r = dims.dt_rank
    var xr = dims.x_proj_rows()
    var m = b * l

    # ---- in_proj (:371). `nn.Linear(d_model, 2*d_inner, bias=use_bias)` with
    #      `use_bias` False, so weight only.
    #      GEMM v1 OP_NT: C[M, 2di] = norm_out[M, dm] . w_in[2di, dm]^T.
    #      `k = d_model <= 128` so `P == 1` -- gemm contract 7.1's serial
    #      ascending chain with nothing to fold.
    identical_gemm(
        ctx,
        stages.in_proj,
        stages.norm_out,
        w.w_in,
        m,
        2 * di,
        dm,
        _gemm_op_nt(),
    )
    trace.record_device[DType.float32](
        ctx, prefix + ".in_proj.out", stages.in_proj, m * 2 * di
    )

    # ---- A = -exp(A_log) (:373). S15.
    ctx.enqueue_function[mamba_a_from_a_log_kernel](
        stages.a_out.unsafe_ptr(),
        w.a_log.unsafe_ptr(),
        Int32(di * D_STATE),
        grid_dim=(_grid(di * D_STATE), 1, 1),
        block_dim=(MAMBA_TPB, 1, 1),
    )
    ctx.synchronize()
    trace.record_device[DType.float32](
        ctx, prefix + ".A.out", stages.a_out, di * D_STATE
    )

    # ---- chunk(2, dim=1) (:396): DEVIATION 730, a column offset, no copy.
    # ---- causal_conv1d_fn (:410-416) + the window update (:415). S13.
    causal_conv1d_fn(
        ctx,
        stages.conv_out,
        stages.silu_out,
        stages.conv_win,
        stages.in_proj,
        w.conv_w,
        w.conv_b,
        state.conv_win,
        b,
        l,
        di,
    )
    ctx.synchronize()
    trace.record_device[DType.float32](
        ctx, prefix + ".conv.out", stages.conv_out, m * di
    )
    trace.record_device[DType.float32](
        ctx, prefix + ".silu.out", stages.silu_out, m * di
    )
    trace.record_device[DType.float32](
        ctx, prefix + ".conv.window", stages.conv_win, b * di * D_CONV
    )
    # The state carries the NEW window into the next call. A copy, and it
    # happens after `conv.window` is recorded so the stage is the window as
    # computed. DEVIATION 726 is why the two buffers are distinct.
    ctx.enqueue_copy(dst_buf=state.conv_win, src_buf=stages.conv_win)
    ctx.synchronize()

    # ---- x_proj (:437). `nn.Linear(d_inner, dt_rank + 2*d_state,
    #      bias=False)`. C[M, xr] = silu_out[M, di] . w_x[xr, di]^T.
    identical_gemm(
        ctx, stages.x_proj, stages.silu_out, w.w_x, m, xr, di, _gemm_op_nt()
    )
    trace.record_device[DType.float32](
        ctx, prefix + ".x_proj.out", stages.x_proj, m * xr
    )

    # ---- torch.split (:437-441). DEVIATION 728: materialized. Copies.
    ctx.enqueue_function[split_x_proj_kernel](
        stages.dt_low.unsafe_ptr(),
        stages.b_mat.unsafe_ptr(),
        stages.c_mat.unsafe_ptr(),
        stages.x_proj.unsafe_ptr(),
        Int32(m),
        Int32(r),
        Int32(xr),
        grid_dim=(_grid(m), 1, 1),
        block_dim=(MAMBA_TPB, 1, 1),
    )
    ctx.synchronize()

    # ---- dt_proj (:443), WITHOUT the bias: their line is
    #      `self.dt_proj.weight @ time_step.transpose(1, 2)`, the weight
    #      alone; the bias is handed to the scan as `delta_bias` (:445, :457)
    #      and enters at S14.
    #      C[M, di] = dt_low[M, r] . w_dt[di, r]^T, `k = dt_rank`.
    #      INHERITED CLAUSE (contract section 4, gemm 9.2(a)): at
    #      `k = dt_rank = 1` the gemm leaf is `ftz(fma(a, b, +0.0))`, so a
    #      `-0.0`-valued product reaches this stage as `+0.0`. That is v1
    #      gemm behavior and this profile inherits it UNCHANGED.
    identical_gemm(
        ctx, stages.dt_proj, stages.dt_low, w.w_dt, m, di, r, _gemm_op_nt()
    )
    trace.record_device[DType.float32](
        ctx, prefix + ".dt_proj.out", stages.dt_proj, m * di
    )

    # ---- the scan and the two steps around it that this file owns
    #      (:461-471). S14, then S5-S11 in the scan file, then S12.
    mamba_selective_scan(ctx, stages, state, w, b, l, trace, prefix)
    # A readable snapshot of `scan.h` for the gates. The STAGE was recorded
    # by `selective_scan_fn` off `state.h`; this copy carries no tag.
    ctx.enqueue_copy(dst_buf=stages.scan_h, src_buf=state.h)
    ctx.synchronize()

    # ---- out_proj (:481). `nn.Linear(d_inner, d_model, bias=use_bias)` with
    #      `use_bias` False. C[M, dm] = gate_out[M, di] . w_out[dm, di]^T.
    identical_gemm(
        ctx,
        stages.out_proj,
        stages.gate_out,
        w.w_out,
        m,
        dm,
        di,
        _gemm_op_nt(),
    )
    trace.record_device[DType.float32](
        ctx, prefix + ".out_proj.out", stages.out_proj, m * dm
    )


def mamba_block_forward(
    ctx: DeviceContext,
    mut stages: MambaDeviceStages,
    mut state: MambaDeviceState,
    mut w: MambaDeviceWeights,
    mut x: DeviceBuffer[DType.float32],
    b: Int,
    l: Int,
    mut trace: IdentityTrace,
    prefix: String,
) raises:
    """`MambaBlock.forward(hidden_states, cache_params)` (:505-530).

        residual = hidden_states                                       :521
        hidden_states = self.norm(hidden_states)                       :522
        hidden_states = self.mixer(hidden_states, ...)                 :526
        hidden_states = residual + hidden_states                       :527

    THE ENTRY POINT of profile `mojolearn.identical.mamba1.fp32.v1`.

    Prefill is a fresh zero `MambaDeviceState`; the decode step is this same
    function at `l == 1` carrying the state. ONE spelling for both paths --
    which is what makes contract section 8's clause (d) (decode == prefill,
    bitwise, at every position) a theorem the gate then verifies, rather than
    a coincidence the gate hopes for.

    Emits every stage of contract section 7 under `prefix`; three of them
    (`scan.y`, `skip.out`, `scan.h`) come out of `selective_scan_fn`, which is
    why nothing here records them.
    """
    if b <= 0 or l <= 0:
        raise Error(
            "mamba_block_forward: B and L must be positive, got B="
            + String(b)
            + " L="
            + String(l)
        )
    if stages.b != b or stages.l != l:
        raise Error(
            "mamba_block_forward: stages were built for B="
            + String(stages.b)
            + " L="
            + String(stages.l)
            + " but the call is B="
            + String(b)
            + " L="
            + String(l)
        )
    if state.b != b or state.d_inner != stages.dims.d_inner:
        raise Error(
            "mamba_block_forward: the state's shape is not the call's (state"
            " B="
            + String(state.b)
            + " d_inner="
            + String(state.d_inner)
            + ")"
        )
    if w.dims.d_model != stages.dims.d_model:
        raise Error(
            "mamba_block_forward: the weights' d_model is not the stages'"
        )

    # Contract section 6, before ANY recorded stage.
    mamba_refuse_bad_inputs(ctx, w, x, state, b, l)

    var dims = stages.dims.copy()
    var dm = dims.d_model
    var m = b * l

    trace.record_device[DType.float32](ctx, prefix + ".input.x", x, m * dm)

    # ---- residual = hidden_states (:521): a NAME, not a copy. `x` is read
    #      again at S16 and nothing writes it.
    # ---- self.norm(...) (:522) == MambaRMSNorm.forward (:495-499). S1-S4.
    mamba_rms_norm(ctx, stages.norm_sumsq, stages.norm_out, x, w.norm_w, m, dm)
    ctx.synchronize()
    trace.record_device[DType.float32](
        ctx, prefix + ".norm.sumsq", stages.norm_sumsq, m
    )
    trace.record_device[DType.float32](
        ctx, prefix + ".norm.out", stages.norm_out, m * dm
    )

    # ---- self.mixer(...) (:526).
    mamba_mixer_forward(ctx, stages, state, w, b, l, trace, prefix)

    # ---- residual + hidden_states (:527). S16.
    ctx.enqueue_function[residual_add_kernel](
        stages.residual_out.unsafe_ptr(),
        x.unsafe_ptr(),
        stages.out_proj.unsafe_ptr(),
        Int32(m * dm),
        grid_dim=(_grid(m * dm), 1, 1),
        block_dim=(MAMBA_TPB, 1, 1),
    )
    ctx.synchronize()
    trace.record_device[DType.float32](
        ctx, prefix + ".residual.out", stages.residual_out, m * dm
    )
