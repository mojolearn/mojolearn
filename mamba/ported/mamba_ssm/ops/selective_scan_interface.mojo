"""`mamba_ssm/ops/selective_scan_interface.py::selective_scan_ref`, on device.

THE ENTRY SIGNATURE, verbatim, so the caller has one place to read it:

    selective_scan_fn(
        ctx: DeviceContext,
        mut out: DeviceBuffer[DType.float32],       # [M, dim]  S11, stage skip.out
        mut y: DeviceBuffer[DType.float32],         # [M, dim]  S5-S10, stage scan.y
        mut h_state: DeviceBuffer[DType.float32],   # [B, dim, 16] IN and OUT, stage scan.h
        mut u: DeviceBuffer[DType.float32],         # [M, dim]      upstream `u`
        mut delta: DeviceBuffer[DType.float32],     # [M, dim]      upstream `delta`, POST-softplus
        mut A: DeviceBuffer[DType.float32],         # [dim, 16]     upstream `A`
        mut B: DeviceBuffer[DType.float32],         # [M, 16]       upstream `B`
        mut C: DeviceBuffer[DType.float32],         # [M, 16]       upstream `C`
        mut D: DeviceBuffer[DType.float32],         # [dim]         upstream `D`
        batch: Int, seqlen: Int, dim: Int,
        z: Bool, delta_bias: Bool,                  # PRESENCE flags; True is REFUSED (DEVIATION 723)
        delta_softplus: Bool,                       # True is REFUSED (DEVIATION 723)
        return_last_state: Bool,                    # False is REFUSED (DEVIATION 723)
        mut trace: IdentityTrace, prefix: String,
        block_size: Int = 64,
    ) raises

`M = B * L` token-major, exactly the layouts
`mamba/mojo_only/mamba_oracle.mojo::selective_scan_oracle` uses for the same
values (`u`, `delta` `[M, d_inner]`; `a` `[d_inner, D_STATE]`; `bmat`, `cmat`
`[M, D_STATE]`; `h_state` `[B, d_inner, D_STATE]`). That oracle is the
INTERFACE AUTHORITY and this file invents no convention beside it: the only
additions are `out`/`D` (the oracle's block applies seam S11 after the scan
returns; the seam is this lane's, so it is applied here and both results are
written) and the `ctx`/`trace`/`block_size` machinery a device launch needs.

Arguments after the outputs are upstream's ORDER and NAMES --
`(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state)`,
`selective_scan_interface.py:127-128`. Outputs come first because the device
kernel shape this file takes does that (`max/kernels/src/state_space/
selective_scan.mojo::selective_scan_fwd_gpu:95-105`, `output` then `x` then
`out_z` then `u`).

WHAT IS MIRRORED, AND FROM WHICH SOURCE
---------------------------------------
- The ARITHMETIC and its rounding ORDER: `selective_scan_ref` (state-spaces/
  mamba `e9594ce`, `mamba_ssm/ops/selective_scan_interface.py:127-193`),
  seams S5-S11 of `mamba/IDENTICAL_MAMBA_CONTRACT.md` section 4, profile
  `mojolearn.identical.mamba1.fp32.v1`.
- The KERNEL SHAPE: `selective_scan_fwd_gpu` (modular/modular `10d978e`,
  `max/kernels/src/state_space/selective_scan.mojo:74+`). One thread per
  `(batch, dim)` pair, serial over the sequence, `DSTATE = 16` held in
  registers. DEVIATION 722 records what of that kernel is taken and what is
  refused.

**THE TWO SOURCES DISAGREE ABOUT ROUNDING AND THE REFERENCE WINS.** At S8 the
CUDA kernel forms `B * (delta * u)` (`csrc/selective_scan/
selective_scan_fwd_kernel.cuh:162` computes `delta_u_vals = delta * u`, `:222`
multiplies `B_vals[i] * delta_u_vals[r][i]`) and MAX follows it
(`selective_scan.mojo:309` `var delta_u = delta_val * u_val`, `:321`
`var b_t = B_vals * delta_u`). The reference pairs the other way, `(delta * B)
* u` -- `torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)` contracts its
operands left to right, and HuggingFace spells the same order explicitly
(`modeling_mamba.py:214-215`, `discrete_B = dt * B` then `deltaB_u =
discrete_B * u`). At S11 the CUDA kernel SEEDS its output accumulator with
`D * u` (`:163`) and adds the C-h terms onto it (`:266`), D FIRST; the
reference adds `u * D` to the finished `y`, D LAST (`:189`). **MAX does NOT
follow the CUDA kernel here** -- `selective_scan.mojo:323`, `:328-329` reduces
the C-h terms first and then does `output_val += D_val * u_val`, D last, which
is the reference's order and this profile's -- so a reader who took S8 from
MAX must not assume S11 came from there too. Contract section 4 pins the
REFERENCE's order at both seams. Fixtures F5 and F6 separate them, and the
sabotage switches below are how this file is shown to be on the right side of
each. **Do not "fix" either seam toward the kernel.**

WHAT THIS FILE DOES NOT DO
---------------------------
S12 (`out * silu(z)`, stage `gate.out`) and S14 (`delta = softplus(dt +
bias)`, stage `softplus.out`) are recorded stages of the BLOCK
(`mamba/ported/transformers/models/mamba/modeling_mamba.mojo`), not of the
scan, so this file refuses `z`, `delta_bias` and `delta_softplus` rather than
carrying a second spelling of a seam it does not own. DEVIATION 723.

`refuse_nonfinite` (contract section 6, row 39) is the DRIVER's door: it runs
once over the inputs and the weights before any recorded stage. Nothing here
re-tests, and nothing here can produce a NaN from finite inputs under the
profile's ranges.

THE DEVIATIONS THIS FILE USES
-----------------------------
**DEVIATION 720** (shared with `mamba_oracle.mojo:41-54`, its host spelling):
`pinned_mul`, a multiply spelled `identical_mul_add(a, b, -0.0)` so no codegen
can contract it into a neighboring add and so a `-0.0` product survives.

**DEVIATION 722**: the KERNEL SHAPE is MAX's `selective_scan_fwd_gpu` and not
the reference's loop structure, and the parts of MAX's kernel that would move
bits are refused. TAKEN: one thread per `(batch, dim)` pair with a linear
thread id and a bounds check (`:174-179`); the whole `DSTATE` state vector in
registers (`:186-190`); A's row for the thread's dim hoisted out of the token
loop (`:215-220`); the sequence walked serially, ascending, one token at a
time (`:246-330`). REFUSED, each because it is a different arithmetic and not
a different schedule: the `exp2(A * LOG2E * delta)` substitution for
`exp(delta * A)` (`:38`, `:215-220`, `:320` -- an extra rounding in the LOG2E
product, and `exp2` is not `exp`; `SAB_S5_EXP2`); the `B * (delta * u)`
pairing at S8 (`:309`, `:321`; `SAB_S8_CUDA_PAIRING`); the SIMD tree fold
`(state * C_vals).reduce_add()` at S10 (`:323`) in place of the serial
ascending fma chain (`SAB_S10_DESCENDING` breaks the same clause a different
way). NOT PORTED AT ALL, because they serve a backward pass and a schedule
this profile does not have: the `TILE_SIZE = 8` staging with its pre-loaded
B/C tiles and buffered stores (`:243-330`), the `cum_a` / `cum_b` running
products, the `x` checkpoint tensor and its chunking, the group/`n_groups`
fan-out, and the `out_z` fused gate.

Also under 722, and it is a departure from the REFERENCE's shape rather than
MAX's: the reference materializes `deltaA` and `deltaB_u` as full
`[B, D, L, N]` tensors BEFORE the recurrence (ref:162, :167) and this kernel
computes each element inside the recurrence, at the moment it is used. Every
one of those elements is a pure function of its own `(b, d, l, n)` inputs, so
moving the computation moves no bit; what it buys is that the kernel needs no
storage that grows with `L`, which is what lets one thread own a whole
sequence.

**DEVIATION 723**: the entry point mirrors upstream's argument order and
names but REFUSES four of the configurations they admit, each refused BY NAME
at the door rather than implemented and left untested (`reached-but-inert`).
`z` and `delta_bias` are presence flags and True raises, because S12 and S14
are the block's recorded stages; `delta_softplus` True raises for the same
reason; `return_last_state` False raises because `h_state` is in-and-out on
every call. Three branches of `selective_scan_ref` are likewise NOT ported
and cannot be reached from this signature: the complex-`A` arm (`:152-156`,
`:185-186` -- FP32 real only, contract section 3), the non-variable `B`/`C`
arm (`:163-164`, `:176-177` -- Mamba-1's `x_proj` always makes both
per-token), and the 4-D grouped `B`/`C` arms with their `repeat` fan-out
(`:168-170`, `:171-172`, `:181-182`), since the profile is `n_groups = 1`.

**DEVIATION 724 IS NOT USED BY THIS FILE** and remains free for the lane.

`[[mojo-buffer-freed-at-last-use]]`: every buffer here is the CALLER's and
the caller keeps it alive past `ctx.synchronize()`.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.math import exp2
from std.sys.compile import is_defined

from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from mojo_only.numerics import ftz, identical_exp, identical_mul_add


# ===========================================================================
# CONSTANTS
# ===========================================================================

comptime MAX_DSTATE = 16
"""`MAX_DSTATE` in BOTH device kernels -- `selective_scan.mojo:39` and
`selective_scan_fwd_kernel.cuh`'s `MAX_DSTATE` -- and `d_state` in contract
section 3. Declared here rather than imported from
`mamba/mojo_only/mamba_fixture.mojo` because both upstream kernels declare it
themselves, and because the device spelling is an INDEPENDENT transcription
of the arithmetic (the oracle's header states that separation); the only
things shared with the host side are the seam functions in
`mojo_only/numerics.mojo`."""


# ===========================================================================
# THE SABOTAGE SWITCHES
# ===========================================================================
# OFF in every build that does not name them. Each is a specific way to break
# a seam this file pins, reachable by a plausible implementer -- three of the
# five are what you get by porting from the CUDA kernel or from MAX instead
# of from the reference, which is exactly the mistake this lane's contract
# was written to prevent. A gate that has never failed is a gate nobody has
# tested.
#
#     tools/with_identical_mode.sh pixi run mojo run \
#         -D MOJOLEARN_MAMBA_SABOTAGE_S8_CUDA_PAIRING=1 \
#         -I . mamba/mojo_only/mamba_check.mojo
#
#: S8 pairs the CUDA way, `B * (delta * u)` (cuh:162 with :222, and MAX
#: `selective_scan.mojo:309`, `:321`), instead of the reference's `(delta * B) *
#: u`. Contract section 4 seam S8, fixture F5.
comptime SAB_S8_CUDA_PAIRING = is_defined[
    "MOJOLEARN_MAMBA_SABOTAGE_S8_CUDA_PAIRING"
]()
#: S9 rounds TWICE -- `ftz(pinned_mul(deltaA, h))` then `ftz(... + deltaB_u)`
#: -- which is the torch reference's own literal spelling (`x = deltaA[:, :,
#: i] * x + deltaB_u[:, :, i]`, ref:175) and is what an implementer gets by
#: writing `a * h + b` on a backend that does not contract. Contract section
#: 4 seam S9 pins the FUSED single rounding; fixture F2.
comptime SAB_S9_UNFUSED = is_defined["MOJOLEARN_MAMBA_SABOTAGE_S9_UNFUSED"]()
#: S11 seeds the fold accumulator with `D * u` and lets the C-h terms
#: accumulate onto it, D FIRST -- the CUDA kernel's shape (cuh:163). The
#: profile adds `u * D` to the FINISHED y, D last. Fixture F6.
comptime SAB_S11_D_FIRST = is_defined[
    "MOJOLEARN_MAMBA_SABOTAGE_S11_D_FIRST"
]()
#: S10 folds n DESCENDING instead of ascending. The einsum's fold order is
#: torch-internal, so the contract pins the CUDA kernel's ascending
#: `state_idx` walk (cuh:168-266); a descending fold is the same set of terms
#: in a different order and rounds differently.
comptime SAB_S10_DESCENDING = is_defined[
    "MOJOLEARN_MAMBA_SABOTAGE_S10_DESCENDING"
]()
#: S6 computes `exp2(A * LOG2E * delta)` instead of `exp(delta * A)` -- MAX's
#: optimization (`selective_scan.mojo:38`, `:215-220`, `:320`) and the CUDA
#: kernel's (`cuh:174-176`, `:222`). A DIFFERENT FUNCTION: the LOG2E product
#: is a rounding the reference does not have, and `exp2` is not `exp`.
#: Contract section 4 seam S5-S6, and row 12's whole argument.
comptime SAB_S5_EXP2 = is_defined["MOJOLEARN_MAMBA_SABOTAGE_S5_EXP2"]()

# WHAT EACH SABOTAGE WAS MEASURED TO DO, 2026-08-23, Apple M4 through MAX,
# IDENTICAL mode, device card diffed against `mamba_block_oracle`'s. Every
# one of the five FAILS the diff, and the SHAPE AT WHICH IT FIRST FAILS is
# the part worth reading:
#
#   S8_CUDA_PAIRING   fails at b=1 l=1 d_model=8,  1 ulp on scan.y and scan.h
#   S11_D_FIRST       fails at b=1 l=1 d_model=8,  and it moves scan.y by the
#                     WHOLE `D * u` term, not by an ulp, because the seed
#                     leaks into the stage recorded before D
#   S10_DESCENDING    fails at b=1 l=1 d_model=8,  1 ulp on scan.y
#   S9_UNFUSED        agrees at every L = 1 shape, first fails at L = 4
#   S5_EXP2           agrees at every L = 1 shape, first fails at L = 4
#
# **A FIXTURE WHOSE ONLY SHAPE IS L = 1 CANNOT SEE S5, S6 OR S9 AT ALL**, and
# that is structural rather than unlucky: at the first token `h` is the zero
# state, so `h = fma(deltaA, 0, deltaB_u)` is `deltaB_u` for EVERY value of
# `deltaA` and for either rounding. `deltaA` -- and with it the whole of the
# exp seam -- is multiplied by zero. The gate shapes in contract section 3
# carry L in {4, 16, 64, 257} and are therefore fine, but gate D's decode arm
# is exactly an L = 1 call, so a decode-only fixture is blind to three of the
# seven seams this file owns and must never be the only arm. Recorded here
# because the next person to trim a fixture for speed will reach for L = 1.
# (The same shape boundary shows in FAST mode, which agrees with the host
# oracle at every L = 1 shape and diverges from L = 4 on: same reason.)

comptime ANY_SABOTAGE = (
    SAB_S8_CUDA_PAIRING
    or SAB_S9_UNFUSED
    or SAB_S11_D_FIRST
    or SAB_S10_DESCENDING
    or SAB_S5_EXP2
)


def mamba_scan_sabotage_name() -> String:
    """Which sabotage this binary compiled with, for a gate's banner."""
    comptime if SAB_S8_CUDA_PAIRING:
        return String("S8_CUDA_PAIRING")
    comptime if SAB_S9_UNFUSED:
        return String("S9_UNFUSED")
    comptime if SAB_S11_D_FIRST:
        return String("S11_D_FIRST")
    comptime if SAB_S10_DESCENDING:
        return String("S10_DESCENDING")
    comptime if SAB_S5_EXP2:
        return String("S5_EXP2")
    return String("none")


# ===========================================================================
# THE PINNED MULTIPLY (DEVIATION 720)
# ===========================================================================


def pinned_mul(a: Float32, b: Float32) -> Float32:
    """DEVIATION 720, the DEVICE spelling of the same deviation
    `mamba/mojo_only/mamba_oracle.mojo:41-54` carries on the host.

    A multiply no codegen may contract into a neighboring add, written
    `identical_mul_add(a, b, -0.0)`. Under IDENTICAL that is `fma(a, b,
    -0.0)`, bit-equal to the correctly rounded product at every input
    including both zero signs (`p + (-0.0) == p` under round-to-nearest,
    while a `+0.0` addend would launder a `-0.0` product -- the gemm lane's
    F6a lesson), and it presents no syntactic multiply for a compiler to
    contract (the gemm README's F3 scar: `var p = a * b; p + c` WAS
    contracted across statements). Under FAST it is `a * b + (-0.0)`.

    Transcribed rather than imported, for the reason `MAX_DSTATE` above is:
    the oracle and the device kernel are two spellings of one arithmetic and
    share only `mojo_only/numerics.mojo`. The two bodies being identical is
    the point, and the gate that diffs the two cards is what holds them so.
    """
    return identical_mul_add(a, b, Float32(-0.0))


# ===========================================================================
# THE KERNEL
# ===========================================================================


def selective_scan_fwd_kernel[
    DSTATE: Int
](
    out_ptr: MutPointer[Float32, MutAnyOrigin],
    y_ptr: MutPointer[Float32, MutAnyOrigin],
    h_ptr: MutPointer[Float32, MutAnyOrigin],
    u_ptr: MutPointer[Float32, MutAnyOrigin],
    delta_ptr: MutPointer[Float32, MutAnyOrigin],
    a_ptr: MutPointer[Float32, MutAnyOrigin],
    b_ptr: MutPointer[Float32, MutAnyOrigin],
    c_ptr: MutPointer[Float32, MutAnyOrigin],
    d_ptr: MutPointer[Float32, MutAnyOrigin],
    batch_in: Int32,
    seqlen_in: Int32,
    dim_in: Int32,
):
    """One `(batch, dim)` pair per thread, the whole sequence in order.

    THE SHAPE IS MAX'S (`selective_scan_fwd_gpu:174-179`): `thread_id =
    block_dim.x * block_idx.x + thread_idx.x`, bounds check, `b, d =
    divmod(thread_id, dim)`, `DSTATE` state values in registers. THE ORDER IS
    THE REFERENCE'S (`selective_scan_ref:160-187`), line for line:

        :162  deltaA   = exp(delta * A)        S5 `pinned_mul`, S6 `identical_exp`
        :167  deltaB_u = (delta * B) * u       S7 then S8, both `pinned_mul`
        :175  h        = deltaA * h + deltaB_u S9, ONE rounding through fma
        :180  y        = sum_n h[n] * C[n]     S10, serial ascending n from +0.0
        :189  out      = y + u * D             S11, PRODUCT then add, unfused

    WHY NOTHING HERE CAN DEPEND ON THE LAUNCH. A thread owns a whole `(b, d)`
    recurrence: every product, every fma and every fold addition of every one
    of its cells happens in its own registers, and no float crosses a thread
    boundary anywhere in this file -- there is no shared memory, no
    reduction, no atomic. So `block_size` and the grid decide only WHICH
    thread holds a pair, never the sequence of values accumulated into it,
    and batch-composition invariance (contract section 8(c)) is a property of
    the kernel's shape rather than of a check that happens to pass. That is
    the same structural argument `gemm/mojo_only/gemm_identical.mojo`'s
    header makes for its FLAT plan, and it is why Mamba's serial-over-L scan
    can have the clause vLLM's batch-invariant mode cannot give it.

    `y_ptr` and `out_ptr` are both written because both are recorded stages
    (`scan.y` before D, `skip.out` after). The reference computes them in one
    expression, `out = y if D is None else y + u * rearrange(D, ...)`; the
    intermediate is the same value either way and writing it costs a store,
    not a rounding.
    """
    var batch = Int(batch_in)
    var seqlen = Int(seqlen_in)
    var dim = Int(dim_in)

    var thread_id = Int(block_dim.x) * Int(block_idx.x) + Int(thread_idx.x)
    if thread_id >= batch * dim:
        return
    var bb = thread_id // dim
    var d = thread_id - bb * dim
    if bb >= batch or d >= dim:
        return

    # `x = A.new_zeros((batch, dim, dstate))` (ref:160) is the caller's zero
    # buffer; a decode step passes the state carried from the previous call
    # (contract section 5). MAX holds the same values in a SIMD register file
    # (`selective_scan.mojo:186-190`, "max dstate 16 to fit in registers").
    var state = SIMD[DType.float32, DSTATE](0.0)
    comptime for n in range(DSTATE):
        state[n] = ftz(h_ptr.unsafe_load((bb * dim + d) * DSTATE + n))

    # A's row for this dim, hoisted -- MAX hoists the same row
    # (`selective_scan.mojo:215-220`). A LOAD AND A FLUSH ARE LOOP INVARIANT,
    # so this moves no bits: the oracle re-reads `a[d * D_STATE + n]` and
    # flushes it inside the token loop and gets the same value every time.
    # WHAT IS NOT TAKEN FROM MAX: it also multiplies each A by LOG2E here so
    # the loop can use `exp2`. That is a DIFFERENT FUNCTION with an extra
    # rounding in it (DEVIATION 722); `SAB_S5_EXP2` is that mistake, on
    # purpose, so a gate can be shown to catch it.
    var a_vals = SIMD[DType.float32, DSTATE](0.0)
    comptime for n in range(DSTATE):
        a_vals[n] = ftz(a_ptr.unsafe_load(d * DSTATE + n))
    var d_val = ftz(d_ptr.unsafe_load(d))

    # `for i in range(u.shape[2])` (ref:174). Serial, ascending, one token at
    # a time -- MAX's shape too (`selective_scan.mojo:246-330`, minus its
    # TILE_SIZE staging, DEVIATION 722).
    for li in range(seqlen):
        var t = bb * seqlen + li
        var uv = ftz(u_ptr.unsafe_load(t * dim + d))
        var dl = ftz(delta_ptr.unsafe_load(t * dim + d))

        comptime for n in range(DSTATE):
            # S5 + S6: `deltaA = torch.exp(einsum('bdl,dn->bdln', delta, A))`
            # (ref:162). One product, then row 12's polynomial.
            var da_arg = ftz(pinned_mul(dl, a_vals[n]))
            var da: Float32
            comptime if SAB_S5_EXP2:
                # SABOTAGE: MAX's and the CUDA kernel's exp2 substitution.
                da = ftz(exp2(ftz(pinned_mul(da_arg, Float32(1.4426950408889634)))))
            else:
                da = ftz(identical_exp(da_arg))

            var bv = ftz(b_ptr.unsafe_load(t * DSTATE + n))
            var dbu: Float32
            comptime if SAB_S8_CUDA_PAIRING:
                # SABOTAGE: `B * (delta * u)`, cuh:162 with :222.
                var du = ftz(pinned_mul(dl, uv))
                dbu = ftz(pinned_mul(bv, du))
            else:
                # S7: `delta * B`, the einsum's first pairing (ref:167), HF's
                # `discrete_B = dt * B` (modeling_mamba.py:214).
                var db = ftz(pinned_mul(dl, bv))
                # S8: `(delta * B) * u`, the second pairing (ref:167), HF's
                # `deltaB_u = discrete_B * u` (modeling_mamba.py:215).
                dbu = ftz(pinned_mul(db, uv))

            # S9: `x = deltaA * x + deltaB_u` (ref:175). ONE rounding. The
            # torch reference rounds twice; fusion is PINNED because only
            # fusion has a portable spelling (gemm contract section 4), and
            # the CUDA scan op and MAX both contract here as well.
            comptime if SAB_S9_UNFUSED:
                # SABOTAGE: the reference's literal two roundings.
                state[n] = ftz(ftz(pinned_mul(da, state[n])) + dbu)
            else:
                state[n] = ftz(identical_mul_add(da, state[n], dbu))

        # S10: `y = einsum('bdn,bn->bd', x, C[:, :, i])` (ref:180). Serial
        # ascending n from +0.0, one fma per term. The einsum's fold order is
        # torch-internal; the order pinned is the CUDA kernel's ascending
        # `state_idx` walk (cuh:168-266). MAX folds this seam as a SIMD
        # tree, `(state * C_vals).reduce_add()` (`selective_scan.mojo:323`),
        # which is a third order again and is NOT taken (DEVIATION 722).
        # The `+0.0` seed makes an all-zero row sum to `+0.0` on every
        # vendor (contract section 6).
        var acc = Float32(0.0)
        comptime if SAB_S11_D_FIRST:
            # SABOTAGE: the CUDA kernel seeds its output accumulator with
            # `D * u` (cuh:163) and folds the C-h terms onto it -- D FIRST.
            acc = ftz(pinned_mul(d_val, uv))
        comptime if SAB_S10_DESCENDING:
            # SABOTAGE: the same terms, folded n descending.
            comptime for nn in range(DSTATE):
                var n = DSTATE - 1 - nn
                acc = ftz(
                    identical_mul_add(
                        ftz(c_ptr.unsafe_load(t * DSTATE + n)), state[n], acc
                    )
                )
        else:
            comptime for n in range(DSTATE):
                acc = ftz(
                    identical_mul_add(
                        ftz(c_ptr.unsafe_load(t * DSTATE + n)), state[n], acc
                    )
                )
        y_ptr.unsafe_store(t * dim + d, acc)

        # S11: `out = y + u * D` (ref:189). UNFUSED -- both the reference and
        # the CUDA kernel round `u * D` as its own product -- and D LAST,
        # which is where the two upstreams part company (cuh:163 puts D
        # first). `SAB_S11_D_FIRST` above is their order.
        comptime if SAB_S11_D_FIRST:
            out_ptr.unsafe_store(t * dim + d, acc)
        else:
            var p = ftz(pinned_mul(uv, d_val))
            out_ptr.unsafe_store(t * dim + d, ftz(acc + p))

    # `last_state = x` at the final token (ref:183-184). Written every call,
    # because it is the recurrent state a decode step carries (contract
    # section 5) and the recorded stage `scan.h`. A copy, not a seam.
    comptime for n in range(DSTATE):
        h_ptr.unsafe_store((bb * dim + d) * DSTATE + n, state[n])


# ===========================================================================
# THE LAUNCHER
# ===========================================================================


def selective_scan_fn(
    ctx: DeviceContext,
    mut out: DeviceBuffer[DType.float32],
    mut y: DeviceBuffer[DType.float32],
    mut h_state: DeviceBuffer[DType.float32],
    mut u: DeviceBuffer[DType.float32],
    mut delta: DeviceBuffer[DType.float32],
    mut A: DeviceBuffer[DType.float32],
    mut B: DeviceBuffer[DType.float32],
    mut C: DeviceBuffer[DType.float32],
    mut D: DeviceBuffer[DType.float32],
    batch: Int,
    seqlen: Int,
    dim: Int,
    z: Bool,
    delta_bias: Bool,
    delta_softplus: Bool,
    return_last_state: Bool,
    mut trace: IdentityTrace,
    prefix: String,
    block_size: Int = 64,
) raises:
    """`selective_scan_fn(u, delta, A, B, C, D, z, delta_bias,
    delta_softplus, return_last_state)` (`selective_scan_interface.py:118-125`)
    under profile `mojolearn.identical.mamba1.fp32.v1`, seams S5-S11.

    ARGUMENT NAMES AND ORDER ARE UPSTREAM'S. `z` and `delta_bias` are
    PRESENCE flags rather than buffers -- `False` is upstream's `None` -- and
    both are REFUSED when True (DEVIATION 723): the profile's block owns S12
    (`gate.out`) and S14 (`softplus.out`) as recorded stages of its own, and
    a second spelling of a seam this file does not own is a second place for
    it to drift. `delta` arrives POST-softplus for the same reason, exactly
    as `selective_scan_oracle` receives it. `return_last_state` is pinned
    True: `h_state` is in-and-out on every call, because it is both the
    recorded stage `scan.h` and the state a decode step carries.

    STAGES RECORDED HERE, and the driver must not record them again (the
    trace's tag-uniqueness invariant raises on a duplicate):

        <prefix>.scan.y      [M, dim]        S5-S10, before D
        <prefix>.skip.out    [M, dim]        S11
        <prefix>.scan.h      [B, dim, 16]    the state after the last token

    `block_size` is an EXECUTION plan quantity in the sense of the gemm
    charter's split: no float crosses a thread boundary in the kernel, so
    varying it must move no bit, and a gate that varies it is testing that
    claim rather than choosing a number. It is never read by any expression
    that reaches a seam.
    """
    if z:
        raise Error(
            "selective_scan_fn: z REFUSED. Seam S12 (out * silu(z), stage"
            " gate.out) belongs to the block, mamba/ported/transformers/"
            "models/mamba/modeling_mamba.mojo, not to the scan"
            " (DEVIATION 723)"
        )
    if delta_bias:
        raise Error(
            "selective_scan_fn: delta_bias REFUSED. Seam S14 (delta ="
            " softplus(dt + bias), stage softplus.out) belongs to the block;"
            " pass delta already softplused, as selective_scan_oracle takes"
            " it (DEVIATION 723)"
        )
    if delta_softplus:
        raise Error(
            "selective_scan_fn: delta_softplus REFUSED. Seam S14 belongs to"
            " the block and softplus.out is its recorded stage"
            " (DEVIATION 723)"
        )
    if not return_last_state:
        raise Error(
            "selective_scan_fn: return_last_state False REFUSED. h_state is"
            " in-and-out on every call: it is the recorded stage scan.h and"
            " the state a decode step carries (contract section 5,"
            " DEVIATION 723)"
        )
    if batch < 0 or seqlen < 0 or dim < 0:
        raise Error(
            "selective_scan_fn: negative shape (batch="
            + String(batch)
            + " seqlen="
            + String(seqlen)
            + " dim="
            + String(dim)
            + ")"
        )
    if block_size < 1:
        raise Error(
            "selective_scan_fn: block_size must be >= 1, got "
            + String(block_size)
        )

    var m = batch * seqlen
    _require(len(u), m * dim, "u", "[M, dim]")
    _require(len(delta), m * dim, "delta", "[M, dim]")
    _require(len(A), dim * MAX_DSTATE, "A", "[dim, 16]")
    _require(len(B), m * MAX_DSTATE, "B", "[M, 16]")
    _require(len(C), m * MAX_DSTATE, "C", "[M, 16]")
    _require(len(D), dim, "D", "[dim]")
    _require(len(y), m * dim, "y", "[M, dim]")
    _require(len(out), m * dim, "out", "[M, dim]")
    _require(
        len(h_state), batch * dim * MAX_DSTATE, "h_state", "[B, dim, 16]"
    )

    var total = batch * dim
    if total > 0:
        comptime kern = selective_scan_fwd_kernel[MAX_DSTATE]
        var grid = (total + block_size - 1) // block_size
        ctx.enqueue_function[kern](
            out.unsafe_ptr(),
            y.unsafe_ptr(),
            h_state.unsafe_ptr(),
            u.unsafe_ptr(),
            delta.unsafe_ptr(),
            A.unsafe_ptr(),
            B.unsafe_ptr(),
            C.unsafe_ptr(),
            D.unsafe_ptr(),
            Int32(batch),
            Int32(seqlen),
            Int32(dim),
            grid_dim=(grid, 1, 1),
            block_dim=(block_size, 1, 1),
        )
        ctx.synchronize()

    trace.record_device(ctx, prefix + ".scan.y", y, m * dim)
    trace.record_device(ctx, prefix + ".skip.out", out, m * dim)
    trace.record_device(
        ctx, prefix + ".scan.h", h_state, batch * dim * MAX_DSTATE
    )


def _require(got: Int, want: Int, name: String, shape: String) raises:
    """A buffer that is the wrong size is a CALLER bug, and it is caught by
    name here rather than by an out-of-bounds device read that returns
    plausible bits."""
    if got < want:
        raise Error(
            "selective_scan_fn: buffer '"
            + name
            + "' holds "
            + String(got)
            + " floats, needs "
            + String(want)
            + " for "
            + shape
        )
