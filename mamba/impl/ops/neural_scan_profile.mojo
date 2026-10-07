# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""NN34 affine-prefix scan component, fp32 numerical profile v2.

Mamba-1 imports this profile through neural_mamba_scan; other Mamba families
retain their own SSD profiles.
It supplies an actual selectable component A/B: candidate uses absolute
32-token chunks and canonical adjacent affine-prefix trees; baseline uses
one ascending recurrent FMA chain. All new switches are OFF by default.

The input is PREPARED affine factors [tokens, chains], not raw Mamba model
inputs: mathematically h[t] = a[t] * h[t-1] + b[t]. Discretization, emission, tree VJP and checkpoint integration are in
`neural_mamba_scan`, modeling_mamba and the Mamba bindings. No quality,
compilation, identity or timing evidence is claimed.

Within a chunk, compose(left, right) means apply left, then right:
  A = ftz(mul(ftz(right.A), ftz(left.A)))
  B = ftz(fma(ftz(right.A), ftz(left.B), ftz(right.B)))
Then evaluate the prefix with ONE rounded FMA on the completed-chunk
boundary. Adjacent equal-sized subtrees merge first, with an unmatched
right subtree carried unchanged. A single leaf has no fabricated identity
composition. All columns call these exact helpers; previous-version bits
need not match.

A checkpoint contains profile id, absolute next-token position, chain
count, boundary[chains], last[chains], and A/B slots [chains, 6]. Its occupied
slot bits are absolute_position % 32. Consumed/inactive slots are canonical
+0. Completed chunks reset the slots and install their final output as the
next boundary. A checkpoint at a nonzero position MUST restore all slots;
last hidden alone is insufficient. Profile ids must not be mixed.

Prefill is three ordered phases: independent per-prefix summaries; ordered
completed-chunk boundary propagation per chain; independent prefix outputs.
The final prefix also produces next slots into DISJOINT state buffers.
Decode pushes one leaf into the same binary slots and uses the same drain
and evaluate graph. Request length and device launch shape never choose
chunk boundaries or the arithmetic tree. Host and device entry points use
identical per-cell functions. The explicit component APIs also remain available for isolated A/B work.
"""
from std.memory import stack_allocation
from std.sys.compile import is_defined, get_defined_int
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext
from checks.numerics import (
    GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_mul, identical_mul_add,
)

comptime NN34FP = MutPointer[Float32, MutAnyOrigin]
comptime NN34_CHUNK = 32
comptime NN34_LEVELS = 6
comptime NN34_TPB = 128
# `-D MOJOLEARN_IDN_M1_SCAN=<arm>` (lane/grid-prune 2026-10-07) is ONE switch
# for the three alternative Mamba-1 scans, formerly three defines whose every
# cross was refused or inert: 1 affine_prefix (NN34, this profile), 2
# state_window (NI38) and 3 persistent (selective_scan_interface.mojo). The
# old define MOJOLEARN_NN34_AFFINE_PREFIX is refused in
# core/six_lane_experiment_guards.mojo.
comptime M1_SCAN_AFFINE_PREFIX = 1
comptime M1_SCAN_STATE_WINDOW = 2
comptime M1_SCAN_PERSISTENT = 3
comptime M1_SCAN = get_defined_int["MOJOLEARN_IDN_M1_SCAN", 0]()
comptime NN34_AFFINE_PREFIX = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and M1_SCAN == M1_SCAN_AFFINE_PREFIX
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
comptime NN34_COMPONENT_PROFILE = 2 if NN34_AFFINE_PREFIX else 1


@always_inline
def nn34_compose(la: Float32, lb: Float32, ra: Float32, rb: Float32) -> Tuple[Float32, Float32]:
    return (
        ftz(identical_mul(ftz(ra), ftz(la))),
        ftz(identical_mul_add(ftz(ra), ftz(lb), ftz(rb))),
    )


@always_inline
def nn34_evaluate(a: Float32, b: Float32, boundary: Float32) -> Float32:
    return ftz(identical_mul_add(ftz(a), ftz(boundary), ftz(b)))


@always_inline
def nn34_clear_slots(sa: NN34FP, sb: NN34FP):
    for level in range(NN34_LEVELS):
        sa.unsafe_store(level, Float32(0.0))
        sb.unsafe_store(level, Float32(0.0))


@always_inline
def nn34_push(count: Int, a: Float32, b: Float32, sa: NN34FP, sb: NN34FP):
    """count is 0..31; slots contain its binary decomposition, oldest left.
    Consumed slots are cleared so a checkpoint has no scheduler-dependent
    uninitialized words. A Float32 leaf is explicitly flushed once."""
    var ca = ftz(a)
    var cb = ftz(b)
    var level = 0
    while (count & (1 << level)) != 0:
        var pair = nn34_compose(sa.unsafe_load(level), sb.unsafe_load(level), ca, cb)
        ca = pair[0]
        cb = pair[1]
        sa.unsafe_store(level, Float32(0.0))
        sb.unsafe_store(level, Float32(0.0))
        level += 1
    sa.unsafe_store(level, ca)
    sb.unsafe_store(level, cb)


@always_inline
def nn34_prefix(count: Int, sa: NN34FP, sb: NN34FP) -> Tuple[Float32, Float32]:
    """count is 1..32. Drain newest small subtree to oldest large subtree;
    each merge puts the older subtree on the LEFT. This is the fixed
    adjacent-pair tree with an unpaired right subtree carried unchanged."""
    var ca = Float32(0.0)
    var cb = Float32(0.0)
    var have = False
    for level in range(NN34_LEVELS):
        if (count & (1 << level)) != 0:
            if have:
                var pair = nn34_compose(sa.unsafe_load(level), sb.unsafe_load(level), ca, cb)
                ca = pair[0]
                cb = pair[1]
            else:
                ca = sa.unsafe_load(level)
                cb = sb.unsafe_load(level)
                have = True
    return (ca, cb)


@always_inline
def nn34_init_chain(chain: Int, initial: NN34FP, boundary: NN34FP, last: NN34FP, sa: NN34FP, sb: NN34FP):
    """Initialize a new sequence at absolute position zero. A restored
    checkpoint is a raw copy of its entire state, never this initializer."""
    var h = ftz(initial.unsafe_load(chain))
    boundary.unsafe_store(chain, h)
    last.unsafe_store(chain, h)
    nn34_clear_slots(sa + chain * NN34_LEVELS, sb + chain * NN34_LEVELS)


@always_inline
def nn34_decode_chain(chain: Int, absolute_position: Int, a: NN34FP, b: NN34FP,
    boundary: NN34FP, last: NN34FP, sa: NN34FP, sb: NN34FP, output: NN34FP):
    """One token, one owner per chain; state updates in place. The caller
    advances the scalar absolute position exactly once after this phase."""
    comptime if NN34_AFFINE_PREFIX:
        var count = absolute_position % NN34_CHUNK
        var aslots = sa + chain * NN34_LEVELS
        var bslots = sb + chain * NN34_LEVELS
        nn34_push(count, a.unsafe_load(chain), b.unsafe_load(chain), aslots, bslots)
        count += 1
        var pair = nn34_prefix(count, aslots, bslots)
        var h = nn34_evaluate(pair[0], pair[1], boundary.unsafe_load(chain))
        output.unsafe_store(chain, h)
        last.unsafe_store(chain, h)
        if count == NN34_CHUNK:
            boundary.unsafe_store(chain, h)
            nn34_clear_slots(aslots, bslots)
    else:
        var h = nn34_evaluate(a.unsafe_load(chain), b.unsafe_load(chain), last.unsafe_load(chain))
        output.unsafe_store(chain, h)
        last.unsafe_store(chain, h)
        boundary.unsafe_store(chain, h)


@always_inline
def nn34_prefill_prefix_cell(cell: Int, a: NN34FP, b: NN34FP,
    in_sa: NN34FP, in_sb: NN34FP, prefix_a: NN34FP, prefix_b: NN34FP,
    out_sa: NN34FP, out_sb: NN34FP, chains: Int, tokens: Int, absolute_start: Int):
    """Parallel phase 1. Each prefix restates the same absolute-chunk
    binary slots as decode. No input from a future token is read. Repeated
    subtree work is deliberate component scope; shared tree staging is a
    later schedule candidate. Output state must not alias the input state."""
    var t = cell // chains
    var chain = cell % chains
    var absolute_t = absolute_start + t
    var chunk_start = absolute_t - absolute_t % NN34_CHUNK
    var local_start = max(0, chunk_start - absolute_start)
    var astack = stack_allocation[NN34_LEVELS, Float32]()
    var bstack = stack_allocation[NN34_LEVELS, Float32]()
    var sa = astack.unsafe_origin_cast[MutAnyOrigin]()
    var sb = bstack.unsafe_origin_cast[MutAnyOrigin]()
    var count = 0
    nn34_clear_slots(sa, sb)
    if chunk_start < absolute_start:
        count = absolute_start % NN34_CHUNK
        for level in range(NN34_LEVELS):
            sa.unsafe_store(level, in_sa.unsafe_load(chain * NN34_LEVELS + level))
            sb.unsafe_store(level, in_sb.unsafe_load(chain * NN34_LEVELS + level))
    for token in range(local_start, t + 1):
        var index = token * chains + chain
        nn34_push(count, a.unsafe_load(index), b.unsafe_load(index), sa, sb)
        count += 1
    var pair = nn34_prefix(count, sa, sb)
    prefix_a.unsafe_store(cell, pair[0])
    prefix_b.unsafe_store(cell, pair[1])
    if t == tokens - 1:
        if count == NN34_CHUNK:
            nn34_clear_slots(sa, sb)
        for level in range(NN34_LEVELS):
            out_sa.unsafe_store(chain * NN34_LEVELS + level, sa.unsafe_load(level))
            out_sb.unsafe_store(chain * NN34_LEVELS + level, sb.unsafe_load(level))


@always_inline
def nn34_prefill_boundary_chain(chain: Int, prefix_a: NN34FP, prefix_b: NN34FP,
    in_boundary: NN34FP, chunk_boundaries: NN34FP, out_boundary: NN34FP,
    chains: Int, tokens: Int, absolute_start: Int):
    """Parallel over independent state chains, ordered over completed
    absolute chunks. The last partial chunk never advances the boundary."""
    var first = absolute_start // NN34_CHUNK
    var chunks = (absolute_start % NN34_CHUNK + tokens + NN34_CHUNK - 1) // NN34_CHUNK
    var carry = ftz(in_boundary.unsafe_load(chain))
    for chunk in range(chunks):
        chunk_boundaries.unsafe_store(chunk * chains + chain, carry)
        var local_end = (first + chunk + 1) * NN34_CHUNK - absolute_start
        if local_end <= tokens:
            var end_cell = (local_end - 1) * chains + chain
            carry = nn34_evaluate(prefix_a.unsafe_load(end_cell), prefix_b.unsafe_load(end_cell), carry)
    out_boundary.unsafe_store(chain, carry)


@always_inline
def nn34_prefill_output_cell(cell: Int, prefix_a: NN34FP, prefix_b: NN34FP,
    chunk_boundaries: NN34FP, output: NN34FP, out_last: NN34FP,
    chains: Int, tokens: Int, absolute_start: Int):
    var t = cell // chains
    var chain = cell % chains
    var chunk = (absolute_start % NN34_CHUNK + t) // NN34_CHUNK
    var h = nn34_evaluate(prefix_a.unsafe_load(cell), prefix_b.unsafe_load(cell), chunk_boundaries.unsafe_load(chunk * chains + chain))
    output.unsafe_store(cell, h)
    if t == tokens - 1:
        out_last.unsafe_store(chain, h)


@always_inline
def nn34_sequential_chain(chain: Int, a: NN34FP, b: NN34FP, in_last: NN34FP,
    output: NN34FP, out_boundary: NN34FP, out_last: NN34FP,
    out_sa: NN34FP, out_sb: NN34FP, chains: Int, tokens: Int):
    """B arm: the prepared-factor recurrent FMA chain. This component
    baseline is not claimed to stand in for a full public-model workload."""
    var h = ftz(in_last.unsafe_load(chain))
    for t in range(tokens):
        var cell = t * chains + chain
        h = nn34_evaluate(a.unsafe_load(cell), b.unsafe_load(cell), h)
        output.unsafe_store(cell, h)
    out_boundary.unsafe_store(chain, h)
    out_last.unsafe_store(chain, h)
    nn34_clear_slots(out_sa + chain * NN34_LEVELS, out_sb + chain * NN34_LEVELS)


def nn34_component_metadata(profile: Int, chains: Int, tokens: Int, absolute_start: Int) raises:
    """Scalar checks only. Tensor buffers must satisfy the documented
    sizes/disjointness; this component does not own or inspect allocations."""
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("NN34 component requires IDENTICAL mode")
    if profile != NN34_COMPONENT_PROFILE:
        raise Error("NN34 checkpoint profile does not match this component build")
    if chains <= 0 or tokens <= 0 or absolute_start < 0:
        raise Error("NN34 requires positive chains/tokens and nonnegative position")
    # Int32 launch extents and Int64 position arithmetic, not a board cap.
    if chains > 2147483647 // tokens:
        raise Error("NN34 component launch extent exceeds Int32")
    if absolute_start > (1 << 62) - tokens - NN34_CHUNK:
        raise Error("NN34 absolute position exceeds component index range")


def nn34_component_host_prefill(profile: Int, chains: Int, tokens: Int, absolute_start: Int,
    a: NN34FP, b: NN34FP, in_boundary: NN34FP, in_last: NN34FP, in_sa: NN34FP, in_sb: NN34FP,
    output: NN34FP, out_boundary: NN34FP, out_last: NN34FP, out_sa: NN34FP, out_sb: NN34FP,
    prefix_a: NN34FP, prefix_b: NN34FP, chunk_boundaries: NN34FP) raises:
    """Host component runner. Buffers: a/b/output/prefix_a/prefix_b each
    tokens*chains; in/out boundary/last each chains; in/out slots each
    chains*6; chunk_boundaries ceil((start%32+tokens)/32)*chains. Input and
    output state are disjoint. Advance metadata only after successful return.
    Empty requests are deliberately not an entry point; retain state as-is."""
    nn34_component_metadata(profile, chains, tokens, absolute_start)
    comptime if NN34_AFFINE_PREFIX:
        for cell in range(tokens * chains):
            nn34_prefill_prefix_cell(cell, a, b, in_sa, in_sb, prefix_a, prefix_b, out_sa, out_sb, chains, tokens, absolute_start)
        for chain in range(chains):
            nn34_prefill_boundary_chain(chain, prefix_a, prefix_b, in_boundary, chunk_boundaries, out_boundary, chains, tokens, absolute_start)
        for cell in range(tokens * chains):
            nn34_prefill_output_cell(cell, prefix_a, prefix_b, chunk_boundaries, output, out_last, chains, tokens, absolute_start)
    else:
        for chain in range(chains):
            nn34_sequential_chain(chain, a, b, in_last, output, out_boundary, out_last, out_sa, out_sb, chains, tokens)


def nn34_component_host_decode(profile: Int, chains: Int, absolute_position: Int,
    a: NN34FP, b: NN34FP, boundary: NN34FP, last: NN34FP, sa: NN34FP, sb: NN34FP, output: NN34FP) raises:
    nn34_component_metadata(profile, chains, 1, absolute_position)
    for chain in range(chains):
        nn34_decode_chain(chain, absolute_position, a, b, boundary, last, sa, sb, output)


def nn34_prefix_kernel(a: NN34FP, b: NN34FP, in_sa: NN34FP, in_sb: NN34FP,
    prefix_a: NN34FP, prefix_b: NN34FP, out_sa: NN34FP, out_sb: NN34FP,
    chains: Int32, tokens: Int32, absolute_start: Int64):
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell < Int(chains) * Int(tokens):
        nn34_prefill_prefix_cell(cell, a, b, in_sa, in_sb, prefix_a, prefix_b, out_sa, out_sb, Int(chains), Int(tokens), Int(absolute_start))


def nn34_boundary_kernel(prefix_a: NN34FP, prefix_b: NN34FP, in_boundary: NN34FP,
    chunk_boundaries: NN34FP, out_boundary: NN34FP, chains: Int32, tokens: Int32, absolute_start: Int64):
    var chain = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if chain < Int(chains):
        nn34_prefill_boundary_chain(chain, prefix_a, prefix_b, in_boundary, chunk_boundaries, out_boundary, Int(chains), Int(tokens), Int(absolute_start))


def nn34_output_kernel(prefix_a: NN34FP, prefix_b: NN34FP, chunk_boundaries: NN34FP,
    output: NN34FP, out_last: NN34FP, chains: Int32, tokens: Int32, absolute_start: Int64):
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell < Int(chains) * Int(tokens):
        nn34_prefill_output_cell(cell, prefix_a, prefix_b, chunk_boundaries, output, out_last, Int(chains), Int(tokens), Int(absolute_start))


def nn34_sequential_kernel(a: NN34FP, b: NN34FP, in_last: NN34FP,
    output: NN34FP, out_boundary: NN34FP, out_last: NN34FP, out_sa: NN34FP, out_sb: NN34FP,
    chains: Int32, tokens: Int32):
    var chain = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if chain < Int(chains):
        nn34_sequential_chain(chain, a, b, in_last, output, out_boundary, out_last, out_sa, out_sb, Int(chains), Int(tokens))


def nn34_decode_kernel(a: NN34FP, b: NN34FP, boundary: NN34FP, last: NN34FP,
    sa: NN34FP, sb: NN34FP, output: NN34FP, chains: Int32, absolute_position: Int64):
    var chain = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if chain < Int(chains):
        nn34_decode_chain(chain, Int(absolute_position), a, b, boundary, last, sa, sb, output)


def nn34_init_kernel(initial: NN34FP, boundary: NN34FP, last: NN34FP,
    sa: NN34FP, sb: NN34FP, chains: Int32):
    var chain = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if chain < Int(chains):
        nn34_init_chain(chain, initial, boundary, last, sa, sb)


def nn34_component_device_prefill(ctx: DeviceContext, profile: Int,
    chains: Int, tokens: Int, absolute_start: Int,
    a: NN34FP, b: NN34FP, in_boundary: NN34FP, in_last: NN34FP, in_sa: NN34FP, in_sb: NN34FP,
    output: NN34FP, out_boundary: NN34FP, out_last: NN34FP, out_sa: NN34FP, out_sb: NN34FP,
    prefix_a: NN34FP, prefix_b: NN34FP, chunk_boundaries: NN34FP) raises:
    """Asynchronous isolated GPU component; same buffer contract as host.
    All three phases share ctx's ordered queue. Caller owns every allocation
    through the final synchronization, consumes output, and then commits
    position=start+tokens and the disjoint output checkpoint. Never recycle
    prefix or input state while any queued phase may still read it."""
    nn34_component_metadata(profile, chains, tokens, absolute_start)
    var cells = chains * tokens
    var grid_cells = (cells + NN34_TPB - 1) // NN34_TPB
    var grid_chains = (chains + NN34_TPB - 1) // NN34_TPB
    comptime if NN34_AFFINE_PREFIX:
        ctx.enqueue_function[nn34_prefix_kernel](a, b, in_sa, in_sb, prefix_a, prefix_b, out_sa, out_sb, Int32(chains), Int32(tokens), Int64(absolute_start), grid_dim=(grid_cells, 1, 1), block_dim=(NN34_TPB, 1, 1))
        ctx.enqueue_function[nn34_boundary_kernel](prefix_a, prefix_b, in_boundary, chunk_boundaries, out_boundary, Int32(chains), Int32(tokens), Int64(absolute_start), grid_dim=(grid_chains, 1, 1), block_dim=(NN34_TPB, 1, 1))
        ctx.enqueue_function[nn34_output_kernel](prefix_a, prefix_b, chunk_boundaries, output, out_last, Int32(chains), Int32(tokens), Int64(absolute_start), grid_dim=(grid_cells, 1, 1), block_dim=(NN34_TPB, 1, 1))
    else:
        ctx.enqueue_function[nn34_sequential_kernel](a, b, in_last, output, out_boundary, out_last, out_sa, out_sb, Int32(chains), Int32(tokens), grid_dim=(grid_chains, 1, 1), block_dim=(NN34_TPB, 1, 1))


def nn34_component_device_decode(ctx: DeviceContext, profile: Int, chains: Int, absolute_position: Int,
    a: NN34FP, b: NN34FP, boundary: NN34FP, last: NN34FP, sa: NN34FP, sb: NN34FP, output: NN34FP) raises:
    """Asynchronous single-token decode. Caller advances position after
    synchronization. A failed in-place decode must discard its state;
    transactional failure handling belongs to the future model wrapper."""
    nn34_component_metadata(profile, chains, 1, absolute_position)
    ctx.enqueue_function[nn34_decode_kernel](a, b, boundary, last, sa, sb, output, Int32(chains), Int64(absolute_position), grid_dim=((chains + NN34_TPB - 1) // NN34_TPB, 1, 1), block_dim=(NN34_TPB, 1, 1))
