# SPDX-License-Identifier: Apache-2.0
"""NN34 Mamba-1 numerical profile: discretization, emission, checkpoints/VJP.

The affine tree lives at absolute 32-token boundaries. Mamba-2/3 keep their
own existing SSD profiles. Zero-state prefill backward differentiates the
actual adjacent-pair prefix tree; it does not reuse the old reverse-chain
state derivative. Same element bodies are restated in the host source.
All code is an uncompiled, unverified source candidate, OFF by default.
"""
from std.memory import stack_allocation
from std.gpu import block_idx, block_dim, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import ftz, identical_mul, identical_mul_add, identical_exp
from mamba.impl.ops.neural_scan_profile import (
    NN34FP, NN34_CHUNK, NN34_LEVELS, NN34_COMPONENT_PROFILE,
    nn34_compose, nn34_evaluate, nn34_clear_slots, nn34_push, nn34_prefix,
    nn34_component_device_prefill,
)

comptime NS = 16
comptime NT = 128


@always_inline
def nn34_factors(t: Int, d: Int, n: Int, dim: Int,
    u: NN34FP, delta: NN34FP, a: NN34FP, b: NN34FP) -> Tuple[Float32, Float32]:
    var dl = ftz(delta.unsafe_load(t * dim + d))
    var av = ftz(a.unsafe_load(d * NS + n))
    var bv = ftz(b.unsafe_load(t * NS + n))
    var uv = ftz(u.unsafe_load(t * dim + d))
    return (ftz(identical_exp(ftz(identical_mul(dl, av)))),
            ftz(identical_mul(ftz(identical_mul(dl, bv)), uv)))


def nn34_prepare_kernel(u: NN34FP, delta: NN34FP, a: NN34FP, b: NN34FP,
    fa: NN34FP, fb: NN34FP, batch: Int32, length: Int32, width: Int32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var chains = Int(batch) * Int(width) * NS
    if i >= chains * Int(length):
        return
    var token = i // chains
    var chain = i % chains
    var n = chain % NS
    var d = (chain // NS) % Int(width)
    var bb = chain // (Int(width) * NS)
    var pair = nn34_factors(bb * Int(length) + token, d, n, Int(width), u, delta, a, b)
    fa.unsafe_store(i, pair[0])
    fb.unsafe_store(i, pair[1])


def nn34_emit_kernel(hidden: NN34FP, c: NN34FP, u: NN34FP, skip: NN34FP,
    y: NN34FP, out: NN34FP, batch: Int32, length: Int32, width: Int32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var di = Int(width)
    var L = Int(length)
    if i >= Int(batch) * L * di:
        return
    var d = i % di
    var t = i // di
    var bb = t // L
    var li = t % L
    var base = li * Int(batch) * di * NS + (bb * di + d) * NS
    var acc = Float32(0)
    for n in range(NS):
        acc = ftz(identical_mul_add(ftz(c.unsafe_load(t * NS + n)), ftz(hidden.unsafe_load(base + n)), acc))
    y.unsafe_store(i, acc)
    # The existing S11 is a separately rounded product followed by add.
    out.unsafe_store(i, ftz(acc + ftz(identical_mul(ftz(u.unsafe_load(i)), ftz(skip.unsafe_load(d))))))


def nn34_mamba_forward(ctx: DeviceContext, u: NN34FP, delta: NN34FP, a: NN34FP, b: NN34FP,
    c: NN34FP, skip: NN34FP, y: NN34FP, out: NN34FP,
    boundary: NN34FP, last: NN34FP, slots_a: NN34FP, slots_b: NN34FP,
    batch: Int, length: Int, dim: Int, absolute_start: Int) raises:
    """Own all temporary storage through completion; commit state only after
    independent output checkpoint and emission have completed successfully.
    This is used at every request length including single-token decode."""
    var chains = batch * dim * NS
    var cells = chains * length
    var chunks = (absolute_start % NN34_CHUNK + length + NN34_CHUNK - 1) // NN34_CHUNK
    var fa = ctx.enqueue_create_buffer[DType.float32](cells)
    var fb = ctx.enqueue_create_buffer[DType.float32](cells)
    var pa = ctx.enqueue_create_buffer[DType.float32](cells)
    var pb = ctx.enqueue_create_buffer[DType.float32](cells)
    var hs = ctx.enqueue_create_buffer[DType.float32](cells)
    var bounds = ctx.enqueue_create_buffer[DType.float32](max(1, chains * chunks))
    var next_boundary = ctx.enqueue_create_buffer[DType.float32](chains)
    var next_last = ctx.enqueue_create_buffer[DType.float32](chains)
    var next_sa = ctx.enqueue_create_buffer[DType.float32](chains * NN34_LEVELS)
    var next_sb = ctx.enqueue_create_buffer[DType.float32](chains * NN34_LEVELS)
    ctx.enqueue_function[nn34_prepare_kernel](u, delta, a, b, fa.unsafe_ptr(), fb.unsafe_ptr(), Int32(batch), Int32(length), Int32(dim), grid_dim=((cells + NT - 1) // NT, 1, 1), block_dim=(NT, 1, 1))
    nn34_component_device_prefill(ctx, NN34_COMPONENT_PROFILE, chains, length, absolute_start,
        fa.unsafe_ptr(), fb.unsafe_ptr(), boundary, last, slots_a, slots_b,
        hs.unsafe_ptr(), next_boundary.unsafe_ptr(), next_last.unsafe_ptr(), next_sa.unsafe_ptr(), next_sb.unsafe_ptr(),
        pa.unsafe_ptr(), pb.unsafe_ptr(), bounds.unsafe_ptr())
    ctx.enqueue_function[nn34_emit_kernel](hs.unsafe_ptr(), c, u, skip, y, out, Int32(batch), Int32(length), Int32(dim), grid_dim=((batch * length * dim + NT - 1) // NT, 1, 1), block_dim=(NT, 1, 1))
    ctx.synchronize()
    # Queue-owned copies avoid depending on any temporary's last-use lifetime.
    ctx.enqueue_function[nn34_copy_kernel](next_boundary.unsafe_ptr(), boundary, Int32(chains), grid_dim=((chains + NT - 1) // NT, 1, 1), block_dim=(NT, 1, 1))
    ctx.enqueue_function[nn34_copy_kernel](next_last.unsafe_ptr(), last, Int32(chains), grid_dim=((chains + NT - 1) // NT, 1, 1), block_dim=(NT, 1, 1))
    ctx.enqueue_function[nn34_copy_kernel](next_sa.unsafe_ptr(), slots_a, Int32(chains * NN34_LEVELS), grid_dim=((chains * NN34_LEVELS + NT - 1) // NT, 1, 1), block_dim=(NT, 1, 1))
    ctx.enqueue_function[nn34_copy_kernel](next_sb.unsafe_ptr(), slots_b, Int32(chains * NN34_LEVELS), grid_dim=((chains * NN34_LEVELS + NT - 1) // NT, 1, 1), block_dim=(NT, 1, 1))
    ctx.synchronize()
    _ = fa^; _ = fb^; _ = pa^; _ = pb^; _ = hs^; _ = bounds^
    _ = next_boundary^; _ = next_last^; _ = next_sa^; _ = next_sb^


def nn34_copy_kernel(src: NN34FP, dst: NN34FP, count: Int32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(count):
        dst.unsafe_store(i, src.unsafe_load(i))


@always_inline
def nn34_checkpoint_chain(chain: Int, hck: NN34FP, initial: NN34FP,
    u: NN34FP, delta: NN34FP, a: NN34FP, b: NN34FP, L: Int, dim: Int):
    var bb = chain // (dim * NS)
    var d = (chain // NS) % dim
    var n = chain % NS
    var boundary = ftz(initial.unsafe_load(chain))
    var sa0 = stack_allocation[NN34_LEVELS, Float32]()
    var sb0 = stack_allocation[NN34_LEVELS, Float32]()
    var sa = sa0.unsafe_origin_cast[MutAnyOrigin]()
    var sb = sb0.unsafe_origin_cast[MutAnyOrigin]()
    nn34_clear_slots(sa, sb)
    hck.unsafe_store((bb * (L + 1) * dim + d) * NS + n, boundary)
    for li in range(L):
        var pair = nn34_factors(bb * L + li, d, n, dim, u, delta, a, b)
        var count = li % NN34_CHUNK
        nn34_push(count, pair[0], pair[1], sa, sb)
        var prefix = nn34_prefix(count + 1, sa, sb)
        var h = nn34_evaluate(prefix[0], prefix[1], boundary)
        hck.unsafe_store(((bb * (L + 1) + li + 1) * dim + d) * NS + n, h)
        if count + 1 == NN34_CHUNK:
            boundary = h
            nn34_clear_slots(sa, sb)


@always_inline
def nn34_tree_vjp_chain(chain: Int, factor_grad: NN34FP, dy: NN34FP, c: NN34FP,
    u: NN34FP, delta: NN34FP, a: NN34FP, b: NN34FP, hck: NN34FP,
    batch: Int, L: Int, dim: Int):
    """Actual tree VJP. Prefix losses are visited newest to oldest; each
    prefix's nodes reverse their forward creation order. A node's right-A
    adjoint receives product-A then product-B contributions separately.
    Chunk carry is attached only to its final prefix. No atomic reductions.
    Layout is [dB (M*dim*16), dA (M*dim*16)] for prepared affine factors.
    Public backward remains zero-state prefill, as its established API."""
    var bb = chain // (dim * NS)
    var d = (chain // NS) % dim
    var n = chain % NS
    var total = batch * L * dim * NS
    var ma0 = stack_allocation[63, Float32]()
    var mb0 = stack_allocation[63, Float32]()
    var ga0 = stack_allocation[63, Float32]()
    var gb0 = stack_allocation[63, Float32]()
    var left0 = stack_allocation[63, Int]()
    var right0 = stack_allocation[63, Int]()
    var ids0 = stack_allocation[32, Int]()
    var sum_a0 = stack_allocation[32, Float32]()
    var sum_b0 = stack_allocation[32, Float32]()
    var ma = ma0.unsafe_origin_cast[MutAnyOrigin]()
    var mb = mb0.unsafe_origin_cast[MutAnyOrigin]()
    var ga = ga0.unsafe_origin_cast[MutAnyOrigin]()
    var gb = gb0.unsafe_origin_cast[MutAnyOrigin]()
    var left = left0.unsafe_origin_cast[MutAnyOrigin]()
    var right = right0.unsafe_origin_cast[MutAnyOrigin]()
    var ids = ids0.unsafe_origin_cast[MutAnyOrigin]()
    var sum_a = sum_a0.unsafe_origin_cast[MutAnyOrigin]()
    var sum_b = sum_b0.unsafe_origin_cast[MutAnyOrigin]()
    var carry = Float32(0)
    var chunks = (L + NN34_CHUNK - 1) // NN34_CHUNK
    for reverse_chunk in range(chunks):
        var chunk = chunks - 1 - reverse_chunk
        var begin = chunk * NN34_CHUNK
        var count = min(NN34_CHUNK, L - begin)
        var boundary = ftz(hck.unsafe_load(((bb * (L + 1) + begin) * dim + d) * NS + n))
        for j in range(count):
            sum_a.unsafe_store(j, Float32(0))
            sum_b.unsafe_store(j, Float32(0))
        var previous_carry = Float32(0)
        for rp in range(count):
            var prefix = count - rp
            for j in range(prefix):
                var pair = nn34_factors(bb * L + begin + j, d, n, dim, u, delta, a, b)
                ma.unsafe_store(j, pair[0]); mb.unsafe_store(j, pair[1])
                ids.unsafe_store(j, j)
            for j in range(63):
                ga.unsafe_store(j, Float32(0)); gb.unsafe_store(j, Float32(0))
            var nodes = prefix
            var width = prefix
            while width > 1:
                var next_width = (width + 1) // 2
                for j in range(next_width):
                    var li = ids.unsafe_load(2 * j)
                    if 2 * j + 1 == width:
                        ids.unsafe_store(j, li)
                    else:
                        var ri = ids.unsafe_load(2 * j + 1)
                        var pair = nn34_compose(ma.unsafe_load(li), mb.unsafe_load(li), ma.unsafe_load(ri), mb.unsafe_load(ri))
                        left.unsafe_store(nodes, li); right.unsafe_store(nodes, ri)
                        ma.unsafe_store(nodes, pair[0]); mb.unsafe_store(nodes, pair[1])
                        ids.unsafe_store(j, nodes)
                        nodes += 1
                width = next_width
            var root = ids.unsafe_load(0)
            var t = bb * L + begin + prefix - 1
            var grad = ftz(identical_mul(ftz(dy.unsafe_load(t * dim + d)), ftz(c.unsafe_load(t * NS + n))))
            if rp == 0 and reverse_chunk != 0:
                grad = ftz(grad + carry)
            ga.unsafe_store(root, ftz(identical_mul(grad, boundary)))
            gb.unsafe_store(root, grad)
            previous_carry = ftz(previous_carry + ftz(identical_mul(grad, ma.unsafe_load(root))))
            for reverse_node in range(nodes - prefix):
                var node = nodes - 1 - reverse_node
                var li = left.unsafe_load(node); var ri = right.unsafe_load(node)
                var da = ga.unsafe_load(node); var db = gb.unsafe_load(node)
                ga.unsafe_store(li, ftz(ga.unsafe_load(li) + ftz(identical_mul(da, ma.unsafe_load(ri)))))
                gb.unsafe_store(li, ftz(gb.unsafe_load(li) + ftz(identical_mul(db, ma.unsafe_load(ri)))))
                var dra = ftz(ga.unsafe_load(ri) + ftz(identical_mul(da, ma.unsafe_load(li))))
                dra = ftz(dra + ftz(identical_mul(db, mb.unsafe_load(li))))
                ga.unsafe_store(ri, dra)
                gb.unsafe_store(ri, ftz(gb.unsafe_load(ri) + db))
            for j in range(prefix):
                sum_a.unsafe_store(j, ftz(sum_a.unsafe_load(j) + ga.unsafe_load(j)))
                sum_b.unsafe_store(j, ftz(sum_b.unsafe_load(j) + gb.unsafe_load(j)))
        for j in range(count):
            var index = ((bb * L + begin + j) * dim + d) * NS + n
            factor_grad.unsafe_store(index, sum_b.unsafe_load(j))
            factor_grad.unsafe_store(total + index, sum_a.unsafe_load(j))
        carry = previous_carry
