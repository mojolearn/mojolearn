"""DEVIATION 210: the plan fold, gated word for word.

`update_partitions_and_plan_kernel` claims to be `update_partitions_after_
split_kernel` + `plan_level_kernel` in one launch, with NO arithmetic moved.
This file makes the claim falsifiable three ways on the same planted
partitions:

1. THE HOST FOLD is the expected value, computed independently from the
   planted borders (per [[gate-against-a-real-accumulator]]: the incumbent
   pair is not its own oracle). Every word of `p_off`/`p_sz`/`hp_off`/
   `hp_sz` and of the plan triple is compared PER CELL against it, for the
   incumbent pair AND for the fused kernel. Offsets and sizes are all
   DISTINCT hashed-looking values so a slot permutation cannot cancel
   ([[uniform-test-data-hides-permutation]]).

2. FUSED vs INCUMBENT, word for word across all seven output arrays. The
   fused kernel computes the same integers from the same operands, so the
   demand is EQUALITY, not tolerance.

3. THE SABOTAGE: the fused body with the pre-PORTING.md-136 tie inversion
   (`small` starts LEFT, moves only on a strictly smaller right). It must
   disagree with the incumbent on the plan words of EXACTLY the tied pairs
   and agree everywhere else -- partitions included, because the sabotage
   touches only the plan tail. A gate no sabotage moves is decorative.

The pair set covers: an exact tie, off-by-one both ways, an all-left split
(empty right child), an all-right split (empty left child), an empty parent
(tie at zero), and two strided leaves (borders past one block's reach at
the narrow launch). Every case runs under TWO grids: the shipped shape
(grid x 4, block 512) and a deliberately narrow one (grid x 1, block 32)
that forces the border past the first stride for the wide leaves.
"""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx

from max.gpu.host import DeviceBuffer, DeviceContext

from std.gpu.intrinsics import ldg
from gbdt.methods.greedy_subsets_searcher.kernel.split_points import (
    update_partitions_after_split_kernel,
    update_partitions_and_plan_kernel,
)
from gbdt.methods.greedy_subsets_searcher.kernel.split_resolve import (
    plan_level_kernel,
)


comptime POISON = UInt32(0xDEADDEAD)


def update_partitions_and_plan_inverted_kernel(
    left_leaves: MutPointer[UInt32, MutAnyOrigin],
    right_leaves: MutPointer[UInt32, MutAnyOrigin],
    leaf_count: Int32,
    sorted_flags: MutPointer[UInt8, MutAnyOrigin],
    part_offset: MutPointer[UInt32, MutAnyOrigin],
    part_size: MutPointer[UInt32, MutAnyOrigin],
    host_offset: MutPointer[UInt32, MutAnyOrigin],
    host_size: MutPointer[UInt32, MutAnyOrigin],
    ids_compute: MutPointer[UInt32, MutAnyOrigin],
    sub_from: MutPointer[UInt32, MutAnyOrigin],
    sub_what: MutPointer[UInt32, MutAnyOrigin],
):
    """THE SABOTAGE: the fused kernel with the pre-136 tie, verbatim except
    that `small` starts LEFT and moves only when the right is STRICTLY
    smaller, so a tie computes the LEFT child. Nothing in the product calls
    this; it exists so the gate above it can be shown to fail on demand."""
    var leaf_slot = Int(block_idx.y)
    var left_leaf = Int(left_leaves.unsafe_load(leaf_slot))
    var right_leaf = Int(right_leaves.unsafe_load(leaf_slot))

    var offset = Int(part_offset.unsafe_load(left_leaf))
    var part_sz = Int(part_size.unsafe_load(left_leaf))

    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)

    while i <= part_sz:
        var flag0 = 1
        if i < part_sz:
            flag0 = Int(ldg(sorted_flags + (offset + i)))
        var flag1 = 0
        if i != 0:
            flag1 = Int(ldg(sorted_flags + (offset + i - 1)))

        if flag0 != flag1:
            part_size.unsafe_store(left_leaf, UInt32(i))
            host_offset.unsafe_store(left_leaf, UInt32(offset))
            host_size.unsafe_store(left_leaf, UInt32(i))

            part_offset.unsafe_store(right_leaf, UInt32(offset + i))
            part_size.unsafe_store(right_leaf, UInt32(part_sz - i))
            host_offset.unsafe_store(right_leaf, UInt32(offset + i))
            host_size.unsafe_store(right_leaf, UInt32(part_sz - i))

            var left_sz = UInt32(i)
            var right_sz = UInt32(part_sz - i)
            var small = UInt32(left_leaf)
            var big = UInt32(right_leaf)
            if right_sz < left_sz:
                small = UInt32(right_leaf)
                big = UInt32(left_leaf)
            ids_compute.unsafe_store(leaf_slot, small)
            sub_from.unsafe_store(leaf_slot, big)
            sub_what.unsafe_store(leaf_slot, small)
            break
        i += stride


def run_path(
    ctx: DeviceContext,
    offs: List[Int],
    sizes: List[Int],
    borders: List[Int],
    n_pairs: Int,
    n_flag_words: Int,
    path: Int,
    grid_x: Int,
    block_x: Int,
) raises -> List[Int]:
    """One full run: plant, launch, read back.

    path 0 = incumbent (`update_partitions_after_split_kernel` then
    `plan_level_kernel`, the two launches the fold retires), 1 = fused,
    2 = the tie-inverted sabotage. Returns the seven arrays concatenated:
    p_off, p_sz, hp_off, hp_sz (2K each), then ids_compute, sub_from,
    sub_what (K each)."""
    var slots = 2 * n_pairs

    var flags = ctx.enqueue_create_buffer[DType.uint8](n_flag_words)
    var h_flags = ctx.enqueue_create_host_buffer[DType.uint8](n_flag_words)
    for j in range(n_pairs):
        for t in range(sizes[j]):
            h_flags.unsafe_ptr().unsafe_store(
                offs[j] + t, UInt8(0) if t < borders[j] else UInt8(1)
            )
    ctx.enqueue_copy(dst_buf=flags, src_ptr=h_flags.unsafe_ptr())

    var p_off = ctx.enqueue_create_buffer[DType.uint32](slots)
    var p_sz = ctx.enqueue_create_buffer[DType.uint32](slots)
    var hp_off = ctx.enqueue_create_buffer[DType.uint32](slots)
    var hp_sz = ctx.enqueue_create_buffer[DType.uint32](slots)
    var d_c = ctx.enqueue_create_buffer[DType.uint32](n_pairs)
    var d_f = ctx.enqueue_create_buffer[DType.uint32](n_pairs)
    var d_w = ctx.enqueue_create_buffer[DType.uint32](n_pairs)
    ctx.enqueue_memset(p_off, POISON)
    ctx.enqueue_memset(p_sz, POISON)
    ctx.enqueue_memset(hp_off, POISON)
    ctx.enqueue_memset(hp_sz, POISON)
    ctx.enqueue_memset(d_c, POISON)
    ctx.enqueue_memset(d_f, POISON)
    ctx.enqueue_memset(d_w, POISON)

    # the parents: left slot j holds the parent partition, exactly the
    # state `split_and_make_sequence` + the stable partition leave behind
    var h_ps = ctx.enqueue_create_host_buffer[DType.uint32](slots)
    var h_po = ctx.enqueue_create_host_buffer[DType.uint32](slots)
    for s in range(slots):
        h_po.unsafe_ptr().unsafe_store(s, POISON)
        h_ps.unsafe_ptr().unsafe_store(s, POISON)
    for j in range(n_pairs):
        h_po.unsafe_ptr().unsafe_store(j, UInt32(offs[j]))
        h_ps.unsafe_ptr().unsafe_store(j, UInt32(sizes[j]))
    ctx.enqueue_copy(dst_buf=p_off, src_ptr=h_po.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=p_sz, src_ptr=h_ps.unsafe_ptr())

    var lids = ctx.enqueue_create_buffer[DType.uint32](n_pairs)
    var rids = ctx.enqueue_create_buffer[DType.uint32](n_pairs)
    var h_l = ctx.enqueue_create_host_buffer[DType.uint32](n_pairs)
    var h_r = ctx.enqueue_create_host_buffer[DType.uint32](n_pairs)
    for j in range(n_pairs):
        h_l.unsafe_ptr().unsafe_store(j, UInt32(j))
        h_r.unsafe_ptr().unsafe_store(j, UInt32(n_pairs + j))
    ctx.enqueue_copy(dst_buf=lids, src_ptr=h_l.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=rids, src_ptr=h_r.unsafe_ptr())
    ctx.synchronize()

    if path == 0:
        ctx.enqueue_function[update_partitions_after_split_kernel](
            lids.unsafe_ptr(), rids.unsafe_ptr(), Int32(n_pairs),
            flags.unsafe_ptr(), p_off.unsafe_ptr(), p_sz.unsafe_ptr(),
            hp_off.unsafe_ptr(), hp_sz.unsafe_ptr(),
            grid_dim=(grid_x, n_pairs, 1),
            block_dim=(block_x, 1, 1),
        )
        # next level's planner, `half = n_pairs`: pair j is exactly
        # (slot j, slot n_pairs + j), the ids planted above
        ctx.enqueue_function[plan_level_kernel](
            p_sz.unsafe_ptr(), Int32(n_pairs),
            d_c.unsafe_ptr(), d_f.unsafe_ptr(), d_w.unsafe_ptr(),
            grid_dim=((n_pairs + 63) // 64, 1, 1),
            block_dim=(64, 1, 1),
        )
    elif path == 1:
        ctx.enqueue_function[update_partitions_and_plan_kernel](
            lids.unsafe_ptr(), rids.unsafe_ptr(), Int32(n_pairs),
            flags.unsafe_ptr(), p_off.unsafe_ptr(), p_sz.unsafe_ptr(),
            hp_off.unsafe_ptr(), hp_sz.unsafe_ptr(),
            d_c.unsafe_ptr(), d_f.unsafe_ptr(), d_w.unsafe_ptr(),
            grid_dim=(grid_x, n_pairs, 1),
            block_dim=(block_x, 1, 1),
        )
    else:
        ctx.enqueue_function[update_partitions_and_plan_inverted_kernel](
            lids.unsafe_ptr(), rids.unsafe_ptr(), Int32(n_pairs),
            flags.unsafe_ptr(), p_off.unsafe_ptr(), p_sz.unsafe_ptr(),
            hp_off.unsafe_ptr(), hp_sz.unsafe_ptr(),
            d_c.unsafe_ptr(), d_f.unsafe_ptr(), d_w.unsafe_ptr(),
            grid_dim=(grid_x, n_pairs, 1),
            block_dim=(block_x, 1, 1),
        )
    ctx.synchronize()

    var out = List[Int]()
    var h_out = ctx.enqueue_create_host_buffer[DType.uint32](slots)
    ctx.enqueue_copy(dst_ptr=h_out.unsafe_ptr(), src_buf=p_off)
    ctx.synchronize()
    for s in range(slots):
        out.append(Int(h_out.unsafe_ptr().unsafe_load(s)))
    ctx.enqueue_copy(dst_ptr=h_out.unsafe_ptr(), src_buf=p_sz)
    ctx.synchronize()
    for s in range(slots):
        out.append(Int(h_out.unsafe_ptr().unsafe_load(s)))
    ctx.enqueue_copy(dst_ptr=h_out.unsafe_ptr(), src_buf=hp_off)
    ctx.synchronize()
    for s in range(slots):
        out.append(Int(h_out.unsafe_ptr().unsafe_load(s)))
    ctx.enqueue_copy(dst_ptr=h_out.unsafe_ptr(), src_buf=hp_sz)
    ctx.synchronize()
    for s in range(slots):
        out.append(Int(h_out.unsafe_ptr().unsafe_load(s)))
    var h_k = ctx.enqueue_create_host_buffer[DType.uint32](n_pairs)
    ctx.enqueue_copy(dst_ptr=h_k.unsafe_ptr(), src_buf=d_c)
    ctx.synchronize()
    for j in range(n_pairs):
        out.append(Int(h_k.unsafe_ptr().unsafe_load(j)))
    ctx.enqueue_copy(dst_ptr=h_k.unsafe_ptr(), src_buf=d_f)
    ctx.synchronize()
    for j in range(n_pairs):
        out.append(Int(h_k.unsafe_ptr().unsafe_load(j)))
    ctx.enqueue_copy(dst_ptr=h_k.unsafe_ptr(), src_buf=d_w)
    ctx.synchronize()
    for j in range(n_pairs):
        out.append(Int(h_k.unsafe_ptr().unsafe_load(j)))

    _ = h_flags^
    _ = h_ps^
    _ = h_po^
    _ = h_l^
    _ = h_r^
    _ = h_out^
    _ = h_k^
    _ = flags^
    _ = p_off^
    _ = p_sz^
    _ = hp_off^
    _ = hp_sz^
    _ = d_c^
    _ = d_f^
    _ = d_w^
    _ = lids^
    _ = rids^
    return out^


def main() raises:
    print("PLAN FUSION (DEVIATION 210): fused vs incumbent vs host fold")
    var ctx = DeviceContext()

    # the pair set; offsets are packed cumulatively so every slot's
    # expected offset is distinct
    var sizes: List[Int] = [100, 101, 101, 64, 64, 0, 977, 1500]
    var borders: List[Int] = [50, 50, 51, 0, 64, 0, 400, 750]
    var n_pairs = len(sizes)
    var offs = List[Int]()
    var acc = 0
    for j in range(n_pairs):
        offs.append(acc)
        acc += sizes[j]
    var n_flag_words = acc
    var slots = 2 * n_pairs

    # THE HOST FOLD: every expected word, from the planted borders alone
    var want = List[Int]()
    for j in range(n_pairs):                      # p_off, left then right
        want.append(offs[j])
    for j in range(n_pairs):
        want.append(offs[j] + borders[j])
    for j in range(n_pairs):                      # p_sz
        want.append(borders[j])
    for j in range(n_pairs):
        want.append(sizes[j] - borders[j])
    for j in range(n_pairs):                      # hp_off
        want.append(offs[j])
    for j in range(n_pairs):
        want.append(offs[j] + borders[j])
    for j in range(n_pairs):                      # hp_sz
        want.append(borders[j])
    for j in range(n_pairs):
        want.append(sizes[j] - borders[j])
    var ties = List[Bool]()
    for j in range(n_pairs):                      # the plan: strict < on
        var l = borders[j]                        # the LEFT, tie -> RIGHT
        var r = sizes[j] - borders[j]
        ties.append(l == r)
        want.append(j if l < r else n_pairs + j)          # ids_compute
    for j in range(n_pairs):
        var l = borders[j]
        var r = sizes[j] - borders[j]
        want.append(n_pairs + j if l < r else j)          # sub_from
    for j in range(n_pairs):
        var l = borders[j]
        var r = sizes[j] - borders[j]
        want.append(j if l < r else n_pairs + j)          # sub_what
    var n_ties = 0
    for j in range(n_pairs):
        if ties[j]:
            n_ties += 1

    var grids_x: List[Int] = [4, 1]
    var blocks_x: List[Int] = [512, 32]
    for g in range(2):
        var gx = grids_x[g]
        var bx = blocks_x[g]
        print(
            "  grid x", gx, "block", bx,
            "(shipped shape)" if g == 0 else "(narrow, forces the stride)",
        )
        var base = run_path(
            ctx, offs, sizes, borders, n_pairs, n_flag_words, 0, gx, bx
        )
        var fused = run_path(
            ctx, offs, sizes, borders, n_pairs, n_flag_words, 1, gx, bx
        )
        var bad = run_path(
            ctx, offs, sizes, borders, n_pairs, n_flag_words, 2, gx, bx
        )

        # 1. both real paths against the host fold, per word
        var wrong_base = 0
        var wrong_fused = 0
        for k in range(len(want)):
            if base[k] != want[k]:
                wrong_base += 1
            if fused[k] != want[k]:
                wrong_fused += 1
        print(
            "    vs host fold: incumbent wrong", wrong_base,
            " fused wrong", wrong_fused, " of", len(want), "words",
        )
        if wrong_base != 0:
            raise Error(
                "the INCUMBENT pair disagrees with the host fold on "
                + String(wrong_base)
                + " words; the fixture or the port is broken and the"
                " fusion comparison below would prove nothing"
            )
        if wrong_fused != 0:
            raise Error(
                "DEVIATION 210 fused kernel disagrees with the host fold"
                " on " + String(wrong_fused) + " words"
            )

        # 2. fused vs incumbent, word for word (redundant given 1, kept so
        # a future fixture edit that breaks the fold shows up HERE)
        var differ = 0
        for k in range(len(base)):
            if base[k] != fused[k]:
                differ += 1
        if differ != 0:
            raise Error(
                "fused and incumbent disagree on " + String(differ)
                + " words with an agreeing host fold; the readback is"
                " broken"
            )

        # 3. the sabotage: plan words flip on EXACTLY the ties, partitions
        # untouched
        var part_words = 4 * slots
        var part_moved = 0
        for k in range(part_words):
            if bad[k] != base[k]:
                part_moved += 1
        var pairs_moved = 0
        var ties_moved = 0
        for j in range(n_pairs):
            var moved = (
                bad[part_words + j] != base[part_words + j]
                or bad[part_words + n_pairs + j]
                != base[part_words + n_pairs + j]
                or bad[part_words + 2 * n_pairs + j]
                != base[part_words + 2 * n_pairs + j]
            )
            if moved:
                pairs_moved += 1
                if ties[j]:
                    ties_moved += 1
        print(
            "    SABOTAGE (pre-136 tie): moved", pairs_moved,
            "pairs' plans,", ties_moved, "of them ties; ties present:",
            n_ties, "; partition words moved:", part_moved,
        )
        if part_moved != 0:
            raise Error(
                "the sabotage moved " + String(part_moved)
                + " partition words; it must only touch the plan tail"
            )
        if pairs_moved != n_ties or ties_moved != n_ties:
            raise Error(
                "the sabotage did not move what it must: expected exactly"
                " the " + String(n_ties) + " tied pairs, it moved "
                + String(pairs_moved) + " (" + String(ties_moved)
                + " tied). A gate no sabotage moves is decorative."
            )

    print(
        "  ok   the fold changes no partition word and no plan word,"
        " under the shipped grid and the narrow one, and the tie"
        " inversion fails exactly the ties"
    )
