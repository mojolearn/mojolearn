"""Does the DEVICE split reduction pick the same candidate as a host loop?

    pixi run mojo run -I . extratrees/mojo_only/split_reduce_check.mojo

NO CUML COUNTERPART -- this is a check. It covers
`extratrees/ported/decisiontree/batched_levelalgo/split.mojo`'s
`split_warp_reduce`, `split_reduce_kernel` and `split_reduce_init_kernel`,
which are `split.cuh:92-152` and `:158-166` and carry DEVIATION BLOCKS
166-169.

WHAT THE ORACLE IS, AND WHY IT IS ALLOWED TO BE ONE
---------------------------------------------------
`Split.update` folded over the same candidates in a plain sequential loop.
That function is already checked, in `split_check.mojo`, against an
independently written lexicographic comparator over 54,289 ordered pairs, so
it is not being used to check itself here. The composed order this reduction
imposes -- exact rational first, then their three arms -- is likewise NOT
re-expressed the way the kernel expresses it: `independent_better` below is a
flat `if` chain with no accumulator and no early-out, and the exact half of it
calls the HOST's `CompareProxyExact`, which multiplies in `Int128`. The device
computes the same comparison from 32-bit limbs (DEVIATION BLOCK 167), so the
two paths through the wide multiply are genuinely different code.

WHY THE COMPARISON CAN BE EXACT, WITH NO TOLERANCE
--------------------------------------------------
Nothing here arithmetic-ally combines anything. `update` SELECTS one of its
two inputs and returns it unchanged, so a maximum under a total order is
identical however the candidates are grouped -- per thread, per warp, per
block, or one at a time on the host. Every assertion below is on bit patterns.

WHAT IS DELIBERATELY ADVERSARIAL ABOUT THE FIXTURE
--------------------------------------------------
1. **The candidates are REAL Gini candidates.** Each one is a left histogram
   over a 1,048,576-row node, scored by `GiniObjectiveFunction.GainPerSplit`
   for `Split.best_metric_val` and by `ProxyImpurityExact` for the key. The
   float and the rational therefore have the relationship the deviation is
   about, rather than one this file invented.
2. **The products the comparator forms EXCEED 2^64.** With n = 2^20 the
   cross-multiply reaches ~2^96, so the high limb of DEVIATION 167's
   hand-widened product is load-bearing; the check counts how many
   comparisons the high word alone decided, and fails if that count is zero.
3. **Ties are planted at BOTH tie-break arms, AT THE TOP.** A tie group that
   is not the argmax leaves the arms live but not DECISIVE, and a check can
   then pass with an arm broken. So the group is built from the exact
   argmax's own key: five candidates share its rational and its float score,
   two of them share a `colid` as well, and the winner is therefore the one
   the `colid` arm and then the `quesval` arm select.
4. **The DEVIATION 145 pair.** For every Gini node the fixture searches for a
   candidate whose `Float32` gain is BIT-EQUAL to the winner's but whose
   exact proxy is strictly smaller, and gives it the largest `colid` in the
   node. A reduction keyed on the float alone takes it; the exact one does
   not. If the search fails the check FAILS -- at that point DEVIATION 145's
   premise would be unsupported at this scale and the entry would need
   re-arguing, not quietly keeping.
5. **Four nodes that are not Gini at all**: an empty one, one whose every
   candidate is rejected by `min_samples_leaf`, one of synthetic keys with
   NEGATIVE numerators (which Gini cannot produce and DEVIATION 167's sign
   branch exists for), and one with the key switched OFF entirely, which is
   DEVIATION 166's regression degeneration and must reduce to exactly what
   `Split.update` alone would pick.
6. **Node sizes that straddle every boundary**: 1, 31, 32, 33, 257 and 1000
   candidates against a 32-wide warp and a 256-thread block -- less than a
   warp, exactly a warp, one past a warp, one past a block, and four blocks.

THE ARMS
--------
  A. PER-CELL, all seven output fields of every node, on bit patterns,
     against the host fold. Run for a 256-thread block AND for a
     one-warp block, which are different reduction shapes.
  B. ORDER INDEPENDENCE. Eight permutations of every node's candidate list,
     each launched separately, every field required bit-identical.
  C. THE PATH, from the device's own counters (`out_n_merges`,
     `out_n_warps`, DEVIATION 169c) -- so "the cross-warp combine ran" is a
     report from the device and not host arithmetic about block sizes.
  D. THE WIDE COMPARATOR, `compare_exact_key` against the host's `Int128`
     `CompareProxyExact`, pairwise, including mixed-sign numerators.
  E. THE FIXTURE ITSELF: the tie counts, the float-collision pair, the
     high-limb count, and the absence of the one payload shape whose order
     WOULD depend on arrival (DEVIATION 166's stated boundary).
  F. SABOTAGE, one per mechanism, seven of them, each selected by a kernel
     ARGUMENT so every arm runs the SAME binary that ships.
"""

from std.math import ceildiv
from std.gpu import WARP_SIZE
from std.sys import has_accelerator
from std.sys.info import (
    has_amd_gpu_accelerator,
    has_apple_gpu_accelerator,
    has_nvidia_gpu_accelerator,
)
from max.gpu.host import DeviceContext
from max.gpu.host.device_attribute import DeviceAttribute

from extratrees.ported.decisiontree.batched_levelalgo.objectives import (
    CountBin,
    GiniObjectiveFunction,
    GiniProxyExact,
)
from extratrees.ported.decisiontree.batched_levelalgo.split import (
    compare_exact_key,
    ExactKey,
    Split,
    SplitExact,
    split_reduce_init_kernel,
    split_reduce_kernel,
    split_reduce_shared_bytes,
    SPLIT_SAB_BLOCK0_ONLY,
    SPLIT_SAB_FLOAT_KEY,
    SPLIT_SAB_NO_COLID_ARM,
    SPLIT_SAB_NO_CROSS_WARP,
    SPLIT_SAB_NO_LOCK,
    SPLIT_SAB_INVERT_GUARD,
    SPLIT_SAB_NO_QUESVAL_ARM,
    SPLIT_SAB_NONE,
)


comptime F = DType.float32
comptime N_ROWS = 1 << 20
"""Rows per node. Large on purpose: `MAX_ROWS_EXACT` is 2^26, and the point of
DEVIATION 145 is counts big enough that two different rationals round to one
`Float32`. It also puts the cross-multiply above 2^64, which is what makes the
high limb of DEVIATION 167's product load-bearing."""

comptime MIN_SAMPLES_LEAF = 2
comptime N_CLASSES = 2
comptime TPB_MULTI = 256
"""Eight warps on a 32-wide target, four on a 64-wide one."""

comptime TPB_ONE = WARP_SIZE
"""ONE warp, whatever the target's warp is. This arm's cross-warp step is a
no-op and its cross-BLOCK step is the busy one; the 256 arm is the reverse."""

comptime SEED: UInt64 = 0xE7_5217_C0FFEE


def mix64(x: UInt64) -> UInt64:
    """Splitmix64's finalizer. Only used to make the fixture unpredictable."""
    var h = x
    h ^= h >> 30
    h *= 0xBF58476D1CE4E5B9
    h ^= h >> 27
    h *= 0x94D049BB133111EB
    h ^= h >> 31
    return h


# ============================================================================
# The oracle side. Written to look nothing like `SplitExact.update`.
# ============================================================================


def host_cmp_key(a: ExactKey, b: ExactKey) -> Int:
    """The comparison the DEVICE does in 32-bit limbs, done in `Int128`.

    This is `objectives.mojo`'s own `CompareProxyExact` with the payload
    repacked -- the host authority for DEVIATION 145, and a completely
    different route through the wide multiply than `compare_exact_key`.
    """
    return GiniObjectiveFunction[F].CompareProxyExact(
        GiniProxyExact(a.num, a.den, 0, a.valid != 0),
        GiniProxyExact(b.num, b.den, 0, b.valid != 0),
    )


def independent_better(a: SplitExact, b: SplitExact) -> Bool:
    """Is `a` strictly better than `b`, as a flat lexicographic chain?

    Deliberately NOT `SplitExact.update`'s shape: no accumulator, no early-out
    flag, no assignment, and the exact half calls the host comparator. If both
    expressions of the order agree everywhere, a slip has to have been made
    twice, in two different arithmetics.
    """
    var c = host_cmp_key(a.key, b.key)
    if c != 0:
        return c > 0
    if a.split.best_metric_val != b.split.best_metric_val:
        return a.split.best_metric_val > b.split.best_metric_val
    if a.split.colid != b.split.colid:
        return a.split.colid > b.split.colid
    return a.split.quesval > b.split.quesval


def host_best(cands: List[SplitExact], begin: Int, count: Int) -> SplitExact:
    """The oracle: a linear maximum under `independent_better`, starting from
    the identity `SplitExact()`, which every real candidate must beat."""
    var best = SplitExact()
    for i in range(begin, begin + count):
        if independent_better(cands[i], best):
            best = cands[i]
    return best^


def host_best_subset(
    cands: List[SplitExact], begin: Int, count: Int, period: Int, width: Int
) -> SplitExact:
    """The oracle restricted to a STRIDED SUBSET of a node's candidates.

    Offset `o` is folded iff `o % period < width`. That is exactly the set a
    sabotaged kernel still sees: `period = TPB * blocks_per_node, width = TPB`
    is block 0's slice, and `period = TPB, width = WARP_SIZE` is warp 0 of
    every block -- both read off the kernel's own loop, `i = begin +
    thread_idx.x + blk * TPB` stepping by `TPB * blocks_per_node`.

    It is what turns "the sabotage moved something" into "the sabotage moved
    EXACTLY these cells", which is the difference between a prediction that
    can fail and one that cannot.
    """
    var best = SplitExact()
    for o in range(count):
        if o % period >= width:
            continue
        if independent_better(cands[begin + o], best):
            best = cands[begin + o]
    return best^


def host_best_float_key(
    cands: List[SplitExact], begin: Int, count: Int
) -> SplitExact:
    """What cuML's OWN reduction would pick: `Split.update` on the float
    field, exact key ignored. Used only to prove the fixture can tell the two
    apart (arm E) -- it is what DEVIATION 145 says is wrong."""
    var best = Split()
    var out = SplitExact()
    for i in range(begin, begin + count):
        if best.update(cands[i].split):
            out = cands[i]
    return out^


def same_payload(a: SplitExact, b: SplitExact) -> Bool:
    """All seven fields, floats on bit patterns."""
    return (
        a.split.quesval.to_bits() == b.split.quesval.to_bits()
        and a.split.colid == b.split.colid
        and a.split.best_metric_val.to_bits()
        == b.split.best_metric_val.to_bits()
        and a.split.n_left == b.split.n_left
        and a.key.num == b.key.num
        and a.key.den == b.key.den
        and a.key.valid == b.key.valid
    )


def show(s: SplitExact) -> String:
    return (
        String("(q=")
        + String(s.split.quesval)
        + " col="
        + String(s.split.colid)
        + " gain="
        + String(s.split.best_metric_val)
        + " nL="
        + String(s.split.n_left)
        + " num="
        + String(s.key.num)
        + " den="
        + String(s.key.den)
        + " v="
        + String(s.key.valid)
        + ")"
    )


# ============================================================================
# The fixture.
# ============================================================================


def gini_cand(
    obj: GiniObjectiveFunction[F],
    mut left_buf: List[CountBin],
    mut tot_buf: List[CountBin],
    l0: Int,
    l1: Int,
    quesval: Float32,
    colid: Int32,
) -> SplitExact:
    """One REAL candidate: cuML's gain in the float field, sklearn's exact
    proxy in the key. Both come from the ported objective, not from here."""
    left_buf[0] = CountBin(Int32(l0))
    left_buf[1] = CountBin(Int32(l1))
    var n = Int32(Int(tot_buf[0].x) + Int(tot_buf[1].x))
    var nl = Int32(l0 + l1)
    var ex = obj.ProxyImpurityExact(
        left_buf.unsafe_ptr(), tot_buf.unsafe_ptr(), n, nl
    )
    var gain = obj.GainPerSplit(
        left_buf.unsafe_ptr(), tot_buf.unsafe_ptr(), n, nl
    )
    var v = Int32(1) if ex.valid else Int32(0)
    return SplitExact(
        Split(quesval, colid, gain, nl), ExactKey(ex.num, ex.den, v)
    )


@fieldwise_init
struct NodeSpec(Copyable, Movable):
    """One cell: what it is, and how many candidates it holds."""

    var name: String
    var count: Int
    var kind: Int
    """0 real Gini, 1 all rejected, 2 synthetic mixed-sign keys, 3 no key."""


def build_candidates(
    specs: List[NodeSpec], mut plant_report: List[Int]
) raises -> List[SplitExact]:
    """Build every node's candidate list, back to back.

    `plant_report` receives, per node: the index of the exact argmax before
    planting, the size of the planted tie group, and whether a
    DEVIATION-145 float-collision candidate was found (-1 if the node has
    none by construction).
    """
    var obj = GiniObjectiveFunction[F](Int32(N_CLASSES), Int32(MIN_SAMPLES_LEAF))
    var left_buf = List[CountBin](length=N_CLASSES, fill=CountBin(0))
    var tot_buf = List[CountBin](length=N_CLASSES, fill=CountBin(0))

    var out = List[SplitExact]()
    for nidx in range(len(specs)):
        var spec = specs[nidx].copy()
        var begin = len(out)
        var h0 = mix64(SEED ^ UInt64(nidx) * 0x9E3779B97F4A7C15)
        # A lopsided class split, different per node, so the optimum is not at
        # the midpoint and the proxy is not symmetric.
        var t0 = Int(N_ROWS // 3) + Int(h0 % UInt64(N_ROWS // 5))
        var t1 = N_ROWS - t0
        tot_buf[0] = CountBin(Int32(t0))
        tot_buf[1] = CountBin(Int32(t1))

        for i in range(spec.count):
            var h = mix64(h0 ^ UInt64(i) * 0xD6E8FEB86659FD93)
            var quesval = Float32(Int(mix64(h ^ 7) % 8388608)) / 64.0
            var colid = Int32(Int(mix64(h ^ 11) % 89))
            if spec.kind == 2:
                # Synthetic keys, NOT Gini: numerators of both signs, so
                # `compare_exact_key`'s sign split is reached on device.
                var num = Int64(Int(h % 0x3FFFFFFFFFFF)) - Int64(0x1FFFFFFFFFFF)
                var den = Int64(1 + Int(mix64(h ^ 3) % 0xFFFFFFF))
                var gain = Float32(Int(mix64(h ^ 5) % 4096)) / 256.0
                out.append(
                    SplitExact(
                        Split(quesval, colid, gain, Int32(i + 1)),
                        ExactKey(num, den, Int32(1)),
                    )
                )
                continue
            if spec.kind == 3:
                # No key at all: DEVIATION 166's regression degeneration.
                var gain = Float32(Int(mix64(h ^ 5) % 100000)) / 512.0
                out.append(
                    SplitExact(
                        Split(quesval, colid, gain, Int32(i + 1)), ExactKey()
                    )
                )
                continue

            var nl: Int
            if spec.kind == 1:
                # Every candidate below `min_samples_leaf`, so every one is
                # rejected and the whole node reduces over invalid keys.
                nl = Int(h % UInt64(MIN_SAMPLES_LEAF))
            else:
                nl = 1 + Int(h % UInt64(N_ROWS - 2))
            var lo = 0
            if nl > t1:
                lo = nl - t1
            var hi = t0
            if nl < hi:
                hi = nl
            var l0 = lo
            if hi > lo:
                l0 = lo + Int(mix64(h ^ 13) % UInt64(hi - lo + 1))
            out.append(
                gini_cand(obj, left_buf, tot_buf, l0, nl - l0, quesval, colid)
            )

        # ---- the planting, per node ------------------------------------
        var argmax = -1
        var tie_group = 0
        var collide = -1
        if spec.kind == 0 and spec.count >= 12:
            # A NEAR-PURE family, and it is the whole reason DEVIATION 145 is
            # reachable at all in this fixture. A random draw's Gini proxy
            # moves by O(1) per sample moved, which is 1e-6 in the gain and
            # 100x an ulp -- so random candidates NEVER collide in `Float32`
            # and a fixture made only of them cannot see the exact
            # comparator. Near a PURE split the proxy is flat: putting one
            # class-1 row on the left instead of leaving one class-0 row on
            # the right changes the proxy by `2/(t0+1) - 2/(t1+1)`, about
            # 2e-6, which is 1e-12 in the gain and FAR below an ulp of it.
            # Those are the candidate pairs whose exact proxies differ and
            # whose floats do not, and they are also the node's best
            # candidates, so the collision sits exactly where it can change
            # the winner.
            var fam = spec.count // 2
            if fam > 40:
                fam = 40
            var fam_slots = List[Int]()
            for j in range(fam):
                var a = j % 15
                var b = j // 15
                if a + b == 0:
                    a = 1
                var slot = (j * 17 + 3) % spec.count
                var already = False
                for k in range(len(fam_slots)):
                    if fam_slots[k] == slot:
                        already = True
                if already:
                    continue
                var hh = mix64(h0 ^ UInt64(j) * 0x2545F4914F6CDD1D)
                out[begin + slot] = gini_cand(
                    obj,
                    left_buf,
                    tot_buf,
                    t0 - a,
                    b,
                    Float32(Int(hh % 8388608)) / 64.0,
                    Int32(Int(mix64(hh ^ 2) % 89)),
                )
                fam_slots.append(slot)

            # The exact argmax, AFTER the family is in and BEFORE any tie is
            # planted. It is a near-pure candidate by construction.
            var best = 0
            for i in range(1, spec.count):
                if independent_better(out[begin + i], out[begin + best]):
                    best = i
            argmax = best
            var a_best = out[begin + best]

            # DEVIATION 145's pair: a DIFFERENT exact proxy whose `Float32`
            # gain has the SAME BITS. Searched for, not assumed -- if the
            # scale ever stops producing one, the check says so.
            var target = a_best.split.best_metric_val.to_bits()
            for i in range(spec.count):
                if i == best:
                    continue
                if out[begin + i].split.best_metric_val.to_bits() != target:
                    continue
                if host_cmp_key(out[begin + i].key, a_best.key) >= 0:
                    continue
                # The largest colid in the node, so that a reduction keyed on
                # the float alone -- cuML's own -- takes it on the tie-break.
                out[begin + i].split.colid = Int32(900)
                collide = i
                break

            # Five copies of the argmax's KEY and SCORE, spread across the
            # list so they land in different warps and different blocks:
            # colid -1 / +0 / +0 / +1 / +1, quesvals all distinct. The winner
            # is then decided by the `colid` arm and then the `quesval` arm,
            # which is what makes those arms decisive and their sabotage red.
            var stride = spec.count // 6
            if stride < 1:
                stride = 1
            var offs = [1, 2, 3, 4, 1]
            var deltas = [Int32(-1), Int32(0), Int32(0), Int32(1), Int32(1)]
            for k in range(5):
                var slot = (best + offs[k] * stride + k) % spec.count
                while slot == best or slot == collide:
                    slot = (slot + 1) % spec.count
                var c = a_best
                c.split.colid = a_best.split.colid + deltas[k]
                c.split.quesval = a_best.split.quesval + Float32(k + 1) * 0.25
                out[begin + slot] = c
                tie_group += 1
        plant_report.append(argmax)
        plant_report.append(tie_group)
        plant_report.append(collide)
    return out^


def permute(count: Int, scheme: Int) -> List[Int]:
    """A permutation of `0..count-1`. Scheme 0 is the identity; the rest are
    a rotation composed with a stride coprime to the count, plus a reversal,
    so the argmax lands in a different lane, warp and block each time."""
    var out = List[Int]()
    if count == 0:
        return out^
    if scheme == 0:
        for i in range(count):
            out.append(i)
        return out^
    var strides = [1, 3, 5, 7, 11, 13, 17, 19]
    var s = strides[scheme % len(strides)]
    while s > 1 and count % s == 0:
        s += 2
    var rot = (scheme * 37) % count
    for i in range(count):
        var j = (rot + i * s) % count
        if scheme % 3 == 2:
            j = count - 1 - j
        out.append(j)
    # A stride coprime with the count is a permutation; assert it rather than
    # believing it, because a fixture that silently visited one candidate
    # twice would make arm B pass about the wrong list.
    var seen = List[Bool](length=count, fill=False)
    for i in range(count):
        seen[out[i]] = True
    for i in range(count):
        if not seen[i]:
            out.clear()
            for k in range(count):
                out.append(k)
            break
    return out^


@fieldwise_init
struct RunOut(Copyable, Movable):
    """One launch's nine output arrays, back on the host."""

    var quesval: List[Float32]
    var colid: List[Int32]
    var metric: List[Float32]
    var nleft: List[Int32]
    var num: List[Int64]
    var den: List[Int64]
    var valid: List[Int32]
    var n_merges: List[Int32]
    var n_warps: List[Int32]

    def at(self, i: Int) -> SplitExact:
        return SplitExact(
            Split(self.quesval[i], self.colid[i], self.metric[i], self.nleft[i]),
            ExactKey(self.num[i], self.den[i], self.valid[i]),
        )


def vendor() -> String:
    comptime if has_apple_gpu_accelerator():
        return "Apple"
    elif has_nvidia_gpu_accelerator():
        return "NVIDIA"
    elif has_amd_gpu_accelerator():
        return "AMD"
    else:
        return "unknown vendor"


def run_arm[
    TPB: Int
](
    ctx: DeviceContext,
    cands: List[SplitExact],
    node_begin: List[Int32],
    node_count: List[Int32],
    bpn: Int,
    sabotage: Int,
) raises -> RunOut:
    """One launch of the SHIPPING kernel, with `sabotage` selecting the arm.

    Buffers are allocated per call rather than reused: this file takes no
    timing numbers (`extratrees/PLAN.md`), and a reused output buffer that
    someone forgot to re-seed is exactly the kind of defect the init kernel
    exists to prevent.
    """
    var n_nodes = len(node_count)
    var n_c = len(cands)

    var d_q = ctx.enqueue_create_buffer[DType.float32](n_nodes)
    var d_c = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var d_m = ctx.enqueue_create_buffer[DType.float32](n_nodes)
    var d_l = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var d_nu = ctx.enqueue_create_buffer[DType.int64](n_nodes)
    var d_de = ctx.enqueue_create_buffer[DType.int64](n_nodes)
    var d_v = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var d_mg = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var d_nw = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var d_mx = ctx.enqueue_create_buffer[DType.int32](n_nodes)

    var c_q = ctx.enqueue_create_buffer[DType.float32](n_c)
    var c_c = ctx.enqueue_create_buffer[DType.int32](n_c)
    var c_m = ctx.enqueue_create_buffer[DType.float32](n_c)
    var c_l = ctx.enqueue_create_buffer[DType.int32](n_c)
    var c_nu = ctx.enqueue_create_buffer[DType.int64](n_c)
    var c_de = ctx.enqueue_create_buffer[DType.int64](n_c)
    var c_v = ctx.enqueue_create_buffer[DType.int32](n_c)
    var d_nb = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var d_nc = ctx.enqueue_create_buffer[DType.int32](n_nodes)

    var h_q = ctx.enqueue_create_host_buffer[DType.float32](n_c)
    var h_c = ctx.enqueue_create_host_buffer[DType.int32](n_c)
    var h_m = ctx.enqueue_create_host_buffer[DType.float32](n_c)
    var h_l = ctx.enqueue_create_host_buffer[DType.int32](n_c)
    var h_nu = ctx.enqueue_create_host_buffer[DType.int64](n_c)
    var h_de = ctx.enqueue_create_host_buffer[DType.int64](n_c)
    var h_v = ctx.enqueue_create_host_buffer[DType.int32](n_c)
    var h_nb = ctx.enqueue_create_host_buffer[DType.int32](n_nodes)
    var h_nc = ctx.enqueue_create_host_buffer[DType.int32](n_nodes)
    ctx.synchronize()

    for i in range(n_c):
        h_q.unsafe_ptr().unsafe_store(i, cands[i].split.quesval)
        h_c.unsafe_ptr().unsafe_store(i, cands[i].split.colid)
        h_m.unsafe_ptr().unsafe_store(i, cands[i].split.best_metric_val)
        h_l.unsafe_ptr().unsafe_store(i, cands[i].split.n_left)
        h_nu.unsafe_ptr().unsafe_store(i, cands[i].key.num)
        h_de.unsafe_ptr().unsafe_store(i, cands[i].key.den)
        h_v.unsafe_ptr().unsafe_store(i, cands[i].key.valid)
    for i in range(n_nodes):
        h_nb.unsafe_ptr().unsafe_store(i, node_begin[i])
        h_nc.unsafe_ptr().unsafe_store(i, node_count[i])

    ctx.enqueue_copy(dst_buf=c_q, src_ptr=h_q.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=c_c, src_ptr=h_c.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=c_m, src_ptr=h_m.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=c_l, src_ptr=h_l.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=c_nu, src_ptr=h_nu.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=c_de, src_ptr=h_de.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=c_v, src_ptr=h_v.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_nb, src_ptr=h_nb.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_nc, src_ptr=h_nc.unsafe_ptr())
    ctx.enqueue_memset(d_mx, Int32(0))
    ctx.synchronize()

    ctx.enqueue_function[split_reduce_init_kernel](
        d_q.unsafe_ptr(),
        d_c.unsafe_ptr(),
        d_m.unsafe_ptr(),
        d_l.unsafe_ptr(),
        d_nu.unsafe_ptr(),
        d_de.unsafe_ptr(),
        d_v.unsafe_ptr(),
        d_mg.unsafe_ptr(),
        d_nw.unsafe_ptr(),
        Int32(n_nodes),
        grid_dim=ceildiv(n_nodes, 64),
        block_dim=64,
    )
    ctx.enqueue_function[split_reduce_kernel[TPB]](
        d_q.unsafe_ptr(),
        d_c.unsafe_ptr(),
        d_m.unsafe_ptr(),
        d_l.unsafe_ptr(),
        d_nu.unsafe_ptr(),
        d_de.unsafe_ptr(),
        d_v.unsafe_ptr(),
        d_mg.unsafe_ptr(),
        d_nw.unsafe_ptr(),
        d_mx.unsafe_ptr(),
        c_q.unsafe_ptr(),
        c_c.unsafe_ptr(),
        c_m.unsafe_ptr(),
        c_l.unsafe_ptr(),
        c_nu.unsafe_ptr(),
        c_de.unsafe_ptr(),
        c_v.unsafe_ptr(),
        d_nb.unsafe_ptr(),
        d_nc.unsafe_ptr(),
        Int32(bpn),
        Int32(sabotage),
        grid_dim=(bpn, n_nodes, 1),
        block_dim=(TPB, 1, 1),
    )

    var o_q = ctx.enqueue_create_host_buffer[DType.float32](n_nodes)
    var o_c = ctx.enqueue_create_host_buffer[DType.int32](n_nodes)
    var o_m = ctx.enqueue_create_host_buffer[DType.float32](n_nodes)
    var o_l = ctx.enqueue_create_host_buffer[DType.int32](n_nodes)
    var o_nu = ctx.enqueue_create_host_buffer[DType.int64](n_nodes)
    var o_de = ctx.enqueue_create_host_buffer[DType.int64](n_nodes)
    var o_v = ctx.enqueue_create_host_buffer[DType.int32](n_nodes)
    var o_mg = ctx.enqueue_create_host_buffer[DType.int32](n_nodes)
    var o_nw = ctx.enqueue_create_host_buffer[DType.int32](n_nodes)
    ctx.enqueue_copy(dst_buf=o_q, src_buf=d_q)
    ctx.enqueue_copy(dst_buf=o_c, src_buf=d_c)
    ctx.enqueue_copy(dst_buf=o_m, src_buf=d_m)
    ctx.enqueue_copy(dst_buf=o_l, src_buf=d_l)
    ctx.enqueue_copy(dst_buf=o_nu, src_buf=d_nu)
    ctx.enqueue_copy(dst_buf=o_de, src_buf=d_de)
    ctx.enqueue_copy(dst_buf=o_v, src_buf=d_v)
    ctx.enqueue_copy(dst_buf=o_mg, src_buf=d_mg)
    ctx.enqueue_copy(dst_buf=o_nw, src_buf=d_nw)
    ctx.synchronize()

    var r_q = List[Float32]()
    var r_c = List[Int32]()
    var r_m = List[Float32]()
    var r_l = List[Int32]()
    var r_nu = List[Int64]()
    var r_de = List[Int64]()
    var r_v = List[Int32]()
    var r_mg = List[Int32]()
    var r_nw = List[Int32]()
    for i in range(n_nodes):
        r_q.append(o_q.unsafe_ptr().unsafe_load(i))
        r_c.append(o_c.unsafe_ptr().unsafe_load(i))
        r_m.append(o_m.unsafe_ptr().unsafe_load(i))
        r_l.append(o_l.unsafe_ptr().unsafe_load(i))
        r_nu.append(o_nu.unsafe_ptr().unsafe_load(i))
        r_de.append(o_de.unsafe_ptr().unsafe_load(i))
        r_v.append(o_v.unsafe_ptr().unsafe_load(i))
        r_mg.append(o_mg.unsafe_ptr().unsafe_load(i))
        r_nw.append(o_nw.unsafe_ptr().unsafe_load(i))
    return RunOut(r_q^, r_c^, r_m^, r_l^, r_nu^, r_de^, r_v^, r_mg^, r_nw^)


def flatten(
    cands: List[SplitExact],
    node_begin: List[Int32],
    node_count: List[Int32],
    scheme: Int,
) -> List[SplitExact]:
    """The same candidates, each node's slice permuted by `scheme`."""
    var out = List[SplitExact]()
    for nid in range(len(node_count)):
        var b = Int(node_begin[nid])
        var c = Int(node_count[nid])
        var p = permute(c, scheme)
        for i in range(c):
            out.append(cands[b + p[i]])
    return out^


def main() raises:
    comptime assert has_accelerator(), "this check must run on a GPU"
    var failures = 0

    # ---------------------------------------------------------------------
    # The device, and the budget the launch is priced against. ONE query per
    # attribute: `get_attribute` costs 1.26 ms a call on Metal
    # (STANDING_ORDERS.md), and nothing here needs it twice.
    # ---------------------------------------------------------------------
    var ctx = DeviceContext()
    # ONE query per attribute. `WARP_SIZE` is not a queryable attribute on
    # every vendor -- Metal refuses it by name -- so the cross-check against
    # the compile-time constant is attempted and REPORTED as unavailable
    # rather than skipped silently.
    var dev_warp: Int
    try:
        dev_warp = Int(ctx.get_attribute(DeviceAttribute.WARP_SIZE))
    except:
        dev_warp = -1
    var max_threads = Int(
        ctx.get_attribute(DeviceAttribute.MAX_THREADS_PER_BLOCK)
    )
    var max_smem = Int(
        ctx.get_attribute(DeviceAttribute.MAX_SHARED_MEMORY_PER_BLOCK)
    )
    var need_smem = split_reduce_shared_bytes(TPB_MULTI, WARP_SIZE)
    print(
        "[device]",
        vendor(),
        "-- queried warp",
        dev_warp,
        ", max threads/block",
        max_threads,
        ", max shared/block",
        max_smem,
        "bytes",
    )
    print(
        "         compile-time WARP_SIZE",
        WARP_SIZE,
        "; this kernel wants",
        need_smem,
        "bytes of shared at TPB",
        TPB_MULTI,
        "and",
        split_reduce_shared_bytes(TPB_ONE, WARP_SIZE),
        "at TPB",
        TPB_ONE,
    )
    if dev_warp < 0:
        print(
            "         (this vendor does not expose WARP_SIZE as a device"
            " attribute, so the compile-time constant is unchecked against"
            " it here; every lane index in the kernel comes from that"
            " constant and never from a literal)"
        )
    elif dev_warp != WARP_SIZE:
        failures += 1
        print(
            "  *** the compile-time WARP_SIZE disagrees with the device's own"
            " report; every lane index in this kernel is derived from the"
            " former"
        )
    if TPB_MULTI > max_threads or need_smem > max_smem:
        failures += 1
        print("  *** the launch does not fit this device's queried budget")

    # ---------------------------------------------------------------------
    # The fixture.
    # ---------------------------------------------------------------------
    var specs: List[NodeSpec] = [
        NodeSpec(String("gini-1"), 1, 0),
        NodeSpec(String("gini-31"), 31, 0),
        NodeSpec(String("gini-32"), 32, 0),
        NodeSpec(String("gini-33"), 33, 0),
        NodeSpec(String("gini-257"), 257, 0),
        NodeSpec(String("gini-1000"), 1000, 0),
        NodeSpec(String("empty"), 0, 0),
        NodeSpec(String("all-rejected"), 40, 1),
        NodeSpec(String("mixed-sign-keys"), 200, 2),
        NodeSpec(String("no-key(regression)"), 300, 3),
    ]
    var plant = List[Int]()
    var cands = build_candidates(specs, plant)
    var n_nodes = len(specs)
    var node_begin = List[Int32]()
    var node_count = List[Int32]()
    var acc = 0
    for i in range(n_nodes):
        node_begin.append(Int32(acc))
        node_count.append(Int32(specs[i].count))
        acc += specs[i].count
    var n_c = len(cands)
    print("")
    print("[fixture]", n_nodes, "nodes,", n_c, "candidates,", N_ROWS,
          "rows a node, min_samples_leaf", MIN_SAMPLES_LEAF)
    for i in range(n_nodes):
        print(
            "          node",
            i,
            specs[i].name,
            "count",
            specs[i].count,
            "argmax",
            plant[3 * i],
            "planted ties",
            plant[3 * i + 1],
            "float-collision slot",
            plant[3 * i + 2],
        )

    # ---------------------------------------------------------------------
    # ARM E -- the fixture is capable of failing. Every clause here is a
    # property the fixture must HAVE for the arms below to mean anything.
    # ---------------------------------------------------------------------
    print("")
    print("[arm E] the fixture itself")

    var want = List[SplitExact]()
    for nid in range(n_nodes):
        want.append(
            host_best(cands, Int(node_begin[nid]), Int(node_count[nid]))
        )

    # E1: both tie-break arms are DECISIVE, not merely present -- the winner
    # of each planted node is a member of the tie group, and the group holds
    # a colid tie and a (colid, quesval) tie at the top.
    var e_colid = 0
    var e_quesval = 0
    for nid in range(n_nodes):
        var b = Int(node_begin[nid])
        var c = Int(node_count[nid])
        var w = want[nid]
        var eq_key = 0
        var eq_key_metric = 0
        var eq_colid = 0
        for i in range(b, b + c):
            if host_cmp_key(cands[i].key, w.key) != 0:
                continue
            eq_key += 1
            if cands[i].split.best_metric_val.to_bits() != w.split.best_metric_val.to_bits():
                continue
            eq_key_metric += 1
            if cands[i].split.colid == w.split.colid:
                eq_colid += 1
        if eq_key_metric > 1:
            e_colid += 1
        if eq_colid > 1:
            e_quesval += 1
    print(
        "  E1 nodes whose winner was decided by the colid arm:",
        e_colid,
        "; by the quesval arm:",
        e_quesval,
    )
    if e_colid < 4 or e_quesval < 4:
        failures += 1
        print(
            "  *** DEFECT IN THE CHECK: without a tie AT THE TOP the tie-break"
            " arms are live but not decisive, and their sabotage cannot move"
            " the answer"
        )

    # E2: the DEVIATION 145 pair exists, and it actually changes the winner.
    var e_split = 0
    var e_gini = 0
    for nid in range(n_nodes):
        if plant[3 * nid + 1] == 0:
            continue
        e_gini += 1
        var fw = host_best_float_key(
            cands, Int(node_begin[nid]), Int(node_count[nid])
        )
        if not same_payload(fw, want[nid]):
            e_split += 1
        else:
            print(
                "  node",
                nid,
                "float key and exact key agree:",
                show(want[nid]),
            )
    print(
        "  E2 of",
        e_gini,
        "Gini nodes, cuML's float key picks a DIFFERENT candidate than the"
        " exact key in",
        e_split,
    )
    if e_split < 4:
        failures += 1
        print(
            "  *** DEFECT IN THE CHECK: DEVIATION 145's premise is that a"
            " float-keyed reduction diverges. A fixture where it does not"
            " cannot see the exact comparator at all."
        )

    # E2b: the raw count of the pairs DEVIATION 145 is about -- same
    # `Float32` score, different exact proxy. It is printed because "the
    # premise holds" is a number, not an opinion.
    var collide_pairs = 0
    var strictly_ordered = 0
    for nid in range(n_nodes):
        var b = Int(node_begin[nid])
        var c = Int(node_count[nid])
        for i in range(b, b + c):
            for j in range(i + 1, b + c):
                if (
                    cands[i].split.best_metric_val.to_bits()
                    == cands[j].split.best_metric_val.to_bits()
                ):
                    if host_cmp_key(cands[i].key, cands[j].key) != 0:
                        collide_pairs += 1
                    else:
                        strictly_ordered += 1
    print(
        "  E2b candidate pairs whose Float32 gain is BIT-EQUAL:",
        collide_pairs + strictly_ordered,
        "of which",
        collide_pairs,
        "have DIFFERENT exact proxies -- those are the pairs a float-keyed"
        " reduction would order by feature index",
    )
    if collide_pairs == 0:
        failures += 1
        print(
            "  *** DEFECT IN THE CHECK: not one pair collides in Float32, so"
            " DEVIATION 145's premise is unsupported at this scale and the"
            " entry must be re-argued, not quietly kept"
        )

    # E3: no two DISTINCT payloads tie in all four ordered fields. That is
    # the ONE shape whose winner would depend on arrival order (DEVIATION
    # BLOCK 166), and the claim of order independence is conditional on its
    # absence -- so it is checked, not assumed.
    var ambiguous = 0
    for nid in range(n_nodes):
        var b = Int(node_begin[nid])
        var c = Int(node_count[nid])
        for i in range(b, b + c):
            for j in range(i + 1, b + c):
                if (
                    host_cmp_key(cands[i].key, cands[j].key) == 0
                    and cands[i].split.best_metric_val.to_bits()
                    == cands[j].split.best_metric_val.to_bits()
                    and cands[i].split.colid == cands[j].split.colid
                    and cands[i].split.quesval.to_bits()
                    == cands[j].split.quesval.to_bits()
                    and not same_payload(cands[i], cands[j])
                ):
                    ambiguous += 1
    print("  E3 payload pairs that tie in all four ordered fields but differ"
          " elsewhere:", ambiguous)
    if ambiguous != 0:
        failures += 1
        print(
            "  *** the fixture contains the one shape whose answer IS"
            " order-dependent; arm B would be asserting something false"
        )

    # E3b: the same question for the FLOAT-ONLY order, which arm F's first
    # sabotage reduces under. Where it holds, that sabotage's moved set can be
    # predicted exactly; where it does not, the node is downgraded to "may
    # move" rather than quietly asserted about.
    var float_ambiguous = List[Bool]()
    for nid in range(n_nodes):
        var b = Int(node_begin[nid])
        var c = Int(node_count[nid])
        var amb = False
        for i in range(b, b + c):
            for j in range(i + 1, b + c):
                if (
                    cands[i].split.best_metric_val.to_bits()
                    == cands[j].split.best_metric_val.to_bits()
                    and cands[i].split.colid == cands[j].split.colid
                    and cands[i].split.quesval.to_bits()
                    == cands[j].split.quesval.to_bits()
                    and not same_payload(cands[i], cands[j])
                ):
                    amb = True
        float_ambiguous.append(amb)
    var amb_n = 0
    for nid in range(n_nodes):
        if float_ambiguous[nid]:
            amb_n += 1
    print("  E3b nodes whose FLOAT-only order is itself ambiguous:", amb_n,
          "(those are predicted as 'may move', not 'must move')")

    # E4: `SplitExact.update` and the flat comparator agree pairwise. The
    # kernel's combine is the former; the oracle is the latter.
    var pairs = 0
    var pair_bad = 0
    var probe_n = 400
    if probe_n > n_c:
        probe_n = n_c
    for i in range(probe_n):
        for j in range(probe_n):
            var a = cands[i]
            var moved = a.update(cands[j], Int32(SPLIT_SAB_NONE))
            if moved != independent_better(cands[j], cands[i]):
                pair_bad += 1
            pairs += 1
    print("  E4 pairwise agreement of SplitExact.update with the flat"
          " comparator:", pairs, "ordered pairs,", pair_bad, "disagreements")
    if pair_bad != 0:
        failures += 1

    # ---------------------------------------------------------------------
    # ARM D -- the hand-widened 128-bit compare against the host's Int128.
    # ---------------------------------------------------------------------
    print("")
    print("[arm D] compare_exact_key (32-bit limbs) against"
          " CompareProxyExact (Int128)")
    var d_cells = 0
    var d_bad = 0
    var d_high = 0
    var d_wide = 0
    var probe_d = 700
    if probe_d > n_c:
        probe_d = n_c
    for i in range(probe_d):
        for j in range(probe_d):
            var got = Int(compare_exact_key(cands[i].key, cands[j].key))
            var expected = host_cmp_key(cands[i].key, cands[j].key)
            if got != expected:
                d_bad += 1
                if d_bad < 5:
                    print(
                        "    MISMATCH",
                        show(cands[i]),
                        "vs",
                        show(cands[j]),
                        "got",
                        got,
                        "want",
                        expected,
                    )
            d_cells += 1
            if cands[i].key.valid != 0 and cands[j].key.valid != 0:
                var p1 = Int128(cands[i].key.num) * Int128(cands[j].key.den)
                var p2 = Int128(cands[j].key.num) * Int128(cands[i].key.den)
                if p1 >= Int128(1) << 64 or p2 >= Int128(1) << 64:
                    d_wide += 1
                if (p1 >> 64) != (p2 >> 64):
                    d_high += 1
    print(
        "  D",
        d_cells,
        "ordered pairs compared;",
        d_bad,
        "disagreements;",
        d_wide,
        "products exceeded 2^64 and",
        d_high,
        "comparisons were decided by the HIGH limb alone",
    )
    if d_bad != 0:
        failures += 1
    if d_high == 0 or d_wide == 0:
        failures += 1
        print(
            "  *** DEFECT IN THE CHECK: every product fitted in 64 bits, so"
            " the widening this arm exists to check was never exercised"
        )

    # ---------------------------------------------------------------------
    # ARMS A, B and C -- the device.
    # ---------------------------------------------------------------------
    var max_count = 0
    for i in range(n_nodes):
        if Int(node_count[i]) > max_count:
            max_count = Int(node_count[i])
    var bpn_multi = ceildiv(max_count, TPB_MULTI)
    var bpn_one = ceildiv(max_count, TPB_ONE)
    print("")
    print(
        "[device] launching (blocks_per_node x nodes) = (",
        bpn_multi,
        "x",
        n_nodes,
        ") at TPB",
        TPB_MULTI,
        "and (",
        bpn_one,
        "x",
        n_nodes,
        ") at TPB",
        TPB_ONE,
    )

    var base_multi = run_arm[TPB_MULTI](
        ctx, flatten(cands, node_begin, node_count, 0), node_begin,
        node_count, bpn_multi, SPLIT_SAB_NONE,
    )
    var base_one = run_arm[TPB_ONE](
        ctx, flatten(cands, node_begin, node_count, 0), node_begin,
        node_count, bpn_one, SPLIT_SAB_NONE,
    )

    print("")
    print("[arm A] every output field of every cell against the host fold,"
          " on bit patterns")
    var a_cells = 0
    var a_bad = 0
    for arm in range(2):
        var run = base_multi.copy() if arm == 0 else base_one.copy()
        var tag = "TPB " + String(TPB_MULTI) if arm == 0 else "TPB " + String(TPB_ONE)
        for nid in range(n_nodes):
            var got = run.at(nid)
            var expected = want[nid]
            a_cells += 7
            if not same_payload(got, expected):
                a_bad += 1
                print(
                    "  MISMATCH",
                    tag,
                    "node",
                    nid,
                    specs[nid].name,
                    "got",
                    show(got),
                    "want",
                    show(expected),
                )
    if a_bad == 0:
        print("  arm A OK:", a_cells, "field comparisons across 2 block"
              " shapes, all bit-identical to the host fold")
    else:
        failures += 1
        print("  arm A FAILED:", a_bad, "cells wrong")

    # A': the no-key node must reduce to exactly what `Split.update` alone
    # picks -- DEVIATION 166's claim that the same kernel IS cuML's reduction
    # when the exact key is absent.
    var reg_nid = n_nodes - 1
    var reg_best = Split()
    for i in range(
        Int(node_begin[reg_nid]),
        Int(node_begin[reg_nid]) + Int(node_count[reg_nid]),
    ):
        _ = reg_best.update(cands[i].split)
    var reg_got = base_multi.at(reg_nid).split
    if (
        reg_got.quesval.to_bits() == reg_best.quesval.to_bits()
        and reg_got.colid == reg_best.colid
        and reg_got.best_metric_val.to_bits() == reg_best.best_metric_val.to_bits()
        and reg_got.n_left == reg_best.n_left
    ):
        print("  arm A' OK: with the key switched off the device answer is"
              " exactly a plain Split.update fold (DEVIATION 166)")
    else:
        failures += 1
        print("  arm A' FAILED: no-key node did not degenerate to Split.update")

    # ---------------------------------------------------------------------
    print("")
    print("[arm B] order independence: eight permutations per node, both"
          " block shapes, every field required bit-identical")
    var b_runs = 0
    var b_bad = 0
    var b_cells = 0
    var identity = flatten(cands, node_begin, node_count, 0)
    for scheme in range(1, 9):
        var ordered = flatten(cands, node_begin, node_count, scheme)
        # A permutation that did not permute would make this arm assert
        # nothing at all, so it is counted rather than assumed.
        var moved_slots = 0
        for i in range(len(ordered)):
            if not same_payload(ordered[i], identity[i]):
                moved_slots += 1
        if moved_slots * 2 < len(ordered):
            failures += 1
            print("  *** DEFECT IN THE CHECK: permutation", scheme,
                  "moved only", moved_slots, "of", len(ordered),
                  "candidate slots")
        var r_m2 = run_arm[TPB_MULTI](
            ctx, ordered, node_begin, node_count, bpn_multi, SPLIT_SAB_NONE
        )
        var r_o2 = run_arm[TPB_ONE](
            ctx, ordered, node_begin, node_count, bpn_one, SPLIT_SAB_NONE
        )
        b_runs += 2
        for nid in range(n_nodes):
            b_cells += 14
            if not same_payload(r_m2.at(nid), base_multi.at(nid)):
                b_bad += 1
                print("  MOVED under permutation", scheme, "TPB", TPB_MULTI,
                      "node", nid, show(r_m2.at(nid)), "was",
                      show(base_multi.at(nid)))
            if not same_payload(r_o2.at(nid), base_one.at(nid)):
                b_bad += 1
                print("  MOVED under permutation", scheme, "TPB", TPB_ONE,
                      "node", nid, show(r_o2.at(nid)), "was",
                      show(base_one.at(nid)))
    if b_bad == 0:
        print("  arm B OK:", b_runs, "launches over 8 permutations,",
              b_cells, "field comparisons, not one bit moved")
    else:
        failures += 1
        print("  arm B FAILED:", b_bad, "cells depended on the input order")

    # ---------------------------------------------------------------------
    print("")
    print("[arm C] which paths ran, from the DEVICE's own counters")
    var c_bad = 0
    var saw_one_warp = 0
    var saw_many_warps = 0
    var saw_one_block = 0
    var saw_many_blocks = 0
    for arm in range(2):
        var run = base_multi.copy() if arm == 0 else base_one.copy()
        var tpb = TPB_MULTI if arm == 0 else TPB_ONE
        var bpn = bpn_multi if arm == 0 else bpn_one
        var nw_max = tpb // WARP_SIZE
        for nid in range(n_nodes):
            var count = Int(node_count[nid])
            var want_blocks = ceildiv(count, tpb)
            if want_blocks > bpn:
                want_blocks = bpn
            var want_warps = ceildiv(count, WARP_SIZE)
            if want_warps > nw_max:
                want_warps = nw_max
            var got_blocks = Int(run.n_merges[nid])
            var got_warps = Int(run.n_warps[nid])
            if got_blocks != want_blocks or got_warps != want_warps:
                c_bad += 1
                print("  PATH WRONG TPB", tpb, "node", nid, specs[nid].name,
                      "count", count, "blocks got", got_blocks, "want",
                      want_blocks, "warps got", got_warps, "want", want_warps)
            if count > 0:
                if got_warps == 1:
                    saw_one_warp += 1
                else:
                    saw_many_warps += 1
                if got_blocks == 1:
                    saw_one_block += 1
                else:
                    saw_many_blocks += 1
    print(
        "  cells reported by the device: intra-warp-only",
        saw_one_warp,
        ", cross-warp",
        saw_many_warps,
        ", single-block",
        saw_one_block,
        ", cross-block",
        saw_many_blocks,
    )
    if (
        c_bad == 0
        and saw_one_warp > 0
        and saw_many_warps > 0
        and saw_one_block > 0
        and saw_many_blocks > 0
    ):
        print("  arm C OK: both warp paths and both block paths RAN, and the"
              " device named each one")
    else:
        failures += 1
        print("  arm C FAILED: c_bad", c_bad, "-- a batch missing a path"
              " cannot speak for the path it did not run")

    # ---------------------------------------------------------------------
    # ARM F -- sabotage, one per mechanism. Every arm states, BEFORE the run,
    # which cells it expects to move: 1 = must move, 0 = must NOT move, 2 =
    # may move (the mechanism's effect is order-dependent, so only the SHAPE
    # is predictable). A prediction of "most of them" cannot fail, so three
    # of the seven predict an exact set, computed rather than guessed.
    # ---------------------------------------------------------------------
    print("")
    print("[arm F] sabotage, one per mechanism, same binary, selected by a"
          " kernel argument")
    var ordered0 = flatten(cands, node_begin, node_count, 0)

    # Which nodes each deterministic mechanism must move, from the host.
    var exp_float = List[Int]()
    var exp_warp = List[Int]()
    var exp_block = List[Int]()
    var exp_colid = List[Int]()
    var exp_quesval = List[Int]()
    var exp_multi = List[Int]()
    for nid in range(n_nodes):
        var b = Int(node_begin[nid])
        var c = Int(node_count[nid])
        var w = want[nid]

        var fw = host_best_float_key(cands, b, c)
        var f = 0
        if not same_payload(fw, w):
            f = 1
        if float_ambiguous[nid]:
            f = 2
        exp_float.append(f)

        var ww = host_best_subset(cands, b, c, TPB_MULTI, WARP_SIZE)
        exp_warp.append(0 if same_payload(ww, w) else 1)

        var bw = host_best_subset(cands, b, c, TPB_MULTI * bpn_multi, TPB_MULTI)
        exp_block.append(0 if same_payload(bw, w) else 1)

        # A tie-break arm can only move a node whose winner was SELECTED by
        # that arm; which candidate it moves to depends on arrival order, so
        # these are "may move" on the nodes that qualify and "must not" on
        # every other.
        var eq_key_metric = 0
        var eq_colid = 0
        for i in range(b, b + c):
            if host_cmp_key(cands[i].key, w.key) != 0:
                continue
            if (
                cands[i].split.best_metric_val.to_bits()
                != w.split.best_metric_val.to_bits()
            ):
                continue
            eq_key_metric += 1
            if cands[i].split.colid == w.split.colid:
                eq_colid += 1
        exp_colid.append(2 if eq_key_metric > 1 else 0)
        exp_quesval.append(2 if eq_colid > 1 else 0)

        # The publish mechanisms can only be visible where more than one
        # block publishes into the cell.
        var blocks = ceildiv(c, TPB_MULTI)
        if blocks > bpn_multi:
            blocks = bpn_multi
        exp_multi.append(2 if blocks > 1 else 0)

    var sabs = [
        SPLIT_SAB_FLOAT_KEY,
        SPLIT_SAB_NO_COLID_ARM,
        SPLIT_SAB_NO_QUESVAL_ARM,
        SPLIT_SAB_NO_CROSS_WARP,
        SPLIT_SAB_INVERT_GUARD,
        SPLIT_SAB_NO_LOCK,
        SPLIT_SAB_BLOCK0_ONLY,
    ]
    var names = [
        String("the EXACT comparator (reduce on best_metric_val alone, which"
               " is cuML's own key and DEVIATION 145's wrong one)"),
        String("update's colid arm (split.cuh:85)"),
        String("update's quesval arm (split.cuh:88)"),
        String("the cross-warp combine (warp 0 reads slot 0 for every lane,"
               " split.cuh:126-131)"),
        String("the publish guard (INVERT split_reg.update's verdict,"
               " split.cuh:143-146)"),
        String("the mutex itself (publish with no lock at all,"
               " split.cuh:135-136), read-modify-write window widened"),
        String("the cross-BLOCK merge (only block 0 publishes)"),
    ]
    var preds = [
        String("moves EXACTLY the cells where a float-keyed fold of the same"
               " candidates picks a different winner -- computed on the host,"
               " per cell"),
        String("moves only cells whose winner was selected by the colid arm,"
               " and at least one of them"),
        String("moves only cells whose winner was selected by the quesval"
               " arm, and at least one of them"),
        String("moves EXACTLY the cells where a fold over warp 0 of every"
               " block differs from the full fold -- the kernel's own loop,"
               " restricted"),
        String("moves EVERY non-empty cell: the first block to reach a"
               " seeded cell always improves it, so an inverted guard never"
               " writes at all. THE FIRST VERSION of this arm published"
               " unconditionally and was red in only 6 runs of 8 -- a"
               " sabotage that is usually red is not evidence, and the fix"
               " belonged in the sabotage, not in the tolerance."),
        String("moves only cells served by more than one block, and at least"
               " one of them, in at least one repetition (RACY by"
               " it is repeated)"),
        String("moves EXACTLY the cells where a fold over block 0's slice"
               " differs from the full fold"),
    ]
    # The inverted publish guard leaves every non-empty cell at its seed.
    var exp_nonempty = List[Int]()
    for nid in range(n_nodes):
        exp_nonempty.append(1 if Int(node_count[nid]) > 0 else 0)

    var expects = [
        exp_float.copy(),
        exp_colid.copy(),
        exp_quesval.copy(),
        exp_warp.copy(),
        exp_nonempty.copy(),
        exp_multi.copy(),
        exp_block.copy(),
    ]
    var reps = [1, 1, 1, 1, 8, 8, 1]
    # HOW MANY OF THOSE REPETITIONS MUST BE RED, and the two are not the same
    # number for every arm. A DETERMINISTIC sabotage must be red in every run;
    # a RACY one -- the arm that removes the mutex -- must be red in at least
    # one, because an unlocked publish sometimes happens to interleave
    # correctly and produce the right answer anyway.
    #
    # THIS DISTINCTION WAS MISSING AND MADE THE CHECK FLAKY, which is worse
    # than the thing it tests: requiring the mutex arm to be red 8 times of 8
    # failed about once in seventeen whole-check runs, so a green run proved
    # nothing and a red one might have been the race rather than a defect.
    # A race that is only sometimes visible is still a race, and the SHAPE
    # requirement below still applies to every repetition -- no cell the
    # mechanism cannot reach may move, in any run -- so nothing is weakened
    # except the count.
    var min_red = [1, 1, 1, 1, 8, 1, 1]
    for k in range(len(sabs)):
        var red_runs = 0
        var shape_bad = 0
        var worst = 0
        var expect = expects[k].copy()
        for _ in range(reps[k]):
            var run = run_arm[TPB_MULTI](
                ctx, ordered0, node_begin, node_count, bpn_multi, sabs[k]
            )
            var wrong = 0
            for nid in range(n_nodes):
                var moved = not same_payload(run.at(nid), want[nid])
                if moved:
                    wrong += 1
                if expect[nid] == 1 and not moved:
                    shape_bad += 1
                    print("      cell", nid, specs[nid].name,
                          "was PREDICTED to move and did not")
                if expect[nid] == 0 and moved:
                    shape_bad += 1
                    print("      cell", nid, specs[nid].name,
                          "moved and the mechanism cannot reach it")
            if wrong > 0:
                red_runs += 1
            if wrong > worst:
                worst = wrong
        var n_must = 0
        var n_may = 0
        for nid in range(n_nodes):
            if expect[nid] == 1:
                n_must += 1
            elif expect[nid] == 2:
                n_may += 1
        print("  -", names[k])
        print("      predicted:", preds[k])
        print("      prediction:", n_must, "cells must move,", n_may,
              "may move, the rest must not")
        if red_runs >= min_red[k] and shape_bad == 0:
            var how = String(" (a race: red in at least one run is the bar)")
            if min_red[k] == reps[k]:
                how = String("")
            print("      RED as required:", worst, "of", n_nodes,
                  "cells wrong, in", red_runs, "of", reps[k],
                  "run(s)", how, ", and the moved set matched the prediction")
        else:
            failures += 1
            print("      *** DEFECT IN THE CHECK ***:", names[k],
                  "-- red in", red_runs,
                  "of", reps[k], "run(s), needed", min_red[k], ";",
                  shape_bad,
                  "cell(s) disagreed with the prediction. A mechanism arm A"
                  " cannot see, or a prediction that is wrong, is a defect in"
                  " this FIXTURE.")

    print("")
    if failures == 0:
        print(
            "split_reduce_check: PASS --",
            n_nodes,
            "cells x 7 fields against a host fold, 16 permuted launches"
            " bit-identical, both warp paths and both block paths named by"
            " the device, and seven mechanisms sabotaged",
        )
    else:
        print("split_reduce_check: FAIL --", failures, "arm(s) red")
        raise Error("split_reduce_check failed")
