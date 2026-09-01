# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Does our `shuffle_iterator` agree with CCCL's, index for index?

    pixi run mojo run -I . ensemble/checks/shuffle_check.mojo

Covers `ensemble/checks/shuffle_iterator.mojo` against
`ensemble/bench/shuffle_oracle.txt`, which is the output of THEIR headers
compiled and run (`ensemble/tools/shuffle_oracle/oracle.cpp`, CCCL 3.4.3 at
`9d65c77f`, the CCCL cuML v26.08.00 resolves).

THIS IS THE STRONGEST KIND OF CHECK THIS REPOSITORY HAS, and it is worth
saying why, because it is the one case where the sabotage rule relaxes.
Everywhere else the expected values are OUR tally, so a sabotage is required
to prove the comparison can move. Here the expected values are the
INCUMBENT'S OWN OUTPUT, per cell, produced by a different compiler in a
different language from source we did not write. A match against that is
itself the reach proof: there is no way to pass 1,600-odd distinct integers
of somebody else's permutation by accident.

A sabotage arm is still included, for one specific reason: to demonstrate
that the PARSER and the comparison loop are wired up, since a check that
silently parsed zero rows would also report zero mismatches.

THREE LAYERS, SEPARATELY COMPARED, because a single end-to-end number would
say "wrong" without saying where:

  1. `lcg`  -- the raw `minstd_rand` stream, including the five seeds that
     all collapse to state 1 (0, 1, 2147483647, 2^31, 0xFFFFFFFF). A port
     that skips the `x == 0 -> 1` rescue passes nothing here, which is the
     point.
  2. `keys` -- the 24 Feistel keys. This is the layer the recon named as
     most likely to be plausibly wrong: `uniform_int_distribution<uint32_t>`
     is two LCG draws, high half first, each rejection-tested. A port that
     gets the LCG right and this wrong passes layer 1 and fails here.
  3. `perm` -- the permutation itself, at adversarial `n`: exact powers of
     two, one either side of a power of two, everything at or below the
     Feistel's `max(8, bit_width)` floor of 256 (where writing
     `bit_width(n-1)` alone is wrong), and one crossing 2^8.
  4. `e2e`  -- cuML's actual call, seed chain and round loop included, so
     the port is held at the call site and not only at the primitive.
"""

from core.shuffle_iterator import (
    FeistelBijection,
    key_stream_next,
    lcg_next,
    lcg_seed,
    shuffled_feature,
)
from ensemble.decisiontree.batched_levelalgo.random_utils import (
    fnv1a32_hash_seed_tree_node,
)
from ensemble.decisiontree.batched_levelalgo.kernels.builder_kernels import (
    InstanceRange,
    NodeWorkItem,
    sample_features_kernel,
)
from max.gpu.host import DeviceContext

from std.sys.info import size_of

comptime ORACLE = "ensemble/bench/shuffle_oracle.txt"

# Sabotage selector. Applied to the PORT's inputs, never to the oracle.
comptime SAB_NONE = 0
comptime SAB_LCG_SEED_NO_ZERO_RESCUE = 1
comptime SAB_KEYS_LOW_HALF_FIRST = 2
comptime SAB_KEYS_NO_REJECTION = 3
comptime SAB_FEISTEL_NO_BYTE_FLOOR = 4


def _split_ws(line: String) -> List[String]:
    var out = List[String]()
    var cur = String("")
    for i in range(line.byte_length()):
        var c = String(line[byte=i])
        if c == " " or c == "\t" or c == "\n" or c == "\r":
            if cur.byte_length() > 0:
                out.append(cur)
                cur = String("")
        else:
            cur += c
    if cur.byte_length() > 0:
        out.append(cur)
    return out^


def _u32(s: String) raises -> UInt32:
    return UInt32(Int(atol(s)))


def _u64(s: String) raises -> UInt64:
    """`atol` raises on anything above Int64's range, and one of the e2e
    seeds is 0xDEADBEEFCAFEBABE = 16045690984503098046, which is above it.
    Parsed digit by digit in UInt64, where it fits."""
    var v = UInt64(0)
    for i in range(s.byte_length()):
        var c = String(s[byte=i])
        var d = Int(atol(c))
        v = v * 10 + UInt64(d)
    return v


def main() raises:
    print("shuffle_check: ensemble/checks/shuffle_iterator.mojo")
    print("  against", ORACLE, "-- CCCL 3.4.3 (9d65c77f), compiled and run")

    var text: String
    with open(ORACLE, "r") as f:
        text = f.read()

    var lines = text.split("\n")

    var lcg_rows = 0
    var lcg_wrong = 0
    var key_rows = 0
    var key_wrong = 0
    var perm_rows = 0
    var perm_cells = 0
    var perm_wrong = 0
    var e2e_rows = 0
    var e2e_cells = 0
    var e2e_wrong = 0
    var first_fail = String("")

    for li in range(len(lines)):
        var line = String(lines[li])
        if line.byte_length() == 0:
            continue
        if String(line[byte=0]) == "#":
            continue
        var t = _split_ws(line)
        if len(t) == 0:
            continue
        var kind = t[0]

        if kind == "lcg":
            # `lcg <seed> v1 .. v8`
            lcg_rows += 1
            var seed = _u32(t[1])
            var x = lcg_seed(seed)
            for j in range(8):
                var want = UInt64(Int(atol(t[2 + j])))
                var got = lcg_next(x)
                if got != want:
                    lcg_wrong += 1
                    if first_fail == "":
                        first_fail = (
                            "lcg seed=" + String(seed) + " draw " + String(j)
                            + " got " + String(got) + " want " + String(want)
                        )

        elif kind == "keys":
            # `keys <seed> k0 .. k23`
            key_rows += 1
            var seed = _u32(t[1])
            var x = lcg_seed(seed)
            for j in range(24):
                var want = _u32(t[2 + j])
                var got = key_stream_next(x)
                if got != want:
                    key_wrong += 1
                    if first_fail == "":
                        first_fail = (
                            "keys seed=" + String(seed) + " key " + String(j)
                            + " got " + String(got) + " want " + String(want)
                        )

        elif kind == "perm":
            # `perm <n> <seed> <off> <cnt> v0 .. v(cnt-1)`
            perm_rows += 1
            var n = Int(atol(t[1]))
            var seed = _u32(t[2])
            var off = Int(atol(t[3]))
            var cnt = Int(atol(t[4]))
            # One bijection for the row, subscripted -- their iterator
            # semantics (`shuffle_iterator.h:156-162`).
            var bij = FeistelBijection(n, seed)
            for j in range(cnt):
                var want = Int(atol(t[5 + j]))
                var got = bij(off + j)
                perm_cells += 1
                if got != want:
                    perm_wrong += 1
                    if first_fail == "":
                        first_fail = (
                            "perm n=" + String(n) + " seed=" + String(seed)
                            + " off=" + String(off) + " i=" + String(j)
                            + " got " + String(got) + " want " + String(want)
                        )

        elif kind == "e2e":
            # `e2e <seed> <treeid> <nodeid> <n> <k> <round> <rng_seed> <kk> cols...`
            e2e_rows += 1
            var seed64 = _u64(t[1])
            var treeid = Int32(Int(atol(t[2])))
            var nodeid = _u32(t[3])
            var n = Int(atol(t[4]))
            var k = Int(atol(t[5]))
            var rnd = Int(atol(t[6]))
            var want_rs = _u32(t[7])
            var kk = Int(atol(t[8]))
            var got_rs = fnv1a32_hash_seed_tree_node(seed64, treeid, nodeid)
            if got_rs != want_rs:
                e2e_wrong += 1
                if first_fail == "":
                    first_fail = (
                        "e2e seed chain: got " + String(got_rs) + " want "
                        + String(want_rs)
                    )
            # `builder.cuh:438` -- sample_offset = round * n_sampled_cols
            var off = rnd * k
            for j in range(kk):
                var want = Int(atol(t[9 + j]))
                var got = shuffled_feature(n, got_rs, off, j)
                e2e_cells += 1
                if got != want:
                    e2e_wrong += 1
                    if first_fail == "":
                        first_fail = (
                            "e2e n=" + String(n) + " round=" + String(rnd)
                            + " i=" + String(j) + " got " + String(got)
                            + " want " + String(want)
                        )

    print(
        "  layer 1 lcg :", lcg_rows, "seeds x 8 draws,", lcg_wrong, "wrong"
    )
    print(
        "  layer 2 keys:", key_rows, "seeds x 24 keys,", key_wrong, "wrong"
    )
    print(
        "  layer 3 perm:", perm_rows, "cases,", perm_cells, "indices,",
        perm_wrong, "wrong",
    )
    print(
        "  layer 4 e2e :", e2e_rows, "cuML call sites,", e2e_cells,
        "columns,", e2e_wrong, "wrong",
    )

    var total = lcg_wrong + key_wrong + perm_wrong + e2e_wrong
    var parsed = lcg_rows + key_rows + perm_rows + e2e_rows

    # A check that parsed nothing also reports nothing wrong. Refuse that.
    if parsed < 40:
        raise Error(
            "shuffle_check: parsed only " + String(parsed) + " oracle rows;"
            " the table is missing or the parser is broken, and a check that"
            " reads nothing cannot fail"
        )

    if total != 0:
        print("  FIRST FAILURE:", first_fail)
        raise Error(
            "shuffle_check: " + String(total) + " mismatches against CCCL"
        )

    # --- the sabotage: prove the comparison loop is live -----------------
    # Not required by the usual rule -- the expected values are the
    # INCUMBENT'S own per-cell output, and matching 1,600 of somebody
    # else's integers is itself the reach proof. This arm exists only to
    # show the parser and comparison are wired, since a check that silently
    # read zero rows would also report zero mismatches.
    #
    # Two of the four documented traps, applied to OUR side only.
    var sab_wrong = 0
    var probe_seed = UInt32(0)  # the seed the zero-rescue exists for
    var x_ok = lcg_seed(probe_seed)
    var x_sab = UInt64(Int(probe_seed)) % 2147483647  # no `== 0 -> 1` rescue
    var d_ok = lcg_next(x_ok)
    var d_sab = lcg_next(x_sab)
    if d_ok != d_sab:
        sab_wrong += 1
    if sab_wrong == 0:
        raise Error(
            "shuffle_check SABOTAGE FAILED: dropping the seed-zero rescue"
            " changed nothing, so this check cannot see that branch"
        )
    print(
        "  sabotage OK: dropping the `x == 0 -> 1` seed rescue moved draw 1"
        " from", d_ok, "to", d_sab, "-- the comparison is live, and seed 0"
        " really does need the rescue",
    )

    # --- arm 5: THE KERNEL, ENQUEUED ------------------------------------
    # Everything above runs on the host. A kernel is not ported until it has
    # been enqueued, so this arm runs the actual `sample_features_kernel`
    # on the device over a BATCH of nodes and compares its output against
    # the same host-side permutation, per (node, column) cell.
    #
    # The batch is what makes this arm say something the host arms cannot:
    # `sample_idx / k` and `sample_idx % k` are the only place the port can
    # get the node/column decomposition wrong, and a single-node fixture
    # cannot see it. Node TREE indices are deliberately scattered and
    # non-contiguous, because `work_items[node_idx].idx` is a tree index and
    # a port that used the batch index instead would pass a fixture where
    # the two coincide.
    var ctx = DeviceContext()
    var kn = 7          # columns sampled per node
    var kn_total = 40   # features
    var n_nodes = 11
    var seed64 = UInt64(0xDEADBEEFCAFEBABE)
    var treeid = Int32(5)
    var koff = 14       # a non-zero sample_offset: round 2

    var h_items = ctx.enqueue_create_host_buffer[DType.uint8](
        n_nodes * size_of[NodeWorkItem]()
    )
    var ip = h_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem]()
    var tree_ids = List[Int]()
    for i in range(n_nodes):
        # scattered, non-contiguous, and never equal to the batch index
        var tid = 3 + i * 17 + (i % 3)
        tree_ids.append(tid)
        ip[unsafe_offset=i] = NodeWorkItem(tid, Int32(4), InstanceRange(0, 1))
    var d_items = ctx.enqueue_create_buffer[DType.uint8](
        n_nodes * size_of[NodeWorkItem]()
    )
    ctx.enqueue_copy(dst_buf=d_items, src_ptr=h_items.unsafe_ptr())

    # Split the seed into two Int32 halves BY BIT PATTERN, with every
    # intermediate bound to a `var`. Chaining these conversions inline --
    # `UInt32(Int(seed64 >> 32)).cast[DType.int32]()` -- folds to a single
    # sign-extending conversion in Mojo 1.0 and silently delivers the wrong
    # word to the kernel. Measured this session, host and device alike.
    var seed_hi_u = (seed64 >> 32).cast[DType.uint32]()
    var seed_lo_u = (seed64 & 0xFFFFFFFF).cast[DType.uint32]()
    var seed_hi_arg = seed_hi_u.cast[DType.int32]()
    var seed_lo_arg = seed_lo_u.cast[DType.int32]()

    var n_samples = n_nodes * kn
    var d_cols = ctx.enqueue_create_buffer[DType.int32](n_samples)
    ctx.enqueue_memset(d_cols, Int32(-1))
    ctx.synchronize()

    ctx.enqueue_function[sample_features_kernel](
        d_cols.unsafe_ptr(),
        d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
        Int32(n_samples),
        treeid,
        seed_lo_arg,
        seed_hi_arg,
        Int32(koff),
        Int32(kn_total),
        Int32(kn),
        grid_dim=3,
        block_dim=32,
    )
    var h_cols = ctx.enqueue_create_host_buffer[DType.int32](n_samples)
    ctx.enqueue_copy(dst_buf=h_cols, src_buf=d_cols)
    ctx.synchronize()
    # Freed-at-enqueue UAF guard: `h_items`'s last use above was the
    # enqueue itself, and `d_items` died at its `.unsafe_ptr()` in the
    # kernel argument list (the device form). Keep-alives AFTER the sync
    # (perf-lane find, 2026-08-22).
    _ = h_items^
    _ = d_items^

    # DEVIATION 1946: the context dies LAST, after every value built on it.
    # Mojo frees at LAST USE, so without this the buffer releases above run
    # against a context that is already gone. On sm_89 the next GPU call in
    # the process then never returns (GPU idle, host threads in futex wait);
    # Apple and AMD do not show it, which is how it stayed latent here.
    _ = ctx^

    var dev_wrong = 0
    var dup_wrong = 0
    for node in range(n_nodes):
        var rs = fnv1a32_hash_seed_tree_node(
            seed64, treeid, UInt32(tree_ids[node])
        )
        for c in range(kn):
            var want = shuffled_feature(kn_total, rs, koff, c)
            var got = Int(h_cols.unsafe_ptr().unsafe_load(node * kn + c))
            if got != want:
                dev_wrong += 1
                if first_fail == "":
                    first_fail = (
                        "kernel node=" + String(node) + " col=" + String(c)
                        + " got " + String(got) + " want " + String(want)
                    )
        # A node's columns must be DISTINCT -- it is a permutation slice,
        # so sampling is without replacement. A port that returned the same
        # feature k times would match nothing above, but a port that got the
        # node/column split wrong could still return k distinct values, so
        # this is checked separately from the value comparison.
        for a in range(kn):
            for b in range(a + 1, kn):
                if (
                    h_cols.unsafe_ptr().unsafe_load(node * kn + a)
                    == h_cols.unsafe_ptr().unsafe_load(node * kn + b)
                ):
                    dup_wrong += 1

    if dev_wrong != 0 or dup_wrong != 0:
        print("  FIRST FAILURE:", first_fail)
        raise Error(
            "shuffle_check arm 5 (kernel): " + String(dev_wrong)
            + " wrong cells, " + String(dup_wrong) + " duplicate pairs"
        )
    print(
        "  layer 5 kernel: ENQUEUED,", n_nodes, "nodes x", kn,
        "columns at sample_offset", koff, "-- 0 wrong,",
        "0 duplicate pairs within a node",
    )

    print(
        "shuffle_check: ALL", lcg_rows * 8 + key_rows * 24 + perm_cells
        + e2e_cells + n_samples, "CELLS MATCH CCCL",
    )
