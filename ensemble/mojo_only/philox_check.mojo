"""Does our Philox agree with RAFT's, draw for draw and index for index?

    tools/with_build_lock.sh pixi run mojo run -I . \\
        ensemble/mojo_only/philox_check.mojo

Covers `ensemble/mojo_only/philox.mojo` against
`ensemble/bench/philox_oracle.txt`, which is the output of THEIR generator
compiled and run (`ensemble/tools/philox_oracle/`): cuRAND's own Philox bytes,
fetched at build time, under RAFT v26.08.00's own wrapper, transcribed line by
line.

WHY THE SABOTAGE RULE RELAXES HERE, and why one is included anyway. The
expected values are the INCUMBENT'S OWN OUTPUT, per cell, produced by a
different compiler in a different language from source we did not write.
Matching ~4,900 of somebody else's integers is itself the reach proof; there is
no way to do it by accident. The sabotage arm at the end exists for one
specific reason: a check that silently parsed zero rows would also report zero
mismatches, so it proves the PARSER and the comparison loop are live by
feeding a deliberately broken generator through the same comparison and
requiring that it fail.

SIX LAYERS, SEPARATELY COUNTED, so a failure names its layer:

  1. `ctr`    -- the state `curand_init` leaves behind: all four counter words,
     both key words, the cached output block and STATE. This is where the
     subsequence has to land in `ctr.z`/`ctr.w` and not in `ctr.x`/`ctr.y`, and
     the subsequences include 2^32-1, 2^32, 2^32+1 and 2^64-1 so the carry in
     `Philox_State_Incr_hi` is exercised in both halves.
  2. `raw`    -- 13 consecutive 32-bit draws, which crosses three 4-word Philox
     blocks and so exercises the `STATE == 4 -> bump the counter, re-run the
     ten rounds` edge twice. A port that emits a block's words in the wrong
     order passes layer 1 and fails here.
  3. `lemire` -- the range reduction ALONE, driven by a scripted draw sequence
     through the same `U32Stream` parameterisation RAFT's own `custom_next`
     template has. Both the VALUE and the NUMBER OF DRAWS CONSUMED are
     compared, which is the only way to hold the rejection loop to anything.
     Adversarial ranges: 1, a power of two, one either side of a power of two,
     INT_MAX, 2^31, 2^31+1 (~half of all draws rejected) and the full 2^32-1
     span.
  4. `uint`   -- ten consecutive reduced draws off one real generator: layers 2
     and 3 joined, with the index mapping still out of the way.
  5. `fill`   -- the whole array, with the mapping. `len` values that are not
     multiples of 4 or of 256, and strides SMALLER than `len` so the
     grid-stride loop actually runs. This is the layer a distributional test
     cannot see. One row is `start = -2, end = INT_MAX` (`diff = 2^31 + 1`),
     which is the only case that reaches the rejection loop from a REAL
     generator with the index mapping in place -- about half its draws are
     rejected -- and is also the only case whose returned value overflows
     `int` (DEVIATION 186). It is device-launchable, so arm 7 reaches both.
  6. `e2e`    -- cuML's actual call: their fnv1a32 seed chain, `start = 0`,
     `end = n_rows`, `len = n_sampled_rows`, for several `tree_id`s. Includes
     `n_rows == 1`, where the reduction returns `start` unconditionally and a
     broken port looks perfect.

  7. THE KERNEL, ENQUEUED. Everything above runs on the host. A kernel is not
     ported until it has been enqueued, so the last arm runs
     `launch_uniform_int` / `launch_uniform_int_ex` on the device over the
     `fill` and `e2e` cases and compares the device's bytes against the same
     oracle rows.
"""

from ensemble.mojo_only.philox import (
    PhiloxState,
    custom_next_uniform_double,
    philox_next_u64,
)
from ensemble.mojo_only.philox import (
    PhiloxState,
    RNG_BLOCK_THREADS,
    RNG_STRIDE,
    U32Stream,
    custom_next_uniform_int_u32,
    launch_uniform_int,
    launch_uniform_int_ex,
    uniform_int_host,
)
from ensemble.decisiontree.batched_levelalgo.random_utils import (
    fnv1a32_hash_seed_tree,
)
from max.gpu.host import DeviceContext

comptime ORACLE = "ensemble/bench/philox_oracle.txt"


@fieldwise_init
struct ScriptedGen(Copyable, Movable, U32Stream):
    """A stand-in for `PhiloxGenerator` that hands out a fixed list of draws.

    RAFT's `custom_next` is a template over `GenType` (`rng_device.cuh:174-175`),
    so substituting the source is their own parameterisation, not a rewrite of
    the function under test -- the loop, the threshold and the rejection test
    compared in layer 3 are the SAME code the kernel runs. Past the end of the
    list it repeats the last value, which is what makes an over-consuming port
    produce a wrong VALUE rather than an out-of-bounds read.
    """

    var xs: List[UInt32]
    var i: Int

    def next_u32(mut self) -> UInt32:
        var j = self.i if self.i < len(self.xs) else len(self.xs) - 1
        self.i += 1
        return self.xs[j]


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


def _i32(s: String) raises -> Int32:
    return Int32(Int(atol(s)))


def _u32(s: String) raises -> UInt32:
    return UInt32(Int(atol(s)))


def _u64(s: String) raises -> UInt64:
    """`atol` raises above Int64's range, and the table carries both
    18446744073709551615 (a subsequence) and 16045690984503098046
    (0xDEADBEEFCAFEBABE, a seed). Parsed digit by digit in UInt64, where they
    fit."""
    var v = UInt64(0)
    for i in range(s.byte_length()):
        var d = Int(atol(String(s[byte=i])))
        v = v * 10 + UInt64(d)
    return v


@always_inline
def _d2b(v: Float64) -> UInt64:
    var b = UInt64(0)
    var p = MutPointer(to=b).unsafe_bitcast[Float64]()
    p[unsafe_offset=0] = v
    return b


def _parse_u64(tok: String) raises -> UInt64:
    var v = UInt64(0)
    for i in range(tok.byte_length()):
        v = v * 10 + UInt64(Int(atol(String(tok[byte=i]))))
    return v


def _hex64_of(tok: String) raises -> UInt64:
    var slash = -1
    for i in range(tok.byte_length()):
        if String(tok[byte=i]) == "/":
            slash = i
            break
    if slash < 0:
        raise Error("token has no /hexbits: " + tok)
    var v = UInt64(0)
    for i in range(slash + 1, tok.byte_length()):
        var c = String(tok[byte=i])
        var d = 0
        if c >= "0" and c <= "9":
            d = Int(atol(c))
        else:
            d = 10 + (Int(ord(c)) - Int(ord("a")))
        v = v * 16 + UInt64(d)
    return v


def main() raises:
    print("philox_check: ensemble/mojo_only/philox.mojo")
    print(
        "  against", ORACLE, "-- RAFT v26.08.00 (ebf9268) over cuRAND's own",
        "Philox, compiled and run",
    )

    var text: String
    with open(ORACLE, "r") as f:
        text = f.read()
    var lines = text.split("\n")

    var ctr_rows = 0
    var ctr_wrong = 0
    var raw_rows = 0
    var raw_cells = 0
    var raw_wrong = 0
    var lem_rows = 0
    var lem_wrong = 0
    var uint_rows = 0
    var uint_cells = 0
    var uint_wrong = 0
    var fill_rows = 0
    var fill_cells = 0
    var fill_wrong = 0
    var e2e_rows = 0
    var e2e_cells = 0
    var e2e_wrong = 0
    var first_fail = String("")

    # Stashed for the sabotage arm: the first `raw` row with a NON-ZERO
    # subsequence, since a zero subsequence cannot distinguish the two counter
    # halves.
    var sab_seed = UInt64(0)
    var sab_sub = UInt64(0)
    var sab_want = List[UInt32]()

    # Stashed for the device arm, so the file is parsed once.
    var dev_seed = List[UInt64]()
    var dev_base = List[UInt64]()
    var dev_stride = List[Int]()
    var dev_start = List[Int32]()
    var dev_end = List[Int32]()
    var dev_vals = List[List[Int32]]()
    var dev_kind = List[Int]()  # 0 = fill (_ex), 1 = e2e via launch_uniform_int

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

        if kind == "ctr":
            # `ctr <seed> <sub> cx cy cz cw kx ky ox oy oz ow STATE`
            ctr_rows += 1
            var seed = _u64(t[1])
            var sub = _u64(t[2])
            var g = PhiloxState.init(seed, sub, UInt64(0))
            var got = List[UInt32]()
            got.append(g.ctr[0])
            got.append(g.ctr[1])
            got.append(g.ctr[2])
            got.append(g.ctr[3])
            got.append(g.key[0])
            got.append(g.key[1])
            got.append(g.output[0])
            got.append(g.output[1])
            got.append(g.output[2])
            got.append(g.output[3])
            got.append(g.state)
            for j in range(11):
                var want = _u32(t[3 + j])
                if got[j] != want:
                    ctr_wrong += 1
                    if first_fail == "":
                        first_fail = (
                            "ctr seed=" + String(seed) + " sub=" + String(sub)
                            + " field " + String(j) + " got " + String(got[j])
                            + " want " + String(want)
                        )

        elif kind == "raw":
            # `raw <seed> <sub> <count> v0 .. v(count-1)`
            raw_rows += 1
            var seed = _u64(t[1])
            var sub = _u64(t[2])
            var cnt = Int(atol(t[3]))
            var g = PhiloxState.init(seed, sub, UInt64(0))
            if sub != 0 and len(sab_want) == 0:
                sab_seed = seed
                sab_sub = sub
                for j in range(cnt):
                    sab_want.append(_u32(t[4 + j]))
            for j in range(cnt):
                var want = _u32(t[4 + j])
                var got = g.next_u32()
                raw_cells += 1
                if got != want:
                    raw_wrong += 1
                    if first_fail == "":
                        first_fail = (
                            "raw seed=" + String(seed) + " sub=" + String(sub)
                            + " draw " + String(j) + " got " + String(got)
                            + " want " + String(want)
                        )

        elif kind == "lemire":
            # `lemire <s> <start> <nx> x0..x(nx-1) <val> <draws>`
            lem_rows += 1
            var s = _u32(t[1])
            var start = _i32(t[2])
            var nx = Int(atol(t[3]))
            var xs = List[UInt32]()
            for j in range(nx):
                xs.append(_u32(t[4 + j]))
            var want_val = _i32(t[4 + nx])
            var want_draws = Int(atol(t[5 + nx]))
            var g = ScriptedGen(xs=xs^, i=0)
            var got_val = custom_next_uniform_int_u32(g, start, s)
            var got_draws = g.i
            if got_val != want_val or got_draws != want_draws:
                lem_wrong += 1
                if first_fail == "":
                    first_fail = (
                        "lemire s=" + String(s) + " start=" + String(start)
                        + " got (" + String(got_val) + "," + String(got_draws)
                        + ") want (" + String(want_val) + ","
                        + String(want_draws) + ")"
                    )

        elif kind == "uint":
            # `uint <seed> <sub> <start> <end> <count> v0 ..`
            uint_rows += 1
            var seed = _u64(t[1])
            var sub = _u64(t[2])
            var start = _i32(t[3])
            var end = _i32(t[4])
            var cnt = Int(atol(t[5]))
            var diff = end.cast[DType.uint32]() - start.cast[DType.uint32]()
            var g = PhiloxState.init(seed, sub, UInt64(0))
            for j in range(cnt):
                var want = _i32(t[6 + j])
                var got = custom_next_uniform_int_u32(g, start, diff)
                uint_cells += 1
                if got != want:
                    uint_wrong += 1
                    if first_fail == "":
                        first_fail = (
                            "uint seed=" + String(seed) + " sub=" + String(sub)
                            + " draw " + String(j) + " got " + String(got)
                            + " want " + String(want)
                        )

        elif kind == "fill":
            # `fill <seed> <base_sub> <stride> <len> <start> <end> v0 ..`
            fill_rows += 1
            var seed = _u64(t[1])
            var base = _u64(t[2])
            var stride = Int(atol(t[3]))
            var n = Int(atol(t[4]))
            var start = _i32(t[5])
            var end = _i32(t[6])
            var want = List[Int32]()
            for j in range(n):
                want.append(_i32(t[7 + j]))
            var got = uniform_int_host(seed, base, stride, n, start, end)
            for j in range(n):
                fill_cells += 1
                if got[j] != want[j]:
                    fill_wrong += 1
                    if first_fail == "":
                        first_fail = (
                            "fill seed=" + String(seed) + " stride="
                            + String(stride) + " i=" + String(j) + " got "
                            + String(got[j]) + " want " + String(want[j])
                        )
            if stride % RNG_BLOCK_THREADS == 0:
                dev_seed.append(seed)
                dev_base.append(base)
                dev_stride.append(stride)
                dev_start.append(start)
                dev_end.append(end)
                dev_vals.append(want^)
                dev_kind.append(0)

        elif kind == "e2e":
            # `e2e <seed> <tree_id> <n_rows> <n_sampled> <stride> <rs> v0 ..`
            e2e_rows += 1
            var seed = _u64(t[1])
            var tree_id = _i32(t[2])
            var n_rows = _i32(t[3])
            var n = Int(atol(t[4]))
            var stride = Int(atol(t[5]))
            var want_rs = _u32(t[6])
            var got_rs = fnv1a32_hash_seed_tree(seed, tree_id)
            if got_rs != want_rs:
                e2e_wrong += 1
                if first_fail == "":
                    first_fail = (
                        "e2e seed chain seed=" + String(seed) + " tree="
                        + String(tree_id) + " got " + String(got_rs)
                        + " want " + String(want_rs)
                    )
            var want = List[Int32]()
            for j in range(n):
                want.append(_i32(t[7 + j]))
            var got = uniform_int_host(
                UInt64(got_rs), UInt64(0), stride, n, Int32(0), n_rows
            )
            for j in range(n):
                e2e_cells += 1
                if got[j] != want[j]:
                    e2e_wrong += 1
                    if first_fail == "":
                        first_fail = (
                            "e2e seed=" + String(seed) + " tree="
                            + String(tree_id) + " i=" + String(j) + " got "
                            + String(got[j]) + " want " + String(want[j])
                        )
            if stride % RNG_BLOCK_THREADS == 0:
                dev_seed.append(UInt64(got_rs))
                dev_base.append(UInt64(0))
                dev_stride.append(stride)
                dev_start.append(Int32(0))
                dev_end.append(n_rows)
                dev_vals.append(want^)
                dev_kind.append(1 if stride == RNG_STRIDE else 0)

    print("  layer 1 ctr   :", ctr_rows, "inits x 11 fields,", ctr_wrong, "wrong")
    print(
        "  layer 2 raw   :", raw_rows, "streams,", raw_cells, "draws,",
        raw_wrong, "wrong",
    )
    print(
        "  layer 3 lemire:", lem_rows, "ranges, value AND draw count,",
        lem_wrong, "wrong",
    )
    print(
        "  layer 4 uint  :", uint_rows, "streams,", uint_cells, "values,",
        uint_wrong, "wrong",
    )
    print(
        "  layer 5 fill  :", fill_rows, "arrays,", fill_cells, "indices,",
        fill_wrong, "wrong",
    )
    print(
        "  layer 6 e2e   :", e2e_rows, "cuML call sites,", e2e_cells, "rows,",
        e2e_wrong, "wrong",
    )

    var parsed = ctr_rows + raw_rows + lem_rows + uint_rows + fill_rows + e2e_rows
    # ---- layer 8: raft::random::uniform<double> -------------------------
    # next_u64's LOW-WORD-FIRST pairing, next_double's (>>11)/2^53, and
    # custom_next's multiply-by-span-then-add-start, each a separable
    # failure mode. Doubles compared as RAW BITS.
    var udbl_rows = 0
    var udbl_cells = 0
    var udbl_wrong = 0
    var u64_wrong = 0
    for li in range(len(lines)):
        var line = String(lines[li])
        if line.byte_length() < 4:
            continue
        var t = _split_ws(line)
        if len(t) == 0:
            continue
        if t[0] == "u64":
            var g = PhiloxState.init(UInt64(12345), UInt64(0), UInt64(0))
            for j in range(6):
                var want = _parse_u64(t[3 + j])
                if philox_next_u64(g) != want:
                    u64_wrong += 1
        elif t[0] == "udbl":
            udbl_rows += 1
            var seed = _parse_u64(t[1])
            var sub = _parse_u64(t[2])
            var lo = _hex64_of(t[5])
            var g2 = PhiloxState.init(seed, sub, UInt64(0))
            # start/end are recoverable from the first row token pair; the
            # oracle prints them in decimal, and both are exactly
            # representable in every case it uses.
            var start = Float64(atof(t[3]))
            var end = Float64(atof(t[4]))
            for j in range(8):
                var want_bits = _hex64_of(t[5 + j])
                var got = custom_next_uniform_double(g2, start, end)
                udbl_cells += 1
                if _d2b(got) != want_bits:
                    udbl_wrong += 1
                    if first_fail == "":
                        first_fail = (
                            "udbl seed=" + String(seed) + " i=" + String(j)
                            + " got bits " + String(_d2b(got))
                            + " want " + String(want_bits)
                        )
            _ = lo
    print(
        "  layer 8 udbl  :", udbl_rows, "streams,", udbl_cells, "values,",
        udbl_wrong, "wrong;  next_u64 pairing:", u64_wrong, "wrong",
    )

    var total = udbl_wrong + u64_wrong + (
        ctr_wrong + raw_wrong + lem_wrong + uint_wrong + fill_wrong + e2e_wrong
    )

    # A check that parsed nothing also reports nothing wrong. Refuse that.
    if parsed < 60:
        raise Error(
            "philox_check: parsed only " + String(parsed) + " oracle rows;"
            " the table is missing or the parser is broken, and a check that"
            " reads nothing cannot fail"
        )
    if len(sab_want) == 0:
        raise Error(
            "philox_check: no `raw` row with a non-zero subsequence was"
            " parsed, so the sabotage arm has nothing to run against"
        )

    if total != 0:
        print("  FIRST FAILURE:", first_fail)
        raise Error(
            "philox_check: " + String(total) + " mismatches against RAFT"
        )

    # --- the sabotage: prove the comparison loop is live -------------------
    # Not required by the usual rule -- the expected values are the
    # INCUMBENT'S own per-cell output. This arm exists only to show that the
    # parser and the comparison are wired, since a check that read zero rows
    # would also report zero mismatches.
    #
    # The break chosen is the single most plausible one in the whole port:
    # putting the subsequence in the LOW half of the 128-bit counter
    # (`Philox_State_Incr`) instead of the HIGH half
    # (`Philox_State_Incr_hi`), which is what `curand_init` -> `skipahead_
    # sequence` actually does. A generator broken that way is still a perfectly
    # good uniform generator; it just gives every thread the wrong stream.
    var ok = PhiloxState.init(sab_seed, sab_sub, UInt64(0))
    var sab = PhiloxState(
        ctr=SIMD[DType.uint32, 4](0, 0, 0, 0),
        key=SIMD[DType.uint32, 2](
            UInt32(sab_seed & 0xFFFFFFFF),
            UInt32((sab_seed >> 32) & 0xFFFFFFFF),
        ),
        output=SIMD[DType.uint32, 4](0, 0, 0, 0),
        state=0,
    )
    sab._incr_n(sab_sub)  # THE SABOTAGE: low half instead of high half
    sab._regen()
    sab.skipahead(UInt64(0))

    var ok_wrong = 0
    var sab_wrong = 0
    for j in range(len(sab_want)):
        if ok.next_u32() != sab_want[j]:
            ok_wrong += 1
        if sab.next_u32() != sab_want[j]:
            sab_wrong += 1
    if ok_wrong != 0:
        raise Error("philox_check: sabotage control arm disagrees with itself")
    if sab_wrong == 0:
        raise Error(
            "philox_check SABOTAGE FAILED: putting the subsequence in the LOW"
            " counter half changed nothing, so this check cannot see which"
            " counter word the subsequence lands in"
        )
    print(
        "  sabotage OK: subsequence into ctr.x/ctr.y instead of ctr.z/ctr.w"
        " moved", sab_wrong, "of", len(sab_want), "draws (seed",
        sab_seed, "sub", sab_sub, ") -- the comparison is live",
    )

    # --- arm 7: THE KERNEL, ENQUEUED --------------------------------------
    var ctx = DeviceContext()
    var dev_cells = 0
    var dev_wrong = 0
    var dev_launches = 0
    var dev_via_entry = 0
    for c in range(len(dev_stride)):
        var n = len(dev_vals[c])
        var d_out = ctx.enqueue_create_buffer[DType.int32](n)
        ctx.enqueue_memset(d_out, Int32(-1))
        ctx.synchronize()
        if dev_kind[c] == 1:
            # THE BRIEFED ENTRY POINT, exactly as `randomforest.cuh:140-142`
            # calls it: base subsequence 0, the shipped stride.
            launch_uniform_int(
                ctx, d_out, n, dev_start[c], dev_end[c], dev_seed[c]
            )
            dev_via_entry += 1
        else:
            launch_uniform_int_ex(
                ctx,
                d_out,
                n,
                dev_start[c],
                dev_end[c],
                dev_seed[c],
                dev_base[c],
                dev_stride[c],
            )
        dev_launches += 1
        var h_out = ctx.enqueue_create_host_buffer[DType.int32](n)
        ctx.enqueue_copy(dst_buf=h_out, src_buf=d_out)
        ctx.synchronize()
        var p = h_out.unsafe_ptr()
        for j in range(n):
            dev_cells += 1
            if p.unsafe_load(j) != dev_vals[c][j]:
                dev_wrong += 1
                if first_fail == "":
                    first_fail = (
                        "kernel case " + String(c) + " stride="
                        + String(dev_stride[c]) + " i=" + String(j) + " got "
                        + String(p.unsafe_load(j)) + " want "
                        + String(dev_vals[c][j])
                    )
        _ = d_out^
        _ = h_out^

    if dev_launches == 0 or dev_via_entry == 0:
        raise Error(
            "philox_check arm 7: nothing was enqueued (" + String(dev_launches)
            + " launches, " + String(dev_via_entry) + " through"
            " launch_uniform_int); a kernel is not ported until it has run"
        )
    if dev_wrong != 0:
        print("  FIRST FAILURE:", first_fail)
        raise Error(
            "philox_check arm 7 (kernel): " + String(dev_wrong)
            + " wrong cells of " + String(dev_cells)
        )
    print(
        "  layer 7 kernel: ENQUEUED,", dev_launches, "launches (",
        dev_via_entry, "through launch_uniform_int itself ),", dev_cells,
        "device-written indices, 0 wrong",
    )

    print(
        "philox_check: ALL",
        ctr_rows * 11 + raw_cells + lem_rows + uint_cells + fill_cells
        + e2e_cells + dev_cells,
        "CELLS MATCH RAFT",
    )
