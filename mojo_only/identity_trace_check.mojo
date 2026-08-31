# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The stage-hash instrument, gated as an instrument rather than as code.

    pixi run check-identity-trace

NO CATBOOST COUNTERPART: a gate, so `mojo_only/`.

WHAT IS UNDER TEST. `core/identity_trace.mojo`, the facility that makes a fit
emit named stage hashes so `tools/identity_trace_diff.py` can say WHICH STAGE
first differs between two backends instead of only "the models differ".

WHY THIS FILE IS NOT OPTIONAL, and the standard it is held to. An instrument
that is wrong is worse than no instrument: it sends the reader to the wrong
stage and every hour after that is spent in the wrong file. So the gates here
are not "does it produce output" -- they are the four ways this particular
instrument could lie.

  T1  SELF-CONSISTENCY. Two runs of the SAME work must produce byte-identical
      trace files. An instrument with any noise of its own reports divergence
      that is its own, and there would be no way to tell that from a real
      one. Run twice, compare the files line for line.

  T2  LOCALIZATION, and this is the one that matters. Perturb ONE cell of ONE
      buffer at ONE stage and the trace must differ AT THAT STAGE AND NOT
      BEFORE. A hash chain that folded the previous record in would flag the
      first record after the perturbation AND everything before it would
      still match -- but a hash that accidentally depended on the whole file
      would flag stage 0. This is `[[verify-reach-not-output]]` applied to an
      instrument: sabotage the path and watch the report move to it.

  T3  IT HASHES CONTENTS, NOT SCHEDULING. The same buffer contents recorded
      after launches at THREE different grid widths must give the same hash.
      If a grid shape could move it, every cross-backend comparison would be
      noise, because grid shape is exactly what differs between backends.
      This is rule 2 of the file's header, tested rather than asserted.

  T4  IT CAN SEE WHAT IT EXISTS TO FIND. A denormal on one side and a zero on
      the other MUST hash differently -- that is the Metal-flushes /
      CUDA-honors divergence of IDENTITY_PATHS row 10, and an instrument that
      normalized it away would report agreement across the single most likely
      real cross-vendor difference. Same for two NaNs with different
      payloads. Both are planted as raw bit patterns.

  T5  THE DUMP AGREES WITH THE HASH. With `dump_match` set, the `.bin` file's
      bytes must re-hash to the recorded hash. The differ recomputes exactly
      this to decide whether it can trust a cell-level comparison; if the
      writer and the reader disagreed, every conclusion drawn from a dump
      would be void.

WHAT IT DOES NOT TEST. Whether the CHECKPOINTS ARE IN THE RIGHT PLACES. A
trace can be self-consistent, localizing, content-only and honest about
denormals while checkpointing the wrong buffers entirely. That judgement
lives with each call site and with `LOSSGUIDE.md`'s stage list.
"""

from max.gpu.host import DeviceContext
from std.memory import bitcast

from core.identity_trace import (
    FNV_OFFSET,
    IdentityTrace,
    first_divergence,
    fnv1a64_bytes,
    read_trace_lines,
)
from gbdt.methods.greedy_subsets_searcher.kernel.compute_scores import (
    LEAFWISE_SCORE_BLOCK_SIZE,
    compute_optimal_split_kernel,
)
from gbdt.options.catboost_options import SCORE_FUNCTION_COSINE


comptime N_BF = 24
comptime STAT_COUNT = 3
comptime N_LEAVES = 4


def hist_cell(l: Int, st: Int, b: Int) -> Int:
    return l * STAT_COUNT * N_BF + st * N_BF + b


def run_pipeline(
    ctx: DeviceContext,
    trace_path: String,
    dump_match: String,
    grid_x: Int,
    poison_cell: Int,
    poison_bits: UInt32,
) raises:
    """One traced run of a real device pipeline: upload a planted histogram,
    score it with the Lossguide kernel, checkpoint at three stages.

    `poison_cell >= 0` overwrites one histogram cell with `poison_bits`
    BEFORE the upload, which is T2's sabotage. The bits go in as a bit
    pattern rather than as a float so T4 can plant a denormal or a NaN
    payload through the same door.
    """
    var tr = IdentityTrace.to_path(trace_path, dump_match)
    tr.header("identity_trace_check, planted fixture")

    var n_cells = N_LEAVES * STAT_COUNT * N_BF
    var h_hist = ctx.enqueue_create_host_buffer[DType.float32](n_cells)
    var h_ps = ctx.enqueue_create_host_buffer[DType.float32](
        N_LEAVES * STAT_COUNT
    )
    var h_skip = ctx.enqueue_create_host_buffer[DType.uint8](N_BF)
    var h_fid = ctx.enqueue_create_host_buffer[DType.uint32](N_BF)
    var h_fw = ctx.enqueue_create_host_buffer[DType.float32](N_BF)

    for b in range(N_BF):
        h_skip.unsafe_ptr().unsafe_store(b, UInt8(0))
        h_fid.unsafe_ptr().unsafe_store(b, UInt32(b))
        h_fw.unsafe_ptr().unsafe_store(b, Float32(1.0))
    for l in range(N_LEAVES):
        var total_w = Float32(30.0) + Float32(l) * Float32(5.0)
        h_ps.unsafe_ptr().unsafe_store(l * STAT_COUNT, total_w)
        for b in range(N_BF):
            h_hist.unsafe_ptr().unsafe_store(
                hist_cell(l, 0, b),
                total_w * Float32(b + 1) / Float32(N_BF + 2),
            )
        for st in range(1, STAT_COUNT):
            var tot = Float32(1.0 + Float32(l) - Float32(st) * 0.7)
            h_ps.unsafe_ptr().unsafe_store(l * STAT_COUNT + st, tot)
            for b in range(N_BF):
                h_hist.unsafe_ptr().unsafe_store(
                    hist_cell(l, st, b),
                    tot * Float32(b + 1) / Float32(N_BF + 2),
                )

    if poison_cell >= 0:
        h_hist.unsafe_ptr().unsafe_store(
            poison_cell, bitcast[DType.float32](poison_bits)
        )

    var d_hist = ctx.enqueue_create_buffer[DType.float32](n_cells)
    var d_ps = ctx.enqueue_create_buffer[DType.float32](
        N_LEAVES * STAT_COUNT
    )
    var d_skip = ctx.enqueue_create_buffer[DType.uint8](N_BF)
    var d_fid = ctx.enqueue_create_buffer[DType.uint32](N_BF)
    var d_fw = ctx.enqueue_create_buffer[DType.float32](N_BF)
    var d_score = ctx.enqueue_create_buffer[DType.float32](grid_x * 2)
    var d_bin = ctx.enqueue_create_buffer[DType.uint32](grid_x * 2)

    ctx.enqueue_copy(dst_buf=d_hist, src_ptr=h_hist.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_ps, src_ptr=h_ps.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_skip, src_ptr=h_skip.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_fid, src_ptr=h_fid.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_fw, src_ptr=h_fw.unsafe_ptr())
    ctx.synchronize()

    # STAGE 1, the inputs. Checkpointing the input as well as the output is
    # what makes the report actionable: a divergence at `hist` is a
    # different bug from a divergence at `score` on identical `hist`.
    tr.record_device[DType.float32](ctx, "fixture.hist", d_hist)
    tr.record_device[DType.float32](ctx, "fixture.part_stats", d_ps)

    ctx.enqueue_function[compute_optimal_split_kernel[SCORE_FUNCTION_COSINE]](
        d_skip.unsafe_ptr(),
        Int32(N_BF),
        d_fid.unsafe_ptr(),
        d_fw.unsafe_ptr(),
        d_hist.unsafe_ptr(),
        d_ps.unsafe_ptr(),
        Int32(STAT_COUNT),
        Int32(1),
        Int32(2),
        Int32(0),
        Float32(1.5),
        Float32(0.0),
        UInt64(0),
        d_score.unsafe_ptr(),
        d_bin.unsafe_ptr(),
        grid_dim=(grid_x, 2, 1),
        block_dim=(LEAFWISE_SCORE_BLOCK_SIZE, 1, 1),
    )
    ctx.synchronize()

    # STAGE 2, the output -- but ONLY the reduced part. `d_score` is
    # `grid_x * 2` long and grid_x is a SCHEDULING row, so hashing all of it
    # would make the tag's meaning depend on the machine, which is rule 3 of
    # the facility's header. What is machine-independent is the per-block-row
    # WINNER, so the host reduce runs first and the trace records that.
    var h_s = ctx.enqueue_create_host_buffer[DType.float32](grid_x * 2)
    var h_b = ctx.enqueue_create_host_buffer[DType.uint32](grid_x * 2)
    ctx.enqueue_copy(dst_ptr=h_s.unsafe_ptr(), src_buf=d_score)
    ctx.enqueue_copy(dst_ptr=h_b.unsafe_ptr(), src_buf=d_bin)
    ctx.synchronize()

    var best_gain = List[Float32]()
    var best_bin = List[Int32]()
    for row in range(2):
        var bg = Float32(-3.4028234663852886e38)
        var bb = Int32(-1)
        for i in range(grid_x):
            var slot = row * grid_x + i
            var g = h_s.unsafe_ptr().unsafe_load(slot)
            var bn = Int32(h_b.unsafe_ptr().unsafe_load(slot))
            if bn == Int32(-1):
                continue
            var take = g > bg
            if g == bg and bn < bb:
                take = True
            if take:
                bg = g
                bb = bn
        best_gain.append(bg)
        best_bin.append(bb)
    tr.record_list_f32("score.best_gain", best_gain)
    tr.record_list_i32("score.best_bin", best_bin)


def check_identity_trace(ctx: DeviceContext) raises:
    var failures = 0
    var dir = String("/private/tmp/claude-501/identity_trace_check")
    _ = dir

    print("-- T1: two identical runs give byte-identical traces --")
    var a = String("/tmp/mojolearn_it_a.trace")
    var b = String("/tmp/mojolearn_it_b.trace")
    run_pipeline(ctx, a, "", 3, -1, UInt32(0))
    run_pipeline(ctx, b, "", 3, -1, UInt32(0))
    var la = read_trace_lines(a)
    var lb = read_trace_lines(b)
    if len(la) != len(lb):
        print("  FAIL line counts differ:", len(la), "vs", len(lb))
        failures += 1
    else:
        var same = True
        for i in range(len(la)):
            if la[i] != lb[i]:
                print("  FAIL line", i, "differs:", la[i], "|", lb[i])
                same = False
                failures += 1
        if same:
            print("  ok  ", len(la), "records, byte identical")

    print()
    print("-- T2: a one-cell perturbation moves EXACTLY its own stage --")
    # Poison one gradient cell of LEAF 1, which the kernel's row 0 scores.
    var poisoned = String("/tmp/mojolearn_it_poison.trace")
    var cell = hist_cell(1, 1, 9)
    run_pipeline(
        ctx, poisoned, "", 3, cell, bitcast[DType.uint32](Float32(99.0))
    )
    var lp = read_trace_lines(poisoned)
    if len(lp) != len(la):
        print("  FAIL the poisoned run has a different record count")
        failures += 1
    else:
        var first_diff = -1
        var n_diff = 0
        for i in range(len(la)):
            if la[i] != lp[i]:
                n_diff += 1
                if first_diff < 0:
                    first_diff = i
        # Records are fixture.hist, fixture.part_stats, score.best_gain,
        # score.best_bin -- `read_trace_lines` drops the `#` provenance
        # lines, which is what makes them provenance and not records. The
        # perturbation is IN the histogram, so the first differing record
        # must be `fixture.hist`, and `part_stats` -- downstream of nothing
        # -- must still match.
        if first_diff < 0:
            print(
                "  FAIL the instrument is BLIND: a poisoned histogram cell"
                " moved no record at all"
            )
            failures += 1
        elif la[first_diff].find("fixture.hist") < 0:
            print(
                "  FAIL first divergence is at '", la[first_diff],
                "', want fixture.hist",
            )
            failures += 1
        else:
            var ps_moved = False
            for i in range(len(la)):
                if la[i].find("fixture.part_stats") >= 0 and la[i] != lp[i]:
                    ps_moved = True
            if ps_moved:
                print(
                    "  FAIL part_stats moved too; the hash is not per-record"
                )
                failures += 1
            else:
                print(
                    "  ok   first divergence at fixture.hist,", n_diff,
                    "records moved, part_stats untouched",
                )

    print()
    print("-- T3: the hash follows CONTENTS, not the grid width --")
    var widths = [1, 3, 8]
    var hist_hashes = List[String]()
    for w in widths:
        var p = String("/tmp/mojolearn_it_w") + String(w) + ".trace"
        run_pipeline(ctx, p, "", w, -1, UInt32(0))
        var lines = read_trace_lines(p)
        for i in range(len(lines)):
            if lines[i].find("fixture.hist") >= 0:
                hist_hashes.append(lines[i])
    var stable = True
    for i in range(1, len(hist_hashes)):
        if hist_hashes[i] != hist_hashes[0]:
            stable = False
    if len(hist_hashes) != 3:
        print("  FAIL expected 3 hist records, got", len(hist_hashes))
        failures += 1
    elif not stable:
        print("  FAIL the hist hash moved with the grid width")
        failures += 1
    else:
        print("  ok   identical at grid widths 1, 3 and 8")

    print()
    print("-- T4: denormals and NaN payloads are VISIBLE --")
    # The whole point of row 10. A denormal against a zero must not hash the
    # same, or the instrument cannot see the one divergence most likely to
    # separate Metal from CUDA.
    var zero_run = String("/tmp/mojolearn_it_zero.trace")
    var denorm_run = String("/tmp/mojolearn_it_denorm.trace")
    var c = hist_cell(2, 2, 5)
    run_pipeline(ctx, zero_run, "", 3, c, UInt32(0x00000000))
    run_pipeline(ctx, denorm_run, "", 3, c, UInt32(0x00000001))
    var lz = read_trace_lines(zero_run)
    var ld = read_trace_lines(denorm_run)
    var denorm_seen = False
    for i in range(len(lz)):
        if lz[i].find("fixture.hist") >= 0 and lz[i] != ld[i]:
            denorm_seen = True
    if not denorm_seen:
        print(
            "  FAIL +0.0 and the smallest denormal hashed the SAME; the"
            " instrument cannot see an FTZ divergence"
        )
        failures += 1
    else:
        var nan_a = String("/tmp/mojolearn_it_nan_a.trace")
        var nan_b = String("/tmp/mojolearn_it_nan_b.trace")
        run_pipeline(ctx, nan_a, "", 3, c, UInt32(0x7FC00001))
        run_pipeline(ctx, nan_b, "", 3, c, UInt32(0x7FC00002))
        var na = read_trace_lines(nan_a)
        var nb = read_trace_lines(nan_b)
        var nan_seen = False
        for i in range(len(na)):
            if na[i].find("fixture.hist") >= 0 and na[i] != nb[i]:
                nan_seen = True
        if not nan_seen:
            print(
                "  FAIL two NaNs with different payloads hashed the same"
            )
            failures += 1
        else:
            print(
                "  ok   +0.0 vs denormal MOVES, and two NaN payloads MOVE"
            )

    print()
    print("-- T5: the .bin dump re-hashes to its recorded hash --")
    var dumped = String("/tmp/mojolearn_it_dump.trace")
    run_pipeline(ctx, dumped, "fixture.hist", 3, -1, UInt32(0))
    var dl = read_trace_lines(dumped)
    var checked = 0
    for i in range(len(dl)):
        if dl[i].find("fixture.hist") < 0:
            continue
        # `<seq>\\t<tag>\\t<dtype>\\t<count>\\t<hash>`
        var parts = dl[i].split("\t")
        var seq = String(parts[0])
        var want = String(parts[4])
        var bin_path = dumped + "." + seq + ".fixture.hist.bin"
        var raw = List[UInt8]()
        with open(bin_path, "r") as fh:
            var body = fh.read_bytes()
            for k in range(len(body)):
                raw.append(body[k])
        var got = fnv1a64_bytes(FNV_OFFSET, raw.unsafe_ptr(), len(raw))
        var got_hex = String("")
        comptime DIGITS = "0123456789abcdef"
        for k in range(16):
            var nib = Int((got >> UInt64(60 - 4 * k)) & UInt64(0xF))
            got_hex += String(DIGITS[byte=nib])
        if got_hex != want:
            print(
                "  FAIL dump for seq", seq, "hashes", got_hex, "but the"
                " record says", want,
            )
            failures += 1
        else:
            checked += 1
        _ = raw^
    if checked == 0:
        print("  FAIL no dump was written; the dump_match arm is UNREACHED")
        failures += 1
    elif failures == 0:
        print("  ok  ", checked, "dump verified against its record")

    print()
    print("-- artifacts for tools/identity_trace_diff.py --")
    # THE END-TO-END PAIR. The Mojo half of this gate proves the writer is
    # honest; it cannot prove the READER agrees with it. So the check leaves
    # behind a clean/poisoned pair WITH dumps, and `pixi run
    # check-identity-trace` runs the differ over them and requires it to name
    # `fixture.hist` and to reach the cell level. Two halves of one
    # instrument, gated together, because a writer and a reader that were
    # only ever tested apart are two programs and not one tool.
    var pair_clean = String("/tmp/mojolearn_it_pair_clean.trace")
    var pair_dirty = String("/tmp/mojolearn_it_pair_dirty.trace")
    var pair_cell = hist_cell(1, 1, 9)
    # THE PAIR IS +0.0 AGAINST THE SMALLEST DENORMAL, which is the SHAPE of
    # a real Metal-versus-CUDA divergence: Metal flushes, CUDA honors, so a
    # cell that is a denormal on one card is a zero on the other
    # (IDENTITY_PATHS row 10). Planting the denormal against the fixture's
    # own 0.5 instead -- which the first version of this block did -- makes
    # the differ report LARGE and exercises none of the classifier that
    # matters. Both sides are therefore poisoned, differently.
    run_pipeline(ctx, pair_clean, "fixture.hist", 3, pair_cell, UInt32(0x00000000))
    run_pipeline(ctx, pair_dirty, "fixture.hist", 3, pair_cell, UInt32(0x00000001))
    print("  wrote", pair_clean, "(+0.0 at cell", pair_cell, ")")
    print("  wrote", pair_dirty, "(smallest denormal at the same cell)")

    if failures != 0:
        raise Error(
            "identity trace check: " + String(failures) + " failures"
        )
    print()
    print("identity trace check: PASS")


def main() raises:
    var ctx = DeviceContext()
    check_identity_trace(ctx)
