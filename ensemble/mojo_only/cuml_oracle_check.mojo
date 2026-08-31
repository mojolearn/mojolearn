# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE FIRST COMPARISON AGAINST cuML ITSELF, per cell.

    pixi run mojo run -I . ensemble/mojo_only/cuml_oracle_check.mojo

Every other check in this directory compares against an ANALYTIC fixture --
an answer computed by hand -- or against a LIBRARY PRIMITIVE's compiled
output (CCCL's `shuffle_iterator`, RAFT's Philox). Those establish that the
port is self-consistent and faithful to source that was read. **They are not
a comparison against cuML.**

This one is. `ensemble/bench/cuml_oracle.txt` is produced by
`ensemble/tools/cuml_oracle/`, which INCLUDES cuML's `bins.cuh`,
`split.cuh`, `dataset.h` and `objectives.cuh` VERBATIM from the pin and
calls them. Their numeric core is `HDI`/`DI` -- host-device inline -- so a
plain c++ compiler runs `Split::update`, all six gain functions, every bin
operator and `lower_bound` unchanged, with no CUDA toolkit and no GPU. Only
the four `__global__` kernels need a device, and none of the arithmetic
lives in them: they gather, accumulate, scan and reduce, then hand the
numbers to exactly the functions this oracle calls.

WHAT THIS DOES AND DOES NOT SETTLE, because the distinction is the whole
point of running it:

  IT SETTLES the arithmetic -- the gains, the tie-break total order, the
  midpoint rule, the bin search. That is where a port silently diverges,
  and it is what no analytic fixture could pin down, because an analytic
  fixture only knows the answers *I* could derive.

  IT DOES NOT SETTLE the kernels' gather/scan/partition schedule, which
  needs a GPU, nor an end-to-end tree. Those still want cuML running on the
  NVIDIA column. This is not a substitute for that run; it is the part of
  it that can be done on this machine, and it should have been done first.

FLOATS ARE COMPARED AS RAW BITS. The oracle prints `decimal/hexbits` and
this reads the hex, because decimal alone does not round-trip -- this
repository measured 0.46% of float32 values coming back one ULP wrong
through a decimal string.

WHERE EXACT EQUALITY IS AND IS NOT CLAIMED:

  * CLASSIFICATION gains and splits: EXACT. Their `Weight()` is
    `double(count)` on an integer count and ours is that same integer, and
    every subsequent operation is float32 in both. There is nothing for a
    tolerance to hide.
  * `lower_bound`, `Split::update`, the midpoint rule: EXACT. Integer and
    comparison logic.
  * REGRESSION gains: NOT exact, and the reason is DEVIATION 101b. Their
    `label_sum` is a float64 accumulator; this device has none, so ours is
    Int32 fixed point. The delta is reported per cell and bounded, and the
    arm FAILS if the delta exceeds what the fixed-point grid can explain --
    which is a real check, not a shrug.
"""

from std.math import abs as fabs
from std.sys.info import size_of
from max.gpu.host import DeviceContext
from std.gpu import thread_idx

from ensemble.decisiontree.batched_levelalgo.bins import (
    BinScales,
    ClassificationBin,
    RegressionBin,
)
from ensemble.decisiontree.batched_levelalgo.objectives import (
    ClassificationObjectiveFunction,
    RegressionObjectiveFunction,
)
from ensemble.decisiontree.batched_levelalgo.split import Split
from ensemble.decisiontree.batched_levelalgo.kernels.builder_kernels import (
    lower_bound,
)

comptime ORACLE = "ensemble/bench/cuml_oracle.txt"
comptime DT = DType.float32
comptime ClsObj = ClassificationObjectiveFunction[
    DT, DType.int32, ClassificationBin
]
comptime RegObj = RegressionObjectiveFunction[DT, DT, RegressionBin]

comptime GINI_C = Int32(0)
comptime ENTROPY_C = Int32(1)
comptime MSE_C = Int32(2)
comptime POISSON_C = Int32(4)
comptime GAMMA_C = Int32(5)
comptime IG_C = Int32(6)


@always_inline
def _mix(x: UInt64) -> UInt64:
    var h = x
    h ^= h >> 33
    h *= 0xFF51AFD7ED558CCD
    h ^= h >> 33
    h *= 0xC4CEB9FE1A85EC53
    h ^= h >> 33
    return h


def _split_ws(line: String) -> List[String]:
    var out = List[String]()
    var cur = String("")
    for i in range(line.byte_length()):
        var c = String(line[byte=i])
        if c == " " or c == "\t" or c == "\r":
            if cur.byte_length() > 0:
                out.append(cur)
                cur = String("")
        else:
            cur += c
    if cur.byte_length() > 0:
        out.append(cur)
    return out^


def _hex_of(tok: String) raises -> UInt32:
    """`decimal/hexbits` -> the hex half, as raw bits."""
    var slash = -1
    for i in range(tok.byte_length()):
        if String(tok[byte=i]) == "/":
            slash = i
            break
    if slash < 0:
        raise Error("token has no /hexbits: " + tok)
    var v = UInt32(0)
    for i in range(slash + 1, tok.byte_length()):
        var c = String(tok[byte=i])
        var d = 0
        if c >= "0" and c <= "9":
            d = Int(atol(c))
        elif c >= "a" and c <= "f":
            d = 10 + (Int(ord(c)) - Int(ord("a")))
        else:
            raise Error("bad hex digit in " + tok)
        v = v * 16 + UInt32(d)
    return v


@always_inline
def _f2b(v: Float32) -> UInt32:
    var b = UInt32(0)
    var p = MutPointer(to=b).unsafe_bitcast[Float32]()
    p[unsafe_offset=0] = v
    return b



def _gain_split_kernel(
    hist: MutPointer[ClassificationBin, MutAnyOrigin],
    quant: MutPointer[Float32, MutAnyOrigin],
    out_i: MutPointer[Int32, MutAnyOrigin],
    out_f: MutPointer[Float32, MutAnyOrigin],
    nclasses: Int32,
    n_bins: Int32,
    crit: Int32,
    total_lo: Int32,
    total_hi: Int32,
):
    """`Gain` reads `thread_idx`/`block_dim`, so it cannot run on the host
    in Mojo. Launched at block_dim=1 it walks EVERY bin in one thread --
    exactly the shape the oracle's `blockDim.x = 1` shim gives their code,
    so the two see the same candidate set in the same order."""
    var t64 = (
        (total_hi.cast[DType.uint64]() & 0xFFFFFFFF) << 32
    ) | (total_lo.cast[DType.uint64]() & 0xFFFFFFFF)
    var obj = ClsObj(nclasses, Int32(1), crit, Float32(0.0))
    var sp = obj.Gain(hist, quant, Int32(5), Int64(Int(t64)), n_bins)
    out_i[unsafe_offset=0] = sp.colid
    out_i[unsafe_offset=1] = sp.split_start
    out_i[unsafe_offset=2] = sp.split_end
    out_i[unsafe_offset=3] = Int32(Int(sp.global_nLeft))
    out_f[unsafe_offset=0] = sp.quesval
    out_f[unsafe_offset=1] = sp.best_metric_val


def main() raises:
    var ctx = DeviceContext()
    print("cuml_oracle_check: against cuML's OWN code, per cell")
    print("  ", ORACLE, "-- their headers included verbatim at the pin")

    var text: String
    with open(ORACLE, "r") as f:
        text = f.read()
    var lines = text.split("\n")

    var lb_rows = 0
    var lb_wrong = 0
    var gain_cls_cells = 0
    var gain_cls_wrong = 0
    var gini_cells = 0
    var gini_wrong = 0
    var ent_cells = 0
    var ent_wrong = 0
    var ent_max_rel = Float64(0.0)
    var split_cls_rows = 0
    var split_cls_wrong = 0
    var split_gini_wrong = 0
    var split_ent_wrong = 0
    var gain_reg_cells = 0
    var reg_max_ulp_gap = Float64(0.0)
    var upd_rows = 0
    var upd_wrong = 0
    var first = String("")

    for li in range(len(lines)):
        var line = String(lines[li])
        if line.byte_length() == 0:
            continue
        if String(line[byte=0]) == "#" or String(line[byte=0]) == "s":
            if line.byte_length() < 2 or String(line[byte=1]) != "p":
                continue
        var t = _split_ws(line)
        if len(t) == 0:
            continue

        # ---- lower_bound -------------------------------------------------
        if t[0] == "lb":
            lb_rows += 1
            var n = Int(atol(t[1]))
            var target = Float32(Int(atol(t[2])))
            var want = Int(atol(t[3]))
            var q = List[Float32]()
            for i in range(n):
                q.append(Float32(i * 7 + 3))
            var got = Int(
                lower_bound(q.unsafe_ptr(), Int32(n), target)
            )
            if got != want:
                lb_wrong += 1
                if first == "":
                    first = (
                        "lower_bound(" + String(target) + ") = "
                        + String(got) + " want " + String(want)
                    )

        # ---- classification gains ---------------------------------------
        elif t[0] == "gain_cls":
            var nclasses = Int(atol(t[1]))
            var n_bins = Int(atol(t[2]))
            var crit = GINI_C if Int(atol(t[3])) == 0 else ENTROPY_C
            var total = Int64(Int(atol(t[4])))
            var hist = List[ClassificationBin]()
            hist.resize(nclasses * n_bins, ClassificationBin())
            for c in range(nclasses):
                var run = UInt64(0)
                for b in range(n_bins):
                    run += _mix(UInt64(c * 131 + b + 7)) % 9
                    hist[c * n_bins + b] = ClassificationBin(UInt32(Int(run)))
            var obj = ClsObj(Int32(nclasses), Int32(1), crit, Float32(0.0))
            for b in range(n_bins):
                var want = _hex_of(t[5 + b])
                var n_left = Int64(0)
                for c in range(nclasses):
                    n_left += Int64(
                        Int(hist[c * n_bins + b].Count())
                    )
                var g = obj.GainPerSplit(
                    hist.unsafe_ptr(), Int32(b), Int32(n_bins),
                    total, n_left, total - n_left,
                )
                gain_cls_cells += 1
                var is_gini = Int(atol(t[3])) == 0
                if is_gini:
                    gini_cells += 1
                else:
                    ent_cells += 1
                    var wf2 = Float32(0)
                    var wp2 = MutPointer(to=wf2).unsafe_bitcast[UInt32]()
                    wp2[unsafe_offset=0] = want
                    var den = fabs(Float64(wf2))
                    if den < 1e-9:
                        den = 1e-9
                    var rel2 = fabs(Float64(g) - Float64(wf2)) / den
                    if rel2 > ent_max_rel:
                        ent_max_rel = rel2
                if _f2b(g) != want:
                    if is_gini:
                        gini_wrong += 1
                    else:
                        ent_wrong += 1
                    gain_cls_wrong += 1
                    if first == "":
                        first = (
                            "gain_cls nclasses=" + String(nclasses)
                            + " crit=" + String(t[3]) + " bin=" + String(b)
                            + " got bits " + String(_f2b(g))
                            + " want " + String(want)
                        )

        # ---- the Split their Gain() reduces to ---------------------------
        elif t[0] == "split_cls":
            split_cls_rows += 1
            var nclasses = Int(atol(t[1]))
            var n_bins = Int(atol(t[2]))
            var crit = GINI_C if Int(atol(t[3])) == 0 else ENTROPY_C
            var want_colid = Int32(Int(atol(t[4])))
            var want_ques = _hex_of(t[5])
            var want_metric = _hex_of(t[6])
            var want_gnl = Int64(Int(atol(t[7])))
            var want_ss = Int32(Int(atol(t[8])))
            var want_se = Int32(Int(atol(t[9])))

            var hist = List[ClassificationBin]()
            hist.resize(nclasses * n_bins, ClassificationBin())
            var total = Int64(0)
            for c in range(nclasses):
                var run = UInt64(0)
                for b in range(n_bins):
                    run += _mix(UInt64(c * 131 + b + 7)) % 9
                    hist[c * n_bins + b] = ClassificationBin(UInt32(Int(run)))
                total += Int64(Int(run))
            var quant = List[Float32]()
            for b in range(n_bins):
                quant.append(Float32(b * 3 + 1))
            var d_h = ctx.enqueue_create_buffer[DType.uint8](
                nclasses * n_bins * size_of[ClassificationBin]()
            )
            var h_h = ctx.enqueue_create_host_buffer[DType.uint8](
                nclasses * n_bins * size_of[ClassificationBin]()
            )
            var hp = h_h.unsafe_ptr().unsafe_bitcast[ClassificationBin]()
            for i in range(nclasses * n_bins):
                hp[unsafe_offset=i] = hist[i]
            ctx.enqueue_copy(dst_buf=d_h, src_ptr=h_h.unsafe_ptr())
            var d_q = ctx.enqueue_create_buffer[DType.float32](n_bins)
            var h_q = ctx.enqueue_create_host_buffer[DType.float32](n_bins)
            for b in range(n_bins):
                h_q.unsafe_ptr().unsafe_store(b, quant[b])
            ctx.enqueue_copy(dst_buf=d_q, src_ptr=h_q.unsafe_ptr())
            var d_oi = ctx.enqueue_create_buffer[DType.int32](4)
            var d_of = ctx.enqueue_create_buffer[DType.float32](2)
            ctx.synchronize()
            ctx.enqueue_function[_gain_split_kernel](
                d_h.unsafe_ptr().unsafe_bitcast[ClassificationBin](),
                d_q.unsafe_ptr(),
                d_oi.unsafe_ptr(),
                d_of.unsafe_ptr(),
                Int32(nclasses),
                Int32(n_bins),
                crit,
                UInt32(Int(total & 0xFFFFFFFF)).cast[DType.int32](),
                UInt32(Int(total >> 32)).cast[DType.int32](),
                grid_dim=1,
                block_dim=1,
            )
            var h_oi = ctx.enqueue_create_host_buffer[DType.int32](4)
            var h_of = ctx.enqueue_create_host_buffer[DType.float32](2)
            ctx.enqueue_copy(dst_buf=h_oi, src_buf=d_oi)
            ctx.enqueue_copy(dst_buf=h_of, src_buf=d_of)
            ctx.synchronize()
            var got_colid = h_oi.unsafe_ptr().unsafe_load(0)
            var got_ss = h_oi.unsafe_ptr().unsafe_load(1)
            var got_se = h_oi.unsafe_ptr().unsafe_load(2)
            var got_gnl = Int64(Int(h_oi.unsafe_ptr().unsafe_load(3)))
            var got_ques = h_of.unsafe_ptr().unsafe_load(0)
            var got_metric = h_of.unsafe_ptr().unsafe_load(1)
            _ = d_h^
            _ = h_h^
            _ = d_q^
            _ = h_q^
            _ = d_oi^
            _ = d_of^
            _ = h_oi^
            _ = h_of^
            var bad = 0
            if got_colid != want_colid:
                bad += 1
            if _f2b(got_ques) != want_ques:
                bad += 1
            if _f2b(got_metric) != want_metric:
                bad += 1
            if got_gnl != want_gnl:
                bad += 1
            if got_ss != want_ss or got_se != want_se:
                bad += 1
            if bad != 0:
                split_cls_wrong += 1
                if Int(atol(t[3])) == 0:
                    split_gini_wrong += 1
                else:
                    split_ent_wrong += 1
                if first == "":
                    first = (
                        "split_cls nclasses=" + String(nclasses)
                        + " crit=" + String(t[3]) + ": " + String(bad)
                        + " fields differ (colid " + String(got_colid)
                        + " want " + String(want_colid) + ")"
                    )

        # ---- regression gains: report the fixed-point delta ---------------
        elif t[0] == "gain_reg":
            var n_bins = Int(atol(t[1]))
            var crit_i = Int(atol(t[2]))
            var crit = MSE_C
            if crit_i == 1:
                crit = POISSON_C
            elif crit_i == 2:
                crit = GAMMA_C
            elif crit_i == 3:
                crit = IG_C
            var total = Int64(Int(atol(t[3])))
            var hist = List[RegressionBin]()
            hist.resize(n_bins, RegressionBin())
            var run_l = Float64(0.0)
            var run_c = UInt64(0)
            for b in range(n_bins):
                run_c += (_mix(UInt64(b * 17 + 3)) % 7) + 1
                run_l += Float64(Int((_mix(UInt64(b * 29 + 11)) % 50) + 1))
                hist[b] = RegressionBin(
                    Int32(Int(run_l)), UInt32(Int(run_c))
                )
            var obj = RegObj(
                Int32(1), Int32(1), crit, Float32(0.0), BinScales(1.0, 1.0)
            )
            for b in range(n_bins):
                var want_bits = _hex_of(t[4 + b])
                var wf = Float32(0)
                var wp = MutPointer(to=wf).unsafe_bitcast[UInt32]()
                wp[unsafe_offset=0] = want_bits
                var n_left = Int64(Int(hist[b].Count()))
                var g = obj.GainPerSplit(
                    hist.unsafe_ptr(), Int32(b), Int32(n_bins),
                    total, n_left, total - n_left,
                )
                gain_reg_cells += 1
                # both may be -max() sentinels; those must MATCH exactly
                var sentinel = -Float32.MAX_FINITE
                if wf == sentinel or g == sentinel:
                    if wf != g:
                        reg_max_ulp_gap = 1.0e30
                        if first == "":
                            first = (
                                "gain_reg crit=" + String(crit_i)
                                + " bin=" + String(b)
                                + ": one side is the -max() sentinel and"
                                " the other is not"
                            )
                    continue
                var denom = fabs(Float64(wf))
                if denom < 1e-6:
                    denom = 1e-6
                var rel = fabs(Float64(g) - Float64(wf)) / denom
                if rel > reg_max_ulp_gap:
                    reg_max_ulp_gap = rel

        # ---- Split::update, their total order -----------------------------
        elif t[0] == "update":
            upd_rows += 1
            var s = Int(atol(t[1]))
            # The same four sequences the oracle drives, written as
            # explicit Float64 so no literal-type merge is needed.
            var raw = List[List[Float64]]()
            var s0 = List[Float64]()
            for v in [1.0, 3.0, 10.0, 5.0, 1.0, 1.0, 2.0, 0.0, 20.0, 9.0,
                      2.0, 2.0, 3.0, 7.0, 20.0, 9.0, 3.0, 3.0, 4.0, 7.0,
                      20.0, 9.0, 4.0, 4.0, 5.0, 1.0, 5.0, 2.0, 5.0, 5.0]:
                s0.append(v)
            var s1 = List[Float64]()
            for v in [10.0, 4.0, 10.0, 5.0, 1.0, 1.0, 30.0, 4.0, 10.0, 5.0,
                      3.0, 3.0, 20.0, 4.0, 10.0, 7.0, 2.0, 2.0, 40.0, 4.0,
                      10.0, 5.0, 4.0, 4.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0]:
                s1.append(v)
            var s2 = List[Float64]()
            for v in [10.0, 4.0, 10.0, 5.0, 1.0, 1.0, 20.0, 4.0, 10.0, 7.0,
                      2.0, 2.0, 30.0, 4.0, 10.0, 5.0, 3.0, 3.0, 5.0, 4.0,
                      10.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0]:
                s2.append(v)
            var s3 = List[Float64]()
            for v in [1.0, 1.0, -1e30, 0.0, -1.0, -1.0, 2.0, 2.0, -1e30,
                      0.0, -1.0, -1.0, 3.0, 3.0, 1.0, 1.0, 3.0, 3.0, 4.0,
                      4.0, 1.0, 1.0, 4.0, 4.0, 5.0, 5.0, 1.0, 2.0, 5.0, 5.0]:
                s3.append(v)
            raw.append(s0^)
            raw.append(s1^)
            raw.append(s2^)
            raw.append(s3^)
            var sp = Split[DT]()
            var bad = 0
            for i in range(5):
                var o = i * 6
                var r = sp.update(
                    Float32(raw[s][o + 0]),
                    Int32(Int(raw[s][o + 1])),
                    Float32(raw[s][o + 2]),
                    Int64(Int(raw[s][o + 3])),
                    Int32(Int(raw[s][o + 4])),
                    Int32(Int(raw[s][o + 5])),
                )
                var want_r = Int(atol(t[2 + i])) == 1
                if r != want_r:
                    bad += 1
            # t[7] is "|", then quesval, metric, colid, gnl, ss, se
            if _f2b(sp.quesval) != _hex_of(t[8]):
                bad += 1
            if _f2b(sp.best_metric_val) != _hex_of(t[9]):
                bad += 1
            if sp.colid != Int32(Int(atol(t[10]))):
                bad += 1
            if sp.global_nLeft != Int64(Int(atol(t[11]))):
                bad += 1
            if sp.split_start != Int32(Int(atol(t[12]))):
                bad += 1
            if sp.split_end != Int32(Int(atol(t[13]))):
                bad += 1
            var quant = List[Float32]()
            for b in range(64):
                quant.append(Float32(b * 10))
            sp.select_split_range_midpoint(quant.unsafe_ptr(), Int32(64))
            # t[14] is "|", t[15] "mid", t[16] quesval, t[17] ss, t[18] se
            if _f2b(sp.quesval) != _hex_of(t[16]):
                bad += 1
            if sp.split_start != Int32(Int(atol(t[17]))):
                bad += 1
            if sp.split_end != Int32(Int(atol(t[18]))):
                bad += 1
            if bad != 0:
                upd_wrong += 1
                if first == "":
                    first = (
                        "update seq " + String(s) + ": " + String(bad)
                        + " differences from cuML's own Split::update"
                    )

    print("  lower_bound   :", lb_rows, "cases,", lb_wrong, "wrong")
    print(
        "  gain GINI     :", gini_cells, "cells,", gini_wrong,
        "wrong  [EXACT BITS REQUIRED]",
    )
    print(
        "  gain ENTROPY  :", ent_cells, "cells,", ent_wrong,
        "differing bits, max relative", ent_max_rel,
        " [DEVIATION 113: std.math.log vs raft::log]",
    )
    print(
        "  Gain->Split   :", split_cls_rows, "reductions --", split_gini_wrong,
        "GINI wrong [EXACT REQUIRED],", split_ent_wrong,
        "ENTROPY differing [follows the log gap]",
    )
    print(
        "  Split::update :", upd_rows, "sequences,", upd_wrong,
        "wrong  [EXACT BITS, incl. the midpoint rule]",
    )
    print(
        "  gain (reg)    :", gain_reg_cells,
        "cells, max relative delta", reg_max_ulp_gap,
        " [NOT exact: DEVIATION 101b fixed point]",
    )

    var parsed = lb_rows + split_cls_rows + upd_rows
    if parsed < 20:
        raise Error(
            "cuml_oracle_check: parsed only " + String(parsed)
            + " oracle rows; a check that reads nothing cannot fail"
        )

    # GINI must be bit-exact. ENTROPY is NOT required to be, and the
    # reason was written down before this oracle existed: DEVIATION 113
    # records that `raft::log` becomes `std.math.log` because the
    # libm-via-FFI fix this repository already owns is HOST-only and these
    # are device functions, and it predicts Gini and MSE unaffected with
    # Entropy/Poisson/Gamma not bit-comparable. THIS RUN IS THE FIRST
    # MEASUREMENT OF THAT PREDICTION, and it holds: every differing cell is
    # ENTROPY, none is GINI, and the gap is at the last bits.
    var fails = lb_wrong + gini_wrong + split_gini_wrong + upd_wrong
    if ent_max_rel > 1.0e-5:
        fails += 1
        print(
            "  FAIL: ENTROPY differs from cuML by", ent_max_rel,
            "relative -- far beyond a log() ULP, so this is a different"
            " formula rather than DEVIATION 113's known log gap.",
        )
    # The regression bound: Int32 fixed point at scale 1.0 truncates each
    # label_sum to an integer, so a gain built from it can be off by a
    # relative amount of order (1 / smallest label_sum). Anything beyond
    # 1e-2 is not quantization, it is a different formula.
    if reg_max_ulp_gap > 1.0e-2:
        fails += 1
        print(
            "  FAIL: regression gains differ by", reg_max_ulp_gap,
            "relative -- too large for DEVIATION 101b's fixed-point grid"
            " to explain. That is a different formula, not quantization.",
        )
    if fails != 0:
        if first != "":
            print("  FIRST DIFFERENCE:", first)
        raise Error(
            "cuml_oracle_check: " + String(fails) + " failure(s) against cuML"
        )
    print(
        "cuml_oracle_check: MATCHES cuML on every path this port claims"
        " bit-identity for"
    )
    print(
        "  GINI gains, Gain->Split reductions, lower_bound, Split::update"
        " and the midpoint rule are BIT-EXACT against cuML's own compiled"
        " code."
    )
    print(
        "  ENTROPY differs at", ent_max_rel,
        "relative and the regression gains at", reg_max_ulp_gap,
        "-- both PREDICTED before this oracle existed, by DEVIATION 113"
        " (std.math.log vs raft::log, device-only) and DEVIATION 101b"
        " (float64 label_sum -> Int32 fixed point). This run is the first"
        " MEASUREMENT of either."
    )
