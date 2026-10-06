# SPDX-License-Identifier: Apache-2.0
"""T15 RF public OOB host counterpart, one version with every GPU column.

NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
Membership uses the fit's exact logical tree draws, including weighted draws.
Prediction sums retain ascending tree order, score terms use shared exact limbs.
This is Mojo host runtime; no Python computation or GPU import is required.
"""
from std.memory import bitcast
from ensemble.host.rf_oracle import RfHostForest, RfHostParams, host_sampled_rows, host_weight_qcdf, n_sampled_rows_host
from checks.numerics import ftz
from checks.soft_f64 import SF64_ONE, SF64_ZERO, SF64_NAN, SF64_SIGN, sf64_add, sf64_div, sf64_sub, sf64_mul, sf64_from_int, sf64_from_f32, sf64_gt, sf64_is_nan
from xtrees.exact_oob_sum import E64_LIMBS, E64_MAX_ROWS, E64_FLAGS, e64_add, e64_round


def rf_host_oob[CLASSIFIER: Bool](
    forest: RfHostForest, p: RfHostParams,
    x: MutPointer[Float32,MutUntrackedOrigin], y_address: Int,
    weights: List[Float32], rows: Int, cols: Int,
    out_address: Int, score_address: Int,
) raises:
    if not p.bootstrap:
        raise Error("out-of-bag estimation requires bootstrap=True")
    if rows > E64_MAX_ROWS:
        raise Error("OOB score: more rows than the exact sum takes")
    var outputs = forest.num_outputs
    var outp = MutPointer[UInt64,MutUntrackedOrigin](unsafe_from_address=out_address)
    var scorep = MutPointer[UInt64,MutUntrackedOrigin](unsafe_from_address=score_address)
    var counts = List[Int32](length=rows,fill=Int32(0))
    for i in range(rows*outputs):
        outp[unsafe_offset=i] = UInt64(0)
    var cdf = List[UInt64]()
    if len(weights) > 0:
        cdf = host_weight_qcdf(weights,rows)
    var sampled = n_sampled_rows_host(p.bootstrap,p.max_samples,rows)
    for tree in range(forest.n_trees):
        var draw = host_sampled_rows(p.seed,tree,True,rows,sampled,weight_qcdf=cdf)
        var membership = List[UInt8](length=rows,fill=UInt8(0))
        for i in range(len(draw)):
            membership[Int(draw[i])] = UInt8(1)
        var base = Int(forest.offsets[tree])
        for row in range(rows):
            if membership[row] != UInt8(0):
                continue
            var node = 0
            var child = Int(forest.left_child[base])
            while child != -1:
                var col = Int(forest.colid[base+node])
                node = child if ftz(x[unsafe_offset=col*rows+row]) <= forest.quesval[base+node] else child+1
                child = Int(forest.left_child[base+node])
            for output in range(outputs):
                var cell = row*outputs+output
                outp[unsafe_offset=cell] = sf64_add(outp[unsafe_offset=cell],sf64_from_f32(forest.leaves[(base+node)*outputs+output]))
            counts[row] += 1
    var valid = 0
    var correct = 0
    for row in range(rows):
        if counts[row] <= 0:
            continue
        valid += 1
        for output in range(outputs):
            var cell = row*outputs+output
            outp[unsafe_offset=cell] = sf64_div(outp[unsafe_offset=cell],sf64_from_int(Int(counts[row])))
        comptime if CLASSIFIER:
            var labels = MutPointer[Int32,MutUntrackedOrigin](unsafe_from_address=y_address)
            var best = 0
            var bestp = outp[unsafe_offset=row*outputs]
            for output in range(1,outputs):
                var value = outp[unsafe_offset=row*outputs+output]
                if not sf64_is_nan(value) and not sf64_is_nan(bestp) and sf64_gt(value,bestp):
                    best = output
                    bestp = value
            if best == Int(labels[unsafe_offset=row]):
                correct += 1
    if valid == 0:
        raise Error("no sample was out of bag for any tree, so there is no OOB score to report")
    comptime if CLASSIFIER:
        scorep[unsafe_offset=0] = sf64_div(sf64_from_int(correct),sf64_from_int(valid))
    else:
        var labels = MutPointer[Float32,MutUntrackedOrigin](unsafe_from_address=y_address)
        var sum = List[Int64](length=E64_LIMBS,fill=Int64(0))
        var den = List[Int64](length=E64_LIMBS,fill=Int64(0))
        var num = List[Int64](length=E64_LIMBS,fill=Int64(0))
        var flags = List[Int32](length=E64_FLAGS,fill=Int32(0))
        for row in range(rows):
            if counts[row] > 0:
                e64_add(sum.unsafe_ptr(),sf64_from_f32(labels[unsafe_offset=row]),flags.unsafe_ptr())
        var mean = sf64_div(e64_round(sum.unsafe_ptr(),flags.unsafe_ptr()),sf64_from_int(valid))
        for row in range(rows):
            if counts[row] > 0:
                var label = sf64_from_f32(labels[unsafe_offset=row])
                var d = sf64_sub(label,mean)
                var residual = sf64_sub(label,outp[unsafe_offset=row])
                e64_add(den.unsafe_ptr(),sf64_mul(d,d),flags.unsafe_ptr())
                e64_add(num.unsafe_ptr(),sf64_mul(residual,residual),flags.unsafe_ptr())
        var dw = e64_round(den.unsafe_ptr(),flags.unsafe_ptr())
        var nw = e64_round(num.unsafe_ptr(),flags.unsafe_ptr())
        var score = SF64_ONE
        if flags[0] != 0 or flags[1] != 0:
            score = SF64_NAN
        elif (nw & ~SF64_SIGN) == UInt64(0):
            score = SF64_ONE
        elif (dw & ~SF64_SIGN) == UInt64(0):
            score = SF64_ZERO
        else:
            score = sf64_sub(SF64_ONE,sf64_div(nw,dw))
        scorep[unsafe_offset=0] = score
