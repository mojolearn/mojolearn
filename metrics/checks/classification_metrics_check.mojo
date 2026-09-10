# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Independent integer/Float64 host oracles, not production reductions."""
from std.math import isfinite
from std.memory import bitcast
from max.gpu.host import DeviceContext
from checks.numerics import numeric_mode_name
from metrics.checks.device_io import upload_i32, download_f32
from metrics.estimator import confusion_matrix_host, precision_recall_fscore_host
from metrics.impl.classification import prf_finish_kernel, confusion_finish_kernel


def close(got: Float32, expected: Float64) raises:
    if not isfinite(got) or abs(Float64(got)-expected) > 2e-6:
        raise Error("classification Float64 oracle mismatch")


def divide(a: Int64, b: Int64, zero: Int) -> Float64:
    return Float64(zero) if b == 0 else Float64(a)/Float64(b)


def check_prf(y: List[Int32], p: List[Int32], k: Int, selected: Int) raises:
    for average in range(5):
        for zero in range(2):
            var positive = k-1
            var actual = precision_recall_fscore_host(y,p,len(y),k,average,positive,zero,selected)
            var again = precision_recall_fscore_host(y,p,len(y),k,average,positive,zero,selected)
            for j in range(len(actual)):
                if bitcast[DType.uint32](actual[j]) != bitcast[DType.uint32](again[j]):
                    raise Error("classification repeat changed bits")
            var width = selected if average == 0 else 1
            for metric in range(3):
                var numerator_sum = Int64(0)
                var denominator_sum = Int64(0)
                var weight_sum = Int64(0)
                var sum = Float64(0)
                var weighted_sum = Float64(0)
                var undefined = False
                var count = 1 if average == 1 else selected
                for c0 in range(count):
                    var c = positive if average == 1 else c0
                    var tp = Int64(0)
                    var true_count = Int64(0)
                    var pred_count = Int64(0)
                    # Deliberately inspect the original pairs for EACH class,
                    # independently of the production marginal histogram.
                    for i in range(len(y)):
                        true_count += Int64(1 if y[i] == Int32(c) else 0)
                        pred_count += Int64(1 if p[i] == Int32(c) else 0)
                        tp += Int64(1 if y[i] == Int32(c) and p[i] == Int32(c) else 0)
                    var numerator = tp if metric < 2 else 2*tp
                    var denominator = pred_count if metric == 0 else (true_count if metric == 1 else true_count+pred_count)
                    var score = divide(numerator,denominator,zero)
                    numerator_sum += numerator
                    denominator_sum += denominator
                    weight_sum += true_count
                    sum += score
                    weighted_sum += score*Float64(true_count)
                    undefined = undefined or denominator == 0
                    if average == 0:
                        close(actual[metric*width+c0],score)
                if average == 2:
                    close(actual[metric],divide(numerator_sum,denominator_sum,zero))
                    undefined = denominator_sum == 0
                elif average == 1:
                    close(actual[metric],sum)
                elif average == 3 or average == 4:
                    var expected = sum/Float64(count)
                    if average == 4 and weight_sum != 0:
                        expected = weighted_sum/Float64(weight_sum)
                    close(actual[metric],expected)
                close(actual[3*width+metric],Float64(1 if undefined else 0))


def check_confusion(y: List[Int32], p: List[Int32], k: Int) raises:
    var expected = List[Int64](length=k*k,fill=0)
    for i in range(len(y)):
        if y[i] >= 0 and p[i] >= 0:
            expected[Int(y[i])*k+Int(p[i])] += 1
    var actual = confusion_matrix_host[DType.int64](y,p,len(y),k,0)
    for i in range(k*k):
        if actual[i] != expected[i]:
            raise Error("integer confusion oracle mismatch")
    for normalization in range(1,4):
        var normalized = confusion_matrix_host[DType.float32](y,p,len(y),k,normalization)
        for row in range(k):
            for col in range(k):
                var denominator = Int64(0)
                for r in range(k):
                    for c in range(k):
                        if normalization == 3 or (normalization == 1 and row == r) or (normalization == 2 and col == c):
                            denominator += expected[r*k+c]
                close(normalized[row*k+col],divide(expected[row*k+col],denominator,0))


def synthetic_large_counts() raises:
    var ctx = DeviceContext()
    # Proves Int64 F1 denominators and explicit Float32 count conversion,
    # without allocating billions of input rows.
    for maximum in [Int32(16777217), Int32(2147483647)]:
        var h: List[Int32] = [maximum,maximum,maximum]
        if maximum == 16777217:
            h[0] = 16777216
        var counts = upload_i32(ctx,h)
        var result = ctx.enqueue_create_buffer[DType.float32](6)
        ctx.enqueue_function[prf_finish_kernel](
            counts.unsafe_ptr(),Int32(1),Int32(1),Int32(2),Int32(0),Int32(0),result.unsafe_ptr(),
            grid_dim=1,block_dim=32,
        )
        var got = download_f32(ctx,result,6)
        for j in range(3):
            if got[j] != 1 or got[3+j] != 0:
                raise Error("large count conversion or F1 denominator overflow")
        var raw = ctx.enqueue_create_buffer[DType.int64](1)
        ctx.enqueue_function[confusion_finish_kernel[DType.int64]](
            counts.unsafe_ptr(),Int32(1),Int32(0),raw.unsafe_ptr(),grid_dim=1,block_dim=32,
        )
        ctx.synchronize()
        with raw.map_to_host() as mapped:
            if mapped[0] != Int64(h[0]):
                raise Error("raw confusion lost integer precision")
        _ = raw^
        _ = result^
        _ = counts^
    _ = ctx^


def main() raises:
    print("numeric_mode",numeric_mode_name())
    var y: List[Int32] = [0,0,1,1,2,2,3,0]
    var p: List[Int32] = [0,1,1,2,0,2,3,3]
    check_confusion(y,p,5)
    check_prf(y,p,5,5)
    check_prf(y,p,5,2) # Retain FP/FN to/from omitted classes.
    var outside: List[Int32] = [2,2]
    var predicted: List[Int32] = [0,2]
    check_prf(outside,predicted,3,2) # Weightedzero-support fallback.
    var swapped_y: List[Int32] = [0,1]
    var swapped_p: List[Int32] = [1,0]
    check_prf(swapped_y,swapped_p,2,2) # F1direct, zero_division1.
    var excluded_y: List[Int32] = [0,-1,1,1]
    var excluded_p: List[Int32] = [0,1,-1,0]
    check_confusion(excluded_y,excluded_p,3)
    var excluded: List[Int32] = [-1,-1]
    check_confusion(excluded,excluded,2) # Allnormalization denominator0.
    var many_y = List[Int32]()
    var many_p = List[Int32]()
    for i in range(1031):
        many_y.append(Int32((i*17+3)%7))
        many_p.append(Int32((i*11+5)%7))
    check_confusion(many_y,many_p,7)
    check_prf(many_y,many_p,7,4)
    synthetic_large_counts()
    for invalid in range(4):
        var rejected = False
        try:
            if invalid == 0:
                _ = confusion_matrix_host[DType.int64](y,p,len(y),4097,0)
            elif invalid == 1:
                _ = precision_recall_fscore_host(y,p,2147483648,5,2,0,0,5)
            elif invalid == 2:
                _ = precision_recall_fscore_host(excluded,predicted,2,3,2,0,0,3)
            else:
                _ = precision_recall_fscore_host(y,p,len(y),5,1,5,0,5)
        except:
            rejected = True
        if not rejected:
            raise Error("classification bound/encoding validation failed")
    print("PASS classification: integer/Float64 oracles, all averages/norms, selected labels, zero support, warning flags, large counts and bounds")
