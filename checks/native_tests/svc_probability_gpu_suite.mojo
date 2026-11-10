# SPDX-License-Identifier: Apache-2.0
"""Importable native GPU SVC probability contracts.

The native SVM entries share _family_ctx, so fixture submissions are ordered.
Thin test kernels call the actual sigmoid/coupling routines, with no duplicated
algorithm. A coupling row is an inherently dependent three-class solve; the
original fixture has one row. Each sigmoid point is an independent GPU thread.
"""
from std.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from svm.estimator import _family_ctx
from svm.impl.svc_epilogue import device_splitmix_perm, device_platt_train
from svm.impl.svc_rows import U64P, I32P, sp_sigmoid_predict, multiclass_probability
from checks.native_tests.svc_probability_cases import SVC_PROBABILITY_FIXTURES, SvcProbabilityCalls, fixture, require


def _sigmoid_contract(ab: U64P, dst: U64P):
    var index = Int(block_idx.x) * 32 + Int(thread_idx.x)
    if index < 2:
        var decision = UInt64(0x4000000000000000) if index == 0 else UInt64(0xC000000000000000)
        dst[index + 1] = sp_sigmoid_predict(decision, ab[1], ab[2])


def _coupling_contract(matrix: U64P, dst: U64P, scratch: U64P, status: I32P):
    var row = Int(block_idx.x) * 32 + Int(thread_idx.x)
    if row == 0:
        status[1] = Int32(multiclass_probability(3, matrix + 1, dst + 1, scratch + 1, scratch + 10))


def _upload(ctx: DeviceContext, address: Int, count: Int) raises -> DeviceBuffer[DType.uint64]:
    var buffer = ctx.enqueue_create_buffer[DType.uint64](count)
    ctx.enqueue_copy(dst_buf=buffer, src_ptr=U64P(unsafe_from_address=address))
    return buffer^


struct DeviceSvcProbability(SvcProbabilityCalls):
    @staticmethod
    def shuffle(dst: Int, count: Int, seed: Int) raises:
        device_splitmix_perm(dst, count, seed, 0)

    @staticmethod
    def platt(decisions: Int, labels: Int, count: Int, dst: Int) raises:
        device_platt_train(decisions, labels, count, dst)

    @staticmethod
    def sigmoid(calibration: Int, dst: Int) raises:
        var ctx = _family_ctx()
        var ab = _upload(ctx, calibration - 8, 4)
        var probabilities = _upload(ctx, dst - 8, 4)
        ctx.enqueue_function[_sigmoid_contract](ab.unsafe_ptr(), probabilities.unsafe_ptr(),
            grid_dim=1, block_dim=32)
        ctx.enqueue_copy(dst_ptr=U64P(unsafe_from_address=dst - 8), src_buf=probabilities)
        ctx.enqueue_copy(dst_ptr=U64P(unsafe_from_address=calibration - 8), src_buf=ab)
        ctx.synchronize()
        _ = ab^
        _ = probabilities^
        _ = ctx^

    @staticmethod
    def couple(matrix: Int, dst: Int, scratch: Int) raises -> Int:
        var ctx = _family_ctx()
        var r = _upload(ctx, matrix - 8, 11)
        var p = _upload(ctx, dst - 8, 5)
        var q = _upload(ctx, scratch - 8, 14)
        var status: List[Int32] = [1234567, -1, 1234567]
        var st = ctx.enqueue_create_buffer[DType.int32](3)
        var sp = I32P(unsafe_from_address=Int(status.unsafe_ptr()))
        ctx.enqueue_copy(dst_buf=st, src_ptr=sp)
        ctx.enqueue_function[_coupling_contract](r.unsafe_ptr(), p.unsafe_ptr(), q.unsafe_ptr(), st.unsafe_ptr(),
            grid_dim=1, block_dim=32)
        ctx.enqueue_copy(dst_ptr=U64P(unsafe_from_address=matrix - 8), src_buf=r)
        ctx.enqueue_copy(dst_ptr=U64P(unsafe_from_address=dst - 8), src_buf=p)
        ctx.enqueue_copy(dst_ptr=U64P(unsafe_from_address=scratch - 8), src_buf=q)
        ctx.enqueue_copy(dst_ptr=sp, src_buf=st)
        ctx.synchronize()
        require(status[0] == 1234567 and status[2] == 1234567, "SVC coupling status guards")
        var result = Int(status[1])
        _ = r^
        _ = p^
        _ = q^
        _ = st^
        _ = ctx^
        return result


def run_checks() raises:
    var failed = 0
    for fixture_id in range(SVC_PROBABILITY_FIXTURES):
        try:
            fixture[DeviceSvcProbability](fixture_id)
        except error:
            failed += 1
            print("FAIL svc_probability_gpu fixture", fixture_id, String(error))
    require(failed == 0, "svc_probability_gpu failed " + String(failed) + " fixtures")
    print("PASS svc_probability_gpu:", SVC_PROBABILITY_FIXTURES, "native probability contracts")
