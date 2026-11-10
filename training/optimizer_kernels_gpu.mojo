# SPDX-License-Identifier: Apache-2.0
"""Pure native GPU optimizer dispatch over the existing tuned operations."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.neural_context import neural_ctx
from training.optimizer_kernels import OptKernels
from training.dev_tensors import train_dev_free
from training.optimizer_public_common import PublicStep, PublicStepOut
from training.optimizer_public import (
    public_step, public_io, public_clip, public_accumulate, resident_download, resident_upload, resident_open, resident_close,
)

comptime _NEURAL_CTX = "MojolearnNeuralTrainingContextIdenticalV3" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnNeuralTrainingContextFastV3"


struct GpuOptKernels(OptKernels):
    @staticmethod
    def open(count: Int) raises -> Int:
        var ctx = neural_ctx[_NEURAL_CTX]()
        return resident_open(ctx, count)

    @staticmethod
    def close(handle: Int) raises -> Int:
        var ctx = neural_ctx[_NEURAL_CTX]()
        return resident_close(ctx, handle)

    @staticmethod
    def free_device(handle: Int) raises -> Int:
        var ctx = neural_ctx[_NEURAL_CTX]()
        train_dev_free(ctx, handle)
        return 0

    @staticmethod
    def is_host() -> Bool:
        return False

    @staticmethod
    def step(req: PublicStep) raises -> PublicStepOut:
        var ctx = neural_ctx[_NEURAL_CTX]()
        return public_step(ctx, req)

    @staticmethod
    def io(req: PublicStep, op_code: Int, tensor: Int) raises -> Tuple[Int, Int]:
        var ctx = neural_ctx[_NEURAL_CTX]()
        return public_io(ctx, req, op_code, tensor)

    @staticmethod
    def clip(addrs: List[Int], nbytes: List[Int], max_norm: Float32) raises -> Tuple[Float32, Float32]:
        var ctx = neural_ctx[_NEURAL_CTX]()
        return public_clip(ctx, addrs, nbytes, max_norm)

    @staticmethod
    def accumulate(in_addrs: List[Int], nbytes: List[Int], out_addrs: List[Int], a_count: Int, tokens: Int) raises -> Int:
        var ctx = neural_ctx[_NEURAL_CTX]()
        return public_accumulate(ctx, in_addrs, nbytes, out_addrs, a_count, tokens)

    @staticmethod
    def download(handle: Int, m_addr: Int, v_addr: Int, n: Int) raises -> Int:
        var ctx = neural_ctx[_NEURAL_CTX]()
        return resident_download(ctx, handle, m_addr, v_addr, n)

    @staticmethod
    def upload(handle: Int, m_addr: Int, v_addr: Int, n: Int) raises -> Int:
        var ctx = neural_ctx[_NEURAL_CTX]()
        return resident_upload(ctx, handle, m_addr, v_addr, n)
