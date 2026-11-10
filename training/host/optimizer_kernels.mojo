# SPDX-License-Identifier: Apache-2.0
"""Pure native host optimizer dispatch with existing update and ownership rules."""
from training.optimizer_kernels import OptKernels
from training.optimizer_public_common import PublicStep, PublicStepOut
from training.host.optimizer_public_host import (
    public_step_host, public_clip_host, public_accumulate_host, host_download, host_upload, host_open, host_close,
)


struct HostOptKernels(OptKernels):
    @staticmethod
    def open(count: Int) raises -> Int:
        return host_open(count)

    @staticmethod
    def close(handle: Int) raises -> Int:
        return host_close(handle)

    @staticmethod
    def free_device(handle: Int) raises -> Int:
        raise Error("RuntimeError: optimizer: a CPU-only install has no device tensor allocation")

    @staticmethod
    def is_host() -> Bool:
        return True

    @staticmethod
    def step(req: PublicStep) raises -> PublicStepOut:
        return public_step_host(req)

    @staticmethod
    def io(req: PublicStep, op_code: Int, tensor: Int) raises -> Tuple[Int, Int]:
        if op_code != 0:
            raise Error("RuntimeError: mojolearn optimizer: this build has no device-resident parameter "
                        "entries (a CPU-only install)")
        return (req.handle, 0)

    @staticmethod
    def clip(addrs: List[Int], nbytes: List[Int], max_norm: Float32) raises -> Tuple[Float32, Float32]:
        return public_clip_host(addrs, nbytes, max_norm)

    @staticmethod
    def accumulate(in_addrs: List[Int], nbytes: List[Int], out_addrs: List[Int], a_count: Int, tokens: Int) raises -> Int:
        return public_accumulate_host(in_addrs, nbytes, out_addrs, a_count, tokens)

    @staticmethod
    def download(handle: Int, m_addr: Int, v_addr: Int, n: Int) raises -> Int:
        return host_download(handle, m_addr, v_addr, n)

    @staticmethod
    def upload(handle: Int, m_addr: Int, v_addr: Int, n: Int) raises -> Int:
        return host_upload(handle, m_addr, v_addr, n)
