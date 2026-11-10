# SPDX-License-Identifier: Apache-2.0
"""Typed optimizer operations. Python admission and GIL management stay outside RUN."""
from training.optimizer_public_common import PublicStep, PublicStepOut


trait OptKernels:
    """The typed kernel entries behind the door, one binding's set."""

    @staticmethod
    def is_host() -> Bool:
        ...

    @staticmethod
    def step(req: PublicStep) raises -> PublicStepOut:
        ...

    @staticmethod
    def io(req: PublicStep, op_code: Int, tensor: Int) raises -> Tuple[Int, Int]:
        ...

    @staticmethod
    def clip(addrs: List[Int], nbytes: List[Int], max_norm: Float32) raises -> Tuple[Float32, Float32]:
        ...

    @staticmethod
    def accumulate(
        in_addrs: List[Int], nbytes: List[Int], out_addrs: List[Int], a_count: Int, tokens: Int
    ) raises -> Int:
        ...

    @staticmethod
    def download(handle: Int, m_addr: Int, v_addr: Int, n: Int) raises -> Int:
        ...

    @staticmethod
    def upload(handle: Int, m_addr: Int, v_addr: Int, n: Int) raises -> Int:
        ...


