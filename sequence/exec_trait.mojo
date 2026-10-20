# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""WHERE AN OPERATION RUNS: the `Exec` interface the lane's algorithms are
written against (`sequence/recurrent.mojo`). `DeviceExec`
(`sequence/exec_device.mojo`) runs each operation as one GPU thread per
element; `HostExec` (`sequence/exec.mojo`, the CPU-only install's executor)
as an ascending host loop. The element body is the same function
(`sequence/ops.mojo::apply`), so the two agree bit for bit by construction.

The trait lives here, apart from `HostExec`, so the GPU modules that take an
`Exec` do not link the host executor (cpu-gpu-cleanup n-seq, 2026-10-02)."""
from sequence.ops import FP, Args
from sequence.execution_profile import ExecutionProfile


trait Exec:
    def alloc(mut self, n: Int) raises -> FP:
        """A zero-filled buffer of n floats that lives as long as the Exec."""
        ...

    def free(mut self, p: FP) raises:
        """Return a buffer that `alloc` handed out before the Exec ends
        (lane ag-fusion: the autograd tape releases its storage). `p` must be
        the exact pointer `alloc` returned and must not be used afterwards;
        work already queued that reads or writes `p` still completes first
        (the device keeps the storage until its next `sync`)."""
        ...

    def upload(mut self, dst: FP, src: FP, n: Int) raises:
        """Host memory `src` -> this Exec's buffer `dst` (n floats)."""
        ...

    def download(mut self, dst: FP, src: FP, n: Int) raises:
        """This Exec's buffer `src` -> host memory `dst` (n floats)."""
        ...

    def download_async(mut self, dst: FP, src: FP, n: Int) raises:
        """`download`, except that `dst` is written by the next `sync()`
        (the device queues its copy; several then share one wait)."""
        ...

    def bind(mut self, src: FP, n: Int) raises -> FP:
        """A buffer of this Exec holding host memory `src`'s n floats, for an
        entry that updates the caller's arrays in place: the device
        allocates and uploads; the host returns `src` itself (no copy), so
        a later `download(src, buf, n)` is the identity."""
        ...

    def launch[OP: Int](mut self, a: Args, n: Int) raises:
        """Run operation OP over elements 0..n-1."""
        ...

    def sync(mut self) raises:
        ...



    def execution_profile_begin(mut self) raises:
        ...

    def execution_profile_end(mut self) raises -> ExecutionProfile:
        ...
