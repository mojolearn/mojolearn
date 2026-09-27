# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""WHERE AN OPERATION RUNS. `Exec` is the one interface the lane's algorithms
are written against (`sequence/recurrent.mojo`); `HostExec` (here, no GPU
import) runs each operation as an ascending host loop over its elements and
`DeviceExec` (`sequence/exec_device.mojo`) as one GPU thread per element.
The element body is the same function (`sequence/ops.mojo::apply`), so the
two agree bit for bit by construction."""
from std.memory import memcpy

from sequence.ops import FP, Args, apply


trait Exec:
    def alloc(mut self, n: Int) raises -> FP:
        """A zero-filled buffer of n floats that lives as long as the Exec."""
        ...

    def upload(mut self, dst: FP, src: FP, n: Int) raises:
        """Host memory `src` -> this Exec's buffer `dst` (n floats)."""
        ...

    def download(mut self, dst: FP, src: FP, n: Int) raises:
        """This Exec's buffer `src` -> host memory `dst` (n floats)."""
        ...

    def launch[OP: Int](mut self, a: Args, n: Int) raises:
        """Run operation OP over elements 0..n-1."""
        ...

    def sync(mut self) raises:
        ...


struct HostExec(Exec):
    var bufs: List[FP]

    def __init__(out self):
        self.bufs = List[FP]()

    def __deinit__(deinit self):
        for i in range(len(self.bufs)):
            self.bufs[i].free()

    def alloc(mut self, n: Int) raises -> FP:
        var count = n if n > 0 else 1
        var p = alloc[Float32](count)
        for i in range(count):
            p.unsafe_store(i, Float32(0.0))
        var q = FP(unsafe_from_address=Int(p))
        self.bufs.append(q)
        return q

    def upload(mut self, dst: FP, src: FP, n: Int) raises:
        if n > 0:
            memcpy(dest=dst, src=src, count=n)

    def download(mut self, dst: FP, src: FP, n: Int) raises:
        if n > 0:
            memcpy(dest=dst, src=src, count=n)

    def launch[OP: Int](mut self, a: Args, n: Int) raises:
        for t in range(n):
            apply[OP](t, a)

    def sync(mut self) raises:
        pass
