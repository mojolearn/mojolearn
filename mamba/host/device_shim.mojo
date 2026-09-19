# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into a CPU host binding (python/mojolearn/host_surface.py names which); product, not only a check.
"""The host side of the names the generated Mamba passes read
(`tools/mamba_host_gen.py`; lane/cpu-training-mamba, 2026-09-15).

HOST ONLY. Nothing here imports `max.gpu` or `std.gpu`.

`DeviceBuffer[dtype]` is a host allocation (or a view of one, from
`create_sub_buffer`) and `DeviceContext` runs every enqueue at once: a copy
copies, a fill fills, `synchronize` has nothing to wait for. The generated
files keep every buffer alive the way the device source does (`_ = buf^`
after the last synchronize), so a view never outlives its owner.

`identical_gemm_into` and `identical_gemm` are `mojolearn.identical.gemm.
fp32.v1` through `gemm/host/gemm_oracle.mojo::gemm_oracle`, THE NORMATIVE
ANSWER every device GEMM plan is gated against bit for bit; the linalg host
binding's gemm lanes serve the same function. The workspace is unused.

`launch_count(grid_dim, block_dim)` is the number of thread indices a
one-axis launch covers; a second or third axis above 1 refuses by name,
because the generated serial loop spells only the first axis.
"""
from std.memory import memcpy
from std.sys import size_of

from gemm.host.identical_gemm import OP_TN, gemm_oracle
# the device GEMM file's host-safe fold helpers, lifted verbatim
from mamba.host.gen.gemm_identical_parts import (
    GEMM_FOLD_LEVELS,
    GEMM_FOLD_SLOTS,
    _fold_drain,
    _fold_push,
    _leaf_at,
    _leaf_bounds,
    contract_partition,
)


def launch_count(grid: Tuple[Int, Int, Int], block: Tuple[Int, Int, Int]) raises -> Int:
    if grid[1] != 1 or grid[2] != 1 or block[1] != 1 or block[2] != 1:
        raise Error(
            "mamba host: a launch with a second or third grid or block axis"
            " has no serial restatement here (tools/mamba_host_gen.py)"
        )
    return grid[0] * block[0]


struct DeviceBuffer[dtype: DType](Movable, Sized):
    var _ptr: MutPointer[Scalar[Self.dtype], MutUntrackedOrigin]
    var _len: Int
    var _owner: Bool

    def __init__(out self, n: Int):
        var count = n
        if count < 1:
            count = 1
        self._ptr = alloc[Scalar[Self.dtype]](count)
        self._len = n
        self._owner = True

    def __init__(out self, ptr: MutPointer[Scalar[Self.dtype], MutUntrackedOrigin], n: Int, *, view: Bool):
        self._ptr = ptr
        self._len = n
        self._owner = False

    def __del__(deinit self):
        if self._owner:
            self._ptr.free()

    def __len__(self) -> Int:
        return self._len

    def unsafe_ptr(self) -> MutPointer[Scalar[Self.dtype], MutAnyOrigin]:
        return MutPointer[Scalar[Self.dtype], MutAnyOrigin](unsafe_from_address=Int(self._ptr))

    def enqueue_fill(mut self, value: Scalar[Self.dtype]) raises:
        for i in range(self._len):
            self._ptr.unsafe_store(i, value)

    def create_sub_buffer[dt: DType](self, offset: Int, n: Int) raises -> DeviceBuffer[dt]:
        if dt != Self.dtype:
            raise Error("mamba host: a sub-buffer of another dtype has no host restatement")
        if offset < 0 or n < 0 or offset + n > self._len:
            raise Error(
                String("mamba host: sub-buffer [") + String(offset) + ", "
                + String(offset + n) + ") outside a buffer of " + String(self._len)
            )
        var p = MutPointer[Scalar[dt], MutUntrackedOrigin](unsafe_from_address=Int(self._ptr) + offset * size_of[Scalar[dt]]())
        return DeviceBuffer[dt](p, n, view=True)


struct DeviceContext(Movable):
    def __init__(out self):
        pass

    def __init__(out self, api: String):
        pass

    def synchronize(self):
        pass

    def enqueue_create_buffer[dt: DType](self, n: Int) raises -> DeviceBuffer[dt]:
        return DeviceBuffer[dt](n)

    def enqueue_create_host_buffer[dt: DType](self, n: Int) raises -> DeviceBuffer[dt]:
        return DeviceBuffer[dt](n)

    def enqueue_copy[dt: DType](self, *, dst_buf: DeviceBuffer[dt], src_buf: DeviceBuffer[dt]) raises:
        var n = len(src_buf)
        if len(dst_buf) < n:
            raise Error("mamba host: copy into a shorter buffer")
        if n > 0 and Int(dst_buf.unsafe_ptr()) != Int(src_buf.unsafe_ptr()):
            memcpy(dest=dst_buf.unsafe_ptr(), src=src_buf.unsafe_ptr(), count=n)

    def enqueue_copy[dt: DType, origin: MutOrigin](self, *, dst_ptr: MutPointer[Scalar[dt], origin], src_buf: DeviceBuffer[dt]) raises:
        var n = len(src_buf)
        if n > 0:
            memcpy(dest=dst_ptr, src=src_buf.unsafe_ptr(), count=n)

    def enqueue_copy[dt: DType, origin: Origin](self, *, dst_buf: DeviceBuffer[dt], src_ptr: UnsafePointer[Scalar[dt], origin]) raises:
        var n = len(dst_buf)
        if n > 0:
            memcpy(dest=dst_buf.unsafe_ptr(), src=src_ptr, count=n)


def _read(buf: DeviceBuffer[DType.float32], n: Int) raises -> List[Float32]:
    if n > len(buf):
        raise Error(
            String("mamba host: a GEMM operand of ") + String(n)
            + " values from a buffer of " + String(len(buf))
        )
    var out = List[Float32](length=n, fill=Float32(0.0))
    if n > 0:
        memcpy(dest=out.unsafe_ptr(), src=buf.unsafe_ptr(), count=n)
    return out^


def identical_gemm_into[allow_vendor: Bool = True](
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
    op: Int,
) raises:
    """`C[m x n] = op(A) . op(B)`: A holds m*k values (k*m under OP_TN), B
    n*k, both row-major, exactly as `gemm_oracle` reads them."""
    var out = gemm_oracle(_read(a, m * k), _read(b, n * k), op, m, n, k)
    if len(c) < m * n:
        raise Error("mamba host: a GEMM output buffer shorter than m * n")
    if m * n > 0:
        memcpy(dest=c.unsafe_ptr(), src=out.unsafe_ptr(), count=m * n)


def identical_gemm[allow_vendor: Bool = True](
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
    op: Int,
) raises:
    var ws = DeviceBuffer[DType.float32](1)
    identical_gemm_into(ctx, c, a, b, ws, m, n, k, op)


def identical_gemm_workspace_max_floats(m: Int, n: Int, k: Int) -> Int:
    return 1
