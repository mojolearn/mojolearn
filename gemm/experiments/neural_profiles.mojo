# SPDX-License-Identifier: Apache-2.0
"""NEURAL G06/G09/V01 experimental adapters. NOT COMPILED OR TESTED.

These explicit entries are isolated from production and classical dispatch.
G06 caches a named-plan decision, G09 changes output-block traversal, and V01
defines a DIFFERENT numerical version with two independent chains per leaf.
Caller integration, cross-column identity, quality and speed are pending.
The owner retains buffers and one in-order context until completion.
"""
from std.sys.compile import is_defined
from std.gpu import block_idx, block_dim, thread_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_mul_add
from gemm.contract import contract_leaf_size, leaf_count
from gemm.checks.gemm_identical import (
    GEMM_FOLD_SLOTS, _fold_push, _fold_drain, _flat_cell_body,
    contract_partition, gemm_operand_strides, choose_gemm_plan,
    identical_gemm_with_plan, identical_gemm_workspace_floats,
)

comptime NI_G06_PLAN_CACHE = (  # NOT TESTED — NOT COMPILED — NOT MEASURED; G06 A, default OFF.
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and is_defined["MOJOLEARN_NI_G06_PLAN_CACHE"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
comptime NI_G09_COLUMN_BLOCKS = (  # NOT TESTED — NOT COMPILED — NOT MEASURED; G09 A, default OFF.
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and is_defined["MOJOLEARN_NI_G09_COLUMN_BLOCKS"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
comptime NI_V01_TWO_CHAIN = (  # NOT TESTED — NOT COMPILED — NOT MEASURED; V01 new version, default OFF.
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and is_defined["MOJOLEARN_NI_V01_TWO_CHAIN"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)


def neural_gemm_profile_name() -> String:
    # Include the leaf profile in provenance; a 64-leaf + two-chain build is
    # not the same version as a 128-leaf + two-chain build.
    var leaf = contract_leaf_size(128)
    comptime if NI_V01_TWO_CHAIN:
        return "mojolearn.neural.gemm.fp32.two-chain.v1.leaf" + String(leaf)
    return "mojolearn.neural.gemm.fp32.serial-leaf.v1.leaf" + String(leaf)


def _neural_cell(
    c: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin],
    m: Int32, n: Int32, k: Int32, leaf: Int32, leaves: Int32,
    asi: Int32, asp: Int32, bsp: Int32, bsj: Int32, cell: Int32,
):
    comptime if not NI_V01_TWO_CHAIN:
        _flat_cell_body(c, a, b, m, n, k, leaf, leaves, asi, asp, bsp, bsj, cell)
    else:
        var i = Int(cell) // Int(n)
        var j = Int(cell) % Int(n)
        var stack = SIMD[DType.float32, GEMM_FOLD_SLOTS](0.0)
        var occupied = 0
        for part in range(Int(leaves)):
            var start = part * Int(leaf)
            var end = min(start + Int(leaf), Int(k))
            var even = Float32(0.0)
            var odd = Float32(0.0)
            var p = start
            # Versioned arithmetic: leaf-relative even and odd indices each
            # ascend, with FTZ around each FMA. Merge even THEN odd once.
            # Chains depend only on K/profile; never on vendor or tile width.
            while p + 1 < end:
                even = ftz(identical_mul_add(
                    ftz(a.unsafe_load(i*Int(asi) + p*Int(asp))),
                    ftz(b.unsafe_load(p*Int(bsp) + j*Int(bsj))), even))
                odd = ftz(identical_mul_add(
                    ftz(a.unsafe_load(i*Int(asi) + (p+1)*Int(asp))),
                    ftz(b.unsafe_load((p+1)*Int(bsp) + j*Int(bsj))), odd))
                p += 2
            if p < end:
                even = ftz(identical_mul_add(
                    ftz(a.unsafe_load(i*Int(asi) + p*Int(asp))),
                    ftz(b.unsafe_load(p*Int(bsp) + j*Int(bsj))), even))
            var value = even
            # Do not insert a +0 merge for a one-element tail.
            if end - start > 1:
                value = ftz(ftz(even) + ftz(odd))
            _ = _fold_push(stack, occupied, ftz(value))
        c.unsafe_store(Int(cell), ftz(_fold_drain(stack, occupied)))


def neural_gemm_kernel(
    c: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin],
    m: Int32, n: Int32, k: Int32, leaf: Int32, leaves: Int32,
    asi: Int32, asp: Int32, bsp: Int32, bsj: Int32,
):
    # 8x16 is an experimental 128-thread ownership layout, not a shape
    # selector. Every positive output shape uses the same masked mapping.
    var row_tiles = (Int(m) + 7) // 8
    var col_tiles = (Int(n) + 15) // 16
    var tile = Int(block_idx.x)
    var tr = tile // col_tiles
    var tc = tile % col_tiles
    comptime if NI_G09_COLUMN_BLOCKS:
        tr = tile % row_tiles
        tc = tile // row_tiles
    var i = tr*8 + Int(thread_idx.x)//16
    var j = tc*16 + Int(thread_idx.x)%16
    if i < Int(m) and j < Int(n):
        _neural_cell(c, a, b, m, n, k, leaf, leaves,
                     asi, asp, bsp, bsj, Int32(i*Int(n)+j))


def neural_gemm_into(
    ctx: DeviceContext, mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32], mut b: DeviceBuffer[DType.float32],
    m: Int, n: Int, k: Int, op: Int,
) raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL, "NEURAL experiment requires IDENTICAL"
    if m < 0 or n < 0 or k < 0 or op < 0 or op > 2:
        raise Error("invalid neural GEMM geometry/orientation")
    if len(c) < m*n or len(a) < m*k or len(b) < n*k:
        raise Error("neural GEMM storage too small")
    if m == 0 or n == 0:
        return
    var part = contract_partition(k)
    var st = gemm_operand_strides(op, m, n, k)
    ctx.enqueue_function[neural_gemm_kernel](c, a, b,
        Int32(m), Int32(n), Int32(k), Int32(part[0]), Int32(part[1]),
        Int32(st[0]), Int32(st[1]), Int32(st[2]), Int32(st[3]),
        grid_dim=(((m+7)//8)*((n+15)//16), 1, 1), block_dim=(128, 1, 1))


struct NeuralNamedPlanWorkspace(Movable):
    """G06 one-entry bounded metadata cache; explicit named-plan adapter.

    Instance belongs to ONE device/context and compiled numerical profile.
    Recreate it when changing context. policy_epoch invalidates caller policy
    changes. This does not cache or replace the production multi-route chooser.
    Scratch grows only after draining the prior consumer; cold setup is owed
    in future timing. A/B both use the same choose_gemm_plan entry.
    """
    var scratch: DeviceBuffer[DType.float32]
    var last_m: Int
    var last_n: Int
    var last_k: Int
    var last_op: Int
    var last_epoch: Int
    var plan: Int

    def __init__(out self, ctx: DeviceContext) raises:
        self.scratch = ctx.enqueue_create_buffer[DType.float32](1)
        self.last_m = -1
        self.last_n = -1
        self.last_k = -1
        self.last_op = -1
        self.last_epoch = -1
        self.plan = 0

    def run(mut self, ctx: DeviceContext, mut c: DeviceBuffer[DType.float32],
            mut a: DeviceBuffer[DType.float32], mut b: DeviceBuffer[DType.float32],
            m: Int, n: Int, k: Int, op: Int, policy_epoch: Int = 0) raises:
        comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL, "NEURAL experiment requires IDENTICAL"
        if m < 0 or n < 0 or k < 0 or op < 0 or op > 2:
            raise Error("invalid named-plan geometry/orientation")
        if len(c) < m*n or len(a) < m*k or len(b) < n*k:
            raise Error("named-plan storage too small")
        if m == 0 or n == 0:
            return
        var hit = False
        comptime if NI_G06_PLAN_CACHE:
            hit = (m == self.last_m and n == self.last_n and k == self.last_k
                   and op == self.last_op and policy_epoch == self.last_epoch)
        if not hit:
            self.plan = choose_gemm_plan(m, n, k)
            self.last_m = m
            self.last_n = n
            self.last_k = k
            self.last_op = op
            self.last_epoch = policy_epoch
        var required = max(1, identical_gemm_workspace_floats(m, n, k, self.plan))
        if required > len(self.scratch):
            ctx.synchronize()
            self.scratch = ctx.enqueue_create_buffer[DType.float32](required)
        identical_gemm_with_plan(ctx, c, a, b, self.scratch, m, n, k, op, self.plan)
