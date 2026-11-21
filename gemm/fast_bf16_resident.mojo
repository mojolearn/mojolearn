# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""BF16 MATRIX-UNIT PRODUCTS OVER RESIDENT BF16 OPERANDS (lane inv-bf16-kernel, 2026-11-20).

EXPERIMENT `-D MOJOLEARN_GEMM_FAST_BF16_RESIDENT`, default OFF, A/B owed with quality beside every time
(docs/apple-fast/EXPERIMENTS.md; arm `fast-bf16-resident` of bench/results/paper_compare/owed-pre0867.json, rows
`kernel-clock-units-*`). Legal only in a `fast` build with the matrix-units route (`-D MOJOLEARN_MODE_FAST_ROWS` on
the NVIDIA or AMD column, no `-D MOJOLEARN_GEMM_FAST_MATRIX_UNITS_OFF`): in a build that promises identical bits the
define is a compile error (here and in core/experiment_guards_neural.mojo).

WHY A NEW ARM. The deleted arm `MOJOLEARN_GEMM_FAST_BF16_UNITS` (docs/TOMBSTONES.md, A/B abf2: median 0.96, dominated by
the TF32 route) narrowed BOTH operands of EVERY product from FP32 into scratch, per call: three launches and one
scratch take where the TF32 route makes one launch. A weight was narrowed again for every product it entered (twice
a training step, once per chunk per layer of an inference call) and nothing ever stayed in bf16. At the sizes of the
board's neural rows a TF32 product is a few microseconds of arithmetic, so the two extra launches cost more than the
unit saved. This module is the other form: an operand is narrowed ONCE into a device-resident bf16 image
(`Bf16Image`) and every later product reads the image, so a product is ONE launch, as on the TF32 route, at half
the operand traffic and on the half-precision instruction.

  - WEIGHTS: `Bf16Resident.image_of` narrows a weight the first time it is asked for (at upload or construction) and
    returns the same image until the owner says the weight was rewritten (`changed`, `changed_all`: an optimizer
    step, a new upload). A stale image is narrowed again in place by the next product that asks: one launch per
    weight per change, however many products read it.
  - ACTIVATIONS: produced in bf16 by the launch that already produces them. `bf16_resident_epilogue` is the
    bias-and-activation follower of a product (the launch the TF32 route already runs as its epilogue tail): it
    reads the FP32 product, adds the bias and applies the activation in FP32, and stores the bf16 image the next
    product reads. No narrow launch exists on the chain. `bf16_resident_product_to_image` is the same for a chain
    with no epilogue between two products (the unit accumulates in FP32 and stores bf16).
  - THE PUBLIC BOUNDARY: FP32. `bf16_resident_product` writes an FP32 output from FP32 accumulation; the last
    product of a chain and every result a caller downloads use it.

WHAT IS RELAXED. Each operand is rounded to bfloat16 (8 significant bits) by round to nearest even with subnormals
flushed (checks/numerics.mojo::f32_to_bf16_bits_rne, contract L-2): relative error at most 2^-9 per operand. The
unit multiplies the rounded operands exactly and accumulates in FP32. For one cell c = sum_k a_k b_k:

    |c_bf16 - c_fp32|  <=  (2 * 2^-9 + 2^-18) * sum_k |a_k b_k|  +  the FP32 accumulation difference

(`bf16_resident_cell_bound`). An activation stored as an image is rounded once, where its consumer would have
rounded it: the chain's error is the per-call arm's, not a compounded one. Masters, optimizer state, the loss,
normalisations and every product not handed to this module stay FP32.

WHAT IS NOT HERE. No size rule and no dispatch: the caller owns the decision which products run here, and the
kernel-clock rows (bench/gemm_units_kernel_clock_main.mojo) measure where the unit gains. The product wiring into
the sequence executor and the neural dispatch is owed to that measurement (the lane's result file names the
sites).

COST PER CALL: `image_of` is a scan of the owner's handful of images (two integer compares each); a product is one
launch; no allocation, lock or wait. Without the define nothing here is compiled into a launch.
"""
from std.sys.compile import is_defined
from layout import TileTensor
from layout.tile_layout import row_major
from linalg.matmul import matmul
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import f32_to_bf16_bits_rne
from gemm.fast_matrix_units import GEMM_FAST_MATRIX_UNITS

comptime _BF16_RESIDENT_DEFINED = is_defined["MOJOLEARN_GEMM_FAST_BF16_RESIDENT"]()


def _bf16_resident_checked() -> Bool:
    """The compile-time refusal outside a fast build with the matrix-units route. True when the arm is on."""
    comptime assert (
        (not _BF16_RESIDENT_DEFINED) or GEMM_FAST_MATRIX_UNITS
    ), "MOJOLEARN_GEMM_FAST_BF16_RESIDENT is legal only in a fast build with the matrix-units route (-D MOJOLEARN_MODE_FAST_ROWS on the NVIDIA or AMD column, without MOJOLEARN_GEMM_FAST_MATRIX_UNITS_OFF); never in a build that promises identical bits"
    return _BF16_RESIDENT_DEFINED


#: The arm: bf16 matrix-unit products over resident bf16 operands in a `fast` build on NVIDIA and AMD.
comptime GEMM_FAST_BF16_RESIDENT = _bf16_resident_checked()

#: threads a block of the narrow and epilogue launches
comptime BF16_RESIDENT_TPB = 256

#: Operand relative error of one bf16 operand: round to nearest at 8 significant bits.
comptime BF16_RESIDENT_OPERAND_REL = Float64(1.0) / Float64(512.0)

#: `bf16_resident_epilogue`'s activations
comptime BF16_EPI_NONE = 0
comptime BF16_EPI_RELU = 1

comptime _BF16 = Scalar[DType.bfloat16]
comptime _F32P = MutPointer[Float32, MutAnyOrigin]
comptime _U16P = MutPointer[UInt16, MutAnyOrigin]


def bf16_resident_cell_bound(abs_sum: Float64, k: Int) -> Float64:
    """The bound on |c_bf16 - c_fp32| for one cell whose absolute sum of terms is `abs_sum` (module docstring):
    both operands of every term rounded to bf16, and the FP32 accumulation of K terms in either order."""
    var operand = (
        Float64(2.0) * BF16_RESIDENT_OPERAND_REL + BF16_RESIDENT_OPERAND_REL * BF16_RESIDENT_OPERAND_REL
    )
    var fold = Float64(2.0) * Float64(k) / Float64(16777216.0)
    return (operand + fold) * abs_sum


def bf16_resident_narrow_kernel(dst: _U16P, src: _F32P, count: Int64):
    """dst[i] = bf16(src[i]), round to nearest even, one element a thread."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(count):
        return
    dst.unsafe_store(i, f32_to_bf16_bits_rne(src.unsafe_load(i)))


def bf16_resident_narrow_t_kernel(dst: _U16P, src: _F32P, rows: Int64, cols: Int64):
    """dst[rows x cols] = bf16(src[cols x rows]^T): the narrow and the transpose of a stored-transposed operand in
    one launch, one element of dst a thread (coalesced store, strided load)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(rows) * Int(cols):
        return
    var r = i // Int(cols)
    var col = i - r * Int(cols)
    dst.unsafe_store(i, f32_to_bf16_bits_rne(src.unsafe_load(col * Int(rows) + r)))


def bf16_resident_epilogue_kernel(dst: _U16P, c: _F32P, bias: _F32P, n: Int64, count: Int64, kind: Int64):
    """dst[i] = bf16(act(c[i] + bias[i mod n])): the follower of a product whose consumer is a bf16 unit. The add
    and the activation are FP32 on the stored FP32 product; only the store is rounded. One element a thread."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(count):
        return
    var v = c.unsafe_load(i) + bias.unsafe_load(i % Int(n))
    if Int(kind) == BF16_EPI_RELU:
        if v < Float32(0.0):
            v = Float32(0.0)
    dst.unsafe_store(i, f32_to_bf16_bits_rne(v))


struct Bf16Image(Movable):
    """A device-resident bf16 image of `cells` FP32 values, row-major in the shape its products read it."""

    var bits: DeviceBuffer[DType.uint16]
    var cells: Int

    def __init__(out self, ctx: DeviceContext, cells: Int) raises:
        self.cells = cells
        self.bits = ctx.enqueue_create_buffer[DType.uint16](max(cells, 1))


def bf16_image_narrow(
    ctx: DeviceContext, mut img: Bf16Image, mut src: DeviceBuffer[DType.float32], cells: Int
) raises:
    """img[0, cells) = bf16(src[0, cells)), elementwise: one asynchronous launch, no wait. The caller keeps `src`
    and `img` alive past its next synchronize."""
    comptime if GEMM_FAST_BF16_RESIDENT:
        if cells < 0 or cells > img.cells or cells > len(src):
            raise Error("bf16 resident: narrow of " + String(cells) + " cells does not fit its image or source")
        if cells == 0:
            return
        var pd = img.bits.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var ps = src.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        ctx.enqueue_function[bf16_resident_narrow_kernel](
            pd,
            ps,
            Int64(cells),
            grid_dim=((cells + BF16_RESIDENT_TPB - 1) // BF16_RESIDENT_TPB, 1, 1),
            block_dim=(BF16_RESIDENT_TPB, 1, 1),
        )
    else:
        raise Error("bf16 resident: built without -D MOJOLEARN_GEMM_FAST_BF16_RESIDENT")


def bf16_image_narrow_transposed(
    ctx: DeviceContext, mut img: Bf16Image, mut src: DeviceBuffer[DType.float32], rows: Int, cols: Int
) raises:
    """img[rows x cols] = bf16(src[cols x rows]^T): one asynchronous launch. For an operand stored transposed
    (the X of a weight gradient X^T dY), since `linalg.matmul` has no `transpose_a`."""
    comptime if GEMM_FAST_BF16_RESIDENT:
        var cells = rows * cols
        if rows < 0 or cols < 0 or cells > img.cells or cells > len(src):
            raise Error("bf16 resident: transposed narrow does not fit its image or source")
        if cells == 0:
            return
        var pd = img.bits.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var ps = src.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        ctx.enqueue_function[bf16_resident_narrow_t_kernel](
            pd,
            ps,
            Int64(rows),
            Int64(cols),
            grid_dim=((cells + BF16_RESIDENT_TPB - 1) // BF16_RESIDENT_TPB, 1, 1),
            block_dim=(BF16_RESIDENT_TPB, 1, 1),
        )
    else:
        raise Error("bf16 resident: built without -D MOJOLEARN_GEMM_FAST_BF16_RESIDENT")


def bf16_resident_product(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut a: Bf16Image,
    mut b: Bf16Image,
    m: Int,
    n: Int,
    k: Int,
    b_transposed: Bool,
) raises:
    """C[m x n] (FP32, dense) = A[m x k] . B, both operands resident bf16 images, B stored k x n (or n x k when
    `b_transposed`). FP32 accumulation and FP32 output. ONE asynchronous launch, no narrow, no scratch, no wait:
    the caller keeps c, a and b alive past its next synchronize."""
    comptime if GEMM_FAST_BF16_RESIDENT:
        if m <= 0 or n <= 0 or k <= 0:
            raise Error("bf16 resident: m, n and k must be positive")
        if len(c) < m * n or a.cells < m * k or b.cells < n * k:
            raise Error("bf16 resident: operand or output storage too short")
        var pa = a.bits.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var pb = b.bits.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var pc = c.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var ha = pa.bitcast[_BF16]().unsafe_origin_cast[MutAnyOrigin]()
        var hb = pb.bitcast[_BF16]().unsafe_origin_cast[MutAnyOrigin]()
        var ta = TileTensor(ptr=ha, layout=row_major(m, k))
        var tc = TileTensor(ptr=pc, layout=row_major(m, n))
        if b_transposed:
            var tbt = TileTensor(ptr=hb, layout=row_major(n, k))
            matmul[transpose_b=True, target="gpu"](tc, ta, tbt, ctx)
        else:
            var tb = TileTensor(ptr=hb, layout=row_major(k, n))
            matmul[transpose_b=False, target="gpu"](tc, ta, tb, ctx)
    else:
        raise Error("bf16 resident: built without -D MOJOLEARN_GEMM_FAST_BF16_RESIDENT")


def bf16_resident_product_to_image(
    ctx: DeviceContext,
    mut c: Bf16Image,
    mut a: Bf16Image,
    mut b: Bf16Image,
    m: Int,
    n: Int,
    k: Int,
    b_transposed: Bool,
) raises:
    """C[m x n] = A . B stored as a bf16 image (FP32 accumulation inside the unit, one rounding at the store): for
    a product whose only consumer is another bf16 product and that has no epilogue. One asynchronous launch."""
    comptime if GEMM_FAST_BF16_RESIDENT:
        if m <= 0 or n <= 0 or k <= 0:
            raise Error("bf16 resident: m, n and k must be positive")
        if c.cells < m * n or a.cells < m * k or b.cells < n * k:
            raise Error("bf16 resident: operand or output image too short")
        var pa = a.bits.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var pb = b.bits.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var pc = c.bits.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var ha = pa.bitcast[_BF16]().unsafe_origin_cast[MutAnyOrigin]()
        var hb = pb.bitcast[_BF16]().unsafe_origin_cast[MutAnyOrigin]()
        var hc = pc.bitcast[_BF16]().unsafe_origin_cast[MutAnyOrigin]()
        var ta = TileTensor(ptr=ha, layout=row_major(m, k))
        var tc = TileTensor(ptr=hc, layout=row_major(m, n))
        if b_transposed:
            var tbt = TileTensor(ptr=hb, layout=row_major(n, k))
            matmul[transpose_b=True, target="gpu"](tc, ta, tbt, ctx)
        else:
            var tb = TileTensor(ptr=hb, layout=row_major(k, n))
            matmul[transpose_b=False, target="gpu"](tc, ta, tb, ctx)
    else:
        raise Error("bf16 resident: built without -D MOJOLEARN_GEMM_FAST_BF16_RESIDENT")


def bf16_resident_epilogue(
    ctx: DeviceContext,
    mut dst: Bf16Image,
    mut c: DeviceBuffer[DType.float32],
    mut bias: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    kind: Int,
) raises:
    """dst[m x n] = bf16(act(C + bias)), bias one value a column, `kind` BF16_EPI_NONE or BF16_EPI_RELU: the
    follower of a product, in FP32, storing the image its consumer reads. One asynchronous launch; it is the
    launch the TF32 route runs as its epilogue tail, so the chain has no narrow launch."""
    comptime if GEMM_FAST_BF16_RESIDENT:
        var cells = m * n
        if m <= 0 or n <= 0 or dst.cells < cells or len(c) < cells or len(bias) < n:
            raise Error("bf16 resident: epilogue storage too short")
        if kind != BF16_EPI_NONE and kind != BF16_EPI_RELU:
            raise Error("bf16 resident: epilogue kind must be 0 (none) or 1 (relu)")
        var pd = dst.bits.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var pc = c.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var pbias = bias.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        ctx.enqueue_function[bf16_resident_epilogue_kernel](
            pd,
            pc,
            pbias,
            Int64(n),
            Int64(cells),
            Int64(kind),
            grid_dim=((cells + BF16_RESIDENT_TPB - 1) // BF16_RESIDENT_TPB, 1, 1),
            block_dim=(BF16_RESIDENT_TPB, 1, 1),
        )
    else:
        raise Error("bf16 resident: built without -D MOJOLEARN_GEMM_FAST_BF16_RESIDENT")


struct Bf16Resident(Movable):
    """One owner's resident bf16 images of its FP32 device operands (a model's weights): narrowed once, read by
    every product until the owner reports a rewrite. Keyed by the source buffer's device address and cell count.
    An image is never dropped while the owner lives (a queued launch may still read it): a rewrite marks it stale
    and the next `image_of` narrows it again in place, stream-ordered after its earlier readers. The owner
    destroys this after its own synchronize, as it does its other device buffers."""

    var addrs: List[Int]
    var counts: List[Int]
    var stale: List[Bool]
    var images: List[Bf16Image]
    #: narrow launches made (tests and the kernel-clock rows read it: one per weight per change)
    var narrows: Int

    def __init__(out self):
        self.addrs = List[Int]()
        self.counts = List[Int]()
        self.stale = List[Bool]()
        self.images = List[Bf16Image]()
        self.narrows = 0

    def image_of(mut self, ctx: DeviceContext, mut src: DeviceBuffer[DType.float32], cells: Int) raises -> Int:
        """The index of the image of `src[0, cells)`, narrowing it now if it has none or is stale. Per call: a
        scan of this owner's images; a launch only on the first use after a change."""
        var addr = Int(src.unsafe_ptr())
        for i in range(len(self.addrs)):  # driver-loop: over this owner's resident images (its weights)
            if self.addrs[i] == addr and self.counts[i] == cells:
                if self.stale[i]:
                    bf16_image_narrow(ctx, self.images[i], src, cells)
                    self.stale[i] = False
                    self.narrows += 1
                return i
        var img = Bf16Image(ctx, cells)
        bf16_image_narrow(ctx, img, src, cells)
        self.addrs.append(addr)
        self.counts.append(cells)
        self.stale.append(False)
        self.images.append(img^)
        self.narrows += 1
        return len(self.addrs) - 1

    def changed(mut self, mut src: DeviceBuffer[DType.float32]):
        """The FP32 values of `src` were rewritten (an optimizer step, an upload): every image whose source range
        starts inside `src` is stale."""
        var lo = Int(src.unsafe_ptr())
        var hi = lo + 4 * len(src)
        for i in range(len(self.addrs)):  # driver-loop: over this owner's resident images (its weights)
            if self.addrs[i] >= lo and self.addrs[i] < hi:
                self.stale[i] = True

    def changed_all(mut self):
        """Every source may have been rewritten: every image is stale."""
        for i in range(len(self.stale)):  # driver-loop: over this owner's resident images (its weights)
            self.stale[i] = True

    def product(
        mut self,
        ctx: DeviceContext,
        mut c: DeviceBuffer[DType.float32],
        mut a: Bf16Image,
        b_index: Int,
        m: Int,
        n: Int,
        k: Int,
        b_transposed: Bool,
    ) raises:
        """C (FP32) = A . image[b_index]: an activation image against a resident weight image, one launch."""
        if b_index < 0 or b_index >= len(self.images):
            raise Error("bf16 resident: no image at index " + String(b_index))
        bf16_resident_product(ctx, c, a, self.images[b_index], m, n, k, b_transposed)
