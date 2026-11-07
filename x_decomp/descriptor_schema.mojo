# SPDX-License-Identifier: Apache-2.0
# The x_decomp native kernel-call table (plan NATIVE_ENTRY_POINTS_AFTER_0866,
# workstream F, lane ne3-f-kernel-table, 2026-11-07), in the shape of
# x_neighbors/descriptor_schema.mojo: an ABI name, an operation count and the
# operation ids. Each image exports one `abi("C")` dispatch function over the
# shared NativeDescriptorCall ABI (core/native_descriptor_abi.mojo) as the
# capsule `x_decomp_native_api_v1`:
#   bindings/_mojolearn_x_decomp.mojo       x_decomp/descriptor_device.mojo
#   bindings/_mojolearn_x_decomp_host.mojo  x_decomp/descriptor_host.mojo
# The kits (bindings/mani_kit.mojo first) connect it once per binding module
# (neural_descriptor_connect_once) and call `kit_call(provider, OP, ...)`
# (bindings/kit_call.mojo) instead of one Python-level entry call, one boxed
# argument tuple and one Python parameter list per primitive operation.
#
# Every operation runs the SAME typed body the matching `*_py` door runs
# (x_decomp/api.mojo, x_decomp/resident.mojo), with the same integer checks,
# the same float32 rounding of scalars and the same operand order: no numeric
# change. Device ids (the x_decomp pool's) travel in the integer span; host
# buffers in the address span; float scalars in the Float64 span.
comptime XD_ABI_NAME = "mojolearn.x_decomp.native.v1"

comptime XD_MODE = 0           # -> GLOBAL_NUMERIC_MODE of the image
comptime XD_CAPABILITIES = 1   # -> XD_CAP_* bits
# Host-address operations (both images; the image's own executor).
comptime XD_EW = 2             # a: [a, b, c, dst]  i: [op, count, d, lb, bm, lc, cm]  f: [s]
comptime XD_GEMM = 3           # a: [a, b, c]       i: [m, k, n, ta, tb]
comptime XD_COLSUM = 4         # a: [a, dst]        i: [n, d]
comptime XD_ROWSUM = 5         # a: [a, dst]        i: [n, d]
comptime XD_SQDIST = 6         # a: [a, b, dst]     i: [na, nb, d, kind]       f: [pw]
comptime XD_MOVE = 7           # a: [src, idx, dst] i: [op, count, a1, a2, a3, ist, ioff, nsrc, ndst, nidx]
# Device-resident operations (GPU image only; the host image refuses them).
comptime XD_DEV_ALLOC = 8      # i: [n]                    -> id
comptime XD_DEV_FREE = 9       # i: [id]
comptime XD_DEV_UPLOAD = 10    # a: [src]  i: [id, n]
comptime XD_DEV_DOWNLOAD = 11  # a: [dst]  i: [id, n]
comptime XD_DEV_EW = 12        # i: [a, b, c, dst, op, count, d, lb, bm, lc, cm]  f: [s]
comptime XD_DEV_GEMM = 13      # i: [a, b, c, m, k, n, ta, tb]
comptime XD_DEV_COLSUM = 14    # i: [a, dst, n, d]
comptime XD_DEV_ROWSUM = 15    # i: [a, dst, n, d]
comptime XD_DEV_SQDIST = 16    # i: [a, b, dst, na, nb, d, kind]  f: [pw]
comptime XD_DEV_MOVE = 17      # i: [src, idx, dst, op, count, a1, a2, a3, ist, ioff, nsrc, ndst, nidx]
# Isomap.transform's geodesic gather-min (row N295): for each neighbor rank
# a < nn, rows im[t * nn + a] of D (dc columns, dn words; im holds imn
# words) plus column a of SQ (q x nn), folded by MIN into G (q x dc),
# exactly the take_rows / take_cols / ew ADD / ew MIN sequence the kit made
# with seven crossings per rank. -> the id of G (q x n), caller-owned.
comptime XD_DEV_ISO_GATHER_MIN = 18  # i: [d, im, sq, q, dc, nn, one, dn, imn]
comptime XD_OPERATION_COUNT = 19

comptime XD_CAP_HOST = 1       # the host-address operations
comptime XD_CAP_RESIDENT = 2   # the device-resident operations

# x_decomp/cells.mojo and moves.mojo codes the fused operation uses (the same
# values bindings/mani_kit.mojo spells as OP_* / MV_*).
comptime XD_CELL_ADD = 0
comptime XD_CELL_MIN = 31
comptime XD_MV_TAKE_ROWS = 0
comptime XD_MV_TAKE_COLS = 4


def xd_schema(op: Int) raises -> Tuple[Int, Int, Int]:
    """(addresses, integers, floats) each operation takes, exactly."""
    if op == XD_MODE or op == XD_CAPABILITIES:
        return (0, 0, 0)
    if op == XD_EW:
        return (4, 7, 1)
    if op == XD_GEMM:
        return (3, 5, 0)
    if op == XD_COLSUM or op == XD_ROWSUM:
        return (2, 2, 0)
    if op == XD_SQDIST:
        return (3, 4, 1)
    if op == XD_MOVE:
        return (3, 10, 0)
    if op == XD_DEV_ALLOC or op == XD_DEV_FREE:
        return (0, 1, 0)
    if op == XD_DEV_UPLOAD or op == XD_DEV_DOWNLOAD:
        return (1, 2, 0)
    if op == XD_DEV_EW:
        return (0, 11, 1)
    if op == XD_DEV_GEMM:
        return (0, 8, 0)
    if op == XD_DEV_COLSUM or op == XD_DEV_ROWSUM:
        return (0, 4, 0)
    if op == XD_DEV_SQDIST:
        return (0, 7, 1)
    if op == XD_DEV_MOVE:
        return (0, 13, 0)
    if op == XD_DEV_ISO_GATHER_MIN:
        return (0, 9, 0)
    raise Error("x_decomp native ABI: unknown operation " + String(op))


def xd_validate_call(op: Int, na: Int, ni: Int, nf: Int, result: Int) raises:
    var want = xd_schema(op)
    if na != want[0] or ni != want[1] or nf != want[2]:
        raise Error("x_decomp native ABI: operation " + String(op) + " takes " + String(want[0])
                    + " addresses, " + String(want[1]) + " integers and " + String(want[2]) + " floats")
    if result == 0:
        raise Error("x_decomp native ABI: null result address")


def xd_dim(v: Int) raises -> Int:
    """The `_n` bound of the `*_py` doors: a dimension in [0, 2^31 - 1]."""
    if v < 0 or v > 2147483647:
        raise Error("x_decomp: dimension out of range")
    return v
