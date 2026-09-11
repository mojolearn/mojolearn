# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Static resources of the shipped GEMM specialization and every step arm
geometry, on ANY vendor (DEVIATION 2543).

    pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 \\
        -D MOJOLEARN_GEMM_ARM_TRIAL=1 -I . bench/gemm_step_resources_main.mojo -o <bin>
    <bin>

`DeviceContext.compile_function[kern]()`, then the runtime's own attributes
(`NUM_REGS`, `LOCAL_SIZE_BYTES`, `SHARED_SIZE_BYTES`, `CONST_SIZE_BYTES`,
`MAX_THREADS_PER_BLOCK`) and `occupancy_max_active_blocks_per_multiprocessor
(256, 0)`, the calls the kNN lane read on an H100
(`bench/results/e1g/2026-09-11_014151-nvidia/remote/knn-kernel-stats/`).
Launches nothing. Each attribute is its own print, in that order, and each
geometry its own try, so a vendor that answers some attributes and raises on
another (the occupancy query is the likeliest) keeps what it answered, and
the raise is printed as GEMM_STEP_RESOURCES_ERROR. On AMD nothing about the
register budget or occupancy is known until this prints (brief section 3).

No vendor instrument is required: no ptxas, no `--emit asm`. Spill counts
stay an NVIDIA confirmation item (tools/gemm_cuda_resources.py).

Rows: the shipped `identical_gemm_tuned_kernel` at 128x128; the trimmed arm
kernel at the same geometry with the shipped lane-wide fold (`LFOLD =
False`, the control: it should read like the shipped row); then `lfold`,
`half`, `half_ks16`, `quarter`, `head` N-wide and `head` M-wide; then
`ksplit_128x128` (DEVIATION 2593), the long-k group kernel
`identical_gemm_ksplit_kernel` at the shipped geometry, which should also
read like the shipped row (docs/lanes/BRIEF_gemm_long_k_2026-09-11.md
section 5.1: same per-window body, two more Int32 arguments, a 2-D grid).

DEVIATION 2595 (that brief's section 10): `shipped_128x128` is the TUNED
128x128 specialization, which the shipped dispatch still runs wherever the
ksplit default does not take a call (every call on a column whose block
parallelism row is 0). On a column whose row is above 0 the calls the group
rule takes run the `ksplit_128x128` row's kernel. The
`GEMM_STEP_RESOURCES_GEOMETRY label=shipped_default` line says which is true
on this build.

DEVIATION 2599 (docs/lanes/BRIEF_gemm_kernel_2026-09-11.md section 6): four
rows of `identical_gemm_kpack_kernel`, `kpack_all` and `kpack_group` at the
128x128 geometry (compare with `shipped_128x128` and `ksplit_128x128`: same
tile, packed page, per-step loads) and `kpack_wide_all` and
`kpack_wide_group` at 128x256 (reg 8x16). Registers, local bytes (spills
show here beside the fold stack) and blocks per SM are what brief section 4.2
reads for C4.
"""
from max.gpu.host import Attribute, DeviceContext

from checks.kernel_matrix import (
    TARGET_COLUMN,
    column_lane_width,
    column_shared_limit,
    lib_smem_pages_for,
)
from checks.numerics import numeric_mode_name
from gemm.checks.gemm_identical import (
    GEMM_GEOM_HALF,
    GEMM_GEOM_HALF_KS16,
    GEMM_GEOM_KPACK,
    GEMM_GEOM_KPACK_WIDE,
    GEMM_KPACKW_CPT,
    GEMM_KPACKW_FS,
    GEMM_KPACKW_KS,
    GEMM_KPACKW_RPT,
    GEMM_KPACK_CPT,
    GEMM_KPACK_FS,
    GEMM_KPACK_KS,
    GEMM_KPACK_PAGE_GUARD_BYTES,
    GEMM_KPACK_RPT,
    GEMM_GEOM_HEAD_M,
    GEMM_GEOM_HEAD_N,
    GEMM_GEOM_KSPLIT,
    GEMM_GEOM_LFOLD,
    GEMM_GEOM_QUARTER,
    GEMM_GEOM_SHIPPED,
    GEMM_HALF_KS,
    GEMM_KSPLIT_CPT,
    GEMM_KSPLIT_KS,
    GEMM_KSPLIT_RPT,
    GEMM_HEAD_W,
    GEMM_HEADM_CPT,
    GEMM_HEADM_RPT,
    GEMM_HEADM_TC,
    GEMM_HEADN_CPT,
    GEMM_HEADN_RPT,
    GEMM_HEADN_TC,
    TUNED_64_KS,
    TUNED_CPT,
    TUNED_FOLD_SLOTS,
    TUNED_RPT,
    TUNED_TC,
    TUNED_TPB,
    TUNED_VECLEN,
    gemm_step_geometry_name,
    identical_gemm_kpack_kernel,
    identical_gemm_ksplit_kernel,
    identical_gemm_step_arm_kernel,
    identical_gemm_tuned_kernel,
)


def _stat_shipped(ctx: DeviceContext, label: String) raises:
    comptime RPT = TUNED_RPT * 2
    comptime CPT = TUNED_CPT * 2
    comptime TR = TUNED_TPB // TUNED_TC
    comptime BM = RPT * TR
    comptime BN = CPT * TUNED_TC
    comptime PAGES = lib_smem_pages_for[TARGET_COLUMN, (BM + BN) * (16 + TUNED_VECLEN) * 4]()
    comptime kern = identical_gemm_tuned_kernel[RPT, CPT, TUNED_TC, 16, TUNED_FOLD_SLOTS, PAGES]
    print("GEMM_STEP_RESOURCES_BEGIN label=", label, " tile=", BM, "x", BN, " pages=", PAGES, sep="")
    var f = ctx.compile_function[kern]()
    print("GEMM_STEP_RESOURCES label=", label, " regs=", f.get_attribute(Attribute.NUM_REGS), sep="")
    print("GEMM_STEP_RESOURCES label=", label, " local=", f.get_attribute(Attribute.LOCAL_SIZE_BYTES), sep="")
    print("GEMM_STEP_RESOURCES label=", label, " shared=", f.get_attribute(Attribute.SHARED_SIZE_BYTES), sep="")
    print("GEMM_STEP_RESOURCES label=", label, " const=", f.get_attribute(Attribute.CONST_SIZE_BYTES), sep="")
    print("GEMM_STEP_RESOURCES label=", label, " max_threads=", f.get_attribute(Attribute.MAX_THREADS_PER_BLOCK), sep="")
    print(
        "GEMM_STEP_RESOURCES label=", label, " blocks_per_sm_256=",
        f.occupancy_max_active_blocks_per_multiprocessor(TUNED_TPB, 0), sep="",
    )


def _stat_arm[
    RPT: Int, CPT: Int, TC: Int, KS: Int, LFOLD: Bool
](ctx: DeviceContext, label: String) raises:
    comptime TR = TUNED_TPB // TC
    comptime BM = RPT * TR
    comptime BN = CPT * TC
    comptime PAGES = lib_smem_pages_for[TARGET_COLUMN, (BM + BN) * (KS + TUNED_VECLEN) * 4]()
    comptime kern = identical_gemm_step_arm_kernel[RPT, CPT, TC, KS, PAGES, LFOLD, False]
    print(
        "GEMM_STEP_RESOURCES_BEGIN label=", label, " tile=", BM, "x", BN, " reg=", RPT, "x", CPT,
        " tc=", TC, " ks=", KS, " pages=", PAGES, " lfold=", LFOLD, sep="",
    )
    var f = ctx.compile_function[kern]()
    print("GEMM_STEP_RESOURCES label=", label, " regs=", f.get_attribute(Attribute.NUM_REGS), sep="")
    print("GEMM_STEP_RESOURCES label=", label, " local=", f.get_attribute(Attribute.LOCAL_SIZE_BYTES), sep="")
    print("GEMM_STEP_RESOURCES label=", label, " shared=", f.get_attribute(Attribute.SHARED_SIZE_BYTES), sep="")
    print("GEMM_STEP_RESOURCES label=", label, " const=", f.get_attribute(Attribute.CONST_SIZE_BYTES), sep="")
    print("GEMM_STEP_RESOURCES label=", label, " max_threads=", f.get_attribute(Attribute.MAX_THREADS_PER_BLOCK), sep="")
    print(
        "GEMM_STEP_RESOURCES label=", label, " blocks_per_sm_256=",
        f.occupancy_max_active_blocks_per_multiprocessor(TUNED_TPB, 0), sep="",
    )


def _stat_ksplit(ctx: DeviceContext, label: String) raises:
    """DEVIATION 2593: the group kernel at the shipped 128x128 geometry, clean."""
    comptime TR = TUNED_TPB // TUNED_TC
    comptime BM = GEMM_KSPLIT_RPT * TR
    comptime BN = GEMM_KSPLIT_CPT * TUNED_TC
    comptime PAGES = lib_smem_pages_for[
        TARGET_COLUMN, (BM + BN) * (GEMM_KSPLIT_KS + TUNED_VECLEN) * 4
    ]()
    comptime kern = identical_gemm_ksplit_kernel[
        GEMM_KSPLIT_RPT, GEMM_KSPLIT_CPT, TUNED_TC, GEMM_KSPLIT_KS, PAGES, False
    ]
    print(
        "GEMM_STEP_RESOURCES_BEGIN label=", label, " tile=", BM, "x", BN, " reg=", GEMM_KSPLIT_RPT,
        "x", GEMM_KSPLIT_CPT, " tc=", TUNED_TC, " ks=", GEMM_KSPLIT_KS, " pages=", PAGES,
        " groups=grid.y", sep="",
    )
    var f = ctx.compile_function[kern]()
    print("GEMM_STEP_RESOURCES label=", label, " regs=", f.get_attribute(Attribute.NUM_REGS), sep="")
    print("GEMM_STEP_RESOURCES label=", label, " local=", f.get_attribute(Attribute.LOCAL_SIZE_BYTES), sep="")
    print("GEMM_STEP_RESOURCES label=", label, " shared=", f.get_attribute(Attribute.SHARED_SIZE_BYTES), sep="")
    print("GEMM_STEP_RESOURCES label=", label, " const=", f.get_attribute(Attribute.CONST_SIZE_BYTES), sep="")
    print("GEMM_STEP_RESOURCES label=", label, " max_threads=", f.get_attribute(Attribute.MAX_THREADS_PER_BLOCK), sep="")
    print(
        "GEMM_STEP_RESOURCES label=", label, " blocks_per_sm_256=",
        f.occupancy_max_active_blocks_per_multiprocessor(TUNED_TPB, 0), sep="",
    )


def _stat_kpack[
    RPT: Int, CPT: Int, TC: Int, KS: Int, FS: Int, GROUP: Bool
](ctx: DeviceContext, label: String) raises:
    """DEVIATION 2599: the packed-page kernel, clean, with the PAGES its
    launcher binds (the matrix row at the packed page bytes plus the guard)."""
    comptime TR = TUNED_TPB // TC
    comptime BM = RPT * TR
    comptime BN = CPT * TC
    comptime PAGE_BYTES = (BM + BN) * KS * 4
    comptime PAGES = lib_smem_pages_for[TARGET_COLUMN, PAGE_BYTES + GEMM_KPACK_PAGE_GUARD_BYTES]()
    comptime kern = identical_gemm_kpack_kernel[RPT, CPT, TC, KS, FS, PAGES, GROUP, False]
    print(
        "GEMM_STEP_RESOURCES_BEGIN label=", label, " tile=", BM, "x", BN, " reg=", RPT, "x", CPT,
        " tc=", TC, " ks=", KS, " fs=", FS, " page_bytes=", PAGE_BYTES, " pages=", PAGES,
        " group=", GROUP, sep="",
    )
    var f = ctx.compile_function[kern]()
    print("GEMM_STEP_RESOURCES label=", label, " regs=", f.get_attribute(Attribute.NUM_REGS), sep="")
    print("GEMM_STEP_RESOURCES label=", label, " local=", f.get_attribute(Attribute.LOCAL_SIZE_BYTES), sep="")
    print("GEMM_STEP_RESOURCES label=", label, " shared=", f.get_attribute(Attribute.SHARED_SIZE_BYTES), sep="")
    print("GEMM_STEP_RESOURCES label=", label, " const=", f.get_attribute(Attribute.CONST_SIZE_BYTES), sep="")
    print("GEMM_STEP_RESOURCES label=", label, " max_threads=", f.get_attribute(Attribute.MAX_THREADS_PER_BLOCK), sep="")
    print(
        "GEMM_STEP_RESOURCES label=", label, " blocks_per_sm_256=",
        f.occupancy_max_active_blocks_per_multiprocessor(TUNED_TPB, 0), sep="",
    )


def main() raises:
    var ctx = DeviceContext()
    print(
        "GEMM_STEP_RESOURCES_DEVICE name=", ctx.name(), " mode=", numeric_mode_name(),
        " column=", TARGET_COLUMN, " lane_width=", column_lane_width(TARGET_COLUMN),
        " shared_limit=", column_shared_limit(TARGET_COLUMN), " head_w=", GEMM_HEAD_W,
        " tpb=", TUNED_TPB, sep="",
    )
    print("GEMM_STEP_RESOURCES_GEOMETRY label=shipped_default ", gemm_step_geometry_name(GEMM_GEOM_SHIPPED), sep="")
    try:
        _stat_shipped(ctx, String("shipped_128x128"))
    except e:
        print("GEMM_STEP_RESOURCES_ERROR label=shipped_128x128 error=", e, sep="")
    try:
        _stat_arm[TUNED_RPT * 2, TUNED_CPT * 2, TUNED_TC, 16, False](ctx, String("control_128x128_lanefold"))
    except e:
        print("GEMM_STEP_RESOURCES_ERROR label=control_128x128_lanefold error=", e, sep="")
    print("GEMM_STEP_RESOURCES_GEOMETRY label=lfold ", gemm_step_geometry_name(GEMM_GEOM_LFOLD), sep="")
    try:
        _stat_arm[TUNED_RPT * 2, TUNED_CPT * 2, TUNED_TC, 16, True](ctx, String("lfold"))
    except e:
        print("GEMM_STEP_RESOURCES_ERROR label=lfold error=", e, sep="")
    print("GEMM_STEP_RESOURCES_GEOMETRY label=half ", gemm_step_geometry_name(GEMM_GEOM_HALF), sep="")
    try:
        _stat_arm[TUNED_RPT, TUNED_CPT * 2, TUNED_TC, GEMM_HALF_KS, True](ctx, String("half"))
    except e:
        print("GEMM_STEP_RESOURCES_ERROR label=half error=", e, sep="")
    print("GEMM_STEP_RESOURCES_GEOMETRY label=half_ks16 ", gemm_step_geometry_name(GEMM_GEOM_HALF_KS16), sep="")
    try:
        _stat_arm[TUNED_RPT, TUNED_CPT * 2, TUNED_TC, 16, True](ctx, String("half_ks16"))
    except e:
        print("GEMM_STEP_RESOURCES_ERROR label=half_ks16 error=", e, sep="")
    print("GEMM_STEP_RESOURCES_GEOMETRY label=quarter ", gemm_step_geometry_name(GEMM_GEOM_QUARTER), sep="")
    try:
        _stat_arm[TUNED_RPT, TUNED_CPT, TUNED_TC, TUNED_64_KS, True](ctx, String("quarter"))
    except e:
        print("GEMM_STEP_RESOURCES_ERROR label=quarter error=", e, sep="")
    print("GEMM_STEP_RESOURCES_GEOMETRY label=head_n ", gemm_step_geometry_name(GEMM_GEOM_HEAD_N), sep="")
    try:
        _stat_arm[GEMM_HEADN_RPT, GEMM_HEADN_CPT, GEMM_HEADN_TC, 16, True](ctx, String("head_n"))
    except e:
        print("GEMM_STEP_RESOURCES_ERROR label=head_n error=", e, sep="")
    print("GEMM_STEP_RESOURCES_GEOMETRY label=head_m ", gemm_step_geometry_name(GEMM_GEOM_HEAD_M), sep="")
    try:
        _stat_arm[GEMM_HEADM_RPT, GEMM_HEADM_CPT, GEMM_HEADM_TC, 16, True](ctx, String("head_m"))
    except e:
        print("GEMM_STEP_RESOURCES_ERROR label=head_m error=", e, sep="")
    print("GEMM_STEP_RESOURCES_GEOMETRY label=ksplit_128x128 ", gemm_step_geometry_name(GEMM_GEOM_KSPLIT), sep="")
    try:
        _stat_ksplit(ctx, String("ksplit_128x128"))
    except e:
        print("GEMM_STEP_RESOURCES_ERROR label=ksplit_128x128 error=", e, sep="")
    print("GEMM_STEP_RESOURCES_GEOMETRY label=kpack ", gemm_step_geometry_name(GEMM_GEOM_KPACK), sep="")
    try:
        _stat_kpack[GEMM_KPACK_RPT, GEMM_KPACK_CPT, TUNED_TC, GEMM_KPACK_KS, GEMM_KPACK_FS, False](
            ctx, String("kpack_all")
        )
    except e:
        print("GEMM_STEP_RESOURCES_ERROR label=kpack_all error=", e, sep="")
    try:
        _stat_kpack[GEMM_KPACK_RPT, GEMM_KPACK_CPT, TUNED_TC, GEMM_KPACK_KS, GEMM_KPACK_FS, True](
            ctx, String("kpack_group")
        )
    except e:
        print("GEMM_STEP_RESOURCES_ERROR label=kpack_group error=", e, sep="")
    print("GEMM_STEP_RESOURCES_GEOMETRY label=kpack_wide ", gemm_step_geometry_name(GEMM_GEOM_KPACK_WIDE), sep="")
    try:
        _stat_kpack[GEMM_KPACKW_RPT, GEMM_KPACKW_CPT, TUNED_TC, GEMM_KPACKW_KS, GEMM_KPACKW_FS, False](
            ctx, String("kpack_wide_all")
        )
    except e:
        print("GEMM_STEP_RESOURCES_ERROR label=kpack_wide_all error=", e, sep="")
    try:
        _stat_kpack[GEMM_KPACKW_RPT, GEMM_KPACKW_CPT, TUNED_TC, GEMM_KPACKW_KS, GEMM_KPACKW_FS, True](
            ctx, String("kpack_wide_group")
        )
    except e:
        print("GEMM_STEP_RESOURCES_ERROR label=kpack_wide_group error=", e, sep="")
    print("GEMM_STEP_RESOURCES_DONE")
