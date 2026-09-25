# lane/nvidia-step-time (2026-09-25): PTX of the GEMM kernels a T3 shard runs
# on NVIDIA, at their shipped instantiations, for `ptxas -v` (registers,
# spill stores and loads, shared memory, stack) on the rented box:
#   tuned128   identical_gemm_tuned_kernel, PLAN_TUNED_128_8X8
#   kpack_all  identical_gemm_kpack_kernel, the kpack-hg body, all leaves
#              (lib_gemm_kernel_body_for NVIDIA = 1: every TUNED call)
#   kpack_grp  the same, GROUP (ksplit) mode: every long-k call (S = 132)
#   pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_NVIDIA \
#       -I . tools/nvidia_step_time/probe_gemm_ptx.mojo -o <bin>
#   <bin> > all.ptx   (one "### <name>" line before each kernel's PTX)
from std.compile import compile_info
from std.gpu.host import get_gpu_target
from gemm.checks.gemm_identical import *
from gemm.checks.gemm_identical import TARGET_COLUMN, lib_smem_pages_for, GEMM_LAUNCH_BOUND

comptime T = get_gpu_target["sm_90a"]()


def main():
    comptime TR = TUNED_TPB // TUNED_TC
    comptime P_t = lib_smem_pages_for[
        TARGET_COLUMN, (2 * TUNED_RPT * TR + 2 * TUNED_CPT * TUNED_TC) * (16 + TUNED_VECLEN) * 4
    ]()
    comptime KB = (GEMM_KPACK_RPT * TR * GEMM_KPACK_KS + TR * GEMM_KPACK_PAD
                   + GEMM_KPACK_CPT * TUNED_TC * GEMM_KPACK_KS + TUNED_TC * GEMM_KPACK_PAD) * 4
    comptime P_k = lib_smem_pages_for[TARGET_COLUMN, KB + GEMM_KPACK_PAGE_GUARD_BYTES]()
    print("### config launch_bound=" + String(GEMM_LAUNCH_BOUND) + " tuned_pages=" + String(P_t)
          + " kpack_pages=" + String(P_k) + " kpack_page_bytes=" + String(KB))
    print("### tuned128")
    print(compile_info[identical_gemm_tuned_kernel[TUNED_RPT * 2, TUNED_CPT * 2, TUNED_TC, 16, TUNED_FOLD_SLOTS, P_t], emission_kind="asm", target=T]().asm)
    print("### kpack_all")
    print(compile_info[identical_gemm_kpack_kernel[
        GEMM_KPACK_RPT, GEMM_KPACK_CPT, TUNED_TC, GEMM_KPACK_KS, GEMM_KPACK_FS, P_k, False, False,
        GEMM_KPACK_PAD, GEMM_KPACK_ALIGN, 0, True, True], emission_kind="asm", target=T]().asm)
    print("### kpack_grp")
    print(compile_info[identical_gemm_kpack_kernel[
        GEMM_KPACK_RPT, GEMM_KPACK_CPT, TUNED_TC, GEMM_KPACK_KS, GEMM_KPACK_FS, P_k, True, False,
        GEMM_KPACK_PAD, GEMM_KPACK_ALIGN, 0, True, True], emission_kind="asm", target=T]().asm)
    print("### end")
