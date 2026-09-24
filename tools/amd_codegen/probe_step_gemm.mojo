# lane/amd-step-time (2026-09-24): gfx942 assembly of the two GEMM kernels
# every call of a T3 shard runs on AMD, at their shipped instantiations:
#   tuned128   identical_gemm_tuned_kernel, PLAN_TUNED_128_8X8 (AMD's k = 768
#              calls with m >= 4096: projections forward, head forward, dA)
#   kpack_all  identical_gemm_kpack_kernel, the kpack-hg body, all leaves
#   kpack_grp  the same, GROUP (ksplit) mode: every long-k call
# Print the whole asm per kernel; grep the .amdhsa_/vgpr/sgpr/scratch/spill/
# lds lines and count v_fma/v_pk_fma/v_cmp_class/s_nop/scratch_ in the loop.
#   pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_AMD \
#       --target-accelerator gfx942 -I . tools/amd_codegen/probe_step_gemm.mojo -o <bin>
from std.compile import compile_info
from std.gpu.host import get_gpu_target
from gemm.checks.gemm_identical import *
from gemm.checks.gemm_identical import TARGET_COLUMN, lib_smem_pages_for, GEMM_EXCP_FAST

comptime T = get_gpu_target["mi300x"]()


def main():
    comptime TR = TUNED_TPB // TUNED_TC
    comptime P_t = lib_smem_pages_for[
        TARGET_COLUMN, (2 * TUNED_RPT * TR + 2 * TUNED_CPT * TUNED_TC) * (16 + TUNED_VECLEN) * 4
    ]()
    comptime KB = (GEMM_KPACK_RPT * TR * GEMM_KPACK_KS + TR * GEMM_KPACK_PAD
                   + GEMM_KPACK_CPT * TUNED_TC * GEMM_KPACK_KS + TUNED_TC * GEMM_KPACK_PAD) * 4
    comptime P_k = lib_smem_pages_for[TARGET_COLUMN, KB + GEMM_KPACK_PAGE_GUARD_BYTES]()
    print("### config excp_fast=" + String(GEMM_EXCP_FAST) + " tuned_pages=" + String(P_t)
          + " kpack_pages=" + String(P_k) + " kpack_page_bytes=" + String(KB))
    print("### tuned128 asm")
    print(compile_info[identical_gemm_tuned_kernel[TUNED_RPT * 2, TUNED_CPT * 2, TUNED_TC, 16, TUNED_FOLD_SLOTS, P_t], emission_kind="asm", target=T]().asm)
    print("### kpack_all asm")
    print(compile_info[identical_gemm_kpack_kernel[
        GEMM_KPACK_RPT, GEMM_KPACK_CPT, TUNED_TC, GEMM_KPACK_KS, GEMM_KPACK_FS, P_k, False, False,
        GEMM_KPACK_PAD, GEMM_KPACK_ALIGN, 0, True, True], emission_kind="asm", target=T]().asm)
    print("### kpack_grp asm")
    print(compile_info[identical_gemm_kpack_kernel[
        GEMM_KPACK_RPT, GEMM_KPACK_CPT, TUNED_TC, GEMM_KPACK_KS, GEMM_KPACK_FS, P_k, True, False,
        GEMM_KPACK_PAD, GEMM_KPACK_ALIGN, 0, True, True], emission_kind="asm", target=T]().asm)
    print("### end")
