# lane/amd-step-time-2 (2026-09-25): gfx942 assembly of the matrix-core GEMM
# kernel (`identical_gemm_mfma_kernel`) at its shipped instantiation, with
# and without the exact admission, whole-leaf and group modes. Compiles on a
# CPU box (no device needed):
#   pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_AMD \
#       -I . tools/amd_codegen/probe_mfma_gemm.mojo -o <bin> && <bin> > mfma.s
# Sections start with `### <name>`; tools/amd_codegen/mfma_census.py counts
# the instructions of each.
from std.compile import compile_info
from std.gpu.host import get_gpu_target
from gemm.checks.gemm_identical import *
from gemm.checks.gemm_identical import TARGET_COLUMN, lib_smem_pages_for

comptime T = get_gpu_target["mi300x"]()


def main():
    comptime KS = 16
    comptime PB = (128 + 128) * (KS + TUNED_VECLEN) * 4
    comptime P = lib_smem_pages_for[TARGET_COLUMN, PB]()
    print("### config pages=" + String(P) + " admit_default=" + String(GEMM_MFMA_ADMIT))
    print("### mfma_plain asm")
    print(compile_info[identical_gemm_mfma_kernel[KS, TUNED_FOLD_SLOTS, P, False, False], emission_kind="asm", target=T]().asm)
    print("### mfma_admit asm")
    print(compile_info[identical_gemm_mfma_kernel[KS, TUNED_FOLD_SLOTS, P, False, True], emission_kind="asm", target=T]().asm)
    print("### mfma_admit_group asm")
    print(compile_info[identical_gemm_mfma_kernel[KS, TUNED_FOLD_SLOTS, P, True, True], emission_kind="asm", target=T]().asm)
    print("### end")
