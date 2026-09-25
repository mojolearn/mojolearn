# lane/amd-step-time-2 (2026-09-25): gfx942 assembly of the attention dq
# kernels (the shipped VALU `fused_bwd_dq_tiled_pf_kernel` and the matrix-core
# `fused_bwd_dq_mfma_kernel`), for tools/amd_codegen/mfma_census.py.
from std.compile import compile_info
from std.gpu.host import get_gpu_target
from transformer.impl.llama.fused_attention import (
    fused_bwd_dq_tiled_pf_kernel,
    fused_bwd_dq_mfma_kernel,
    fused_bwd_dkdv_r2_kernel,
    fused_bwd_dkdv_mfma_kernel,
)

comptime T = get_gpu_target["mi300x"]()


def main():
    print("### dq_valu asm")
    print(compile_info[fused_bwd_dq_tiled_pf_kernel[64, True], emission_kind="asm", target=T]().asm)
    print("### dq_mfma asm")
    print(compile_info[fused_bwd_dq_mfma_kernel[64, True], emission_kind="asm", target=T]().asm)
    print("### dkdv_valu asm")
    print(compile_info[fused_bwd_dkdv_r2_kernel[64, 32, False, True], emission_kind="asm", target=T]().asm)
    print("### dkdv_mfma asm")
    print(compile_info[fused_bwd_dkdv_mfma_kernel[64, True], emission_kind="asm", target=T]().asm)
    print("### end")
