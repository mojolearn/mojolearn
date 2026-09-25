# tools/release_0819/apple_compile_probe.mojo -- lane/release-0819
# (2026-09-25): compile-only check, for the Apple Metal target, of the kernels
# the merged 0.8.19 source changed in code every column compiles: the
# one-block embedding run-start scan (new kernel, every column) and the two
# attention kernels that now carry a launch-bound declaration (1024 on Apple).
# Emits LLVM (AIR) for target "apple-m4" and prints each kernel's attribute
# lines. Build with -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_APPLE -I .
# on any host (no GPU needed); a Metal library is NOT produced here.
from std.compile import compile_info
from std.gpu.host import get_gpu_target

from embedding.checks.embedding_identical import emb_run_begin_block_kernel
from transformer.impl.llama.fused_attention import (
    fused_attn_forward_r2_kernel,
    fused_bwd_dq_tiled_pf_kernel,
)


def _attrs(name: String, ir: String):
    print("### " + name + " bytes=" + String(len(ir)))
    for line in ir.split("\n"):
        if line.startswith("attributes #") or line.startswith("target triple"):
            print(line)


def main():
    comptime t = get_gpu_target["apple-m4"]()
    _attrs("emb_run_begin_block_kernel", String(compile_info[emb_run_begin_block_kernel, emission_kind="llvm", target=t]().asm))
    _attrs("fused_attn_forward_r2_kernel[64,32,True,True,False]",
           String(compile_info[fused_attn_forward_r2_kernel[64, 32, True, True, False], emission_kind="llvm", target=t]().asm))
    _attrs("fused_bwd_dq_tiled_pf_kernel[64]",
           String(compile_info[fused_bwd_dq_tiled_pf_kernel[64], emission_kind="llvm", target=t]().asm))
    print("APPLE_COMPILE_PROBE done")
