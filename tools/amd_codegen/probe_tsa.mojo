# Emit the gfx942 LLVM IR (optimized) and assembly of the Holt-Winters device
# kernels, one section per kernel, so cold compiles can be compared stage by stage.
from std.compile import compile_info
from std.gpu.host import get_gpu_target
from holtwinters.impl.internal.hw_eval import holtwinters_eval_gpu_global_kernel
from holtwinters.impl.internal.hw_optim import holtwinters_optim_gpu_global_kernel
from holtwinters.impl.internal.hw_estimate import (
    holtwinters_estimate_gpu_kernel,
    holtwinters_estimate_block_kernel,
    holtwinters_estimate_finish_kernel,
)

def main():
    comptime T = get_gpu_target["mi300x"]()
    print("### optim llvm"); print(compile_info[holtwinters_optim_gpu_global_kernel, emission_kind="llvm", target=T]().asm)
    print("### est llvm"); print(compile_info[holtwinters_estimate_gpu_kernel, emission_kind="llvm", target=T]().asm)
    print("### eval llvm-opt"); print(compile_info[holtwinters_eval_gpu_global_kernel, emission_kind="llvm-opt", target=T]().asm)
    print("### eval asm"); print(compile_info[holtwinters_eval_gpu_global_kernel, emission_kind="asm", target=T]().asm)
    print("### optim llvm-opt"); print(compile_info[holtwinters_optim_gpu_global_kernel, emission_kind="llvm-opt", target=T]().asm)
    print("### optim asm"); print(compile_info[holtwinters_optim_gpu_global_kernel, emission_kind="asm", target=T]().asm)
    print("### est llvm-opt"); print(compile_info[holtwinters_estimate_gpu_kernel, emission_kind="llvm-opt", target=T]().asm)
    print("### est asm"); print(compile_info[holtwinters_estimate_gpu_kernel, emission_kind="asm", target=T]().asm)
    print("### estblock llvm-opt"); print(compile_info[holtwinters_estimate_block_kernel, emission_kind="llvm-opt", target=T]().asm)
    print("### estblock asm"); print(compile_info[holtwinters_estimate_block_kernel, emission_kind="asm", target=T]().asm)
    print("### estfinish llvm-opt"); print(compile_info[holtwinters_estimate_finish_kernel, emission_kind="llvm-opt", target=T]().asm)
    print("### estfinish asm"); print(compile_info[holtwinters_estimate_finish_kernel, emission_kind="asm", target=T]().asm)
