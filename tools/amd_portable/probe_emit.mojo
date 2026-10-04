# AMD portable-path probe 1 (compile only): emit unoptimized/optimized LLVM IR
# and ISA of pkernels.mojo for several AMD targets. Output stream uses
# "=== <kernel> <target> <kind>" separators; split with split_emit.py.
from std.compile import compile_info
from std.gpu.host import get_gpu_target
from pkernels import k_muladd, k_pinned, k_math, k_dot




def emit[target_name: StaticString]():
    comptime T = get_gpu_target[target_name]()
    comptime for kind in ["llvm", "llvm-opt", "asm"]:
        print("=== k_muladd", target_name, kind)
        print(compile_info[k_muladd, emission_kind=kind, target=T]().asm)
        print("=== k_pinned", target_name, kind)
        print(compile_info[k_pinned, emission_kind=kind, target=T]().asm)
        print("=== k_math", target_name, kind)
        print(compile_info[k_math, emission_kind=kind, target=T]().asm)
        print("=== k_dot", target_name, kind)
        print(compile_info[k_dot, emission_kind=kind, target=T]().asm)


def main():
    emit["gfx942"]()
    emit["gfx90a"]()
    emit["gfx1100"]()
    emit["gfx1201"]()
