# AMD portable-path probe 5: can a normal build hand us the LLVM IR of the
# kernels it embeds? Tries DeviceContext.compile_function's dump_llvm hook.
from max.gpu.host import DeviceContext
from std.pathlib import Path
from pkernels import k_muladd, k_dot


def main() raises:
    var ctx = DeviceContext()
    _ = ctx.compile_function[k_muladd, dump_llvm=Path("/root/lq/amd-portable/dump/k_muladd.ll")]()
    _ = ctx.compile_function[k_dot, dump_llvm=Path("/root/lq/amd-portable/dump/k_dot.ll")]()
    print("ok")
