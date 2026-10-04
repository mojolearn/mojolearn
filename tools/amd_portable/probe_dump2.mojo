# AMD portable-path probe 6: does enqueue_function take the dump_llvm hook,
# so an EXTRACTION build (never shipped) embeds every launched kernel's
# optimized IR as text that extract_embedded_ir.py can read without running?
from max.gpu.host import DeviceContext
from std.pathlib import Path
from std.sys import argv
from pkernels import k_muladd, k_dot


def main() raises:
    var ctx = DeviceContext()
    var o = ctx.enqueue_create_buffer[DType.float32](64)
    var a = ctx.enqueue_create_buffer[DType.float32](64)
    var b = ctx.enqueue_create_buffer[DType.float32](64)
    var c = ctx.enqueue_create_buffer[DType.float32](64)
    var op = o.unsafe_ptr()
    var ap = a.unsafe_ptr()
    var bp = b.unsafe_ptr()
    var cp = c.unsafe_ptr()
    if len(argv()) > 5:  # not launched in the probe run; IR must be embedded anyway
        ctx.enqueue_function[k_muladd, dump_llvm=Path("/root/lq/amd-portable/dump2/k_muladd.ll")](
            op, ap, bp, cp, Int64(0), grid_dim=(1, 1, 1), block_dim=(64, 1, 1))
        ctx.enqueue_function[k_dot, dump_llvm=Path("/root/lq/amd-portable/dump2/k_dot.ll")](
            op, ap, bp, Int64(0), grid_dim=(1, 1, 1), block_dim=(64, 1, 1))
    ctx.synchronize()
    print("ok")
