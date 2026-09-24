# lane/amd-step-time: does `@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=...)`
# reach the gfx942 kernel attributes, and does an EMPTY tuple emit nothing
# (the spelling a column that wants no bound would use)?
from std.compile import compile_info
from std.gpu.host import get_gpu_target
from std.gpu import thread_idx, MAX_THREADS_PER_BLOCK_METADATA
from std.utils import StaticTuple

comptime T = get_gpu_target["mi300x"]()
comptime TN = get_gpu_target["sm_90a"]()


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256))
def k1(p: MutPointer[Float32, MutAnyOrigin]):
    p.unsafe_store(Int(thread_idx.x), Float32(1.0))


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 0]())
def k0(p: MutPointer[Float32, MutAnyOrigin]):
    p.unsafe_store(Int(thread_idx.x), Float32(1.0))


def k2(p: MutPointer[Float32, MutAnyOrigin]):
    p.unsafe_store(Int(thread_idx.x), Float32(1.0))


def main():
    print("## k1 amd"); print(compile_info[k1, emission_kind="llvm", target=T]().asm)
    print("## k0 amd"); print(compile_info[k0, emission_kind="llvm", target=T]().asm)
    print("## k2 amd"); print(compile_info[k2, emission_kind="llvm", target=T]().asm)
    print("## k0 nv"); print(compile_info[k0, emission_kind="llvm", target=TN]().asm)
    print("## k2 nv"); print(compile_info[k2, emission_kind="llvm", target=TN]().asm)
