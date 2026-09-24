# lane/amd-step-time: which `@__llvm_metadata` spellings reach the gfx942
# kernel attributes: the launch bound (MAX_THREADS_PER_BLOCK_METADATA, which
# lowers to rocdl.flat_work_group_size) and a waves-per-EU floor
# (rocdl.waves_per_eu -> "amdgpu-waves-per-eu").
from std.compile import compile_info
from std.gpu.host import get_gpu_target
from std.gpu import thread_idx, MAX_THREADS_PER_BLOCK_METADATA
from std.utils import StaticTuple

comptime T = get_gpu_target["mi300x"]()


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256))
def k1(p: MutPointer[Float32, MutAnyOrigin]):
    p.unsafe_store(Int(thread_idx.x), Float32(1.0))


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256), `rocdl.waves_per_eu`=Int32(2))
def k3(p: MutPointer[Float32, MutAnyOrigin]):
    p.unsafe_store(Int(thread_idx.x), Float32(1.0))


def main():
    print("## k1 amd"); print(compile_info[k1, emission_kind="llvm", target=T]().asm)
    print("## k3 amd"); print(compile_info[k3, emission_kind="llvm", target=T]().asm)
