# lane/amd-step-time: does a kernel carrying the launch-bound decorator the
# GEMM kernels now carry still compile for target "metal"?
from std.compile import compile_info
from std.gpu.host import get_gpu_target
from std.gpu import thread_idx, MAX_THREADS_PER_BLOCK_METADATA
from std.utils import StaticTuple


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256))
def k1(p: MutPointer[Float32, MutAnyOrigin]):
    p.unsafe_store(Int(thread_idx.x), Float32(1.0))


def main():
    print(compile_info[k1, emission_kind="llvm", target=get_gpu_target["metal"]()]().asm)
