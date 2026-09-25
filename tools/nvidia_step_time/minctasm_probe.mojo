# lane/nvidia-step-time (2026-09-25): which `@__llvm_metadata` spellings of a
# minimum-blocks-per-SM bound reach the sm_90a PTX (`.minnctapersm`)? Compile
# only; prints the PTX directive lines of each kernel.
#   pixi run mojo build tools/nvidia_step_time/minctasm_probe.mojo -o <bin> && <bin>
from std.compile import compile_info
from std.gpu.host import get_gpu_target
from std.gpu import thread_idx, MAX_THREADS_PER_BLOCK_METADATA
from std.utils import StaticTuple

comptime T = get_gpu_target["sm_90a"]()


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256), `nvvm.minctasm`=StaticTuple[Int32, 1](2))
def k_minctasm_i32(p: MutPointer[Float32, MutAnyOrigin]):
    p.unsafe_store(Int(thread_idx.x), Float32(1.0))


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256), `nvvm.maxnreg`=StaticTuple[Int32, 1](128))
def k_maxnreg(p: MutPointer[Float32, MutAnyOrigin]):
    p.unsafe_store(Int(thread_idx.x), Float32(1.0))


def show(name: String, asm: String):
    print("### " + name)
    for line in asm.split("\n"):
        if line.find(".maxntid") >= 0 or line.find(".minnctapersm") >= 0 or line.find(".maxnreg") >= 0 or line.find(".entry") >= 0:
            print(line)


def main():
    show("minctasm_i32", compile_info[k_minctasm_i32, emission_kind="asm", target=T]().asm)
    show("maxnreg", compile_info[k_maxnreg, emission_kind="asm", target=T]().asm)
