# SPDX-License-Identifier: Apache-2.0
"""Two aligned logical 32-lane groups per AMD wave using supported XOR.

No ballot, width-32 mask or NVIDIA-only warp assumption is involved. XOR
strides smaller than 32 cannot cross the aligned logical-group boundary.
The control uses a whole physical wave for one logical group (unused lanes
carry zero), and tests empty/tail members with distinct adjacent-group tags.
"""
from std.gpu import block_idx,block_dim,thread_idx,WARP_SIZE
from std.gpu.primitives.warp import shuffle_xor
from max.gpu.host import DeviceContext


# A04 experiment: NEVER RUN — PENDING VALIDATION; incumbent defaults retained.
# A04 PENDING: compile evidence alone does not qualify device correctness, quality or speed.
# Explicit subwave fixture kernel only; no production scheduler admission.
def membership_kernel[PAIRED: Bool](output: MutPointer[UInt32,MutAnyOrigin],active: Int32,groups: Int32):
    var tid = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    var width = WARP_SIZE
    comptime if PAIRED:
        width = 32
    var group = tid//width
    var lane = tid%width
    var value = UInt32(0)
    if group < Int(groups) and lane < 32 and lane < Int(active):
        value = UInt32((group+1)*100+lane)
    var offset = 1
    while offset < width:
        value += shuffle_xor(value,UInt32(offset))
        offset *= 2
    if group < Int(groups) and lane == 0:
        output.unsafe_store(group,value)


def main() raises:
    var ctx = DeviceContext()
    var counts: List[Int] = [0,1,17,31,32]
    var groups = 4
    var output = ctx.enqueue_create_buffer[DType.uint32](groups)
    var host = ctx.enqueue_create_host_buffer[DType.uint32](groups)
    ctx.synchronize()
    for fixture in range(len(counts)):
        for arm in range(2):
            if arm == 0:
                ctx.enqueue_function[membership_kernel[False]](output,Int32(counts[fixture]),Int32(groups),
                    grid_dim=(1,1,1),block_dim=(groups*WARP_SIZE,1,1))
            else:
                ctx.enqueue_function[membership_kernel[True]](output,Int32(counts[fixture]),Int32(groups),
                    grid_dim=(1,1,1),block_dim=(groups*32,1,1))
            ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(),src_buf=output)
            ctx.synchronize()
            for group in range(groups):
                var active = counts[fixture]
                var expected = UInt32(active*(group+1)*100+active*(active-1)//2)
                if host.unsafe_ptr().unsafe_load(group) != expected:
                    raise Error("cross-group contamination or incomplete membership")
            print("SUBWAVE_PASS arm="+String(arm)+" active="+String(counts[fixture])+" physical_width="+String(WARP_SIZE))
