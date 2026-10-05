# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Completion witness for Apple launches (lane/neural-pass97, 2026-10-02).

macOS aborts a Metal command buffer that holds a contended GPU too long and
reports nothing: the cut kernel's writes stay partly stale, the work queued
after it is dropped, and `synchronize` returns as if all ran (the M2 under
two other Metal jobs: GLM and Bayes fits returned wrong or another fit's
coefficients). So on Apple every guarded launch ends each block with a
device-ordering barrier and thread 0's store of the launch's nonce into the
block's own word; the host reads the words back into a list preset to -1
and accepts the launch only if every block's word is the nonce (a cut
kernel leaves words stale, a dropped copy leaves -1). Guarded launches are
idempotent (they rebuild their outputs from inputs the launch does not
write), so a failed check reruns them: up to WITNESS_TRIES times, then the
fit raises. Off Apple the words are never written and `ok` is True without
a wait. `MOJOLEARN_METAL_WITNESS_LOG=1` prints each rerun to stdout; any
other non-empty value is a file path each rerun appends one line to (a
harness that swallows the worker's output reads the file)."""
from std.gpu import block_idx, thread_idx
from std.atomic import Atomic
from std.os import getenv
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator, is_apple_gpu
from std.time import perf_counter_ns
from max.gpu.host import DeviceBuffer, DeviceContext
from x_linear.ops import IP, sti
from x_linear.team import team_barrier

comptime WITNESS_TRIES = 4
comptime WITNESS_ABORT = (
    "mojolearn: the Metal GPU aborted this fit's work repeatedly (macOS cuts long GPU work when the GPU is"
    " shared); no result was returned. Refit when fewer GPU jobs are running."
)


@always_inline
def witness_end(wf: IP, woff: Int32, nonce: Int32):
    """Every thread of a guarded kernel calls this last (uniformly)."""
    comptime if is_apple_gpu():
        team_barrier()
        if Int(thread_idx.x) == 0:
            var v = Int(nonce)
            comptime if is_defined["MOJOLEARN_WITNESS_SABOTAGE"]():
                # the check's own test: block 0 never reports done
                if Int(block_idx.x) == 0:
                    v = v ^ 1
            sti(wf, Int(woff) + Int(block_idx.x), v)


def witness_check_kernel(wf: IP, count_in: Int32, nonce: Int32, dst: IP):
    """One thread a witness word (lane cpu3-core: the host no longer walks
    them). dst[0]: count minus the first block whose word is not the nonce
    (atomic max from 0, so 0 means every word matched); dst[1]: how many
    words were checked (atomic count), so a cut check reads as incomplete
    rather than as a pass. Both start at 0 (a memset)."""
    var i = Int(block_idx.x) * 256 + Int(thread_idx.x)
    if i < Int(count_in):
        if wf.unsafe_load(i) != nonce:
            _ = Atomic.max(dst, count_in - Int32(i))
        _ = Atomic.fetch_add(dst.unsafe_offset(1), Int32(1))


struct Witness(Movable):
    var buf: DeviceBuffer[DType.int32]
    var chk: DeviceBuffer[DType.int32]
    var host: List[Int32]
    var nonce: Int32
    var retries: Int
    var log: Bool
    var log_path: String

    def __init__(out self, mut ctx: DeviceContext, cap: Int) raises:
        self.buf = ctx.enqueue_create_buffer[DType.int32](max(cap, 1))
        # the device check's two words (first bad block, words checked)
        self.chk = ctx.enqueue_create_buffer[DType.int32](2)
        self.host = List[Int32](length=2, fill=Int32(-1))
        # a per-process start, so a word left from an earlier fit never matches
        self.nonce = Int32(Int(perf_counter_ns()) & 0x3FFFFFFF)
        self.retries = 0
        var lv = String(getenv("MOJOLEARN_METAL_WITNESS_LOG"))
        self.log = lv == "1"
        self.log_path = lv if (lv != "" and lv != "1") else String("")

    def p(self) -> IP:
        return IP(unsafe_from_address=Int(self.buf.unsafe_ptr()))

    def begin(mut self) -> Int32:
        self.nonce = Int32((Int(self.nonce) + 1) & 0x3FFFFFFF)
        return self.nonce

    def ok(mut self, mut ctx: DeviceContext, count: Int, what: String) raises -> Bool:
        """Waits and checks words [0, count); off Apple True at once."""
        comptime if not has_apple_gpu_accelerator():
            return True
        if count <= 0:
            return True
        # the words are checked on the device; two words come home
        ctx.enqueue_memset(self.chk, Int32(0))
        ctx.enqueue_function[witness_check_kernel](
            self.p(), Int32(count), self.nonce, self.chk.unsafe_ptr(),
            grid_dim=(count + 255) // 256, block_dim=256,
        )
        self.host[0] = Int32(-1)
        self.host[1] = Int32(-1)
        ctx.enqueue_copy(dst_ptr=self.host.unsafe_ptr(), src_buf=self.chk)
        ctx.synchronize()
        # out[0] = count - (first bad block), 0 when every word matched
        var bad = count - Int(self.host[0])
        if Int(self.host[1]) != count and bad >= count:
            bad = 0  # the check itself was cut: report it as incomplete
        if bad < count:
            self.retries += 1
            var line = ("mojolearn: Metal witness: " + what + " incomplete (block " + String(bad) + " of "
                        + String(count) + "), rerunning; retries " + String(self.retries))
            if self.log:
                print(line)
            if self.log_path != "":
                try:
                    with open(self.log_path, "a") as f:
                        f.write(line + "\n")
                except:
                    pass
            return False
        return True

    def fail(self) raises:
        if self.log_path != "":
            try:
                with open(self.log_path, "a") as f:
                    f.write("mojolearn: Metal witness: raised after " + String(self.retries) + " reruns\n")
            except:
                pass
        raise Error(WITNESS_ABORT)
