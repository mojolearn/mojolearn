# SPDX-License-Identifier: Apache-2.0
"""An OWNED executor (`DeviceExec(ctx)`: the Mamba block programs, the byte
LM / Samba / CausalLM programs, the transformer step arms) reuses its own
storage across calls (lane fable-perf-neural-0855, 2026-11-13):

  * a device buffer freed before a sync serves the next `alloc` of the same
    count after that sync (the address comes back; `idle_bytes` returns to
    zero when it is taken);
  * the pinned stage of an upload is kept at the sync that drained it and
    serves the next upload or download that fits, so a steady call pattern
    creates no pinned HostBuffer after its first call;
  * the words that come back through a reused stage are the words sent.

Before this every owned-executor upload and download created a pinned
HostBuffer and dropped it at the sync (cudaHostAlloc / cudaFreeHost per
call on NVIDIA). GPU columns only; the test owns its context."""
from std.memory import bitcast
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_CPU
from max.gpu.host import DeviceContext
from sequence.exec_device import DeviceExec
from sequence.ops import FP


def _require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _words(n: Int, seed: Int) -> List[Float32]:
    var out = List[Float32](length=n, fill=0.0)
    for i in range(n):  # test fixture: n words
        out[i] = Float32(((i * 31 + seed * 7) % 101) - 50) / 16.0
    return out^


def _same_words(got: List[Float32], want: List[Float32], what: String) raises:
    _require(len(got) == len(want), what + ": lengths differ")
    for i in range(len(got)):  # test fixture: compare every word
        if bitcast[DType.uint32](got[i]) != bitcast[DType.uint32](want[i]):
            raise Error(what + ": word " + String(i) + " differs")


def _device_buffer_reuse(ctx: DeviceContext) raises:
    var ex = DeviceExec(ctx)
    var n = 4096
    var p1 = ex.alloc(n)
    ex.free(p1)
    _require(ex.idle_bytes == 0, "a freed buffer is kept only at the sync that drains its queue")
    ex.sync()
    _require(ex.idle_bytes == n * 4, "the drained buffer is kept")
    var p2 = ex.alloc(n)
    _require(Int(p2) == Int(p1), "the same-count alloc after the sync takes the kept buffer")
    _require(ex.idle_bytes == 0, "the kept buffer left the idle list")
    var p3 = ex.alloc(n + 1)
    _require(Int(p3) != Int(p2), "a different count is a new buffer")
    ex.free(p2)
    ex.free(p3)
    ex.sync()
    _require(ex.idle_bytes == (2 * n + 1) * 4, "both drained buffers are kept")
    print("PASS exec_owned_reuse device buffers")


def _stage_reuse(ctx: DeviceContext) raises:
    var ex = DeviceExec(ctx)
    var n = 1 << 16
    var sent = _words(n, 1)
    var dst = ex.alloc(n)
    ex.upload(dst, FP(unsafe_from_address=Int(sent.unsafe_ptr())), n)
    _require(len(ex.idle_host) == 0, "the stage of a queued upload is in flight, not idle")
    ex.sync()
    _require(len(ex.idle_host) == 1 and ex.idle_host_n[0] == n, "the drained upload stage is kept")
    _require(ex.idle_bytes == n * 4, "the kept stage is counted")
    # a second upload takes the kept stage (no new pinned buffer)
    var sent2 = _words(n, 2)
    ex.upload(dst, FP(unsafe_from_address=Int(sent2.unsafe_ptr())), n)
    _require(len(ex.idle_host) == 0 and ex.idle_bytes == 0, "the second upload took the kept stage")
    ex.sync()
    _require(len(ex.idle_host) == 1, "one stage serves the steady upload pattern")
    # a synchronous download takes it and keeps it again; the words are the
    # second upload's
    var back = List[Float32](length=n, fill=0.0)
    ex.download(FP(unsafe_from_address=Int(back.unsafe_ptr())), dst, n)
    _require(len(ex.idle_host) == 1 and ex.idle_host_n[0] == n, "the download stage is kept after its copy")
    _same_words(back, sent2, "download through a reused stage")
    # a smaller download fits the kept stage (smallest fit); a larger one
    # is a new stage, kept beside it
    var half = n // 2
    var small = List[Float32](length=half, fill=0.0)
    ex.download(FP(unsafe_from_address=Int(small.unsafe_ptr())), dst, half)
    _require(len(ex.idle_host) == 1, "a smaller download fits the kept stage")
    for i in range(half):  # test fixture: the first half of the words
        _require(bitcast[DType.uint32](small[i]) == bitcast[DType.uint32](sent2[i]), "smaller download: word differs")
    var big = ex.alloc(2 * n)
    var sent3 = _words(2 * n, 3)
    ex.upload(big, FP(unsafe_from_address=Int(sent3.unsafe_ptr())), 2 * n)
    ex.sync()
    _require(len(ex.idle_host) == 2, "a larger upload makes a second stage, kept beside the first")
    var back3 = List[Float32](length=2 * n, fill=0.0)
    ex.download_async(FP(unsafe_from_address=Int(back3.unsafe_ptr())), big, 2 * n)
    _require(len(ex.idle_host) == 1, "the async download took the fitting stage")
    ex.sync()
    _require(len(ex.idle_host) == 2, "the async download's stage is kept at the sync that wrote its words")
    _same_words(back3, sent3, "async download through a reused stage")
    _ = len(sent)
    _ = len(sent2)
    _ = len(sent3)
    print("PASS exec_owned_reuse pinned stages")


def run_checks() raises:
    comptime assert TARGET_COLUMN != COLUMN_CPU
    var ctx = DeviceContext()
    _device_buffer_reuse(ctx)
    _stage_reuse(ctx)
    ctx.synchronize()
