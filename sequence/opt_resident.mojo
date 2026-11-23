# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE RESIDENT OPTIMIZER STATE of the sequence lane's torch-style
optimizers (lane gap-optimizers, 2026-10-02; device binding only).

`optimizer_step` and `lamb_step` took every state slot from the caller's
host arrays each step: up to three n-float moments uploaded and downloaded
again around one element-wise launch (Adamax, RMSprop, NAdam: five or six
64 MB transfers a step at the board's 16,777,216 parameters, against
torch's zero), and LAMB on top of that downloaded its per-tensor gradient
norms in the middle of the step to fold the clip on the host. Here a
Python optimizer opens a handle once: its state slots live on the shared
sequence context (`sequence_ctx`) across steps, and a step moves only what
the API must: the parameters and gradients up (each tensor straight into
its place in the flat device buffer, no host concatenation), ONE
element-wise launch over every tensor (`opt_step`, the same `op_opt`) or
LAMB's launch-only `lamb_core`, the parameters down. The same element
statements on the same values: no bit moves against the per-call entries.
`opt_resident_get` / `_set` move a slot for a state read or a
`load_state_dict`. Storage is `std.ffi._Global`, one slot per tier; a
handle indexes the pool and a closed handle's slot is reused. The Python
entries (`opt_resident_open_py` ...) are `providers/python/sequence_opt_resident_pyapi.mojo`'s;
this file takes typed handles, addresses and parameters only."""
from std.memory import bitcast, unsafe_memcpy
from std.sys import is_defined
from core.accelerator_target import TARGET_HAS_APPLE_GPU
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, NUMERIC_IDENTICAL
from sequence.exec_device import (
    DeviceExec, TPB, _args_i32_out, _pack_ff, _pack_ii, _pool_host, _pool_locked, _pool_unlock, _pool_release_host,
    seq_kernel,
    sequence_ctx,
)
from sequence.ops import FP, OP_OPT, OPT_NADAM, OPT_SGD, OPT_SK_ADAM, OPT_SK_SGD, Args
from sequence.recurrent import OptConfig, OptState, opt_scalars, opt_step
from sequence.ops_impl import adafactor_core, opt_slots
from sequence.lamb_core import lamb_bias, lamb_core, lamb_table
from sequence.seq_tensor import seq_tensor_ptr, seq_tensor_view
from core.resident_upload_count import RUC_OPTIMIZER_STATE, resident_upload_note
from core.shared_slot import SharedSlot, shared_slot_wait_lock, shared_slot_unlock

#: Apple FAST transport switches of the resident step (lane
#: apple-fast-optspeed, 2026-10-03). At the board's one tensor of 16,777,216
#: floats a step is three 64 MB transfers around one ~1 ms element-wise
#: launch, and the time is the transport: each upload was a single-thread
#: memcpy into a pinned stage plus its DMA (a raw host-pointer copy is
#: faster on Metal, memory metal-transfer-costs-on-apple), and the download
#: was one DMA into a pinned stage, then one single-thread read of that
#: write-combined memory (~15-25 ms per 64 MB, the step's largest cost since
#: cpu-gpu-cleanup n-seq made `_pcopy` one memcpy). Copies only: no bit
#: moves, the same launches on the same values.
#:  MOJOLEARN_OPT_RAW_UP: the parameter and gradient uploads are raw
#:    host-pointer copies straight into the device buffers (no stage).
#:  MOJOLEARN_OPT_PIPE_DOWN: the parameter download goes in OPT_PIPE_CH
#:    chunks through two pinned halves: the DMA of chunk i overlaps the read
#:    of chunk i - 1.
#:  MOJOLEARN_OPT_ZERO_OPEN: the open reports its slots zero filled on the
#:    device (a third return value), so the Python side skips uploading its
#:    still untouched zero host copies on the first step.
#: IDENTICAL and the other vendors compile the main path unchanged.
comptime _OPT_APPLE_FAST = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and TARGET_HAS_APPLE_GPU
#: OUTCOME (M3 afc_ab_def, full board size, 1 run per arm, 2026-10-04, lane/
#: apple-fast-rec-ab2 @ 40027eb8e): RAW_UP measured as a pair with
#: OPT_FAST_STREAM below (see there): KEEP, the FAST + Apple default since
#: then (the rollback define was deleted 2026-11-15).
# TOMBSTONE: MOJOLEARN_OPT_RAW_UP_OFF (old path; the FAST+Apple default is the measured pair with OPT_FAST_STREAM, EXP:1204) deleted 2026-11-15 by lane/expdel2-30-neural; code recoverable at e39daf79ad.
# Restore: git apply experiments/removed/MOJOLEARN_OPT_RAW_UP_OFF.patch
comptime OPT_RAW_UP = _OPT_APPLE_FAST
#: PIPE_DOWN and ZERO_OPEN are the FAST+Apple default since the M3 A/B of
#: lane/apple-fast-optspeed f2b6491a8 (n=1, synthetic, output digest
#: identical): all three switches adagrad 317 -> 187 ms, lamb 340 -> 206,
#: adamax 326 -> 195; PIPE_DOWN alone 318 -> 191. Off with
#: MOJOLEARN_OPT_PIPE_DOWN_OFF / MOJOLEARN_OPT_ZERO_OPEN_OFF; the old
#: -D names are harmless. RAW_UP alone measured noise (310 -> 307); it is
#: the default since 2026-10-04 as the pair with OPT_FAST_STREAM (below).
#: lane idn-opt-resident (2026-10-04): IDENTICAL takes PIPE_DOWN and
#: ZERO_OPEN on every vendor (copies only, and zeros not uploaded over
#: zeros: no bit moves). -D MOJOLEARN_IDN_OPT_PIPE_DOWN_OFF /
#: -D MOJOLEARN_IDN_OPT_ZERO_OPEN_OFF restore IDENTICAL's old transport.
from checks.mode_policy import KERNEL_FORM_IDENTICAL as _OPT_IDN
# TOMBSTONE: MOJOLEARN_OPT_PIPE_DOWN_OFF (old path; FAST+Apple default, EXP:589 KEPT on Apple M3 only; NVIDIA/AMD A/B owed) deleted 2026-11-15 by lane/expdel2-30-neural; code recoverable at e39daf79ad.
# Restore: git apply experiments/removed/MOJOLEARN_OPT_PIPE_DOWN_OFF.patch
# TOMBSTONE: MOJOLEARN_IDN_OPT_PIPE_DOWN_OFF (old path, no A/B recorded) deleted 2026-11-15 by lane/expdel2-30-neural; code recoverable at e39daf79ad.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_OPT_PIPE_DOWN_OFF.patch
# TOMBSTONE: MOJOLEARN_OPT_ZERO_OPEN_OFF (old path; FAST+Apple default, EXP:590 KEPT on Apple M3 only; NVIDIA/AMD A/B owed) deleted 2026-11-15 by lane/expdel2-30-neural; code recoverable at e39daf79ad.
# Restore: git apply experiments/removed/MOJOLEARN_OPT_ZERO_OPEN_OFF.patch
# TOMBSTONE: MOJOLEARN_IDN_OPT_ZERO_OPEN_OFF (old path, no A/B recorded) deleted 2026-11-15 by lane/expdel2-30-neural; code recoverable at e39daf79ad.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_OPT_ZERO_OPEN_OFF.patch
# TOMBSTONE: MOJOLEARN_IDN_ALL_OFF (umbrella old-path switch, EXP:2438) read folded here (non-owner) 2026-11-15 by lane/expdel2-30-neural; code recoverable at e39daf79ad.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_ALL_OFF.expdel2-30-neural.patch
comptime OPT_PIPE_DOWN = _OPT_APPLE_FAST or _OPT_IDN
comptime OPT_ZERO_OPEN = _OPT_APPLE_FAST or _OPT_IDN
#: Experiment MOJOLEARN_OPT_FAST_ASYNC_DEV_STEP (lane fb12-rows, 2026-11-17; A/B owed, NVIDIA
#: L40S; default off): the element-wise step on resident device params and device grads
#: (`opt_resident_step_dev`) returns without draining the queue. (lane fb15-sync-counts,
#: 2026-11-18: the arm was inert as written: the step's `DeviceExec` drained the queue in its
#: destructor whether or not `ex.sync()` ran. The step now enqueues its launch on the shared
#: context directly (`_opt_launch`), so the define does remove the step's one wait.) The launch takes every scalar
#: by value (`_opt_args`), allocates nothing and frees nothing, so nothing after it waits for the
#: kernel except a read of the params or state, which drains first (`seq_tensor_download`,
#: `opt_resident_move`, `opt_resident_close`, `seq_tensor_free` all synchronize the one
#: in-order context). Lets the next step's host door overlap this step's kernel instead of
#: running between drained steps (rows: algos rmsprop/adagrad/adamax/nadam/lion synthetic fit,
#: 16 Mi floats x 10 steps; neighbor: any parameter count; a kernel fault surfaces at the next
#: drain instead of in this step). Same launches, same order: no bit moves.
comptime OPT_ASYNC_DEV_STEP = is_defined["MOJOLEARN_OPT_FAST_ASYNC_DEV_STEP"]()
#: the pipelined download's chunk, floats (8 MB), a transfer granularity.
# TOMBSTONE: MOJOLEARN_OPT_FAST_PIPE_CH (=524288 arm; the rab7 optimizer read-back batch DROPPED-slower, this arm untried)
# deleted 2026-10-09 on lane/owed-deletions-D2: the -D override of the chunk; the default 1 << 21 is unchanged;
# code recoverable at b639a2bd2. Restore: git apply experiments/removed/MOJOLEARN_OPT_FAST_PIPE_CH.patch
comptime OPT_PIPE_CH = 1 << 21
#: lane apple-fast-gap-optim (2026-10-03, docs/apple-fast/notes/gap-optim.md): its two
#: parameter read-back candidates were deleted as losers.
# TOMBSTONE: MOJOLEARN_OPT_FAST_MAP_DOWN (DROPPED-slower: M3 rab7-optfastmapdo rmsprop/adagrad/adamax/nadam +81% .. +86%)
# deleted 2026-10-09 on lane/owed-deletions-D2 (the define and _map_download); code recoverable at b639a2bd2.
# Restore: git apply experiments/removed/MOJOLEARN_OPT_FAST_MAP_DOWN.patch
# TOMBSTONE: MOJOLEARN_OPT_FAST_RAW_DOWN (DROPPED-slower: M3 rab7-optfastrawdo rmsprop/adagrad/adamax/nadam +76% .. +80%)
# deleted 2026-10-09 on lane/owed-deletions-D2 (the define and _raw_download); code recoverable at b639a2bd2.
# Restore: git apply experiments/removed/MOJOLEARN_OPT_FAST_RAW_DOWN.patch

#: lane apple-fast-gap-optim (2026-10-03), FAST + Apple only:
#:  MOJOLEARN_AF_FAST_RESIDENT: Adafactor's second moment (row_var and
#:    col_var, or a vector's full variance) lives in a handle's slots on
#:    the device across steps (`adafactor_resident_*`); a step moves the
#:    parameter and gradient up and the parameter down only (the board's
#:    1-D tensor: three 64 MB transfers instead of five). The same launches
#:    (`sequence/ops_impl.mojo::adafactor_core`) on the same values.
#: AF_FAST_RESIDENT OUTCOME (M3 afc_ab_def, full board size, 1 run per arm,
#: 2026-10-05, rab10-afresident): adafactor 392.8 -> 205.2 ms, digest
#: identical A == B. KEEP: the FAST + Apple default since then.
# TOMBSTONE: MOJOLEARN_AF_FAST_RESIDENT_OFF (old path, superseded by the FAST+Apple default) deleted 2026-11-15 by lane/expdel-06-base-cluster-kernel; code recoverable at a656258c20.
# Restore: git apply experiments/removed/MOJOLEARN_AF_FAST_RESIDENT_OFF.patch; record in docs/TOMBSTONES.md.
#: lane idn-opt-resident (2026-10-04): the IDENTICAL default on every
#: vendor (the per-call entry moved P, G and the moment up and two of them
#: down each step; the -D MOJOLEARN_IDN_AF_RESIDENT_OFF rollback was deleted
#: 2026-11-15).
#: The same `adafactor_core` launches on the same values: no bit moves.
# TOMBSTONE: MOJOLEARN_IDN_AF_RESIDENT_OFF (old path, no A/B recorded) deleted 2026-11-15 by lane/expdel2-30-neural; code recoverable at e39daf79ad.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_AF_RESIDENT_OFF.patch
comptime AF_RESIDENT = _OPT_APPLE_FAST or _OPT_IDN

#: MOJOLEARN_OPT_FAST_STREAM (lane apple-fast-rec-optim, 2026-10-04), default OFF,
#: FAST + Apple only: the element-wise resident step (rmsprop, adagrad, adamax,
#: nadam, lion, sgd/adam of the sequence lane) STREAMED in OPT_PIPE_CH chunks
#: instead of three whole-buffer phases. Per chunk: the parameter and gradient
#: chunk up, ONE `op_opt` launch over that chunk's range (the same Args the
#: whole-buffer `opt_step` builds; the host scalars computed ONCE per step), the
#: chunk DMAd into a pinned half; the host reads chunk c - 1 out of the other half
#: while the GPU does chunk c. Main runs upload-all, launch, then the pipelined
#: download, so the host idles through both 64 MB DMAs and the launch (~5-6 ms of
#: a ~20 ms step at the board's 16,777,216 floats, docs/apple-fast/notes/
#: gap-optim.md); here only the first chunk's GPU work is exposed. `op_opt` is
#: per element (sequence/ops.mojo), so a chunked launch computes the same bits.
#: Chunk = OPT_PIPE_CH (8 MB): a transfer
#: granularity, not a board-size window. Uploads follow OPT_RAW_UP (raw host
#: pointers) when that is named, else a memcpy into two pinned stage pairs.
#: Prior: none (new). Source: this branch.
#: Wins over the reference only on the host-bound transport; the floor stays
#: the single-thread read of write-combined pinned memory (~13 ms / 64 MB).
#: OUTCOME (M3 afc_ab_def, full board size, 1 run per arm, 2026-10-04, lane/
#: apple-fast-rec-ab2 @ 40027eb8e): STREAM together with OPT_RAW_UP (the pair):
#: rmsprop 185.3 -> 179.1 ms, adagrad 185.6 -> 179.3, adamax 194.8 -> 190.1,
#: nadam 194.8 -> 185.7; output digests identical A == B. STREAM alone (staged
#: uploads) was slower: rmsprop 185.6 -> 187.3, adagrad 185.5 -> 195.6, adamax
#: 192.4 -> 217.3, nadam 192.7 -> 201.9. KEEP the pair: both are the FAST + Apple
#: default since then; rollback -D MOJOLEARN_OPT_FAST_STREAM_OFF and
#: -D MOJOLEARN_OPT_RAW_UP_OFF, each on its own (STREAM_OFF alone leaves the
#: measured-noise RAW_UP; RAW_UP_OFF alone leaves the slower staged STREAM).
# TOMBSTONE: MOJOLEARN_OPT_FAST_STREAM_OFF (old path; the FAST+Apple default is the measured pair with OPT_RAW_UP, EXP:1204) deleted 2026-11-15 by lane/expdel2-30-neural; code recoverable at e39daf79ad.
# Restore: git apply experiments/removed/MOJOLEARN_OPT_FAST_STREAM_OFF.patch
comptime OPT_STREAM = _OPT_APPLE_FAST

#: the handle kinds
comptime RES_ELEMENTWISE = 1
comptime RES_LAMB = 2
comptime RES_ADAFACTOR = 3


#: The live resident optimizer handles one process may hold at once (each open
#: fit or session holds one; a close frees it for reuse): a count of live
#: owners, not a model or data size. Past it an open is refused in words.
comptime _RES_HANDLES = 4096


struct _ResPool(Defaultable, Movable):
    #: the three state slots (a 1-float placeholder for a slot not kept)
    var s0: List[DeviceBuffer[DType.float32]]
    var s1: List[DeviceBuffer[DType.float32]]
    var s2: List[DeviceBuffer[DType.float32]]
    var used: List[Int]
    #: LAMB's table (`sequence/ops_impl.mojo::lamb_table`), on the device
    var tab: List[DeviceBuffer[DType.float32]]
    #: floats per slot (0 marks a free handle), kind, tensors, blocks
    var n: List[Int]
    var kind: List[Int]
    var nt: List[Int]
    var nb: List[Int]
    #: LAMB's tensor offsets (nt + 1)
    var offs: List[List[Int]]
    #: One lock word (lane r5-proving, 2026-11-23): `_open_sized` and
    #: `opt_resident_close` take it for the free-handle search and the cell
    #: writes. Before, two threads (device entries run with the GIL released)
    #: could take the same free handle or append while another thread read a
    #: cell. The lists are made at their full capacity (_RES_HANDLES) and never
    #: grow past it, so a cell never moves: a step reads its own handle's cells
    #: without the word, as before.
    var lock: List[Int64]

    def __init__(out self):
        self.s0 = List[DeviceBuffer[DType.float32]](capacity=_RES_HANDLES)
        self.s1 = List[DeviceBuffer[DType.float32]](capacity=_RES_HANDLES)
        self.s2 = List[DeviceBuffer[DType.float32]](capacity=_RES_HANDLES)
        self.used = List[Int](capacity=_RES_HANDLES)
        self.tab = List[DeviceBuffer[DType.float32]](capacity=_RES_HANDLES)
        self.n = List[Int](capacity=_RES_HANDLES)
        self.kind = List[Int](capacity=_RES_HANDLES)
        self.nt = List[Int](capacity=_RES_HANDLES)
        self.nb = List[Int](capacity=_RES_HANDLES)
        self.offs = List[List[Int]](capacity=_RES_HANDLES)
        self.lock = List[Int64](length=1, fill=Int64(0))


# V2: the lock word and the fixed capacity changed the layout.
comptime _RES_NAME = "MojolearnXSequenceOptResidentIdenticalV2" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnXSequenceOptResidentFastV2"
comptime _RES = SharedSlot[_ResPool, _RES_NAME]


def _res_lock(pool: UnsafePointer[_ResPool, MutUntrackedOrigin]):
    shared_slot_wait_lock(rebind[MutPointer[Int64, MutUntrackedOrigin]](pool[].lock.unsafe_ptr()))


def _res_unlock(pool: UnsafePointer[_ResPool, MutUntrackedOrigin]):
    shared_slot_unlock(rebind[MutPointer[Int64, MutUntrackedOrigin]](pool[].lock.unsafe_ptr()))


def _slot_buf(n: Int, keep: Bool) raises -> DeviceBuffer[DType.float32]:
    var ctx = sequence_ctx()
    var b = ctx.enqueue_create_buffer[DType.float32](n if keep else 1)
    # lane fb15-sync-counts (2026-11-18): only a kept slot is zero filled. A
    # slot the configuration does not use is a 1-float placeholder that no
    # launch and no move reads (`opt_resident_move` returns 0 for it, the
    # step passes the parameter pointer in its place): its fill was a
    # memset per open (2 of the 3 per Adagrad fit). No bit moves.
    if keep:
        b.enqueue_fill(Float32(0.0))
    return b^


def _open(kind: Int, n: Int, used: Int, nt: Int, var offs: List[Int], tab: List[Float32]) raises -> Int:
    return _open_sized(kind, n, n, n, n, used, nt, offs^, tab)


def _open_sized(kind: Int, n: Int, n0: Int, n1: Int, n2: Int, used: Int, nt: Int, var offs: List[Int],
                tab: List[Float32]) raises -> Int:
    """`_open` with each slot's own float count (n0, n1, n2); n is the
    handle's parameter count."""
    var pool = _RES.get_or_create_ptr()
    var ctx = sequence_ctx()
    var tb = ctx.enqueue_create_buffer[DType.float32](max(len(tab), 1))
    if len(tab) > 0:
        ctx.enqueue_copy(dst_buf=tb, src_ptr=tab.unsafe_ptr())
        resident_upload_note(RUC_OPTIMIZER_STATE, len(tab))  # check-only count (compiles to nothing by default)
    var nb = 0
    if len(tab) > 0:
        nb = Int(bitcast[DType.int32](tab[len(tab) - 1]))
    # The slots are made before the lock word is taken (an allocation and a
    # fill are queued on the context, never under the word).
    var b0 = _slot_buf(n0, (used & 1) != 0)
    var b1 = _slot_buf(n1, (used & 2) != 0)
    var b2 = _slot_buf(n2, (used & 4) != 0)
    var h = -1
    _res_lock(pool)
    try:
        for j in range(len(pool[].n)):  # small-loop(pool: resident optimizer handles): free handle slot search, no data
            if pool[].n[j] == 0 and h < 0:
                h = j
        if h < 0:
            if len(pool[].n) >= _RES_HANDLES:
                raise Error("RuntimeError: mojolearn: too many live resident optimizer handles")
            pool[].s0.append(b0^)
            pool[].s1.append(b1^)
            pool[].s2.append(b2^)
            pool[].used.append(used)
            pool[].tab.append(tb^)
            pool[].kind.append(kind)
            pool[].nt.append(nt)
            pool[].nb.append(nb)
            pool[].offs.append(offs^)
            # `n` last: a nonzero count is what marks the handle taken.
            pool[].n.append(n)
            h = len(pool[].n) - 1
        else:
            pool[].s0[h] = b0^
            pool[].s1[h] = b1^
            pool[].s2[h] = b2^
            pool[].used[h] = used
            pool[].tab[h] = tb^
            pool[].kind[h] = kind
            pool[].nt[h] = nt
            pool[].nb[h] = nb
            pool[].offs[h] = offs^
            pool[].n[h] = n
    finally:
        _res_unlock(pool)
    # the fills and the table copy are ordered before every later use on
    # the one in-order context. Only LAMB's table copy reads a host list
    # (`tab`, freed when the caller returns), so only it waits here (lane
    # fb15-sync-counts, 2026-11-18: every open waited, one cuStreamSynchronize
    # per element-wise or Adafactor fit with nothing host-side to protect).
    if len(tab) > 0:
        ctx.synchronize()
    return h


def _host_fp(a: Int, what: String) raises -> FP:
    """A caller's host float32 array at address a (`providers/python/sequence_pyapi.mojo`
    `fptr`'s check)."""
    if a == 0:
        raise Error("sequence: null float32 buffer for " + what)
    return FP(unsafe_from_address=a)


def _opt_of(ip: List[Int], at: Int, fp: List[Float64], fat: Int) raises -> OptConfig:
    """`providers/python/sequence_pyapi.mojo` `opt_of` on typed parameters: ip[at] kind,
    ip[at + 1] flags, fp[fat .. fat + 4] as float32."""
    var kind = ip[at]
    if kind < OPT_SGD or kind > OPT_NADAM or kind == OPT_SK_ADAM or kind == OPT_SK_SGD:
        raise Error("sequence: unknown optimizer kind " + String(kind))
    return OptConfig(kind, ip[at + 1], Float32(fp[fat]), Float32(fp[fat + 1]), Float32(fp[fat + 2]),
                     Float32(fp[fat + 3]), Float32(fp[fat + 4]))


def _lamb_offsets(ip: List[Int], at: Int, nt: Int) raises -> List[Int]:
    """`providers/python/sequence_pyapi.mojo` `lamb_offsets` on typed parameters: ip[at ..
    at + nt], rising strictly from 0, below 2^31."""
    var offs = List[Int]()
    for k in range(nt + 1):  # small-loop(nt: parameter tensors): offset admission, no data
        var o = ip[at + k]
        if (k == 0 and o != 0) or (k > 0 and o <= offs[k - 1]) or o >= 2147483647:
            raise Error("lamb_step: offsets must rise strictly from 0, below 2^31 - 1")
        offs.append(o)
    return offs^


def _handle(handle: Int, kind: Int) raises -> Int:
    var h = handle
    var pool = _RES.get_or_create_ptr()
    if h < 0 or h >= len(pool[].n) or pool[].n[h] == 0:
        raise Error("optimizer_resident: handle " + String(h) + " is not open")
    if kind > 0 and pool[].kind[h] != kind:
        raise Error("optimizer_resident: handle " + String(h) + " belongs to another optimizer kind")
    return h


def _slot_ptr(h: Int, slot: Int) raises -> FP:
    var pool = _RES.get_or_create_ptr()
    if slot == 0:
        return FP(unsafe_from_address=Int(pool[].s0[h].unsafe_ptr()))
    if slot == 1:
        return FP(unsafe_from_address=Int(pool[].s1[h].unsafe_ptr()))
    return FP(unsafe_from_address=Int(pool[].s2[h].unsafe_ptr()))


def _pair(a: Int, b: Int) -> List[Int]:
    """[a, b] (the shell returns it as a Python list)."""
    var res = List[Int]()
    res.append(a)
    res.append(b)
    comptime if OPT_ZERO_OPEN:
        # the slots are zero filled on the device (`_slot_buf`): a caller
        # whose host copies are still the zeros it built need not upload them
        res.append(1)
    return res^


def opt_resident_open(ip: List[Int], fp: List[Float64]) raises -> List[Int]:
    """An element-wise optimizer's handle: ip = [n, kind, flags], fp = [lr,
    f1, f2, eps, weight_decay, f7] (the step's configuration; it decides
    which slots `op_opt` uses, `opt_slots`, and only those are kept, zero
    filled). Returns [handle, used slot mask]."""
    if len(ip) != 3 or len(fp) != 6:
        raise Error("optimizer_resident_open: requires 3 integer and 6 float parameters")
    var n = ip[0]
    if n < 1:
        raise Error("optimizer_resident_open: n must be >= 1")
    var cfg = _opt_of(ip, 1, fp, 1)
    var u = opt_slots(cfg)
    var used = (1 if u[0] else 0) | (2 if u[1] else 0) | (4 if u[2] else 0)
    var h = _open(RES_ELEMENTWISE, n, used, 0, List[Int](), List[Float32]())
    return _pair(h, used)


def lamb_resident_open(ip: List[Int]) raises -> List[Int]:
    """A LAMB handle: ip = [n_tensors, off_0, ..., off_n]; exp_avg and
    exp_avg_sq (slots 0 and 1) zero filled, the table on the device.
    Returns [handle, 3]."""
    if len(ip) < 3:
        raise Error("lamb_resident_open: requires n_tensors and n_tensors + 1 offsets")
    var nt = ip[0]
    if nt < 1 or len(ip) != 2 + nt:
        raise Error("lamb_resident_open: n_tensors >= 1 and n_tensors + 1 offsets")
    var offs = _lamb_offsets(ip, 1, nt)
    var tab = List[Float32]()
    _ = lamb_table(offs, tab)
    var n = offs[nt]
    var h = _open(RES_LAMB, n, 3, nt, offs^, tab)
    return _pair(h, 3)


def opt_resident_close(handle: Int) raises -> Int:
    """Free a handle's slots (after the queue drains)."""
    var h = _handle(handle, 0)
    var ctx = sequence_ctx()
    ctx.synchronize()
    var pool = _RES.get_or_create_ptr()
    var p0 = ctx.enqueue_create_buffer[DType.float32](1)
    var p1 = ctx.enqueue_create_buffer[DType.float32](1)
    var p2 = ctx.enqueue_create_buffer[DType.float32](1)
    var pt = ctx.enqueue_create_buffer[DType.float32](1)
    _res_lock(pool)
    pool[].s0[h] = p0^
    pool[].s1[h] = p1^
    pool[].s2[h] = p2^
    pool[].tab[h] = pt^
    pool[].offs[h] = List[Int]()
    pool[].used[h] = 0
    # `n` last, under the word: the handle is free for the next open.
    pool[].n[h] = 0
    _res_unlock(pool)
    return h


def opt_resident_move(handle: Int, slot: Int, addr: Int, up: Bool) raises -> Int:
    """Slot `slot` (0, 1, 2) to the host array at `addr` (up = 0) or from
    it (up = 1), n floats. Returns 1, or 0 for a slot the handle does not
    keep (nothing moves)."""
    var h = _handle(handle, 0)
    var s = slot
    if s < 0 or s > 2:
        raise Error("optimizer_resident: slot must be 0, 1 or 2")
    var pool = _RES.get_or_create_ptr()
    if (pool[].used[h] & (1 << s)) == 0:
        return 0
    var hp = _host_fp(addr, "state")
    var ctx = sequence_ctx()
    var to_dev = up
    if to_dev:
        resident_upload_note(RUC_OPTIMIZER_STATE, pool[].n[h])  # check-only count (compiles to nothing by default)
    if s == 0:
        if to_dev:
            ctx.enqueue_copy(dst_buf=pool[].s0[h], src_ptr=hp)
        else:
            ctx.enqueue_copy(dst_ptr=hp, src_buf=pool[].s0[h])
    elif s == 1:
        if to_dev:
            ctx.enqueue_copy(dst_buf=pool[].s1[h], src_ptr=hp)
        else:
            ctx.enqueue_copy(dst_ptr=hp, src_buf=pool[].s1[h])
    else:
        if to_dev:
            ctx.enqueue_copy(dst_buf=pool[].s2[h], src_ptr=hp)
        else:
            ctx.enqueue_copy(dst_ptr=hp, src_buf=pool[].s2[h])
    ctx.synchronize()
    return 1


def _tensors(addrs: List[Int], J: Int, sizes: List[Int], n: Int) raises -> Tuple[List[Int], List[Int]]:
    """The J parameter and J gradient addresses (addrs[0:J], addrs[J:2J]),
    checked against the sizes that must add up to n."""
    var tot = 0
    for j in range(J):
        if sizes[j] < 1:
            raise Error("optimizer_resident_step: every tensor holds at least one value")
        tot += sizes[j]
    if tot != n:
        raise Error("optimizer_resident_step: the tensors hold " + String(tot) + " values, the handle " + String(n))
    var ps = List[Int]()
    var gs = List[Int]()
    for j in range(J):
        ps.append(Int(_host_fp(addrs[j], "params")))
        gs.append(Int(_host_fp(addrs[J + j], "grads")))
    return (ps^, gs^)


def _upload_all(mut ex: DeviceExec, P: FP, G: FP, ps: List[Int], gs: List[Int], sizes: List[Int]) raises:
    comptime if OPT_RAW_UP:
        # raw host-pointer copies into the executor's buffers, queued; the
        # caller's arrays live until the download's wait ends the call
        var o = 0
        for j in range(len(sizes)):
            var fp_ = ex._find(P + o, sizes[j])
            var vp = ex._sub(fp_[0], fp_[1], sizes[j])
            ex.ctx.enqueue_copy(dst_buf=vp, src_ptr=FP(unsafe_from_address=ps[j]))
            _ = vp^
            var fg = ex._find(G + o, sizes[j])
            var vg = ex._sub(fg[0], fg[1], sizes[j])
            ex.ctx.enqueue_copy(dst_buf=vg, src_ptr=FP(unsafe_from_address=gs[j]))
            _ = vg^
            o += sizes[j]
        return
    var off = 0
    for j in range(len(sizes)):  # small-loop(sizes: parameter tensors of the model): one upload per tensor, no host arithmetic
        ex.upload(P + off, FP(unsafe_from_address=ps[j]), sizes[j])
        ex.upload(G + off, FP(unsafe_from_address=gs[j]), sizes[j])
        off += sizes[j]


def _pipe_download(mut ex: DeviceExec, P: FP, ps: List[Int], sizes: List[Int]) raises:
    """`_download_all` through two pinned halves of OPT_PIPE_CH floats: the
    DMA of chunk i runs into one half while the host reads chunk i - 1 out
    of the other. Every chunk's wait comes before its half is reused (the
    half of chunk i was last read for chunk i - 2, before the previous
    wait). Returns with every byte in the caller's arrays."""
    var h0 = _pool_host(ex.ctx, OPT_PIPE_CH)
    var h1 = _pool_host(ex.ctx, OPT_PIPE_CH)
    # The sequence pool's lists under its lock word (lane r5-proving): the stages' addresses only.
    var pool = _pool_locked()
    var st0 = FP(unsafe_from_address=Int(pool[].host[h0].unsafe_ptr()))
    var st1 = FP(unsafe_from_address=Int(pool[].host[h1].unsafe_ptr()))
    _pool_unlock(pool)
    var have_prev = False
    var prev_dst = 0
    var prev_cnt = 0
    var prev_half = 0
    var half = 0
    var off = 0
    for j in range(len(sizes)):
        var done = 0
        while done < sizes[j]:
            var cnt = min(OPT_PIPE_CH, sizes[j] - done)
            var f = ex._find(P + off + done, cnt)
            var v = ex._sub(f[0], f[1], cnt)
            ex.ctx.enqueue_copy(dst_ptr=st0 if half == 0 else st1, src_buf=v)
            _ = v^
            if have_prev:
                # overlaps the DMA just queued, into the other half
                unsafe_memcpy(dest=FP(unsafe_from_address=prev_dst), src=st0 if prev_half == 0 else st1,
                       count=prev_cnt)
            ex.ctx.synchronize()
            have_prev = True
            prev_dst = ps[j] + done * 4
            prev_cnt = cnt
            prev_half = half
            half = 1 - half
            done += cnt
        off += sizes[j]
    if have_prev:
        unsafe_memcpy(dest=FP(unsafe_from_address=prev_dst), src=st0 if prev_half == 0 else st1, count=prev_cnt)
    _pool_release_host(h0)
    _pool_release_host(h1)
    # the executor's own bookkeeping (no copy is pending: the queue is empty)
    ex.sync()


# TOMBSTONE: MOJOLEARN_OPT_FAST_MAP_DOWN (DROPPED-slower, rab7 +81% .. +86%) deleted 2026-10-09 on lane/owed-deletions-D2:
# _map_download (and its _download_all branch); code recoverable at b639a2bd2.
# Restore: git apply experiments/removed/MOJOLEARN_OPT_FAST_MAP_DOWN.patch
# TOMBSTONE: MOJOLEARN_OPT_FAST_RAW_DOWN (DROPPED-slower, rab7 +76% .. +80%) deleted 2026-10-09 on lane/owed-deletions-D2:
# _raw_download (and its _download_all branch); code recoverable at b639a2bd2.
# Restore: git apply experiments/removed/MOJOLEARN_OPT_FAST_RAW_DOWN.patch
def _download_all(mut ex: DeviceExec, P: FP, ps: List[Int], sizes: List[Int]) raises:
    comptime if OPT_PIPE_DOWN:
        _pipe_download(ex, P, ps, sizes)
        return
    var off = 0
    for j in range(len(sizes)):  # small-loop(sizes: parameter tensors of the model): one async download per tensor
        ex.download_async(FP(unsafe_from_address=ps[j]), P + off, sizes[j])
        off += sizes[j]
    ex.sync()


def _opt_args(cfg: OptConfig, mut st: OptState, t: Int, lr: Float32, P: FP, G: FP, s1: FP, s2: FP,
              s3: FP) -> Args:
    """OPT_STREAM: the Args `sequence/recurrent.mojo::opt_step` builds for one
    whole-buffer launch, field for field (the scalars advanced once)."""
    var a = Args()
    a.p0 = P
    a.p1 = G
    a.p2 = s1
    a.p3 = s2
    a.p4 = s3
    a.i0 = cfg.kind
    a.i1 = t
    a.i2 = cfg.flags
    a.f0 = lr
    a.f1 = cfg.f1
    a.f2 = cfg.f2
    a.f3 = cfg.eps
    a.f4 = cfg.wd
    var sc = opt_scalars(cfg, st, t, lr)
    a.f5 = sc[0]
    a.f6 = sc[1]
    a.f7 = sc[2]
    return a


def _stream_step(mut ex: DeviceExec, a0: Args, ps: List[Int], gs: List[Int], sizes: List[Int]) raises:
    """OPT_STREAM: chunk c's uploads, its `op_opt` launch and its DMA into a
    pinned half are queued, then the host reads chunk c - 1 out of the other
    half, then one wait. A stage of parity c & 1 was last used by chunk c - 2,
    whose DMA ended at that chunk's wait (uploads) and whose host read came
    before chunk c - 1's wait (download). Returns with every byte in the
    caller's arrays. a0.p0 .. p4 are the flat P, G, s1, s2, s3."""
    var hi = List[Int]()
    for _ in range(6):
        hi.append(_pool_host(ex.ctx, OPT_PIPE_CH))
    # The sequence pool's lists under its lock word (lane r5-proving): the stages' addresses only.
    var pool = _pool_locked()
    # 0, 1: parameter stages; 2, 3: gradient stages; 4, 5: download halves
    var stg = List[Int]()
    for k in range(6):
        stg.append(Int(pool[].host[hi[k]].unsafe_ptr()))
    _pool_unlock(pool)
    var P = a0.p0
    var G = a0.p1
    var c = 0
    var have_prev = False
    var prev_dst = 0
    var prev_cnt = 0
    var prev_half = 0
    var off = 0
    for j in range(len(sizes)):
        var done = 0
        while done < sizes[j]:
            var cnt = min(OPT_PIPE_CH, sizes[j] - done)
            var par = c & 1
            var o = off + done
            var hp = FP(unsafe_from_address=ps[j] + done * 4)
            var hg = FP(unsafe_from_address=gs[j] + done * 4)
            var fpp = ex._find(P + o, cnt)
            var vp = ex._sub(fpp[0], fpp[1], cnt)
            var fgg = ex._find(G + o, cnt)
            var vg = ex._sub(fgg[0], fgg[1], cnt)
            comptime if OPT_RAW_UP:
                ex.ctx.enqueue_copy(dst_buf=vp, src_ptr=hp)
                ex.ctx.enqueue_copy(dst_buf=vg, src_ptr=hg)
            else:
                var sp = FP(unsafe_from_address=stg[par])
                var sg = FP(unsafe_from_address=stg[2 + par])
                unsafe_memcpy(dest=sp, src=hp, count=cnt)
                unsafe_memcpy(dest=sg, src=hg, count=cnt)
                ex.ctx.enqueue_copy(dst_buf=vp, src_ptr=sp)
                ex.ctx.enqueue_copy(dst_buf=vg, src_ptr=sg)
            var a = a0
            a.p0 = a0.p0 + o
            a.p1 = a0.p1 + o
            a.p2 = a0.p2 + o
            a.p3 = a0.p3 + o
            a.p4 = a0.p4 + o
            ex.launch[OP_OPT](a, cnt)
            ex.ctx.enqueue_copy(dst_ptr=FP(unsafe_from_address=stg[4 + par]), src_buf=vp)
            _ = vg^
            _ = vp^
            if have_prev:
                # overlaps chunk c's uploads, launch and DMA
                unsafe_memcpy(dest=FP(unsafe_from_address=prev_dst), src=FP(unsafe_from_address=stg[4 + prev_half]),
                       count=prev_cnt)
            ex.ctx.synchronize()
            have_prev = True
            prev_dst = ps[j] + done * 4
            prev_cnt = cnt
            prev_half = par
            done += cnt
            c += 1
        off += sizes[j]
    if have_prev:
        unsafe_memcpy(dest=FP(unsafe_from_address=prev_dst), src=FP(unsafe_from_address=stg[4 + prev_half]),
               count=prev_cnt)
    for k in range(6):
        _pool_release_host(hi[k])
    # the executor's own bookkeeping (the queue is empty)
    ex.sync()


def opt_resident_step(handle: Int, addrs: List[Int], ip: List[Int],
                      fp: List[Float64]) raises -> Int:
    """One element-wise step with the state on the device.
    addrs = [param_0 .. param_{J-1}, grad_0 .. grad_{J-1}, scalars];
    ip = [J, kind, flags, t, t0, size_0 .. size_{J-1}]; fp = [lr, f1, f2,
    eps, weight_decay, f7]; scalars as `opt_step_py`'s (float32[3] after
    step t0, advanced through t and written back). Every tensor is updated
    in place. Returns n."""
    var h = _handle(handle, RES_ELEMENTWISE)
    if len(ip) < 6 or len(fp) != 6:
        raise Error("optimizer_resident_step: requires >= 6 integer and 6 float parameters")
    var J = ip[0]
    if J < 1 or len(ip) != 5 + J or len(addrs) != 2 * J + 1:
        raise Error("optimizer_resident_step: J >= 1 tensors, 2 J + 1 addresses and J sizes")
    var t = ip[3]
    var t0 = ip[4]
    if t < 1 or t0 < 0 or t0 >= t:
        raise Error("optimizer_resident_step: the one-based step t >= 1 and the scalars' step 0 <= t0 < t")
    var pool = _RES.get_or_create_ptr()
    var n = pool[].n[h]
    var used = pool[].used[h]
    var sizes = List[Int]()
    for j in range(J):
        sizes.append(ip[5 + j])
    var pg = _tensors(addrs, J, sizes, n)
    var cfg = _opt_of(ip, 1, fp, 1)
    var u = opt_slots(cfg)
    var want = (1 if u[0] else 0) | (2 if u[1] else 0) | (4 if u[2] else 0)
    if (want & used) != want:
        raise Error("optimizer_resident_step: the configuration uses a state slot the handle did not open")
    var sc = _host_fp(addrs[2 * J], "scalars")
    var st = OptState()
    var k0 = 1
    if t0 > 0:
        st.pw1 = sc.unsafe_load(0)
        st.pw2 = sc.unsafe_load(1)
        st.mu_prod = sc.unsafe_load(2)
        k0 = t0 + 1
    for k in range(k0, t):
        _ = opt_scalars(cfg, st, k, Float32(fp[0]))
    var ex = DeviceExec()
    var P = ex._alloc(n, False)
    var G = ex._alloc(n, False)
    comptime if OPT_STREAM:
        # an unused slot names P (never read by op_opt for that kind)
        var a0 = _opt_args(cfg, st, t, Float32(fp[0]), P, G, _slot_ptr(h, 0) if u[0] else P,
                           _slot_ptr(h, 1) if u[1] else P, _slot_ptr(h, 2) if u[2] else P)
        _stream_step(ex, a0, pg[0], pg[1], sizes)
    else:
        _upload_all(ex, P, G, pg[0], pg[1], sizes)
        var s1 = _slot_ptr(h, 0) if u[0] else P
        var s2 = _slot_ptr(h, 1) if u[1] else P
        var s3 = _slot_ptr(h, 2) if u[2] else P
        opt_step(ex, cfg, st, t, Float32(fp[0]), P, G, s1, s2, s3, n)
        _download_all(ex, P, pg[0], sizes)
    sc.unsafe_store(0, st.pw1)
    sc.unsafe_store(1, st.pw2)
    sc.unsafe_store(2, st.mu_prod)
    return n


def lamb_resident_step(handle: Int, addrs: List[Int], ip: List[Int],
                       fp: List[Float64]) raises -> Int:
    """One LAMB step with exp_avg / exp_avg_sq and the table on the device
    (`lamb_core`). addrs = [param_0 .. param_{J-1}, grad_0 .. grad_{J-1},
    scalars]; ip = [J, t, flags]; fp = [lr, beta1, beta2, eps,
    weight_decay, max_grad_norm, t0]; scalars as `lamb_step_impl`'s. The
    tensor sizes are the handle's offsets. Returns n."""
    var h = _handle(handle, RES_LAMB)
    if len(ip) != 3 or len(fp) != 7:
        raise Error("lamb_resident_step: requires 3 integer and 7 float parameters")
    var pool = _RES.get_or_create_ptr()
    var nt = pool[].nt[h]
    var n = pool[].n[h]
    var nb = pool[].nb[h]
    var J = ip[0]
    var t = ip[1]
    var flags = ip[2]
    if J != nt or len(addrs) != 2 * J + 1 or t < 1:
        raise Error("lamb_resident_step: the handle's " + String(nt) + " tensors, 2 J + 1 addresses and t >= 1")
    var sizes = List[Int]()
    for k in range(nt):  # small-loop(nt: parameter tensors of the handle): per-tensor sizes from the offset table
        sizes.append(pool[].offs[h][k + 1] - pool[].offs[h][k])
    var pg = _tensors(addrs, J, sizes, n)
    var t0 = 0
    if (flags & 8) != 0:
        t0 = Int(fp[6])
        if t0 < 0 or t0 >= t or Float64(t0) != fp[6]:
            raise Error("lamb_resident_step: the scalars' step t0 must be an integer with 0 <= t0 < t")
    var sc = _host_fp(addrs[2 * J], "scalars")
    var bias = lamb_bias(flags, t, t0, Float32(fp[1]), Float32(fp[2]), sc, True)
    var ex = DeviceExec()
    var P = ex._alloc(n, False)
    var G = ex._alloc(n, False)
    _upload_all(ex, P, G, pg[0], pg[1], sizes)
    var U = ex._alloc(n, False)
    var partsA = ex._alloc(nb, False)
    var partsB = ex._alloc(nb, False)
    var nrm = ex._alloc(nt, False)
    var ratio = ex._alloc(nt, False)
    var scal = ex._alloc(1, False)
    var TAB = FP(unsafe_from_address=Int(pool[].tab[h].unsafe_ptr()))
    lamb_core(ex, P, G, _slot_ptr(h, 0), _slot_ptr(h, 1), U, TAB, partsA, partsB, nrm, ratio, scal,
              n, nt, nb, flags, Float32(fp[0]), Float32(fp[1]), Float32(fp[2]), Float32(fp[3]), Float32(fp[4]),
              Float32(fp[5]), bias[0], bias[1])
    _download_all(ex, P, pg[0], sizes)
    if (flags & 8) != 0:
        sc.unsafe_store(0, bias[2])
        sc.unsafe_store(1, bias[3])
    return n


def adafactor_resident_open(ip: List[Int]) raises -> List[Int]:
    """AF_RESIDENT: one Adafactor tensor's handle, ip = [R, C] (C = 0 for a
    vector of R values): slot 0 = row_var (R) or the variance (R), slot 1 =
    col_var (C, matrices only), zero filled. Returns [handle, used mask]
    (and 1 under OPT_ZERO_OPEN)."""
    comptime if not AF_RESIDENT:
        raise Error("adafactor_resident_open: not built (IDENTICAL default, FAST + Apple default)")
    if len(ip) != 2:
        raise Error("adafactor_resident_open: requires [R, C]")
    var R = ip[0]
    var C = ip[1]
    if R < 1 or C < 0:
        raise Error("adafactor_resident_open: R >= 1 and C >= 0")
    var n = R * C if C > 0 else R
    var used = 3 if C > 0 else 1
    var h = _open_sized(RES_ADAFACTOR, n, R, C if C > 0 else 1, 1, used, R, List[Int](), List[Float32]())
    var pool = _RES.get_or_create_ptr()
    pool[].nb[h] = C
    return _pair(h, used)


def adafactor_resident_step(handle: Int, addrs: List[Int], ip: List[Int],
                            fp: List[Float64]) raises -> Int:
    """AF_RESIDENT: one Adafactor step of the handle's tensor with its
    second moment on the device. addrs = [param, grad]; ip = [R, C, t];
    fp = `adafactor_step`'s six. The parameter is updated in place.
    Returns n."""
    comptime if not AF_RESIDENT:
        raise Error("adafactor_resident_step: not built (IDENTICAL default, FAST + Apple default)")
    var h = _handle(handle, RES_ADAFACTOR)
    if len(addrs) != 2 or len(ip) != 3 or len(fp) != 6:
        raise Error("adafactor_resident_step: requires 2 addresses, 3 integer and 6 float parameters")
    var pool = _RES.get_or_create_ptr()
    var R = ip[0]
    var C = ip[1]
    var t = ip[2]
    if R != pool[].nt[h] or C != pool[].nb[h] or t < 1:
        raise Error("adafactor_resident_step: the handle's shape and a one-based step t >= 1")
    var n = pool[].n[h]
    var sizes = List[Int]()
    sizes.append(n)
    var pg = _tensors(addrs, 1, sizes, n)
    var ex = DeviceExec()
    var P = ex._alloc(n, False)
    var G = ex._alloc(n, False)
    _upload_all(ex, P, G, pg[0], pg[1], sizes)
    adafactor_core(ex, P, G, _slot_ptr(h, 0), _slot_ptr(h, 1), R, C, t, fp)
    _download_all(ex, P, pg[0], sizes)
    return n


# ---- device-parameter steps (lane gap-neural-io, 2026-10-08) -----------------------
# Plan plan gaps-2026-10-08 (git history) Section 4: the board's torch columns time
# ten `opt.step()` calls on a parameter and gradients already on the device;
# every resident step above still moved P and G up and P down (3 x 67 MB at
# 16,777,216 floats, ~15-25 ms) around ~1 ms of launches. The `_dev` forms
# take the parameters and gradients as resident sequence tensors
# (`sequence/seq_tensor.mojo` handles, `python/mojolearn/_x_sequence_device.py`)
# and update the parameters where they live: no transfer at all, one wait
# (the executor's drain when the call returns). The launches are the host
# forms' (`opt_step`'s Args through `_opt_args`, `lamb_core`,
# `adafactor_core`) on the same values: no bit moves. Several element-wise
# tensors are one `op_opt` launch each over their own range of the flat
# state slots (`op_opt` is per element, the scalars computed once, as
# OPT_STREAM's chunks); LAMB's per-tensor norms read one flat buffer, so its
# tensors are copied device to device into it and the parameters back.


def _dev_ptrs(handles: List[Int], J: Int, sizes: List[Int], n: Int) raises -> Tuple[List[Int], List[Int]]:
    """The J parameter and J gradient device addresses (handles[0:J],
    handles[J:2J]), each tensor's size checked, the sizes adding up to n."""
    var tot = 0
    for j in range(J):
        if sizes[j] < 1:
            raise Error("optimizer_resident_step_dev: every tensor holds at least one value")
        tot += sizes[j]
    if tot != n:
        raise Error("optimizer_resident_step_dev: the tensors hold " + String(tot) + " values, the handle " + String(n))
    var ps = List[Int]()
    var gs = List[Int]()
    for j in range(J):  # small-loop(J: parameter tensors of the model): one handle lookup per tensor, no data
        ps.append(Int(seq_tensor_ptr(handles[j], sizes[j], "params")))
        gs.append(Int(seq_tensor_ptr(handles[J + j], sizes[j], "grads")))
    return (ps^, gs^)


def _opt_launch(ctx: DeviceContext, a: Args, n: Int) raises:
    """The one `op_opt` launch of `DeviceExec.launch[OP_OPT]` (its generic
    `seq_kernel` route: one thread per element, TPB a block, the integers and
    floats packed two to a word), enqueued on the shared context with no
    executor around it, so nothing drains when it returns (lane
    fb15-sync-counts, 2026-11-18)."""
    if n <= 0:
        return
    if _args_i32_out(a):
        raise Error("optimizer_resident_step_dev: an integer argument does not fit Int32")
    ctx.enqueue_function[seq_kernel[OP_OPT]](
        a.p0, a.p1, a.p2, a.p3, a.p4, a.p5, a.p6, a.p7, a.p8, a.p9, a.p10, a.p11,
        _pack_ii(a.i0, a.i1), _pack_ii(a.i2, a.i3), _pack_ii(a.i4, a.i5),
        _pack_ii(a.i6, a.i7), _pack_ii(a.i8, a.i9), _pack_ii(a.i10, a.i11),
        _pack_ff(a.f0, a.f1), _pack_ff(a.f2, a.f3), _pack_ff(a.f4, a.f5), _pack_ff(a.f6, a.f7),
        Int64(n),
        grid_dim=((n + TPB - 1) // TPB, 1, 1),
        block_dim=(TPB, 1, 1),
    )


def opt_resident_step_dev(handle: Int, handles: List[Int], ip: List[Int],
                          fp: List[Float64]) raises -> Int:
    """`opt_resident_step` on resident tensors: handles = [param_0 ..
    param_{J-1}, grad_0 .. grad_{J-1}] (seq_tensor handles), then the host
    address of the float32[3] scalars; ip and fp as there. Every parameter
    tensor is updated in place on the device. Returns n."""
    var h = _handle(handle, RES_ELEMENTWISE)
    if len(ip) < 6 or len(fp) != 6:
        raise Error("optimizer_resident_step_dev: requires >= 6 integer and 6 float parameters")
    var J = ip[0]
    if J < 1 or len(ip) != 5 + J or len(handles) != 2 * J + 1:
        raise Error("optimizer_resident_step_dev: J >= 1 tensors, 2 J + 1 handles and J sizes")
    var t = ip[3]
    var t0 = ip[4]
    if t < 1 or t0 < 0 or t0 >= t:
        raise Error("optimizer_resident_step_dev: the one-based step t >= 1 and the scalars' step 0 <= t0 < t")
    var pool = _RES.get_or_create_ptr()
    var n = pool[].n[h]
    var used = pool[].used[h]
    var sizes = List[Int]()
    for j in range(J):
        sizes.append(ip[5 + j])
    var pg = _dev_ptrs(handles, J, sizes, n)
    var cfg = _opt_of(ip, 1, fp, 1)
    var u = opt_slots(cfg)
    var want = (1 if u[0] else 0) | (2 if u[1] else 0) | (4 if u[2] else 0)
    if (want & used) != want:
        raise Error("optimizer_resident_step_dev: the configuration uses a state slot the handle did not open")
    var sc = _host_fp(handles[2 * J], "scalars")
    var st = OptState()
    var k0 = 1
    if t0 > 0:
        st.pw1 = sc.unsafe_load(0)
        st.pw2 = sc.unsafe_load(1)
        st.mu_prod = sc.unsafe_load(2)
        k0 = t0 + 1
    for k in range(k0, t):
        _ = opt_scalars(cfg, st, k, Float32(fp[0]))
    var ctx = sequence_ctx()
    var P0 = FP(unsafe_from_address=pg[0][0])
    # an unused slot names the first parameter tensor (never read by op_opt
    # for that kind); the scalars advance once for the whole step
    var a0 = _opt_args(cfg, st, t, Float32(fp[0]), P0, FP(unsafe_from_address=pg[1][0]),
                       _slot_ptr(h, 0) if u[0] else P0, _slot_ptr(h, 1) if u[1] else P0,
                       _slot_ptr(h, 2) if u[2] else P0)
    var off = 0
    for j in range(J):  # small-loop(J: parameter tensors of the model): one element-wise launch per tensor
        var a = a0
        var pj = FP(unsafe_from_address=pg[0][j])
        a.p0 = pj
        a.p1 = FP(unsafe_from_address=pg[1][j])
        a.p2 = (a0.p2 + off) if u[0] else pj
        a.p3 = (a0.p3 + off) if u[1] else pj
        a.p4 = (a0.p4 + off) if u[2] else pj
        _opt_launch(ctx, a, sizes[j])
        off += sizes[j]
    # lane fb15-sync-counts (2026-11-18): ONE wait per step. The step ran on a
    # `DeviceExec`, whose `sync()` drained the queue and whose destructor then
    # drained it again (2 cuStreamSynchronize per step, 20 per 10-step fit).
    # The launch now goes straight onto the shared in-order context: it
    # allocates, stages and frees nothing, so this wait is the step's only
    # one (none under OPT_ASYNC_DEV_STEP: every reader of the parameters or
    # state drains first). No bit moves: the same kernel, Args and grid.
    comptime if not OPT_ASYNC_DEV_STEP:
        ctx.synchronize()
    sc.unsafe_store(0, st.pw1)
    sc.unsafe_store(1, st.pw2)
    sc.unsafe_store(2, st.mu_prod)
    return n


def lamb_resident_step_dev(handle: Int, handles: List[Int], ip: List[Int],
                           fp: List[Float64]) raises -> Int:
    """`lamb_resident_step` on resident tensors: handles = [param_0 ..
    param_{J-1}, grad_0 .. grad_{J-1}] (seq_tensor handles), then the host
    address of the scalars; ip and fp as there. One tensor is updated where
    it lives; several are copied device to device into the flat buffers
    `lamb_core` folds and the parameters copied back. Returns n."""
    var h = _handle(handle, RES_LAMB)
    if len(ip) != 3 or len(fp) != 7:
        raise Error("lamb_resident_step_dev: requires 3 integer and 7 float parameters")
    var pool = _RES.get_or_create_ptr()
    var nt = pool[].nt[h]
    var n = pool[].n[h]
    var nb = pool[].nb[h]
    var J = ip[0]
    var t = ip[1]
    var flags = ip[2]
    if J != nt or len(handles) != 2 * J + 1 or t < 1:
        raise Error("lamb_resident_step_dev: the handle's " + String(nt) + " tensors, 2 J + 1 handles and t >= 1")
    var sizes = List[Int]()
    for k in range(nt):  # small-loop(nt: parameter tensors of the handle): per-tensor sizes from the offset table
        sizes.append(pool[].offs[h][k + 1] - pool[].offs[h][k])
    var pg = _dev_ptrs(handles, J, sizes, n)
    var t0 = 0
    if (flags & 8) != 0:
        t0 = Int(fp[6])
        if t0 < 0 or t0 >= t or Float64(t0) != fp[6]:
            raise Error("lamb_resident_step_dev: the scalars' step t0 must be an integer with 0 <= t0 < t")
    var sc = _host_fp(handles[2 * J], "scalars")
    var bias = lamb_bias(flags, t, t0, Float32(fp[1]), Float32(fp[2]), sc, True)
    var ex = DeviceExec()
    var P: FP
    var G: FP
    if J == 1:
        P = FP(unsafe_from_address=pg[0][0])
        G = FP(unsafe_from_address=pg[1][0])
    else:
        P = ex._alloc(n, False)
        G = ex._alloc(n, False)
        var o = 0
        for j in range(J):  # small-loop(J: parameter tensors of the model): one device copy per tensor into the flat buffers
            var fpp = ex._find(P + o, sizes[j])
            var vp = ex._sub(fpp[0], fpp[1], sizes[j])
            var sp = seq_tensor_view(handles[j], sizes[j], "params")
            ex.ctx.enqueue_copy(dst_buf=vp, src_buf=sp)
            var fgg = ex._find(G + o, sizes[j])
            var vg = ex._sub(fgg[0], fgg[1], sizes[j])
            var sg = seq_tensor_view(handles[J + j], sizes[j], "grads")
            ex.ctx.enqueue_copy(dst_buf=vg, src_buf=sg)
            _ = vp^
            _ = sp^
            _ = vg^
            _ = sg^
            o += sizes[j]
    var U = ex._alloc(n, False)
    var partsA = ex._alloc(nb, False)
    var partsB = ex._alloc(nb, False)
    var nrm = ex._alloc(nt, False)
    var ratio = ex._alloc(nt, False)
    var scal = ex._alloc(1, False)
    var TAB = FP(unsafe_from_address=Int(pool[].tab[h].unsafe_ptr()))
    lamb_core(ex, P, G, _slot_ptr(h, 0), _slot_ptr(h, 1), U, TAB, partsA, partsB, nrm, ratio, scal,
              n, nt, nb, flags, Float32(fp[0]), Float32(fp[1]), Float32(fp[2]), Float32(fp[3]), Float32(fp[4]),
              Float32(fp[5]), bias[0], bias[1])
    if J > 1:
        var o2 = 0
        for j in range(J):  # small-loop(J: parameter tensors of the model): one device copy back per tensor
            var fpp2 = ex._find(P + o2, sizes[j])
            var vp2 = ex._sub(fpp2[0], fpp2[1], sizes[j])
            var dp = seq_tensor_view(handles[j], sizes[j], "params")
            ex.ctx.enqueue_copy(dst_buf=dp, src_buf=vp2)
            _ = vp2^
            _ = dp^
            o2 += sizes[j]
    # lane fb15-sync-counts (2026-11-18): the executor's destructor drains the
    # queue (here, at its last use above) before its buffers go; the explicit
    # `ex.sync()` before it was a second cuStreamSynchronize per step on an
    # empty queue (no staged upload or pending download to release). The
    # scalars below are host words. One wait per step. No bit moves.
    if (flags & 8) != 0:
        sc.unsafe_store(0, bias[2])
        sc.unsafe_store(1, bias[3])
    return n


def adafactor_resident_step_dev(handle: Int, handles: List[Int], ip: List[Int],
                                fp: List[Float64]) raises -> Int:
    """AF_RESIDENT: `adafactor_resident_step` on resident tensors:
    handles = [param, grad] (seq_tensor handles); ip = [R, C, t]; fp =
    `adafactor_step`'s six. The parameter is updated where it lives (the
    same `adafactor_core` launches). Returns n."""
    comptime if not AF_RESIDENT:
        raise Error("adafactor_resident_step_dev: not built (IDENTICAL default, FAST + Apple default)")
    var h = _handle(handle, RES_ADAFACTOR)
    if len(handles) != 2 or len(ip) != 3 or len(fp) != 6:
        raise Error("adafactor_resident_step_dev: requires 2 handles, 3 integer and 6 float parameters")
    var pool = _RES.get_or_create_ptr()
    var R = ip[0]
    var C = ip[1]
    var t = ip[2]
    if R != pool[].nt[h] or C != pool[].nb[h] or t < 1:
        raise Error("adafactor_resident_step_dev: the handle's shape and a one-based step t >= 1")
    var n = pool[].n[h]
    var P = seq_tensor_ptr(handles[0], n, "param")
    var G = seq_tensor_ptr(handles[1], n, "grad")
    var ex = DeviceExec()
    adafactor_core(ex, P, G, _slot_ptr(h, 0), _slot_ptr(h, 1), R, C, t, fp)
    # lane fb15-sync-counts (2026-11-18): one wait per step, the executor's
    # destructor drain (was `ex.sync()` and then that drain again on an empty
    # queue). No bit moves.
    return n
