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
handle indexes the pool and a closed handle's slot is reused."""
from std.ffi import _Global
from std.memory import bitcast, memcpy
from std.sys.compile import is_defined
from std.sys.defines import get_defined_int
from std.sys.info import has_apple_gpu_accelerator
from std.python import Python, PythonObject
from max.gpu.host import DeviceBuffer

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, NUMERIC_IDENTICAL
from sequence.exec_device import DeviceExec, X_SEQUENCE_POOL, _pool_host, _pool_release_host, sequence_ctx
from sequence.ops import FP
from sequence.recurrent import OptState, opt_scalars, opt_step
from sequence.pyapi import fptr, fval, ival, lamb_bias, lamb_core, lamb_offsets, lamb_table, opt_of, opt_slots

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
comptime _OPT_APPLE_FAST = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
comptime OPT_RAW_UP = _OPT_APPLE_FAST and is_defined["MOJOLEARN_OPT_RAW_UP"]()
#: PIPE_DOWN and ZERO_OPEN are the FAST+Apple default since the M3 A/B of
#: lane/apple-fast-optspeed f2b6491a8 (n=1, synthetic, output digest
#: identical): all three switches adagrad 317 -> 187 ms, lamb 340 -> 206,
#: adamax 326 -> 195; PIPE_DOWN alone 318 -> 191. Off with
#: MOJOLEARN_OPT_PIPE_DOWN_OFF / MOJOLEARN_OPT_ZERO_OPEN_OFF; the old
#: -D names are harmless. RAW_UP measured noise (310 -> 307): opt-in.
comptime OPT_PIPE_DOWN = _OPT_APPLE_FAST and not is_defined["MOJOLEARN_OPT_PIPE_DOWN_OFF"]()
comptime OPT_ZERO_OPEN = _OPT_APPLE_FAST and not is_defined["MOJOLEARN_OPT_ZERO_OPEN_OFF"]()
#: the pipelined download's chunk, floats (8 MB; lane apple-fast-gap-optim:
#: -D MOJOLEARN_OPT_FAST_PIPE_CH=<floats> for the A/B)
comptime OPT_PIPE_CH = get_defined_int["MOJOLEARN_OPT_FAST_PIPE_CH", 1 << 21]()
#: lane apple-fast-gap-optim (2026-10-03, docs/apple-fast/notes/gap-optim.md),
#: default OFF, FAST + Apple only: the parameter read-back.
#:  MOJOLEARN_OPT_FAST_MAP_DOWN: each tensor is read through
#:    `DeviceBuffer.map_to_host` and one memcpy (no pinned halves).
#:  MOJOLEARN_OPT_FAST_RAW_DOWN: each tensor is DMAd straight into the
#:    caller's array in OPT_PIPE_CH chunks, all queued, one wait.
#: Copies only: the same bytes.
comptime OPT_MAP_DOWN = _OPT_APPLE_FAST and is_defined["MOJOLEARN_OPT_FAST_MAP_DOWN"]()
comptime OPT_RAW_DOWN = _OPT_APPLE_FAST and is_defined["MOJOLEARN_OPT_FAST_RAW_DOWN"]() and not OPT_MAP_DOWN

#: the handle kinds
comptime RES_ELEMENTWISE = 1
comptime RES_LAMB = 2


struct _ResPool(Defaultable, Movable):
    #: the three state slots (a 1-float placeholder for a slot not kept)
    var s0: List[DeviceBuffer[DType.float32]]
    var s1: List[DeviceBuffer[DType.float32]]
    var s2: List[DeviceBuffer[DType.float32]]
    var used: List[Int]
    #: LAMB's table (`sequence/pyapi.mojo::lamb_table`), on the device
    var tab: List[DeviceBuffer[DType.float32]]
    #: floats per slot (0 marks a free handle), kind, tensors, blocks
    var n: List[Int]
    var kind: List[Int]
    var nt: List[Int]
    var nb: List[Int]
    #: LAMB's tensor offsets (nt + 1)
    var offs: List[List[Int]]

    def __init__(out self):
        self.s0 = List[DeviceBuffer[DType.float32]]()
        self.s1 = List[DeviceBuffer[DType.float32]]()
        self.s2 = List[DeviceBuffer[DType.float32]]()
        self.used = List[Int]()
        self.tab = List[DeviceBuffer[DType.float32]]()
        self.n = List[Int]()
        self.kind = List[Int]()
        self.nt = List[Int]()
        self.nb = List[Int]()
        self.offs = List[List[Int]]()


comptime _RES_NAME = "MojoXSequenceOptResidentIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoXSequenceOptResidentFast"
comptime _RES = _Global[StorageType=_ResPool, name=_RES_NAME, init_fn=_ResPool.__init__]


def _slot_buf(n: Int, keep: Bool) raises -> DeviceBuffer[DType.float32]:
    var ctx = sequence_ctx()
    var b = ctx.enqueue_create_buffer[DType.float32](n if keep else 1)
    b.enqueue_fill(Float32(0.0))
    return b^


def _open(kind: Int, n: Int, used: Int, nt: Int, var offs: List[Int], tab: List[Float32]) raises -> Int:
    var pool = _RES.get_or_create_ptr()
    var ctx = sequence_ctx()
    var tb = ctx.enqueue_create_buffer[DType.float32](max(len(tab), 1))
    if len(tab) > 0:
        ctx.enqueue_copy(dst_buf=tb, src_ptr=tab.unsafe_ptr())
    var nb = 0
    if len(tab) > 0:
        nb = Int(bitcast[DType.int32](tab[len(tab) - 1]))
    var h = -1
    for j in range(len(pool[].n)):
        if pool[].n[j] == 0 and h < 0:
            h = j
    if h < 0:
        pool[].s0.append(_slot_buf(n, (used & 1) != 0))
        pool[].s1.append(_slot_buf(n, (used & 2) != 0))
        pool[].s2.append(_slot_buf(n, (used & 4) != 0))
        pool[].used.append(used)
        pool[].tab.append(tb^)
        pool[].n.append(n)
        pool[].kind.append(kind)
        pool[].nt.append(nt)
        pool[].nb.append(nb)
        pool[].offs.append(offs^)
        h = len(pool[].n) - 1
    else:
        pool[].s0[h] = _slot_buf(n, (used & 1) != 0)
        pool[].s1[h] = _slot_buf(n, (used & 2) != 0)
        pool[].s2[h] = _slot_buf(n, (used & 4) != 0)
        pool[].used[h] = used
        pool[].tab[h] = tb^
        pool[].n[h] = n
        pool[].kind[h] = kind
        pool[].nt[h] = nt
        pool[].nb[h] = nb
        pool[].offs[h] = offs^
    # the fills and the table copy are ordered before every later use on
    # the one in-order context; the table's host list must outlive its copy
    ctx.synchronize()
    return h


def _handle(handle: PythonObject, kind: Int) raises -> Int:
    var h = Int(py=handle)
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


def _pair(a: Int, b: Int) raises -> PythonObject:
    """A Python list [a, b]. `PythonObject([a, b])` does not build a list
    of two ints (the caller got a bare int back and its `h, used = ...`
    unpack raised TypeError)."""
    var out = Python.list()
    out.append(PythonObject(a))
    out.append(PythonObject(b))
    comptime if OPT_ZERO_OPEN:
        # the slots are zero filled on the device (`_slot_buf`): a caller
        # whose host copies are still the zeros it built need not upload them
        out.append(PythonObject(1))
    return out


def opt_resident_open_py(ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    """An element-wise optimizer's handle: ip = [n, kind, flags], fp = [lr,
    f1, f2, eps, weight_decay, f7] (the step's configuration; it decides
    which slots `op_opt` uses, `opt_slots`, and only those are kept, zero
    filled). Returns [handle, used slot mask]."""
    if len(ip) != 3 or len(fp) != 6:
        raise Error("optimizer_resident_open: requires 3 integer and 6 float parameters")
    var n = ival(ip, 0)
    if n < 1:
        raise Error("optimizer_resident_open: n must be >= 1")
    var cfg = opt_of(ip, 1, fp, 1)
    var u = opt_slots(cfg)
    var used = (1 if u[0] else 0) | (2 if u[1] else 0) | (4 if u[2] else 0)
    var h = _open(RES_ELEMENTWISE, n, used, 0, List[Int](), List[Float32]())
    return _pair(h, used)


def lamb_resident_open_py(ip: PythonObject) raises -> PythonObject:
    """A LAMB handle: ip = [n_tensors, off_0, ..., off_n]; exp_avg and
    exp_avg_sq (slots 0 and 1) zero filled, the table on the device.
    Returns [handle, 3]."""
    if len(ip) < 3:
        raise Error("lamb_resident_open: requires n_tensors and n_tensors + 1 offsets")
    var nt = ival(ip, 0)
    if nt < 1 or len(ip) != 2 + nt:
        raise Error("lamb_resident_open: n_tensors >= 1 and n_tensors + 1 offsets")
    var offs = lamb_offsets(ip, 1, nt)
    var tab = List[Float32]()
    _ = lamb_table(offs, tab)
    var n = offs[nt]
    var h = _open(RES_LAMB, n, 3, nt, offs^, tab)
    return _pair(h, 3)


def opt_resident_close_py(handle: PythonObject) raises -> PythonObject:
    """Free a handle's slots (after the queue drains)."""
    var h = _handle(handle, 0)
    var ctx = sequence_ctx()
    ctx.synchronize()
    var pool = _RES.get_or_create_ptr()
    pool[].s0[h] = ctx.enqueue_create_buffer[DType.float32](1)
    pool[].s1[h] = ctx.enqueue_create_buffer[DType.float32](1)
    pool[].s2[h] = ctx.enqueue_create_buffer[DType.float32](1)
    pool[].tab[h] = ctx.enqueue_create_buffer[DType.float32](1)
    pool[].offs[h] = List[Int]()
    pool[].used[h] = 0
    pool[].n[h] = 0
    return PythonObject(h)


def opt_resident_move_py(handle: PythonObject, slot: PythonObject, addr: PythonObject,
                         up: PythonObject) raises -> PythonObject:
    """Slot `slot` (0, 1, 2) to the host array at `addr` (up = 0) or from
    it (up = 1), n floats. Returns 1, or 0 for a slot the handle does not
    keep (nothing moves)."""
    var h = _handle(handle, 0)
    var s = Int(py=slot)
    if s < 0 or s > 2:
        raise Error("optimizer_resident: slot must be 0, 1 or 2")
    var pool = _RES.get_or_create_ptr()
    if (pool[].used[h] & (1 << s)) == 0:
        return PythonObject(0)
    var hp = fptr(addr, "state")
    var ctx = sequence_ctx()
    var to_dev = Int(py=up) != 0
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
    return PythonObject(1)


def _tensors(addrs: PythonObject, J: Int, sizes: List[Int], n: Int) raises -> Tuple[List[Int], List[Int]]:
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
        ps.append(Int(fptr(addrs[j], "params")))
        gs.append(Int(fptr(addrs[J + j], "grads")))
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
    for j in range(len(sizes)):
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
    var pool = X_SEQUENCE_POOL.get_or_create_ptr()
    var st0 = FP(unsafe_from_address=Int(pool[].host[h0].unsafe_ptr()))
    var st1 = FP(unsafe_from_address=Int(pool[].host[h1].unsafe_ptr()))
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
                memcpy(dest=FP(unsafe_from_address=prev_dst), src=st0 if prev_half == 0 else st1,
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
        memcpy(dest=FP(unsafe_from_address=prev_dst), src=st0 if prev_half == 0 else st1, count=prev_cnt)
    _pool_release_host(h0)
    _pool_release_host(h1)
    # the executor's own bookkeeping (no copy is pending: the queue is empty)
    ex.sync()


def _map_download(mut ex: DeviceExec, P: FP, ps: List[Int], sizes: List[Int]) raises:
    """OPT_MAP_DOWN: every tensor through `map_to_host` (the mapping waits
    for the queue) and one memcpy into the caller's array."""
    var off = 0
    for j in range(len(sizes)):
        var f = ex._find(P + off, sizes[j])
        var v = ex._sub(f[0], f[1], sizes[j])
        with v.map_to_host() as h:
            memcpy(dest=FP(unsafe_from_address=ps[j]), src=FP(unsafe_from_address=Int(h.unsafe_ptr())),
                   count=sizes[j])
        _ = v^
        off += sizes[j]
    ex.sync()


def _raw_download(mut ex: DeviceExec, P: FP, ps: List[Int], sizes: List[Int]) raises:
    """OPT_RAW_DOWN: every chunk DMAd straight into the caller's array, all
    queued behind the step's launch, one wait."""
    var off = 0
    for j in range(len(sizes)):
        var done = 0
        while done < sizes[j]:
            var cnt = min(OPT_PIPE_CH, sizes[j] - done)
            var f = ex._find(P + off + done, cnt)
            var v = ex._sub(f[0], f[1], cnt)
            ex.ctx.enqueue_copy(dst_ptr=FP(unsafe_from_address=ps[j] + done * 4), src_buf=v)
            _ = v^
            done += cnt
        off += sizes[j]
    ex.sync()


def _download_all(mut ex: DeviceExec, P: FP, ps: List[Int], sizes: List[Int]) raises:
    comptime if OPT_MAP_DOWN:
        _map_download(ex, P, ps, sizes)
        return
    comptime if OPT_RAW_DOWN:
        _raw_download(ex, P, ps, sizes)
        return
    comptime if OPT_PIPE_DOWN:
        _pipe_download(ex, P, ps, sizes)
        return
    var off = 0
    for j in range(len(sizes)):
        ex.download_async(FP(unsafe_from_address=ps[j]), P + off, sizes[j])
        off += sizes[j]
    ex.sync()


def opt_resident_step_py(handle: PythonObject, addrs: PythonObject, ip: PythonObject,
                         fp: PythonObject) raises -> PythonObject:
    """One element-wise step with the state on the device.
    addrs = [param_0 .. param_{J-1}, grad_0 .. grad_{J-1}, scalars];
    ip = [J, kind, flags, t, t0, size_0 .. size_{J-1}]; fp = [lr, f1, f2,
    eps, weight_decay, f7]; scalars as `opt_step_py`'s (float32[3] after
    step t0, advanced through t and written back). Every tensor is updated
    in place. Returns n."""
    var h = _handle(handle, RES_ELEMENTWISE)
    if len(ip) < 6 or len(fp) != 6:
        raise Error("optimizer_resident_step: requires >= 6 integer and 6 float parameters")
    var J = ival(ip, 0)
    if J < 1 or len(ip) != 5 + J or len(addrs) != 2 * J + 1:
        raise Error("optimizer_resident_step: J >= 1 tensors, 2 J + 1 addresses and J sizes")
    var t = ival(ip, 3)
    var t0 = ival(ip, 4)
    if t < 1 or t0 < 0 or t0 >= t:
        raise Error("optimizer_resident_step: the one-based step t >= 1 and the scalars' step 0 <= t0 < t")
    var pool = _RES.get_or_create_ptr()
    var n = pool[].n[h]
    var used = pool[].used[h]
    var sizes = List[Int]()
    for j in range(J):
        sizes.append(ival(ip, 5 + j))
    var pg = _tensors(addrs, J, sizes, n)
    var cfg = opt_of(ip, 1, fp, 1)
    var u = opt_slots(cfg)
    var want = (1 if u[0] else 0) | (2 if u[1] else 0) | (4 if u[2] else 0)
    if (want & used) != want:
        raise Error("optimizer_resident_step: the configuration uses a state slot the handle did not open")
    var sc = fptr(addrs[2 * J], "scalars")
    var st = OptState()
    var k0 = 1
    if t0 > 0:
        st.pw1 = sc.unsafe_load(0)
        st.pw2 = sc.unsafe_load(1)
        st.mu_prod = sc.unsafe_load(2)
        k0 = t0 + 1
    for k in range(k0, t):
        _ = opt_scalars(cfg, st, k, fval(fp, 0))
    var ex = DeviceExec()
    var P = ex._alloc(n, False)
    var G = ex._alloc(n, False)
    _upload_all(ex, P, G, pg[0], pg[1], sizes)
    var s1 = _slot_ptr(h, 0) if u[0] else P
    var s2 = _slot_ptr(h, 1) if u[1] else P
    var s3 = _slot_ptr(h, 2) if u[2] else P
    opt_step(ex, cfg, st, t, fval(fp, 0), P, G, s1, s2, s3, n)
    _download_all(ex, P, pg[0], sizes)
    sc.unsafe_store(0, st.pw1)
    sc.unsafe_store(1, st.pw2)
    sc.unsafe_store(2, st.mu_prod)
    return PythonObject(n)


def lamb_resident_step_py(handle: PythonObject, addrs: PythonObject, ip: PythonObject,
                          fp: PythonObject) raises -> PythonObject:
    """One LAMB step with exp_avg / exp_avg_sq and the table on the device
    (`lamb_core`). addrs = [param_0 .. param_{J-1}, grad_0 .. grad_{J-1},
    scalars]; ip = [J, t, flags]; fp = [lr, beta1, beta2, eps,
    weight_decay, max_grad_norm, t0]; scalars as `lamb_step_py`'s. The
    tensor sizes are the handle's offsets. Returns n."""
    var h = _handle(handle, RES_LAMB)
    if len(ip) != 3 or len(fp) != 7:
        raise Error("lamb_resident_step: requires 3 integer and 7 float parameters")
    var pool = _RES.get_or_create_ptr()
    var nt = pool[].nt[h]
    var n = pool[].n[h]
    var nb = pool[].nb[h]
    var J = ival(ip, 0)
    var t = ival(ip, 1)
    var flags = ival(ip, 2)
    if J != nt or len(addrs) != 2 * J + 1 or t < 1:
        raise Error("lamb_resident_step: the handle's " + String(nt) + " tensors, 2 J + 1 addresses and t >= 1")
    var sizes = List[Int]()
    for k in range(nt):
        sizes.append(pool[].offs[h][k + 1] - pool[].offs[h][k])
    var pg = _tensors(addrs, J, sizes, n)
    var t0 = 0
    if (flags & 8) != 0:
        t0 = Int(Float64(py=fp[6]))
        if t0 < 0 or t0 >= t or Float64(t0) != Float64(py=fp[6]):
            raise Error("lamb_resident_step: the scalars' step t0 must be an integer with 0 <= t0 < t")
    var sc = fptr(addrs[2 * J], "scalars")
    var bias = lamb_bias(flags, t, t0, fval(fp, 1), fval(fp, 2), sc, True)
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
              n, nt, nb, flags, fval(fp, 0), fval(fp, 1), fval(fp, 2), fval(fp, 3), fval(fp, 4),
              fval(fp, 5), bias[0], bias[1])
    _download_all(ex, P, pg[0], sizes)
    if (flags & 8) != 0:
        sc.unsafe_store(0, bias[2])
        sc.unsafe_store(1, bias[3])
    return PythonObject(n)
