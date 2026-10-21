# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The device half of the public sequence optimizers' native calls
(sequence/opt_public.mojo has the words, the parsing and the host routes).

Here the route is chosen: device parameters (resident sequence tensors)
step where they live; otherwise the state stays resident on the device
once a handle is open, or when `MOJOLEARN_OPTIMIZER_RESIDENT` is not "0"
(the default); otherwise the per-call step runs on a `DeviceExec`. The
resident handle is opened here, the host copies of the state are uploaded
before a step when they are the newest (a reader may have written them)
and downloaded for a state read, and host gradients for device parameters
are uploaded into temporary resident tensors and freed after the step.
Every step is the existing entry of sequence/opt_resident.mojo or
sequence/ops_impl.mojo on the same values in the same order: no bit
moves. No Python here."""
from std.os import getenv

from sequence.exec_device import DeviceExec
from sequence.opt_public import (
    IP64,
    SeqStep,
    StateCopy,
    W_FLAGS,
    W_FRESH,
    W_HANDLE,
    W_N,
    W_OWNED,
    W_PRISTINE,
    W_SC_T,
    W_T,
    W_USED,
    adafactor_step_host,
    begin_step,
    elementwise_step_host,
    export_copy,
    import_copy,
    lamb_offsets_of,
    lamb_step_host,
    set_word,
    word,
    words_ptr,
)
from sequence.opt_resident import (
    AF_RESIDENT,
    adafactor_resident_open,
    adafactor_resident_step,
    adafactor_resident_step_dev,
    lamb_resident_open,
    lamb_resident_step,
    lamb_resident_step_dev,
    opt_resident_close,
    opt_resident_move,
    opt_resident_open,
    opt_resident_step,
    opt_resident_step_dev,
)
from sequence.seq_tensor import seq_tensor_alloc, seq_tensor_free, seq_tensor_upload


def _resident_env() -> Bool:
    """`MOJOLEARN_OPTIMIZER_RESIDENT=0` keeps the per-call entries (the
    before arm of the resident A/B)."""
    return getenv("MOJOLEARN_OPTIMIZER_RESIDENT", "1") != "0"


def _handles_ptr(addr: Int) raises -> IP64:
    if addr == 0:
        raise Error("sequence optimizer: null handle buffer")
    return IP64(unsafe_from_address=addr)


def _opened(w: IP64, zeroed: Bool, init_zero: Bool):
    """After an open: the host copies need no upload when the device slots
    are zero filled and the host copies are still the constructor's zeros."""
    set_word(w, W_HANDLE, 1)
    set_word(w, W_OWNED, 0 if (zeroed and word(w, W_PRISTINE) == 1 and init_zero) else 1)


def _upload_slots(w: IP64, h: Int, slots: List[Int]) raises:
    """The host copies into the handle's slots when they are the newest;
    then the device holds the newest state."""
    if word(w, W_OWNED) != 0:
        for k in range(len(slots)):
            if slots[k] != 0:
                _ = opt_resident_move(h, k, slots[k], True)
        set_word(w, W_OWNED, 0)
    set_word(w, W_FRESH, 0)


def _grad_tensors(a: SeqStep, keep: List[Int]) raises -> Tuple[List[Int], List[Int]]:
    """Device parameters: each kept gradient as a resident tensor handle,
    a host gradient uploaded into a temporary one. Returns (handles,
    temporaries to free)."""
    var hs = List[Int]()
    var tmp = List[Int]()
    for i in range(len(keep)):
        var j = keep[i]
        if a.gk[j] == 0:
            hs.append(a.gs[j])
        else:
            var h = seq_tensor_alloc(a.sizes[j], False)
            tmp.append(h)
            _ = seq_tensor_upload(h, a.gs[j], a.sizes[j])
            hs.append(h)
    return (hs^, tmp^)


def _free(tmp: List[Int]) raises:
    for i in range(len(tmp)):
        _ = seq_tensor_free(tmp[i])


def seq_optimizer_step_device(words: Int, handles: Int, addrs: List[Int], ip: List[Int],
                              fp: List[Float64]) raises -> Int:
    """One element-wise step (RMSprop, Adagrad, Lion, Adamax, NAdam).
    addrs = [p_0 .., g_0 .., state1, state2, state3, scalars]; ip = [J,
    dev, gk .., size .., kind, flags]; fp = [lr, f1, f2, eps,
    weight_decay, f7, initial_accumulator]. Returns n."""
    var w = words_ptr(words)
    var a = SeqStep(addrs, ip)
    if len(a.rest_i) != 2 or len(a.rest_a) != 4:
        raise Error("sequence optimizer: a step requires kind, flags, 3 state addresses and the scalars")
    var kind = a.rest_i[0]
    var flags = a.rest_i[1]
    if not (a.dev or word(w, W_HANDLE) >= 0 or _resident_env()):
        var ex = DeviceExec()
        return elementwise_step_host(ex, w, a, kind, flags, fp)
    var n = a.total()
    if n != word(w, W_N):
        raise Error("sequence optimizer: the params were resized under the optimizer")
    var hp = _handles_ptr(handles)
    var sc = a.rest_a[3]
    var t = begin_step(w, sc, 3)
    var fp6 = List[Float64]()
    for k in range(6):
        fp6.append(fp[k])
    if word(w, W_HANDLE) < 0:
        var ret = opt_resident_open([n, kind, flags], fp6)
        hp[0] = Int64(ret[0])
        set_word(w, W_USED, ret[1])
        _opened(w, len(ret) > 2 and ret[2] == 1, fp[6] == 0.0)
    var h = Int(hp[0])
    var slots: List[Int] = [a.rest_a[0], a.rest_a[1], a.rest_a[2]]
    _upload_slots(w, h, slots)
    # empty tensors hold nothing to move or update
    var keep = List[Int]()
    for j in range(a.J):
        if a.sizes[j] > 0:
            keep.append(j)
    var ip2: List[Int] = [len(keep), kind, flags, t, word(w, W_SC_T)]
    for i in range(len(keep)):
        ip2.append(a.sizes[keep[i]])
    var at = List[Int]()
    for i in range(len(keep)):
        at.append(a.ps[keep[i]])
    if a.dev:
        var gt = _grad_tensors(a, keep)
        for i in range(len(gt[0])):
            at.append(gt[0][i])
        at.append(sc)
        try:
            _ = opt_resident_step_dev(h, at, ip2, fp6)
        except e:
            _free(gt[1])
            raise e
        _free(gt[1])
    else:
        for i in range(len(keep)):
            at.append(a.gs[keep[i]])
        at.append(sc)
        _ = opt_resident_step(h, at, ip2, fp6)
    set_word(w, W_T, t)
    set_word(w, W_SC_T, t)
    return n


def seq_lamb_step_device(words: Int, handles: Int, addrs: List[Int], ip: List[Int],
                         fp: List[Float64]) raises -> Int:
    """One LAMB step. addrs = [p_0 .., g_0 .., exp_avg, exp_avg_sq,
    scalars]; ip = [J, dev, gk .., size ..]; fp = [lr, beta1, beta2, eps,
    weight_decay, max_grad_norm]. Returns n."""
    var w = words_ptr(words)
    var a = SeqStep(addrs, ip)
    if not (a.dev or word(w, W_HANDLE) >= 0 or _resident_env()):
        var ex = DeviceExec()
        return lamb_step_host(ex, w, a, fp)
    if len(a.rest_a) != 3:
        raise Error("LAMB: a step requires 2 state addresses and the scalars")
    var n = a.total()
    if n != word(w, W_N):
        raise Error("LAMB: the params were resized under the optimizer")
    var hp = _handles_ptr(handles)
    var sc = a.rest_a[2]
    var flags = word(w, W_FLAGS)
    var t = begin_step(w, sc, 2)
    if word(w, W_HANDLE) < 0:
        var oip: List[Int] = [a.J]
        var offs = lamb_offsets_of(a.sizes)
        for k in range(len(offs)):
            oip.append(offs[k])
        var ret = lamb_resident_open(oip)
        hp[0] = Int64(ret[0])
        set_word(w, W_USED, ret[1])
        _opened(w, len(ret) > 2 and ret[2] == 1, True)
    var h = Int(hp[0])
    var slots: List[Int] = [a.rest_a[0], a.rest_a[1]]
    _upload_slots(w, h, slots)
    var fp7 = List[Float64]()
    for k in range(6):
        fp7.append(fp[k])
    fp7.append(Float64(word(w, W_SC_T)))
    var ip2: List[Int] = [a.J, t, flags]
    var at = List[Int]()
    for j in range(a.J):
        at.append(a.ps[j])
    if a.dev:
        var keep = List[Int]()
        for j in range(a.J):
            keep.append(j)
        var gt = _grad_tensors(a, keep)
        for i in range(len(gt[0])):
            at.append(gt[0][i])
        at.append(sc)
        try:
            _ = lamb_resident_step_dev(h, at, ip2, fp7)
        except e:
            _free(gt[1])
            raise e
        _free(gt[1])
    else:
        for j in range(a.J):
            at.append(a.gs[j])
        at.append(sc)
        _ = lamb_resident_step(h, at, ip2, fp7)
    set_word(w, W_T, t)
    if (flags & 8) != 0:
        set_word(w, W_SC_T, t)
    return n


def seq_adafactor_step_device(words: Int, handles: Int, addrs: List[Int], ip: List[Int],
                              fp: List[Float64]) raises -> Int:
    """One Adafactor step, tensor by tensor. addrs = [p_0 .., g_0 .., s1_0
    .., s2_0 ..]; ip = [J, dev, gk .., size .., R_0 .., C_0 ..]; fp =
    `adafactor_step`'s six. With the state resident, each tensor has its
    own handle (`handles`, J words). Returns J."""
    var w = words_ptr(words)
    var a = SeqStep(addrs, ip)
    comptime if not AF_RESIDENT:
        if a.dev:
            raise Error("Adafactor: this binding has no device-parameter step; pass NumPy arrays")
        var ex = DeviceExec()
        return adafactor_step_host(ex, w, a, fp)
    else:
        if not (a.dev or word(w, W_HANDLE) >= 0 or _resident_env()):
            var ex = DeviceExec()
            return adafactor_step_host(ex, w, a, fp)
        if len(a.rest_a) != 2 * a.J or len(a.rest_i) != 2 * a.J:
            raise Error("Adafactor: a step requires two state addresses and (R, C) per tensor")
        var hp = _handles_ptr(handles)
        var t = word(w, W_T) + 1
        if word(w, W_HANDLE) < 0:
            var zeroed = True
            for j in range(a.J):
                var oip: List[Int] = [a.rest_i[j], a.rest_i[a.J + j]]
                var ret = adafactor_resident_open(oip)
                hp[j] = Int64(ret[0])
                zeroed = zeroed and len(ret) > 2 and ret[2] == 1
            _opened(w, zeroed, True)
        if word(w, W_OWNED) != 0:
            for j in range(a.J):
                _ = opt_resident_move(Int(hp[j]), 0, a.rest_a[j], True)
                if a.rest_i[a.J + j] > 0:
                    _ = opt_resident_move(Int(hp[j]), 1, a.rest_a[a.J + j], True)
            set_word(w, W_OWNED, 0)
        set_word(w, W_FRESH, 0)
        var f = List[Float64]()
        for k in range(6):
            f.append(fp[k])
        for j in range(a.J):
            var ip2: List[Int] = [a.rest_i[j], a.rest_i[a.J + j], t]
            if a.dev:
                var one: List[Int] = [j]
                var gt = _grad_tensors(a, one)
                var hs2: List[Int] = [a.ps[j], gt[0][0]]
                try:
                    _ = adafactor_resident_step_dev(Int(hp[j]), hs2, ip2, f)
                except e:
                    _free(gt[1])
                    raise e
                _free(gt[1])
            else:
                var at: List[Int] = [a.ps[j], a.gs[j]]
                _ = adafactor_resident_step(Int(hp[j]), at, ip2, f)
        set_word(w, W_T, t)
        return a.J


def seq_optimizer_state_device(words: Int, handles: Int, addrs: List[Int]) raises -> Int:
    """A state read: the resident slots into the host copies when the device
    holds the newest values (addrs = three slot addresses per handle, 0 for
    a slot not kept); the reader may then write into them, so the next step
    uploads them first. Returns the number of handles read."""
    var w = words_ptr(words)
    var nh = len(addrs) // 3
    if nh * 3 != len(addrs) or nh < 1:
        raise Error("sequence optimizer: a state read requires three slot addresses per handle")
    var moved = 0
    if word(w, W_HANDLE) >= 0 and word(w, W_FRESH) == 0:
        var hp = _handles_ptr(handles)
        for j in range(nh):
            for k in range(3):
                if addrs[3 * j + k] != 0:
                    _ = opt_resident_move(Int(hp[j]), k, addrs[3 * j + k], False)
            moved += 1
        set_word(w, W_FRESH, 1)
    set_word(w, W_OWNED, 1)
    set_word(w, W_PRISTINE, 0)
    return moved


def _refresh(w: IP64, handles: Int, c: StateCopy) raises:
    """The resident slots into the host copies when the device holds the
    newest values (one handle)."""
    if word(w, W_HANDLE) >= 0 and word(w, W_FRESH) == 0:
        var hp = _handles_ptr(handles)
        for k in range(c.k):
            _ = opt_resident_move(Int(hp[0]), k, c.slots[k], False)
        set_word(w, W_FRESH, 1)


def seq_optimizer_export_device(words: Int, handles: Int, addrs: List[Int], ip: List[Int]) raises -> Int:
    """`state_dict`'s copies (StateCopy's arguments), the resident slots
    downloaded first when they are the newest."""
    var w = words_ptr(words)
    var c = StateCopy(addrs, ip)
    _refresh(w, handles, c)
    return export_copy(w, c)


def seq_optimizer_import_device(words: Int, handles: Int, addrs: List[Int], ip: List[Int]) raises -> Int:
    """`load_state_dict`'s copies (StateCopy's arguments), the resident
    slots downloaded first so a slot the sources leave out keeps its newest
    value."""
    var w = words_ptr(words)
    var c = StateCopy(addrs, ip)
    _refresh(w, handles, c)
    return import_copy(w, c)


def seq_optimizer_close_device(words: Int, handles: Int, nh: Int) raises -> Int:
    """Free every open resident handle (best effort: a handle that fails to
    close is left open for the process). Returns the number closed."""
    var w = words_ptr(words)
    var hp = _handles_ptr(handles)
    var closed = 0
    for j in range(nh):
        var h = Int(hp[j])
        if h >= 0:
            try:
                _ = opt_resident_close(h)
                closed += 1
            except:
                pass
            hp[j] = Int64(-1)
    set_word(w, W_HANDLE, -1)
    return closed
