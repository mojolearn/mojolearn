# SPDX-License-Identifier: Apache-2.0
# Python doors for model and session checkpoints (model_checkpoint.mojo):
# argument unpacking and Python results around the native checkpoint records.
from std.python import Python, PythonObject
from std.memory import bitcast, memcpy
from sequence.exec_trait import Exec
from sequence.ops import FP
from sequence.pyapi import fptr, fval, ival
from autograd.model import (
    FIT_STATE_WORDS, FS_SEED_HI, FS_SEED_LO, FS_STEP, IP64, LAYER_LAYER_NORM, LAYER_LINEAR, ModelSpec,
    parse_model,
)
from autograd.pyapi import _floats, _ints, cast_in_of, program_info
from autograd.program import program_from_words, program_words
from autograd.ops_api import OpsSession, ops_session
from core.cast_in import CAST_KIND_FLOAT, CastIn
from training.checkpoint import (
    CKPT_REFUSE_SHAPE, CKPT_SECTION_PROGRAM, CKPT_SECTION_SEQUENTIAL, CKPT_SECTION_SESSION, Checkpoint,
    read_checkpoint_unchecked,
    save_checkpoint,
)
from autograd.model_checkpoint import (
    BP, SEQ_FIT_FLOATS, SEQ_MODEL_INTS, SEQ_OPT_INTS, SEQ_RECORD_VERSION, SEQ_SAVE_INTS,
    SESSION_PER_PARAM_FLOATS, SESSION_PER_PARAM_INTS, SESSION_RECORD_VERSION, _floats_of, _put_floats,
    _read_sequential, _seq_record, _session_record, _upload_slice, model_layout,
)


def _addr(addrs: PythonObject, i: Int) raises -> Int:
    return Int(py=addrs[i])


def model_save_checkpoint_py(
    path: PythonObject, addrs: PythonObject, prog: PythonObject, ip: PythonObject, fp: PythonObject,
    classes: PythonObject,
) raises -> PythonObject:
    """Write a Sequential checkpoint to `path`.

    addrs = [params (n_params float32), m, v (n_params float32 each),
    state (FIT_STATE_WORDS int64), loss curve (epochs run float32)]; with
    ip[0] = 1 (an inference export) only params is read and the others may
    be 0. prog = the description; ip = [export, loss, target, optimizer,
    optimizer flags, schedule, schedule step, n_iter_no_change, batch size,
    shuffle, n_features, n_outputs, y_1d, epochs run, class count]; fp =
    model_fit's fp (9 options, then one float per layer); classes = the
    classes_ input descriptor (autograd/pyapi.mojo cast_in_of), or None
    when the class count is 0. Returns the file's byte count."""
    if len(addrs) != 5 or len(ip) != SEQ_SAVE_INTS or len(fp) < SEQ_FIT_FLOATS + 1:
        raise Error("model_save_checkpoint: requires 5 addresses, " + String(SEQ_SAVE_INTS)
                    + " integers and 9 floats plus one per layer")
    var file = String(py=path)
    var words = _ints(prog)
    var table = _floats(fp, SEQ_FIT_FLOATS)
    var spec = parse_model(words, table)
    var is_export = ival(ip, 0) != 0
    var n_run = ival(ip, 1 + SEQ_OPT_INTS + SEQ_MODEL_INTS)
    var k = ival(ip, 2 + SEQ_OPT_INTS + SEQ_MODEL_INTS)
    var np_ = spec.n_params
    var ck = Checkpoint()
    ck.section_kind = CKPT_SECTION_SEQUENTIAL
    model_layout(spec, ck.offsets, ck.names)
    for _ in range(len(ck.names)):  # driver-loop: over parameter tensors
        ck.buf_initialized.append(False)
    ck.param = _floats_of(fptr(addrs[0], "params"), np_)
    var state = List[Int64](length=FIT_STATE_WORDS, fill=Int64(0))
    if is_export:
        ck.m_state = List[Float32](length=np_, fill=Float32(0))
        ck.v_state = List[Float32](length=np_, fill=Float32(0))
        n_run = 0
    else:
        ck.m_state = _floats_of(fptr(addrs[1], "m"), np_)
        ck.v_state = _floats_of(fptr(addrs[2], "v"), np_)
        var sa = _addr(addrs, 3)
        if sa == 0:
            raise Error("model_save_checkpoint: null training-state buffer")
        memcpy(dest=state.unsafe_ptr(), src=IP64(unsafe_from_address=sa), count=FIT_STATE_WORDS)
    var opt = ival(ip, 3)
    var flags = ival(ip, 4)
    ck.t = Int(state[FS_STEP])
    ck.seed = (UInt64(state[FS_SEED_HI]) << 32) | UInt64(state[FS_SEED_LO])
    ck.opt_kind = opt
    ck.lr = fval(fp, 0)
    if opt == 0:
        ck.momentum = fval(fp, 1)
    else:
        ck.beta1 = fval(fp, 1)
        ck.beta2 = fval(fp, 2)
    ck.eps = fval(fp, 3)
    ck.weight_decay = fval(fp, 4)
    ck.dampening = fval(fp, 5)
    ck.nesterov = (flags & 1) != 0
    # the extension record
    var xi = List[Int64]()
    xi.append(Int64(SEQ_RECORD_VERSION))
    xi.append(Int64(1) if is_export else Int64(0))
    xi.append(Int64(len(words)))
    for i in range(len(words)):  # driver-loop: over description words
        xi.append(Int64(words[i]))
    for i in range(SEQ_OPT_INTS + SEQ_MODEL_INTS):  # driver-loop: over option words
        xi.append(Int64(ival(ip, 1 + i)))
    xi.append(bitcast[DType.int64](Float64(py=fp[6])))
    for i in range(FIT_STATE_WORDS):  # driver-loop: over state words
        xi.append(state[i])
    xi.append(Int64(n_run))
    xi.append(Int64(k))
    if k > 0:
        var c = cast_in_of(classes, "classes")
        if c.count() != k or c.itemsize > 8 or not c.is_c_order():
            raise Error("model_save_checkpoint: classes must be " + String(k) + " contiguous values")
        xi.append(Int64(c.kind))
        xi.append(Int64(c.itemsize))
        var bp = BP(unsafe_from_address=c.addr)
        for j in range(k):  # driver-loop: over classes
            var w = UInt64(0)
            for b in range(c.itemsize):  # driver-loop: over the value's bytes
                w = w | (UInt64(bp[j * c.itemsize + b]) << UInt64(8 * b))
            xi.append(bitcast[DType.int64](w))
    else:
        xi.append(Int64(0))
        xi.append(Int64(0))
    var xf = List[Float32]()
    for i in range(6):  # driver-loop: over option floats (lr .. f7)
        xf.append(fval(fp, i))
    xf.append(fval(fp, 7))
    xf.append(fval(fp, 8))
    for i in range(len(table)):  # driver-loop: over layers
        xf.append(table[i])
    if n_run > 0:
        var curve = _floats_of(fptr(addrs[4], "loss_curve"), n_run)
        for e in range(n_run):  # driver-loop: over epochs
            xf.append(curve[e])
    ck.ext_ints = xi^
    ck.ext_floats = xf^
    var h = save_checkpoint(file, ck)
    return PythonObject(h.bytes)


def model_checkpoint_info_py(path: PythonObject) raises -> PythonObject:
    """The description and options of a Sequential checkpoint, for Python to
    rebuild the model object: [export (0/1), layers, ints, floats], layers =
    one [code, arg0, arg1, float] per layer, ints = [the 9 option words,
    n_features, n_outputs, y_1d, epochs run, class count, class kind, class
    item size, n_params], floats = [lr, f1, f2, eps, weight_decay, f7, tol,
    schedule gamma, huber delta] (model_fit's fp head)."""
    var file = String(py=path)
    var ck = _read_sequential(file)
    var r = _seq_record(ck, file)
    var spec = parse_model(r.prog, r.table)
    var out = Python.list()
    out.append(PythonObject(1 if r.is_export else 0))
    var layers = Python.list()
    for i in range(spec.n_layers()):  # driver-loop: over layers
        var rec = Python.list()
        rec.append(PythonObject(spec.kinds[i]))
        rec.append(PythonObject(spec.arg0[i]))
        rec.append(PythonObject(spec.arg1[i]))
        rec.append(PythonObject(Float64(spec.farg[i])))
        layers.append(rec)
    out.append(layers)
    var ints = Python.list()
    for i in range(len(r.opts)):  # driver-loop: over option words
        ints.append(PythonObject(r.opts[i]))
    ints.append(PythonObject(r.n_run))
    ints.append(PythonObject(r.k))
    ints.append(PythonObject(r.class_kind))
    ints.append(PythonObject(r.class_size))
    ints.append(PythonObject(spec.n_params))
    out.append(ints)
    var floats = Python.list()
    for i in range(6):  # driver-loop: over option floats (lr .. f7)
        floats.append(PythonObject(Float64(r.floats[i])))
    floats.append(PythonObject(bitcast[DType.float64](r.tol_bits)))
    floats.append(PythonObject(Float64(r.floats[6])))
    floats.append(PythonObject(Float64(r.floats[7])))
    out.append(floats)
    return out


def model_load_checkpoint_py(path: PythonObject, addrs: PythonObject) raises -> PythonObject:
    """Fill the caller's buffers from a Sequential checkpoint: addrs =
    [params (n_params float32), m, v (n_params float32 each), state
    (FIT_STATE_WORDS int64), loss curve (epochs run float32), classes
    (class count values of the class kind and size)]; every address but
    params may be 0 and is then skipped. Returns n_params."""
    if len(addrs) != 6:
        raise Error("model_load_checkpoint: requires 6 addresses")
    var file = String(py=path)
    var ck = _read_sequential(file)
    var r = _seq_record(ck, file)
    _put_floats(fptr(addrs[0], "params"), ck.param)
    if _addr(addrs, 1) != 0:
        _put_floats(fptr(addrs[1], "m"), ck.m_state)
    if _addr(addrs, 2) != 0:
        _put_floats(fptr(addrs[2], "v"), ck.v_state)
    if _addr(addrs, 3) != 0:
        memcpy(dest=IP64(unsafe_from_address=_addr(addrs, 3)), src=r.state.unsafe_ptr(), count=FIT_STATE_WORDS)
    if _addr(addrs, 4) != 0 and r.n_run > 0:
        var curve = fptr(addrs[4], "loss_curve")
        memcpy(dest=curve, src=ck.ext_floats.unsafe_ptr() + r.curve_at, count=r.n_run)
    if _addr(addrs, 5) != 0 and r.k > 0:
        var bp = BP(unsafe_from_address=_addr(addrs, 5))
        for j in range(r.k):  # driver-loop: over classes
            var w = bitcast[DType.uint64](ck.ext_ints[r.class_at + j])
            for b in range(r.class_size):  # driver-loop: over the value's bytes
                bp[j * r.class_size + b] = UInt8((w >> UInt64(8 * b)) & UInt64(0xFF))
    return PythonObject(ck.n_total())


def _handles(p: PythonObject) raises -> List[Int]:
    var hs = List[Int]()
    for i in range(len(p)):  # driver-loop: over parameter handles
        hs.append(Int(py=p[i]))
    return hs^


def ops_save_state_py[E: Exec & Movable & Deinitable, KIND: Int](
    sid: PythonObject, path: PythonObject, params: PythonObject, optimizer: PythonObject,
    counters: PythonObject,
) raises -> PythonObject:
    """Write the session tensors `params` (handles), the state of
    `optimizer` (its handle, or 0 for none) for each of them, read through
    OpsSession.optimizer_state, and counters = [seed low, seed high, epoch,
    step] to `path`. Returns the byte count."""
    if len(counters) != 4:
        raise Error("ops_save_state: counters are [seed low, seed high, epoch, step]")
    var file = String(py=path)
    var hs = _handles(params)
    var K = len(hs)
    if K < 1:
        raise Error("ops_save_state: no parameters")
    var opt = Int(py=optimizer)
    var p = ops_session[E, KIND](Int(py=sid))
    var ck = Checkpoint()
    ck.section_kind = CKPT_SECTION_SESSION
    var xi = List[Int64]()
    xi.append(Int64(SESSION_RECORD_VERSION))
    xi.append(Int64(K))
    var n = 0
    ck.offsets.append(0)
    for i in range(K):  # driver-loop: over parameter handles
        var shape = p[].shape(hs[i])
        xi.append(Int64(shape[0]))
        xi.append(Int64(shape[1]))
        n += shape[0] * shape[1]
        ck.offsets.append(n)
        ck.names.append(String("p") + String(i))
        ck.buf_initialized.append(False)
    ck.param = List[Float32](length=n, fill=Float32(0))
    ck.m_state = List[Float32](length=n, fill=Float32(0))
    ck.v_state = List[Float32](length=n, fill=Float32(0))
    xi.append(Int64(1) if opt != 0 else Int64(0))
    var xf = List[Float32]()
    var first_step = 0
    for i in range(K):  # driver-loop: over parameter handles
        var off = ck.offsets[i]
        p[].download(hs[i], FP(unsafe_from_address=Int(ck.param.unsafe_ptr())) + off)
        if opt == 0:
            xi.append(Int64(0))
            xi.append(Int64(0))
            xi.append(Int64(0))
            xf.append(Float32(1))
            xf.append(Float32(1))
            xf.append(Float32(1))
            continue
        # (step, beta1^t, beta2^t, mu product, buffer copies): the words
        # load_optimizer_state takes back
        var w = p[].optimizer_state(opt, hs[i])
        if w[6] != 0:
            raise Error("ops_save_state: this optimizer keeps a third state buffer, which the "
                        + "session checkpoint does not hold")
        if w[4] != 0:
            p[].download(w[4], FP(unsafe_from_address=Int(ck.m_state.unsafe_ptr())) + off)
            p[].free(w[4])
        if w[5] != 0:
            p[].download(w[5], FP(unsafe_from_address=Int(ck.v_state.unsafe_ptr())) + off)
            p[].free(w[5])
        if i == 0:
            first_step = w[0]
        xi.append(Int64(w[0]))
        xi.append(Int64(1) if w[4] != 0 else Int64(0))
        xi.append(Int64(1) if w[5] != 0 else Int64(0))
        xf.append(w[1])
        xf.append(w[2])
        xf.append(w[3])
    var lo = ival(counters, 0)
    var hi = ival(counters, 1)
    xi.append(Int64(lo))
    xi.append(Int64(hi))
    xi.append(Int64(ival(counters, 2)))
    xi.append(Int64(ival(counters, 3)))
    ck.seed = (UInt64(hi) << 32) | UInt64(lo)
    ck.t = first_step
    ck.ext_ints = xi^
    ck.ext_floats = xf^
    var h = save_checkpoint(file, ck)
    return PythonObject(h.bytes)


def ops_load_state_py[E: Exec & Movable & Deinitable, KIND: Int](
    sid: PythonObject, path: PythonObject, requires_grad: PythonObject, optimizer: PythonObject,
) raises -> PythonObject:
    """Upload the parameters of a session checkpoint as new tensors and,
    with `optimizer` (a handle, or 0), restore each one's saved optimizer
    state through OpsSession.load_optimizer_state (the buffers are copied
    from temporary tensors, freed after). Returns [handles, has optimizer
    state, seed low, seed high, epoch, step]."""
    var file = String(py=path)
    var ck = read_checkpoint_unchecked(file)
    var xi = _session_record(ck, file)
    var K = Int(xi[1])
    var has_opt = xi[2 + 2 * K] != 0
    var opt = Int(py=optimizer)
    if opt != 0 and not has_opt:
        raise Error("mojolearn: '" + file + "' holds no optimizer state")
    var p = ops_session[E, KIND](Int(py=sid))
    var grad = Int(py=requires_grad) != 0
    var hs = Python.list()
    var at = 2 + 2 * K + 1
    for i in range(K):  # driver-loop: over parameter tensors
        var rows = Int(xi[2 + 2 * i])
        var cols = Int(xi[3 + 2 * i])
        var off = ck.offsets[i]
        var h = _upload_slice(p[], ck.param, off, rows, cols, grad)
        hs.append(PythonObject(h))
        if opt == 0:
            continue
        var base = at + SESSION_PER_PARAM_INTS * i
        var hm = 0
        var hv = 0
        if xi[base + 1] != 0:
            hm = _upload_slice(p[], ck.m_state, off, rows, cols, False)
        if xi[base + 2] != 0:
            hv = _upload_slice(p[], ck.v_state, off, rows, cols, False)
        var ints = List[Int]()
        ints.append(Int(xi[base]))
        ints.append(hm)
        ints.append(hv)
        ints.append(0)
        var fs = List[Float32]()
        for k in range(SESSION_PER_PARAM_FLOATS):  # driver-loop: over the three bias-correction words
            fs.append(ck.ext_floats[SESSION_PER_PARAM_FLOATS * i + k])
        p[].load_optimizer_state(opt, h, ints, fs)
        if hm != 0:
            p[].free(hm)
        if hv != 0:
            p[].free(hv)
    var c = at + SESSION_PER_PARAM_INTS * K
    var res = Python.list()
    res.append(hs)
    res.append(PythonObject(1 if has_opt else 0))
    for i in range(4):  # driver-loop: over the four counters
        res.append(PythonObject(Int(xi[c + i])))
    _ = ck^
    return res


def ops_save_program_py[E: Exec & Movable & Deinitable, KIND: Int](
    sid: PythonObject, pid: PythonObject, path: PythonObject,
) raises -> PythonObject:
    """Write program `pid` of session `sid` and its parameters' current
    values to `path`. Returns the byte count."""
    var file = String(py=path)
    var p = ops_session[E, KIND](Int(py=sid))
    var prog = p[].program(Int(py=pid))
    var K = len(prog.params)
    var ck = Checkpoint()
    ck.section_kind = CKPT_SECTION_PROGRAM
    var n = 0
    ck.offsets.append(0)
    for i in range(K):  # driver-loop: over program parameters
        n += prog.param_shapes[i].count()
        ck.offsets.append(n)
        ck.names.append(String("p") + String(i))
        ck.buf_initialized.append(False)
    # a program without parameters still writes one (unused) word
    var n_store = n if n > 0 else 1
    ck.param = List[Float32](length=n_store, fill=Float32(0))
    ck.m_state = List[Float32](length=n_store, fill=Float32(0))
    ck.v_state = List[Float32](length=n_store, fill=Float32(0))
    if K == 0:
        ck.offsets.append(1)
        ck.names.append(String("unused"))
        ck.buf_initialized.append(False)
    for i in range(K):  # driver-loop: over program parameters
        p[].download(prog.params[i], FP(unsafe_from_address=Int(ck.param.unsafe_ptr())) + ck.offsets[i])
    var words = program_words(prog)
    ck.ext_ints = words[0].copy()
    ck.ext_floats = words[1].copy()
    ck.seed = 0
    ck.t = 0
    var h = save_checkpoint(file, ck)
    return PythonObject(h.bytes)


def ops_load_program_py[E: Exec & Movable & Deinitable, KIND: Int](
    sid: PythonObject, path: PythonObject,
) raises -> PythonObject:
    """Read a program checkpoint into session `sid`: its parameters become
    new session tensors and the program runs over them. Returns
    program_info (autograd/pyapi.mojo)."""
    var file = String(py=path)
    var ck = read_checkpoint_unchecked(file)
    if ck.section_kind != CKPT_SECTION_PROGRAM:
        raise Error("mojolearn: '" + file + "' is not a captured-program checkpoint")
    var prog = program_from_words(ck.ext_ints, ck.ext_floats, "mojolearn: '" + file + "'")
    var K = len(prog.params)
    if (K > 0 and ck.j_count() != K) or (K == 0 and ck.j_count() != 1):
        raise Error("mojolearn: '" + file + "' holds a damaged program record")
    var p = ops_session[E, KIND](Int(py=sid))
    if p[].recording():
        raise Error("mojolearn: programs run in an inference session (Session(record=False))")
    for i in range(K):  # driver-loop: over program parameters
        var sh = prog.param_shapes[i]
        var off = ck.offsets[i]
        if ck.offsets[i + 1] - off != sh.count():
            raise Error(String(CKPT_REFUSE_SHAPE) + ": '" + file + "' tensor p" + String(i)
                        + " does not hold its recorded shape")
        var c = CastIn(Int(ck.param.unsafe_ptr()) + 4 * off, CAST_KIND_FLOAT, 4, sh.rows(), sh.cols(), 4 * sh.cols(), 4)
        prog.params[i] = p[].upload_nd(c, sh, False)
    var pid = p[].add_program(prog^)
    var res = program_info(pid, p[].program(pid))
    _ = ck^
    return res
