# SPDX-License-Identifier: Apache-2.0
# One checkpoint container for a conditional or bounded loop. Components share
# one parameter table: restoring a handle once preserves aliasing across all
# branches and the predicate. Validation completes before session mutation.
from std.python import Python, PythonObject
from sequence.exec_trait import Exec
from sequence.ops import FP
from autograd.ops_api import ops_session
from autograd.program import OpsProgram, _Reader, _put_shape, program_words, program_from_words, same_shape
from autograd.program_control import _signature
from autograd.program_pyapi import _integer
from autograd.control_training_checkpoint import ControlTraining, save_control_training, read_control_training, restore_control_training
from autograd.pyapi import program_info
from autograd.rankn import ShapeN
from core.cast_in import CastIn, CAST_KIND_FLOAT, CAST_KIND_SIGNED
from bindings.native_json import native_path
from training.checkpoint import Checkpoint, CKPT_SECTION_CONTROL_PROGRAM, read_checkpoint_unchecked, save_checkpoint


def control_save_py[E: Exec & Movable & Deinitable, KIND: Int](
    sid: PythonObject, body: PythonObject, condition: PythonObject, otherwise: PythonObject,
    maximum: PythonObject, path: PythonObject, optimizer: Int = 0, seed: UInt64 = 0,
) raises -> PythonObject:
    if Int(py=body._s._id) != Int(py=sid) or Int(py=condition._s._id) != Int(py=sid):
        raise Error("ValueError: control-flow programs must belong to the same session")
    var loop = otherwise is PythonObject(None)
    var limit = 0
    if loop:
        limit = _integer(maximum, "max_iterations", 0)
    elif maximum is not PythonObject(None):
        raise Error("ValueError: a conditional checkpoint does not take max_iterations")
    elif Int(py=otherwise._s._id) != Int(py=sid):
        raise Error("ValueError: control-flow programs must belong to the same session")
    var s = ops_session[E, KIND](Int(py=sid))
    var programs = List[OpsProgram]()
    programs.append(s[].program(Int(py=body._id)))
    programs.append(s[].program(Int(py=condition._id)))
    if not loop:
        programs.append(s[].program(Int(py=otherwise._id)))
    for j in range(1, len(programs)):  # driver-loop: over control components
        _signature(programs[0], programs[j])
    if loop and len(programs[0].outs) != programs[0].n_in:
        raise Error("ValueError: while_loop body must return one tensor per input")
    if len(programs[1].outs) != 1:
        raise Error("ValueError: control-flow condition must return one scalar")
    var handles = List[Int]()
    var shapes = List[ShapeN]()
    var kinds = List[Int]()
    var index = Dict[Int, Int]()
    for j in range(len(programs)):  # driver-loop: over component parameters
        for k in range(len(programs[j].params)):  # driver-loop: over parameter references
            var h = programs[j].params[k]
            var slot = s[]._slot(h)
            if s[].owned[slot] == 0 or not same_shape(s[].shapes[slot], programs[j].param_shapes[k]):
                raise Error("ValueError: control checkpoint parameter is not an unchanged uploaded tensor")
            if h not in index:
                index[h] = len(handles)
                handles.append(h)
                shapes.append(s[].shapes[slot])
                kinds.append(Int(s[].int_words[slot]))
    var ck = Checkpoint()
    ck.section_kind = CKPT_SECTION_CONTROL_PROGRAM
    ck.offsets.append(0)
    var n = 0
    ck.ext_ints.append(Int64(2) if optimizer != 0 else Int64(1))
    ck.ext_ints.append(Int64(Int(loop)))
    ck.ext_ints.append(Int64(limit))
    ck.ext_ints.append(Int64(len(handles)))
    for k in range(len(handles)):  # driver-loop: over unique parameter storage
        n += shapes[k].count()
        ck.offsets.append(n)
        ck.names.append("p" + String(k))
        ck.buf_initialized.append(False)
        _put_shape(ck.ext_ints, shapes[k])
        ck.ext_ints.append(Int64(kinds[k]))
    if len(handles) == 0:
        ck.offsets.append(1)
        ck.names.append("unused")
        ck.buf_initialized.append(False)
    var stored = max(n, 1)
    ck.param = List[Float32](length=stored, fill=Float32(0))
    ck.m_state = List[Float32](length=stored, fill=Float32(0))
    ck.v_state = List[Float32](length=stored, fill=Float32(0))
    for k in range(len(handles)):  # driver-loop: over unique parameter downloads
        s[].download(handles[k], FP(unsafe_from_address=Int(ck.param.unsafe_ptr())) + ck.offsets[k])
    for j in range(len(programs)):  # driver-loop: over serialized control components
        var words = program_words(programs[j])
        ck.ext_ints.append(Int64(len(words[0])))
        ck.ext_ints.append(Int64(len(words[1])))
        ck.ext_ints.append(Int64(len(programs[j].params)))
        for k in range(len(programs[j].params)):  # driver-loop: over shared parameter references
            ck.ext_ints.append(Int64(index[programs[j].params[k]]))
        for k in range(len(words[0])):  # driver-loop: over serialized graph metadata
            ck.ext_ints.append(words[0][k])
        for k in range(len(words[1])):  # driver-loop: over serialized scalar options
            ck.ext_floats.append(words[1][k])
    if optimizer != 0:
        save_control_training(s[], handles, shapes, ck, optimizer, seed)
    var normalized = native_path(path)
    var written = save_checkpoint(String(py=normalized), ck)
    return PythonObject(written.bytes)


def control_load_py[E: Exec & Movable & Deinitable, KIND: Int](
    sid: PythonObject, path: PythonObject,
) raises -> PythonObject:
    var normalized = native_path(path)
    var ck = read_checkpoint_unchecked(String(py=normalized))
    var what = String("ValueError: malformed control checkpoint")
    if ck.section_kind != CKPT_SECTION_CONTROL_PROGRAM:
        raise Error("ValueError: not a control-flow checkpoint")
    var r = _Reader(len(ck.ext_ints))
    var version = r.take(ck.ext_ints, what)
    if version != 1 and version != 2:
        raise Error(what + ": unknown version")
    var kind = r.take(ck.ext_ints, what)
    var limit = r.take(ck.ext_ints, what)
    var count = r.take(ck.ext_ints, what)
    if (kind != 0 and kind != 1) or limit < 0 or (kind == 0 and limit != 0) or count < 0:
        raise Error(what + ": invalid kind, iteration bound or parameter count")
    if ck.j_count() != max(count, 1):
        raise Error(what + ": wrong parameter table length")
    var shapes = List[ShapeN]()
    var kinds = List[Int]()
    for k in range(count):  # driver-loop: over parameter metadata
        var sh = r.shape(ck.ext_ints, what)
        var dtype = r.take(ck.ext_ints, what)
        if (dtype != 0 and dtype != 1) or ck.names[k] != "p" + String(k) or ck.offsets[k + 1] - ck.offsets[k] != sh.count():
            raise Error(what + ": invalid parameter shape, kind or name")
        shapes.append(sh)
        kinds.append(dtype)
    if count == 0 and (ck.names[0] != "unused" or ck.n_total() != 1):
        raise Error(what + ": invalid parameter-free storage")
    var components = 2 if kind == 1 else 3
    var programs = List[OpsProgram]()
    var float_at = 0
    var used = List[Bool](length=count, fill=False)
    for j in range(components):  # driver-loop: over control component records
        var ni = r.take(ck.ext_ints, what)
        var nf = r.take(ck.ext_ints, what)
        var nk = r.take(ck.ext_ints, what)
        if ni < 1 or nf < 0 or nk < 0 or nk > count or nk > len(ck.ext_ints) - r.at:
            raise Error(what + ": invalid component lengths")
        var refs = List[Int]()
        for k in range(nk):  # driver-loop: over shared parameter indices
            var index = r.take(ck.ext_ints, what)
            if index < 0 or index >= count:
                raise Error(what + ": shared parameter reference out of range")
            refs.append(index)
            used[index] = True
        if ni > len(ck.ext_ints) - r.at or nf > len(ck.ext_floats) - float_at:
            raise Error(what + ": truncated component")
        var xi = List[Int64]()
        var xf = List[Float32]()
        for k in range(ni):  # driver-loop: over serialized graph integer metadata
            xi.append(Int64(r.take(ck.ext_ints, what)))
        for k in range(nf):  # driver-loop: over serialized graph scalar metadata
            xf.append(ck.ext_floats[float_at + k])
        float_at += nf
        var p = program_from_words(xi, xf, what)
        if len(p.params) != nk:
            raise Error(what + ": parameter reference count does not match graph")
        for k in range(nk):  # driver-loop: over validated parameter references
            var index = refs[k]
            if not same_shape(p.param_shapes[k], shapes[index]):
                raise Error(what + ": shared parameter shapes disagree")
            p.params[k] = index
        programs.append(p^)
    var training = ControlTraining()
    if version == 2:
        training = read_control_training(ck, kinds, r, float_at)
    if r.at != len(ck.ext_ints) or float_at != len(ck.ext_floats):
        raise Error(what + ": trailing component metadata")
    for k in range(count):  # driver-loop: over referenced parameter table entries
        if not used[k]:
            raise Error(what + ": unused parameter table entry")
    for j in range(1, components):  # driver-loop: over validated control signatures
        _signature(programs[0], programs[j])
    if len(programs[1].outs) != 1 or (kind == 1 and len(programs[0].outs) != programs[0].n_in):
        raise Error(what + ": invalid predicate or loop output count")
    var s = ops_session[E, KIND](Int(py=sid))
    if s[].recording():
        raise Error("ValueError: control programs load into an inference session")
    var handles = List[Int]()
    var ids = List[Int]()
    var optimizer = 0
    var seed = ck.seed
    try:
        for k in range(count):  # driver-loop: over unique restored parameter buffers
            var sh = shapes[k]
            var source = CastIn(Int(ck.param.unsafe_ptr()) + 4 * ck.offsets[k],
                               CAST_KIND_SIGNED if kinds[k] == 1 else CAST_KIND_FLOAT,
                               4, sh.rows(), sh.cols(), sh.cols() * 4, 4)
            var h = s[].upload_ids(source, sh) if kinds[k] == 1 else s[].upload_nd(source, sh, False)
            handles.append(h)
        for j in range(components):  # driver-loop: over restored graph bindings
            for k in range(len(programs[j].params)):  # driver-loop: over shared handle rebinding
                programs[j].params[k] = handles[programs[j].params[k]]
            ids.append(s[].add_program(programs[j].copy()))
        if training.present:
            optimizer = restore_control_training(s[], handles, shapes, ck, training)
        _ = ck^  # parameter and optimizer uploads have now consumed the payload
    except error:
        for j in range(len(ids)):  # driver-loop: over partial restored graphs
            s[].program_free(ids[j])
        for k in range(len(handles)):  # driver-loop: over partial parameter uploads
            s[].free(handles[k])
        raise error
    var out = Python.import_module("builtins").dict()
    out["kind"] = "while" if kind == 1 else "if"
    out["max_iterations"] = PythonObject(limit) if kind == 1 else PythonObject(None)
    out["body"] = program_info(ids[0], s[].program(ids[0]))
    out["condition"] = program_info(ids[1], s[].program(ids[1]))
    out["otherwise"] = program_info(ids[2], s[].program(ids[2])) if kind == 0 else PythonObject(None)
    out["optimizer"] = PythonObject(optimizer) if training.present else PythonObject(None)
    out["seed"] = PythonObject(Int(seed >> 32)).__lshift__(32).__or__(Int(seed & 0xffffffff))
    return out
