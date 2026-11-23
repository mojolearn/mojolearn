# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""Checkpoint tensor names of any length (lane nn-fix-4, 2026-11-23).

training/checkpoint.mojo format version 1 held a tensor name in a 16-byte NUL-padded field, so
`l1.in_proj_weight` (17 bytes) was refused at save and a Sequential model with multihead_attention could not
be saved. Version 2 length-prefixes the name. Each case:
  - a Sequential model with multihead_attention (its in_proj_weight / out_proj.weight tensors) is saved
    through autograd/model_checkpoint.mojo model_save_checkpoint and loaded back
    (training_sequential_common.mojo SeqModel.save / load): the parameter words come back bit for bit;
  - a raw checkpoint whose first tensor has a 40-character name is saved and loaded through
    training/checkpoint.mojo save_checkpoint / load_checkpoint: the names, offsets and payload round trip,
    a re-save is byte-identical, and the file is the formula's length;
  - a file whose format_version word says 1 is refused with CKPT_REFUSE_VERSION.
Host column only: the format is host bytes (no device in training/checkpoint.mojo).
"""
from std.ffi import external_call
from std.memory import bitcast
from std.os import getenv, remove
from std.testing import TestSuite
from autograd.nn_attention import LAYER_NN_MHA
from autograd.model import FIT_STATE_WORDS, LOSS_MSE
from autograd.model_checkpoint import model_checkpoint_info
from training.checkpoint import (
    CKPT_REFUSE_VERSION, Checkpoint, checkpoint_file_bytes, checkpoint_layout_bytes, compare_checkpoint_files,
    load_checkpoint, save_checkpoint,
)
from checks.native_tests.training_sequential_common import SeqModel, L_LINEAR, L_FLATTEN

comptime _SELF = -2


def _fail(message: String) raises:
    print("FAIL CHECKPOINT_NAMES", message)
    raise Error(message)


def _path(name: String) -> String:
    var base = getenv("TMPDIR", "/tmp")
    return base + "/mojolearn-check-ckpt-names-" + String(external_call["getpid", Int32]()) + "-" + name + ".ckpt"


def _drop(path: String):
    try:
        remove(path)
    except:
        pass


def _mha_model() raises -> SeqModel:
    # (4, 6) -> linear(8) -> multihead_attention(2 heads, bias) -> flatten -> linear(2)
    var m = SeqModel()
    m.sample = [4, 6]
    m.layer(L_LINEAR, 8, 1)
    m.layer(LAYER_NN_MHA, 2, 1, _SELF, 8, 0, 0, f=0.0)
    m.layer(L_FLATTEN)
    m.layer(L_LINEAR, 2, 1)
    m.compile(LOSS_MSE, "adam", 0.01)
    return m^


def test_checkpoint_saves_multihead_attention() raises:
    var m = _mha_model()
    var spec = m.spec(24)
    var n = spec.n_params
    m.params = List[Float32](length=n, fill=Float32(0))
    for i in range(n):  # small-loop(n: parameter words)
        m.params[i] = Float32((i * 7) % 61 - 30) / Float32(64)
    m.m = List[Float32](length=n, fill=Float32(0.125))
    m.v = List[Float32](length=n, fill=Float32(0.25))
    m.state = List[Int64](length=FIT_STATE_WORDS, fill=Int64(0))
    m.curve = List[Float32](length=1, fill=Float32(0))
    var path = _path("mha")
    try:
        _ = m.save(path, 24, False)
        var info = model_checkpoint_info(path)
        if len(info.layer_ints) != 3 * 4:
            _fail("the multihead_attention checkpoint describes " + String(len(info.layer_ints) // 3) + " layers")
        var back = _mha_model()
        back.load(path, 24)
        if len(back.params) != n:
            _fail("the loaded model has " + String(len(back.params)) + " parameters, not " + String(n))
        for i in range(n):  # small-loop(n: parameter words)
            if bitcast[DType.uint32](back.params[i]) != bitcast[DType.uint32](m.params[i]):
                _fail("parameter " + String(i) + " did not round trip")
    except e:
        _drop(path)
        raise e^
    _drop(path)
    print("PASS CHECKPOINT_NAMES multihead_attention saved and loaded,", n, "parameters")


def _raw(names: List[String], counts: List[Int]) -> Checkpoint:
    var ck = Checkpoint()
    ck.t = 3
    ck.seed = UInt64(11)
    ck.lr = Float32(0.001)
    ck.offsets.append(0)
    var total = 0
    for k in range(len(counts)):  # small-loop(the tensors)
        total += counts[k]
        ck.offsets.append(total)
        ck.names.append(names[k].copy())
        ck.buf_initialized.append(k % 2 == 0)
    for i in range(total):  # small-loop(total: payload words)
        ck.param.append(Float32(i) + Float32(0.5))
        ck.m_state.append(Float32(-i))
        ck.v_state.append(Float32(2 * i + 1))
    return ck^


def test_checkpoint_forty_character_name() raises:
    var long = String("layer.with.a.forty_character_long_name-X")
    if long.byte_length() != 40:
        _fail("the fixture name is " + String(long.byte_length()) + " bytes, not 40")
    var names: List[String] = [long.copy(), String("b")]
    var counts: List[Int] = [5, 3]
    var ck = _raw(names, counts)
    var path = _path("long")
    var again = _path("long-again")
    var old = _path("long-v1")
    try:
        var h = save_checkpoint(path, ck)
        if h.bytes != checkpoint_file_bytes(checkpoint_layout_bytes(names), 8):
            _fail("the file is " + String(h.bytes) + " bytes, not the formula's "
                  + String(checkpoint_file_bytes(checkpoint_layout_bytes(names), 8)))
        var back = load_checkpoint(path, ck.offsets, names)
        if back.names[0] != long or back.names[1] != "b":
            _fail("the names came back as '" + back.names[0] + "', '" + back.names[1] + "'")
        for i in range(8):  # small-loop(8: payload words)
            if bitcast[DType.uint32](back.param[i]) != bitcast[DType.uint32](ck.param[i]):
                _fail("param " + String(i) + " did not round trip")
            if bitcast[DType.uint32](back.v_state[i]) != bitcast[DType.uint32](ck.v_state[i]):
                _fail("v " + String(i) + " did not round trip")
        _ = save_checkpoint(again, back)
        var same = compare_checkpoint_files(path, again)
        if same != "":
            _fail("re-saving the loaded checkpoint changed its bytes: " + same)
        # a version-1 file (format_version word 1) is refused by version, never parsed
        var raw: List[UInt8]
        with open(path, "r") as f:
            raw = f.read_bytes()
        raw[8] = UInt8(1)
        with open(old, "w") as f:
            f.write_bytes(Span(raw))
        _ = len(raw)
        var refused = False
        try:
            _ = load_checkpoint(old, ck.offsets, names)
        except e:
            refused = String(e).find(String(CKPT_REFUSE_VERSION)) >= 0
            if not refused:
                _fail("a version-1 file was refused for another reason: " + String(e))
        if not refused:
            _fail("a version-1 file loaded")
    except e:
        _drop(path)
        _drop(again)
        _drop(old)
        raise e^
    _drop(path)
    _drop(again)
    _drop(old)
    print("PASS CHECKPOINT_NAMES 40-character name saved and loaded; version 1 refused")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
