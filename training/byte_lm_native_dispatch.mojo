# SPDX-License-Identifier: Apache-2.0
# Typed ByteLM engine table over the common native descriptor provider.
# Session handles are minted only by its compiled boundary and retain their
# exact ByteLMSession owner there; no CPython object layout enters this file.
from std.ffi import _Global
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_words, native_descriptor_report
from training.byte_lm_binding_impl import ByteLMSession
from training.byte_lm_session_ops import byte_owner_step, byte_owner_logits, byte_owner_export_state, byte_owner_export_gradients, byte_scatter_named
from training.byte_lm_binding_impl import _validate_slot_table
from training.byte_lm_stateless_run import byte_lm_run_native, byte_lm_run_finalize
from training.byte_lm_config import ByteConfig
from training.byte_lm_stateless_logits import byte_lm_stateless_logits_native
from checks.vendor import COMPILED_VENDOR
from training.byte_lm_afn import BYTE_LM_FAST_APPLE
from training.checks.optimizer_contract import OptimizerConfig
from training.byte_lm import byte_validate_optimizer
from training.byte_lm_logits import BYTE_LOGITS_MAX_BATCH, BYTE_LOGITS_MAX_CELLS

comptime BYTE_LM_NATIVE_ABI_NAME = "mojolearn.training.byte_lm.v1"
comptime BYTE_LM_NATIVE_OPERATION_COUNT = 11


def byte_lm_native_operation(op: Int, a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    if op == 0:
        return 1 if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else 0
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL and not BYTE_LM_FAST_APPLE:
        raise Error("byte LM binding requires IDENTICAL (or FAST on Apple)")
    if String(COMPILED_VENDOR) != "cuda" and String(COMPILED_VENDOR) != "hip" and String(COMPILED_VENDOR) != "metal":
        raise Error("byte LM binding requires CUDA, HIP or Metal")
    if op != 7 and (len(a) < 1 or a[0] == 0):
        raise Error("byte LM native ABI: missing admitted session")
    var owner = MutPointer[ByteLMSession, MutUntrackedOrigin](unsafe_from_address=a[0] if len(a) > 0 else 0)
    if op == 9:
        byte_lm_run_finalize(owner[])
        return 0
    if op == 4:
        owner[].close()
        return 0
    if op == 5:
        if len(a) != 2 or a[1] == 0:
            raise Error("byte LM native ABI: missing native status output")
        var out = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=a[1])
        out[0] = owner[].trainer.value().completed_steps if owner[].trainer else -1
        out[1] = owner[].trainer.value().grad_step if owner[].trainer else -1
        out[2] = Int(owner[].usable)
        out[3] = Int(Bool(owner[].trainer))
        return 4
    if len(p) != 17 or len(f) != 8:
        raise Error("byte LM native ABI: invalid configuration descriptor")
    if p[9] != 0 and p[9] != 1:
        raise Error("byte LM: chunked head selector must be 0 or 1")
    var shape = ByteConfig(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9] == 1)
    shape.validate()
    if op == 2 or op == 6 or op == 7:
        var batch = p[14]
        var length = p[15]
        if batch < 1 or batch > BYTE_LOGITS_MAX_BATCH or length < 1 or length > shape.length:
            raise Error("byte LM logits: batch or length outside admitted bounds")
        if batch * length > BYTE_LOGITS_MAX_CELLS // shape.vocab_size:
            raise Error("byte LM logits: batch * length * vocab exceeds the admitted span")
    if op == 7:
        return byte_lm_stateless_logits_native(a, p[14], p[15], shape)
    var action = p[10]
    var completed = p[11]
    if action != 0 and action != 1 or p[13] != 0 and p[13] != 1:
        raise Error("byte LM: action/nesterov must be 0 or 1")
    if completed < 0 or completed >= 1000000:
        raise Error("byte LM: completed step outside admitted bound")
    var cfg = OptimizerConfig(p[12], Float32(f[0]), Float32(f[1]), Float32(f[2]), Float32(f[3]), Float32(f[4]),
        Float32(f[5]), Float32(f[6]), p[13] == 1, Float32(f[7]))
    byte_validate_optimizer(cfg)
    var addresses = List[Int]()
    for i in range(1, len(a)):  # small-loop(the admitted operation's buffer descriptors)
        addresses.append(a[i])
    if op == 10:
        return byte_owner_export_gradients(owner[], addresses, completed, shape)
    if op == 8:
        if len(addresses) < 11:
            raise Error("byte LM native ABI: missing stateless operation addresses")
        var admitted = List[Int]()
        var named = List[Int]()
        for i in range(11):  # small-loop(the stateless input/output descriptors)
            admitted.append(addresses[i])
        for i in range(11, len(addresses)):  # driver-loop: named gradient descriptors
            named.append(addresses[i])
        if len(named) > 0:
            if action != 1:
                raise Error("byte LM trainer: an evaluation returns no gradient")
            if len(named) != shape.n_tensors():
                raise Error("byte LM: expected one named output per parameter tensor")
            var slots: List[Int] = [admitted[8]]
            var cells: List[Int] = [shape.n_total()]
            for i in range(len(named)):  # driver-loop: validate configured tensor spans
                slots.append(named[i])
                cells.append(shape.param_count(i))
            _validate_slot_table(slots, cells, 0)
        var result = byte_lm_run_native(admitted, action, completed, cfg, shape, owner[], False, p[16] == 1)
        if len(named) > 0:
            byte_scatter_named(admitted[8], named, shape)
        return result
    if op == 1:
        return byte_owner_step(owner[], addresses, completed, cfg, shape, action == 1)
    if op == 2 or op == 6:
        var batch = p[14]
        var length = p[15]
        if batch < 1 or batch > BYTE_LOGITS_MAX_BATCH or length < 1 or length > shape.length:
            raise Error("byte LM logits: batch or length outside admitted bounds")
        if batch * length > BYTE_LOGITS_MAX_CELLS // shape.vocab_size:
            raise Error("byte LM logits: batch * length * vocab exceeds the admitted span")
        return byte_owner_logits(owner[], addresses, batch, length, completed, cfg, shape, op == 6)
    if op == 3:
        if p[16] != 0 and p[16] != 1:
            raise Error("byte LM native ABI: invalid release selector")
        return byte_owner_export_state(owner[], addresses, completed, shape, p[16] == 1)
    raise Error("byte LM native ABI: unknown operation")


def byte_lm_native_v1(op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                     result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        if op < 0 or op >= BYTE_LM_NATIVE_OPERATION_COUNT:
            raise Error("byte LM native ABI: unknown operation")
        if result == 0 or (nf > 0 and ff == 0):
            raise Error("byte LM native ABI: null scalar or result span")
        if op == 0 and (na != 0 or np != 0 or nf != 0):
            raise Error("byte LM native ABI: metadata takes no descriptors")
        if (op == 4 or op == 9) and (na != 1 or np != 0 or nf != 0):
            raise Error("byte LM native ABI: close takes one admitted session")
        if op == 5 and (na != 2 or np != 0 or nf != 0):
            raise Error("byte LM native ABI: status takes session and output")
        var a = native_descriptor_words(aa, na)
        var p = native_descriptor_words(pp, np)
        var f = List[Float64]()
        var fp = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
        for i in range(nf):  # small-loop(the eight optimizer scalars)
            f.append(fp[i])
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = byte_lm_native_operation(op, a, p, f)
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct ByteLmNativeTable(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(BYTE_LM_NATIVE_ABI_NAME)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(byte_lm_native_v1)


comptime BYTE_LM_NATIVE_TABLE = _Global[StorageType=ByteLmNativeTable,
    name="MojoByteLmNativeTableV1", init_fn=ByteLmNativeTable.__init__]
