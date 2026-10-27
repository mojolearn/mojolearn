# SPDX-License-Identifier: Apache-2.0
# CPU implementation of the same ByteLM descriptor table as GPU trainers.
# Only the session/host execution owner differs; metadata transport is shared.
from std.ffi import _Global
from std.memory import memcpy, memcmp
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_words, native_descriptor_report
from core.hostptr import f32_ptr, i32_ptr, read_f32, read_i32
from training.byte_lm_host_io import all_finite_f32_scan
from training.byte_lm_config import ByteConfig
from training.byte_lm_host_session import ByteHostSession, host_optimizer, host_state_validate, host_state_finite, host_tokens, host_loss_bits, host_copy_f32, host_copy_i32, host_scatter
from training.byte_lm_host_execution import byte_host_train_native, byte_host_loss_native, byte_host_logits_native

comptime BYTE_HOST_TRAINER_ABI = "mojolearn.training.byte_lm.v1"
comptime BYTE_HOST_TRAINER_COUNT = 12


def host_trainer_recover(mut owner: ByteHostSession, completed: Int):
    try:
        if owner.opened:
            _ = owner.rollback()
    except:
        owner.usable = False
        return
    if owner.opened and (not owner.usable or owner.completed != completed):
        owner.usable = False


def host_trainer_step(mut owner: ByteHostSession, a: List[Int], cfg: ByteConfig,
                      opt: List[Float64], completed: Int, train: Bool) raises -> Int:
    if len(a) < 8:
        raise Error("byte LM trainer: expected at least 8 addresses")
    var full = a[7] != 0
    if full and not train:
        raise Error("byte LM trainer: an evaluation returns no gradient")
    if len(a) != (8 + cfg.n_tensors() if full else 8):
        raise Error("byte LM trainer: expected one named gradient output per parameter tensor")
    if train and completed >= 999999:
        raise Error("byte LM: completed step outside admitted bound")
    if not train and a[6] != 0:
        raise Error("byte LM trainer: an evaluation requires a null flags output")
    host_tokens(a[0], cfg.batch * (cfg.length + 1), cfg.vocab_size)
    if not owner.opened:
        if a[2] == 0 and a[3] == 0 and a[4] == 0:
            raise Error("byte LM: session is not open")
        _ = owner.open([a[2], a[3], a[4], a[1]], cfg, opt, completed)
    try:
        var step = 0
        if train:
            step = owner.step([a[0], a[1], a[5], a[6]], cfg, opt, completed)
            if step != completed + 1:
                raise Error("byte LM: successful result has wrong completed step")
            if full:
                _ = owner.export_gradients(a[7], cfg)
                var named = List[Int]()
                for i in range(8, len(a)):  # driver-loop: over named gradient output descriptors
                    named.append(a[i])
                host_scatter(a[7], named, cfg)
        else:
            step = owner.evaluate([a[0], a[1], a[5]], cfg, opt, completed)
            if step != completed:
                raise Error("byte LM: successful result has wrong completed step")
        return step
    except error:
        host_trainer_recover(owner, completed)
        raise error


def host_trainer_run(a: List[Int], cfg: ByteConfig, opt: List[Float64], completed: Int, train: Bool) raises -> Int:
    if len(a) < 11:
        raise Error("byte LM: expected 11 addresses and 12 scalar parameters")
    if train and completed >= 999999:
        raise Error("byte LM: completed step outside admitted bound")
    if not train and a[8] != 0:
        raise Error("byte LM: evaluation requires a null gradient address")
    if train and a[8] == 0:
        raise Error("byte LM: training requires a gradient address")
    for i in range(11):  # driver-loop: over stateless input/output buffer descriptors
        if i != 8 and a[i] == 0:
            raise Error("byte LM: null input or output address")
    if len(a) != 11 and (not train or len(a) != 11 + cfg.n_tensors()):
        raise Error("byte LM: expected one named output per parameter tensor")
    var n = cfg.n_total()
    var p = read_f32(a[0], n)
    var m = read_f32(a[1], n)
    var v = read_f32(a[2], n)
    var flags = read_i32(a[3], cfg.n_tensors())
    host_state_validate(p, m, v, flags, completed)
    host_tokens(a[4], cfg.batch * (cfg.length + 1), cfg.vocab_size)
    var loss = 0
    if train:
        var grad = List[Float32](length=n, fill=0)
        var out_p = List[Float32](length=n, fill=0)
        var out_m = List[Float32](length=n, fill=0)
        var out_v = List[Float32](length=n, fill=0)
        loss = host_loss_bits(byte_host_train_native([Int(p.unsafe_ptr()), Int(m.unsafe_ptr()), Int(v.unsafe_ptr()),
            a[4], Int(grad.unsafe_ptr()), Int(out_p.unsafe_ptr()), Int(out_m.unsafe_ptr()), Int(out_v.unsafe_ptr())],
            cfg, opt, completed))
        host_state_validate(out_p, out_m, out_v, flags, completed + 1)
        host_state_finite(grad, "returned gradient")
        host_copy_f32(a[8], grad)
        host_copy_f32(a[5], out_p)
        host_copy_f32(a[6], out_m)
        host_copy_f32(a[7], out_v)
        if len(a) > 11:
            var named = List[Int]()
            for i in range(11, len(a)):  # driver-loop: over named stateless gradient descriptors
                named.append(a[i])
            host_scatter(a[8], named, cfg)
    else:
        loss = host_loss_bits(byte_host_loss_native([Int(p.unsafe_ptr()), a[4]], cfg, 0, 1))
        host_copy_f32(a[5], p)
        host_copy_f32(a[6], m)
        host_copy_f32(a[7], v)
    host_copy_i32(a[9], flags)
    MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=a[10])[0] = UInt32(loss)
    return completed + Int(train)


def host_trainer_operation(op: Int, a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    if op == 0:
        return 1
    if op != 7 and (len(a) < 1 or a[0] == 0):
        raise Error("byte LM native ABI: missing admitted session")
    var owner = MutPointer[ByteHostSession, MutUntrackedOrigin](unsafe_from_address=a[0] if len(a) > 0 else 0)
    if op == 4 or op == 9:
        owner[].close()
        return 0
    if op == 5 or op == 11:
        var count = 10 if op == 11 else 4
        if len(a) != 2 or a[1] == 0 or (op == 11 and (len(p) != 1 or p[0] < count)):
            raise Error("byte LM native ABI: invalid native status output")
        var out = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=a[1])
        out[0] = owner[].completed if owner[].opened else -1
        out[1] = owner[].grad_step if owner[].opened else -1
        out[2] = Int(owner[].usable)
        out[3] = Int(owner[].opened)
        for i in range(4, count):  # driver-loop: over CPU attention-stage metadata (no device stash)
            out[i] = 0
        return count
    if len(p) != 17 or len(f) != 8 or (p[9] != 0 and p[9] != 1):
        raise Error("byte LM native ABI: invalid configuration descriptor")
    var cfg = ByteConfig(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9] == 1)
    cfg.validate()
    if op == 7 or op == 2 or op == 6:
        if p[14] < 1 or p[14] > 1024 or p[15] < 1 or p[15] > cfg.length:
            raise Error("byte LM logits: batch or length outside admitted bounds")
        if p[14] * p[15] > 268435456 // cfg.vocab_size:
            raise Error("byte LM logits: batch * length * vocab exceeds the admitted span")
    if op == 7:
        if len(a) != 3:
            raise Error("byte LM: logits expects 3 addresses")
        host_tokens(a[1], p[14] * p[15], cfg.vocab_size)
        if not all_finite_f32_scan(f32_ptr(a[0]), cfg.n_total()):
            raise Error("byte LM logits: nonfinite parameters")
        var count = byte_host_logits_native(a, [p[14], p[15]], cfg, 0, 1)
        if count != p[14] * p[15] * cfg.vocab_size:
            raise Error("byte LM logits: wrong logits length")
        if not all_finite_f32_scan(f32_ptr(a[2]), count):
            raise Error("byte LM logits: nonfinite logit")
        return count
    if (p[10] != 0 and p[10] != 1) or p[11] < 0 or p[11] >= 1000000:
        raise Error("byte LM: completed step or action outside admitted bound")
    var opt = host_optimizer(p, f)
    var addresses = List[Int]()
    for i in range(1, len(a)):  # driver-loop: over admitted buffer descriptors
        addresses.append(a[i])
    if op == 8:
        return host_trainer_run(addresses, cfg, opt, p[11], p[10] == 1)
    if op == 1:
        return host_trainer_step(owner[], addresses, cfg, opt, p[11], p[10] == 1)
    if op == 10:
        if len(addresses) != 1 and len(addresses) != 1 + cfg.n_tensors():
            raise Error("byte LM trainer: expected the flat gradient and optional named outputs")
        var step = owner[].export_gradients(addresses[0], cfg)
        if step != p[11]:
            raise Error("Byte-LM gradient export returned a different step")
        if len(addresses) > 1:
            var named = List[Int]()
            for i in range(1, len(addresses)):  # driver-loop: over named gradient descriptors
                named.append(addresses[i])
            host_scatter(addresses[0], named, cfg)
        return step
    if op == 3:
        if len(addresses) != 5 or (p[16] != 0 and p[16] != 1):
            raise Error("byte LM trainer: expected 5 addresses and a release selector")
        var expected = read_i32(addresses[0], cfg.n_tensors())
        var step = 0
        try:
            step = owner[].export_state([addresses[1], addresses[2], addresses[3], addresses[4]], cfg)
            if step != p[11]:
                raise Error("Byte-LM state export returned a different completed step")
            if memcmp(i32_ptr(addresses[4]), expected.unsafe_ptr(), len(expected)) != 0:
                raise Error("Byte-LM exported flags differ from the admitted flags")
        except error:
            if p[16] == 1:
                owner[].close()
            raise error
        if p[16] == 1:
            owner[].close()
        return step
    if op == 2 or op == 6:
        if len(addresses) != 6:
            raise Error("byte LM trainer: expected 6 addresses")
        host_tokens(addresses[0], p[14] * p[15], cfg.vocab_size)
        if not owner[].opened:
            if addresses[3] == 0 and addresses[4] == 0 and addresses[5] == 0:
                raise Error("byte LM: session is not open")
            _ = owner[].open([addresses[3], addresses[4], addresses[5], addresses[2]], cfg, opt, p[11])
        try:
            return owner[].logits([addresses[0], addresses[1]], p[14], p[15], cfg, p[11], op == 6)
        except error:
            if owner[].opened and (not owner[].usable or owner[].completed != p[11]):
                owner[].usable = False
            raise error
    raise Error("byte LM native ABI: unknown operation")


def host_trainer_native_v1(op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                          result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        if op < 0 or op >= BYTE_HOST_TRAINER_COUNT or result == 0 or nf < 0 or (nf > 0 and ff == 0):
            raise Error("byte LM native ABI: invalid descriptor")
        var scalars = List[Float64]()
        var raw = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
        for i in range(nf):  # driver-loop: over optimizer descriptor scalars
            scalars.append(raw[i])
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = host_trainer_operation(
            op, native_descriptor_words(aa, na), native_descriptor_words(pp, np), scalars)
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


struct ByteHostTrainerTable(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(BYTE_HOST_TRAINER_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(host_trainer_native_v1)


comptime BYTE_HOST_TRAINER_TABLE = _Global[StorageType=ByteHostTrainerTable,
    name="MojoByteHostTrainerTableV1", init_fn=ByteHostTrainerTable.__init__]
