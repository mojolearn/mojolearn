# SPDX-License-Identifier: Apache-2.0
# Native CPU counterpart of the former _byte_lm_trainer_host session owner.
# This owns admission, transactional state and rollback, never new arithmetic.
# Lane x7b-step-ops-host-arm (2026-11-09): training and evaluation run the
# trainer's captured program on a resident host session (ByteHostProgram,
# training/byte_lm_program.mojo), the GPU trainer's algorithm and its call
# shape: `train_steps` is ONE Program.fit and a refused call commits none of
# its steps; logits and next bytes keep byte_lm_host_execution's forward.
from std.math import isfinite
from std.memory import memcpy, memcmp, bitcast
from core.hostptr import read_f32, read_i32, f32_ptr, i32_ptr
from training.byte_lm_config import ByteConfig
from training.byte_lm_host_io import all_finite_f32_scan
from training.byte_lm_host_execution import byte_host_logits_native, byte_host_next_native, byte_host_optimizer
from training.byte_lm_program import (
    ByteHostProgram, byte_host_program_open, byte_host_program_load, byte_host_program_store, byte_program_fit,
    byte_program_eval,
)
from core.training_state_validation import training_admit_moments_and_flags
from core.host_helpers_impl import argmax_rows_f32


def host_state_finite(values: List[Float32], name: String) raises:
    if not all_finite_f32_scan(f32_ptr(Int(values.unsafe_ptr())), len(values)):
        raise Error("byte LM: nonfinite " + name)


def host_state_validate(p: List[Float32], m: List[Float32], v: List[Float32],
                        flags: List[Int32], completed: Int) raises:
    if completed < 0 or completed >= 1000000:
        raise Error("byte LM: completed step outside admitted bound")
    host_state_finite(p, "parameters")
    host_state_finite(m, "m")
    host_state_finite(v, "v")
    training_admit_moments_and_flags(Int(v.unsafe_ptr()), len(v), Int(flags.unsafe_ptr()), len(flags))


def host_tokens(address: Int, cells: Int, vocab: Int) raises:
    var values = i32_ptr(address)
    for i in range(cells):  # cpu-route: native CPU token admission, no arithmetic fallback
        if values[i] < 0 or Int(values[i]) >= vocab:
            raise Error("byte LM: token id outside [0, vocab)")


def host_loss_bits(bits: Int) raises -> Int:
    var word = UInt32(bits)
    if (word & 0x7f800000) == 0x7f800000:
        raise Error("byte LM: nonfinite returned loss")
    return bits


def host_shape_words(cfg: ByteConfig) -> List[Int]:
    return [cfg.batch, cfg.length, cfg.d_model, cfg.n_heads, cfg.n_kv, cfg.head_dim,
        cfg.intermediate, cfg.n_layers, cfg.vocab_size, Int(cfg.chunked_lm_head_v2)]


def host_words_equal(a: List[Int], b: List[Int]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):  # driver-loop: over admitted shape metadata
        if a[i] != b[i]:
            return False
    return True


def host_optimizer(words: List[Int], scalars: List[Float64]) raises -> List[Float64]:
    if len(scalars) != 8 or words[12] != 2 or words[13] != 0:
        raise Error("byte LM: the first profile admits positive-lr AdamW without clipping or SGD options")
    var result = List[Float64]()
    for i in range(8):  # driver-loop: over fixed optimizer metadata, same f32 narrowing as the legacy CPU proxy
        result.append(Float64(Float32(scalars[i])))
    if (not result[0] > 0 or not result[3] > 0 or not result[1] >= 0 or not result[1] < 1
            or not result[2] >= 0 or not result[2] < 1 or not result[4] >= 0
            or result[5] != 0 or result[6] != 0 or result[7] != 0):
        raise Error("byte LM: the first profile admits positive-lr AdamW without clipping or SGD options")
    return result^


def host_copy_f32(address: Int, values: List[Float32]) raises:
    memcpy(dest=f32_ptr(address), src=values.unsafe_ptr(), count=len(values))


def host_copy_i32(address: Int, values: List[Int32]) raises:
    memcpy(dest=i32_ptr(address), src=values.unsafe_ptr(), count=len(values))


def host_scatter(flat: Int, named: List[Int], cfg: ByteConfig) raises:
    if len(named) != cfg.n_tensors():
        raise Error("byte LM: expected one named output per parameter tensor")
    var offset = 0
    for i in range(len(named)):  # driver-loop: over named parameter tensor spans
        var count = cfg.param_count(i)
        memcpy(dest=f32_ptr(named[i]), src=f32_ptr(flat) + offset, count=count)
        offset += count


struct ByteHostSession(Movable, Writable):
    var p: List[Float32]
    var m: List[Float32]
    var v: List[Float32]
    var flags: List[Int32]
    var grad: List[Float32]
    var shape: List[Int]
    var optimizer: List[Float64]
    var shadow_p: List[Float32]
    var shadow_m: List[Float32]
    var shadow_v: List[Float32]
    var shadow_flags: List[Int32]
    var completed: Int
    var grad_step: Int
    var shadow_step: Int
    var opened: Bool
    var usable: Bool
    var busy: Bool
    #: the trainer's program (opened at the first training or evaluation
    #: call) and whether it holds the committed p, m, v and step
    var program: Optional[ByteHostProgram]
    var synced: Bool

    def __init__(out self):
        self.p = List[Float32]()
        self.m = List[Float32]()
        self.v = List[Float32]()
        self.flags = List[Int32]()
        self.grad = List[Float32]()
        self.shape = List[Int]()
        self.optimizer = List[Float64]()
        self.shadow_p = List[Float32]()
        self.shadow_m = List[Float32]()
        self.shadow_v = List[Float32]()
        self.shadow_flags = List[Int32]()
        self.completed = -1
        self.grad_step = -1
        self.shadow_step = -1
        self.opened = False
        self.usable = True
        self.busy = False
        self.program = Optional[ByteHostProgram]()
        self.synced = False

    def write_to(self, mut writer: Some[Writer]):
        writer.write("ByteHostSession")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("ByteHostSession")

    def close(mut self) raises:
        if self.busy:
            raise Error("byte LM: session is busy")
        self.p = List[Float32]()
        self.m = List[Float32]()
        self.v = List[Float32]()
        self.flags = List[Int32]()
        self.grad = List[Float32]()
        self.shadow_p = List[Float32]()
        self.shadow_m = List[Float32]()
        self.shadow_v = List[Float32]()
        self.shadow_flags = List[Int32]()
        self.program = None
        self.synced = False
        self.opened = False
        self.usable = False
        self.completed = -1
        self.grad_step = -1
        self.shadow_step = -1

    def require_open(self) raises:
        if self.busy:
            raise Error("byte LM: session is busy")
        if not self.opened:
            raise Error("byte LM: session is not open")
        if not self.usable:
            raise Error("byte LM: session is not usable; close and reopen")

    def require_shape(self, cfg: ByteConfig, completed: Int) raises:
        self.require_open()
        if not host_words_equal(self.shape, host_shape_words(cfg)):
            raise Error("byte LM: resident model shape mismatch")
        if completed != self.completed:
            raise Error("byte LM: resident completed-step mismatch")

    def open(mut self, a: List[Int], cfg: ByteConfig, opt: List[Float64], completed: Int) raises -> Int:
        if self.busy:
            raise Error("byte LM: session is busy")
        if self.opened:
            raise Error("byte LM: session is already open; close it first")
        if len(a) != 4:
            raise Error("byte LM: byte_lm_session_open expects 4 addresses")
        var n = cfg.n_total()
        var p = read_f32(a[0], n)
        var m = read_f32(a[1], n)
        var v = read_f32(a[2], n)
        var flags = read_i32(a[3], cfg.n_tensors())
        host_state_validate(p, m, v, flags, completed)
        self.p = p^
        self.m = m^
        self.v = v^
        self.flags = flags^
        self.shape = host_shape_words(cfg)
        self.optimizer = opt.copy()
        self.completed = completed
        self.grad_step = -1
        self.shadow_step = -1
        self.grad = List[Float32]()
        self.program = None
        self.synced = False
        self.opened = True
        self.usable = True
        return completed

    def admit(self, cfg: ByteConfig, opt: List[Float64], completed: Int, flags: Int) raises:
        self.require_shape(cfg, completed)
        for i in range(len(opt)):  # driver-loop: over admitted optimizer settings
            if opt[i] != self.optimizer[i]:
                raise Error("byte LM: resident optimizer mismatch")
        if memcmp(self.flags.unsafe_ptr(), i32_ptr(flags), len(self.flags)) != 0:
            raise Error("byte LM: resident momentum flags mismatch")

    def clear_shadow(mut self):
        self.shadow_step = -1
        self.shadow_p = List[Float32]()
        self.shadow_m = List[Float32]()
        self.shadow_v = List[Float32]()
        self.shadow_flags = List[Int32]()

    def _cfg(self) raises -> ByteConfig:
        if len(self.shape) != 10:
            raise Error("byte LM: resident model shape mismatch")
        return ByteConfig(self.shape[0], self.shape[1], self.shape[2], self.shape[3], self.shape[4],
            self.shape[5], self.shape[6], self.shape[7], self.shape[8], self.shape[9] == 1)

    def _ready(mut self) raises:
        """The program, opened at its first use and loaded from the
        committed lists whenever they were written outside it (open,
        rollback, a refused call)."""
        var cfg = self._cfg()
        if not self.program:
            self.program = byte_host_program_open(cfg, byte_host_optimizer(self.optimizer))
            self.synced = False
        if not self.synced:
            byte_host_program_load(self.program.value(), Int(self.p.unsafe_ptr()), Int(self.m.unsafe_ptr()),
                                   Int(self.v.unsafe_ptr()), self.completed)
            self.synced = True

    def _fit(mut self, ids: Int, n_steps: Int, losses: Int, completed: Int) raises -> Int:
        """`n_steps` training steps as ONE Program.fit (the GPU trainer's
        `byte_steps_device`): `ids` the n_steps batches of batch * (length +
        1) int32 ids in order, `losses` n_steps float32 step losses. A
        non-finite loss or a state that fails the after-scans refuses the
        whole call: nothing of it is committed. Returns completed + n_steps."""
        var cfg = self._cfg()
        var n = cfg.n_total()
        self.clear_shadow()
        self.grad_step = -1
        self._ready()
        self.synced = False
        self.busy = True
        var p = List[Float32](length=n, fill=0)
        var m = List[Float32](length=n, fill=0)
        var v = List[Float32](length=n, fill=0)
        var grad = List[Float32](length=n, fill=0)
        # AdamW writes every tensor's moments: every momentum flag is set
        var flags = List[Int32](length=len(self.flags), fill=Int32(1))
        try:
            var mean = byte_program_fit(self.program.value(), ids, n_steps, losses)
            if not isfinite(mean):
                raise Error("byte LM: nonfinite loss")
            byte_host_program_store(self.program.value(), Int(p.unsafe_ptr()), Int(m.unsafe_ptr()),
                                    Int(v.unsafe_ptr()), Int(grad.unsafe_ptr()))
            host_state_validate(p, m, v, flags, completed + n_steps)
        except error:
            self.busy = False
            raise error
        self.busy = False
        var old_p = self.p^
        self.p = p^
        self.shadow_p = old_p^
        var old_m = self.m^
        self.m = m^
        self.shadow_m = old_m^
        var old_v = self.v^
        self.v = v^
        self.shadow_v = old_v^
        var old_flags = self.flags^
        self.flags = flags^
        self.shadow_flags = old_flags^
        self.shadow_step = completed
        self.grad = grad^
        self.completed = completed + n_steps
        self.grad_step = self.completed
        self.synced = True
        return self.completed

    def step(mut self, a: List[Int], cfg: ByteConfig, opt: List[Float64], completed: Int) raises -> Int:
        """One training step: a = [ids, flags_in, out_loss, out_flags]."""
        if len(a) != 4 or a[2] == 0 or a[3] == 0:
            raise Error("byte LM: byte_lm_session_step expects four non-null addresses")
        if completed >= 999999:
            raise Error("byte LM: completed step outside admitted bound")
        host_tokens(a[0], cfg.batch * (cfg.length + 1), cfg.vocab_size)
        self.admit(cfg, opt, completed, a[1])
        var step = self._fit(a[0], 1, a[2], completed)
        host_copy_i32(a[3], self.flags)
        return step

    def steps(mut self, a: List[Int], n_steps: Int, completed: Int) raises -> Int:
        """`n_steps` training steps in ONE call from the shape, optimizer and
        flags this session admitted at open. a = [in_ids_i32 (n_steps
        batches of batch * (length + 1), one after the other), out_loss
        (n_steps), out_flags_i32 (n_tensors)]. Every batch is admitted
        first; a refused call commits none of its steps (the GPU trainer's
        rule). Returns completed + n_steps."""
        self.require_open()
        if len(a) != 3 or a[0] == 0 or a[1] == 0 or a[2] == 0:
            raise Error("byte LM: the multi-step entry expects three non-null addresses")
        if n_steps < 1:
            raise Error("byte LM trainer: the multi-step entry requires n_steps >= 1")
        if completed != self.completed:
            raise Error("byte LM: resident completed-step mismatch")
        if completed + n_steps > 999999:
            raise Error("byte LM: completed step outside admitted bound")
        var cfg = self._cfg()
        host_tokens(a[0], n_steps * cfg.batch * (cfg.length + 1), cfg.vocab_size)
        var step = self._fit(a[0], n_steps, a[1], completed)
        host_copy_i32(a[2], self.flags)
        return step

    def evaluate(mut self, a: List[Int], cfg: ByteConfig, opt: List[Float64], completed: Int) raises -> Int:
        if len(a) != 3 or a[2] == 0:
            raise Error("byte LM: byte_lm_session_eval expects three non-null addresses")
        host_tokens(a[0], cfg.batch * (cfg.length + 1), cfg.vocab_size)
        self.admit(cfg, opt, completed, a[1])
        self.clear_shadow()
        self._ready()
        var loss = host_loss_bits(Int(bitcast[DType.uint32](byte_program_eval(self.program.value(), a[0]))))
        MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=a[2])[0] = UInt32(loss)
        return self.completed

    def rollback(mut self) raises -> Int:
        if not self.opened:
            raise Error("byte LM: session is not open")
        if self.shadow_step >= 0:
            self.p = self.shadow_p^
            self.shadow_p = List[Float32]()
            self.m = self.shadow_m^
            self.shadow_m = List[Float32]()
            self.v = self.shadow_v^
            self.shadow_v = List[Float32]()
            self.flags = self.shadow_flags^
            self.shadow_flags = List[Int32]()
            self.completed = self.shadow_step
            self.shadow_step = -1
            self.grad = List[Float32]()
            self.grad_step = -1
        # the program holds the call's state, not the committed one
        self.synced = False
        return self.completed

    def export_state(self, a: List[Int], cfg: ByteConfig) raises -> Int:
        self.require_shape(cfg, self.completed)
        if len(a) != 4:
            raise Error("byte LM: byte_lm_session_export_state expects 4 addresses")
        host_state_validate(self.p, self.m, self.v, self.flags, self.completed)
        host_copy_f32(a[0], self.p)
        host_copy_f32(a[1], self.m)
        host_copy_f32(a[2], self.v)
        host_copy_i32(a[3], self.flags)
        return self.completed

    def export_gradients(self, address: Int, cfg: ByteConfig) raises -> Int:
        self.require_shape(cfg, self.completed)
        if self.grad_step < 0 or self.grad_step != self.completed:
            raise Error("byte LM: no gradient to export; complete a step first")
        host_state_finite(self.grad, "returned gradient")
        host_copy_f32(address, self.grad)
        return self.grad_step

    def next_rows(self, a: List[Int], lengths_addr: Int, batch: Int, length: Int, cfg: ByteConfig,
                  completed: Int) raises -> Int:
        """The host column of the ragged device greedy pick (plan rows N091,
        N092): the full logits through `byte_host_logits_native` (the same
        arithmetic as `logits`), then each row's first-max argmax at its last
        real position lengths[b] - 1 (`argmax_rows_f32`, the rule the
        binding applied on the host before). a = [ids, out int32 (batch)];
        `lengths_addr` holds batch int32 lengths in [1, length]."""
        self.require_shape(cfg, completed)
        if len(a) != 2 or a[1] == 0 or lengths_addr == 0:
            raise Error("byte LM: next bytes expects non-null ids, output and lengths")
        host_tokens(a[0], batch * length, cfg.vocab_size)
        var lengths = read_i32(lengths_addr, batch)
        for row in range(batch):  # driver-loop: over the admitted row lengths
            if lengths[row] < 1 or Int(lengths[row]) > length:
                raise Error("byte LM next bytes: row length outside [1, length]")
        var vocab = cfg.vocab_size
        var count = batch * length * vocab
        var full = List[Float32](length=count, fill=Float32(0))
        var descriptors: List[Int] = [Int(self.p.unsafe_ptr()), a[0], Int(full.unsafe_ptr())]
        if byte_host_logits_native(descriptors, [batch, length], cfg, 0, 1) != count:
            raise Error("byte LM logits: wrong logits length")
        if not all_finite_f32_scan(f32_ptr(Int(full.unsafe_ptr())), count):
            raise Error("byte LM logits: nonfinite logit")
        var picked = List[Int64](length=batch, fill=0)
        var out = i32_ptr(a[1])
        for row in range(batch):  # driver-loop: over the batch rows (one argmax row each)
            var base = (row * length + Int(lengths[row]) - 1) * vocab
            argmax_rows_f32(f32_ptr(Int(full.unsafe_ptr()) + base * 4), 1, vocab,
                            MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(picked.unsafe_ptr()) + row * 8))
            out[row] = Int32(picked[row])
        _ = full^
        return batch

    def logits(self, a: List[Int], batch: Int, length: Int, cfg: ByteConfig,
               completed: Int, next_bytes: Bool) raises -> Int:
        self.require_shape(cfg, completed)
        if len(a) != 2 or a[1] == 0:
            raise Error("byte LM: logits expects two non-null addresses")
        host_tokens(a[0], batch * length, cfg.vocab_size)
        var descriptors: List[Int] = [Int(self.p.unsafe_ptr()), a[0], a[1]]
        if next_bytes:
            return byte_host_next_native(descriptors, [batch, length], cfg, 1)
        var count = byte_host_logits_native(descriptors, [batch, length], cfg, 0, 1)
        if count != batch * length * cfg.vocab_size:
            raise Error("byte LM logits: wrong logits length")
        if not all_finite_f32_scan(f32_ptr(a[1]), count):
            raise Error("byte LM logits: nonfinite logit")
        return count
