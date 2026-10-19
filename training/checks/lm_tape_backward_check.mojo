# SPDX-License-Identifier: Apache-2.0
"""The byte LM's tape backward (autograd/rules_lm.mojo) against the hand walk, word for word.

For each config (2 layers, d_model 64, vocab 256, B 2, L 32; the v1 head and
the chunked v2 head), runs one train step (forward, backward, certified
optimizer update) on two fresh trainers with the same parameters and ids,
one through the hand walk and one through the tape, and requires the loss,
every parameter gradient word and every updated parameter word to be equal.
Prints one FNV-1a digest per gradient tensor so columns can be compared
across vendors.

    mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . training/checks/lm_tape_backward_check.mojo
"""
from std.memory import bitcast
from max.gpu.host import DeviceContext

from training.byte_lm import ByteTrainer, byte_gradient_device, byte_names, byte_update_device
from training.byte_lm_config import ByteConfig
from training.checks.optimizer_contract import OPT_ADAMW, OptimizerConfig
from training.checks.train_loop import download_f32


def digest(values: List[Float32], lo: Int, hi: Int) -> UInt64:
    var hash = UInt64(1469598103934665603)
    for i in range(lo, hi):
        hash = (hash ^ UInt64(bitcast[DType.uint32](values[i]))) * UInt64(1099511628211)
    return hash


def params(config: ByteConfig) raises -> List[Float32]:
    # norm weights near 1, everything else small and deterministic
    var out = List[Float32]()
    var offsets = config.offsets()
    for j in range(config.n_tensors()):
        var norm = j > 0 and j < config.n_tensors() - 1 and ((j - 1) % 9 == 0 or (j - 1) % 9 == 5)
        for i in range(offsets[j], offsets[j + 1]):
            var x = Float32((i * 17 + j * 13) % 101 - 50) / Float32(5000.0)
            out.append(x + Float32(1.0) if norm else x)
    return out^


def ids(config: ByteConfig) -> List[Int32]:
    var out = List[Int32]()
    for i in range(config.batch * (config.length + 1)):
        out.append(Int32((i * 127 + 11) % config.vocab_size))
    return out^


def step[tape: Bool](ctx: DeviceContext, config: ByteConfig) raises -> Tuple[Float32, List[Float32], List[Float32]]:
    """One step on a fresh trainer: (loss, gradient, updated parameters)."""
    var p = params(config)
    var z = List[Float32](length=len(p), fill=Float32(0.0))
    var flags = List[Bool](length=config.n_tensors(), fill=False)
    var opt = OptimizerConfig(OPT_ADAMW, Float32(1e-3), Float32(0.9),
        Float32(0.999), Float32(1e-8), Float32(0.0), Float32(0.0),
        Float32(0.0), False, Float32(0.0))
    var tr = ByteTrainer(ctx, p, z, z.copy(), flags, 0, opt, config)
    # _byte_step_device's composition with the backward path chosen here
    var loss = byte_gradient_device[False, tape](ctx, tr, ids(config))
    byte_update_device(ctx, tr)
    var grad = download_f32(ctx, tr.buffers.grad, config.n_total())
    var after = download_f32(ctx, tr.buffers.param, config.n_total())
    return (loss, grad^, after^)


def compare_words(tag: String, name: String, hand: List[Float32], tape: List[Float32], lo: Int, hi: Int) -> Int:
    var diff = 0
    var first = -1
    for i in range(lo, hi):
        if bitcast[DType.uint32](hand[i]) != bitcast[DType.uint32](tape[i]):
            diff += 1
            if first < 0:
                first = i - lo
    print("LM_TAPE", tag, name, hi - lo, "hand", digest(hand, lo, hi), "tape", digest(tape, lo, hi),
          "SAME" if diff == 0 else "DIFFER words=" + String(diff) + " first=" + String(first))
    return diff


def run_case(ctx: DeviceContext, tag: String, config: ByteConfig) raises -> Int:
    var hand = step[False](ctx, config)
    var tape = step[True](ctx, config)
    var bad = 0
    var lh = bitcast[DType.uint32](hand[0])
    var lt = bitcast[DType.uint32](tape[0])
    print("LM_TAPE", tag, "loss hand", lh, "tape", lt, "SAME" if lh == lt else "DIFFER")
    if lh != lt:
        bad += 1
    if len(hand[1]) != len(tape[1]) or len(hand[2]) != len(tape[2]):
        raise Error(tag + ": gradient or parameter length differs")
    var offsets = config.offsets()
    var names = byte_names(config)
    for j in range(config.n_tensors()):
        bad += compare_words(tag, String("grad.") + names[j], hand[1], tape[1], offsets[j], offsets[j + 1])
    bad += compare_words(tag, "params_after", hand[2], tape[2], 0, len(hand[2]))
    return bad


def main() raises:
    var ctx = DeviceContext()
    var bad = 0
    # batch, length, d_model, n_heads, n_kv, head_dim, intermediate, n_layers, vocab, chunked v2
    bad += run_case(ctx, "v1", ByteConfig(2, 32, 64, 8, 4, 8, 128, 2, 256, False))
    bad += run_case(ctx, "chunked_v2", ByteConfig(2, 32, 64, 8, 4, 8, 128, 2, 256, True))
    if bad != 0:
        raise Error("LM_TAPE_BACKWARD_FAIL " + String(bad) + " differing loss / gradient / parameter tensors")
    print("LM_TAPE_BACKWARD_PASS every gradient and updated parameter word equals the hand path")
