# SPDX-License-Identifier: Apache-2.0
"""The byte LM's tape backward (autograd/rules_lm.mojo) against stored digests and the host oracle, word for word.

For each config (2 layers, d_model 64, vocab 256, B 2, L 32; the v1 head and
the chunked v2 head), runs one train step (forward, backward, certified
optimizer update) on a fresh trainer and requires
  1. the loss bits, every parameter gradient digest and the updated parameter
     digest to equal the stored digests (lm_tape_backward_digests.tsv, recorded
     when the tape equalled the old hand walk word for word on Metal, NVIDIA and
     AMD), and
  2. the loss, every gradient word and every updated parameter word to equal
     the host oracle step (training/byte_lm_host_backward.mojo
     byte_host_train_step, the host column's composition of the oracles).
Prints one FNV-1a digest per tensor so columns can be compared across vendors.

    mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . training/checks/lm_tape_backward_check.mojo

Run from the repository root; MOJOLEARN_LM_TAPE_DIGESTS overrides the digest
file's path.
"""
# TOMBSTONE: MOJOLEARN_AUTOGRAD_LM_OFF (hand LM backward walk; superseded by the tape, NV 1.02 / AMD 1.00, same bits) deleted 2026-10-19 by a54a5fb2e; code recoverable at a54a5fb2e^.
# Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_LM_OFF.patch; record in docs/TOMBSTONES.md.
# (The hand-vs-tape arm of this check went with the hand walk; the stored digests are that arm's agreed answer.)
from std.memory import bitcast
from std.os import getenv
from max.gpu.host import DeviceContext

from training.byte_lm import ByteTrainer, byte_gradient_device, byte_names, byte_update_device
from training.byte_lm_config import ByteConfig
from training.byte_lm_host_backward import byte_host_train_step
from training.checks.optimizer_contract import OPT_ADAMW, OptimizerConfig
from training.checks.train_loop import download_f32

comptime DIGESTS_PATH = "training/checks/lm_tape_backward_digests.tsv"


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


def optimizer() -> OptimizerConfig:
    return OptimizerConfig(OPT_ADAMW, Float32(1e-3), Float32(0.9),
        Float32(0.999), Float32(1e-8), Float32(0.0), Float32(0.0),
        Float32(0.0), False, Float32(0.0))


def step(ctx: DeviceContext, config: ByteConfig) raises -> Tuple[Float32, List[Float32], List[Float32]]:
    """One device step on a fresh trainer: (loss, gradient, updated parameters)."""
    var p = params(config)
    var z = List[Float32](length=len(p), fill=Float32(0.0))
    var flags = List[Bool](length=config.n_tensors(), fill=False)
    var tr = ByteTrainer(ctx, p, z, z.copy(), flags, 0, optimizer(), config)
    # _byte_step_device's composition
    var loss = byte_gradient_device(ctx, tr, ids(config))
    byte_update_device(ctx, tr)
    var grad = download_f32(ctx, tr.buffers.grad, config.n_total())
    var after = download_f32(ctx, tr.buffers.param, config.n_total())
    return (loss, grad^, after^)


def load_digests() raises -> Dict[String, String]:
    """`"<case>\\t<name>" -> "<words>\\t<digest>"` from the stored digest file."""
    var path = String(getenv("MOJOLEARN_LM_TAPE_DIGESTS"))
    if path.byte_length() == 0:
        path = String(DIGESTS_PATH)
    var text: String
    with open(path, "r") as f:
        text = f.read()
    var out = Dict[String, String]()
    for line_slice in text.split("\n"):
        var line = String(line_slice)
        if line.byte_length() == 0 or line.startswith("#") or line.startswith("case\t"):
            continue
        var parts = List[String]()
        for tok in line.split("\t"):
            parts.append(String(tok))
        if len(parts) != 4:
            raise Error("lm tape digests: malformed row '" + line + "'")
        out[parts[0] + "\t" + parts[1]] = parts[2] + "\t" + parts[3]
    return out^


def check_stored(stored: Dict[String, String], tag: String, name: String, words: Int, got: String) raises -> Int:
    var key = tag + "\t" + name
    if key not in stored:
        print("LM_TAPE", tag, name, words, "tape", got, "MISSING from the stored digests")
        return 1
    var want = String(stored[key])
    var have = String(words) + "\t" + got
    var ok = want == have
    print("LM_TAPE", tag, name, words, "tape", got, "stored", want.replace("\t", " "), "SAME" if ok else "DIFFER")
    return 0 if ok else 1


def compare_oracle(tag: String, name: String, oracle: List[Float32], tape: List[Float32], lo: Int, hi: Int) -> Int:
    var diff = 0
    var first = -1
    for i in range(lo, hi):
        if bitcast[DType.uint32](oracle[i]) != bitcast[DType.uint32](tape[i]):
            diff += 1
            if first < 0:
                first = i - lo
    print("LM_ORACLE", tag, name, hi - lo, "host", digest(oracle, lo, hi), "tape", digest(tape, lo, hi),
          "SAME" if diff == 0 else "DIFFER words=" + String(diff) + " first=" + String(first))
    return 0 if diff == 0 else 1


def run_case(ctx: DeviceContext, stored: Dict[String, String], tag: String, config: ByteConfig) raises -> Int:
    var tape = step(ctx, config)
    var bad = 0
    var offsets = config.offsets()
    var names = byte_names(config)
    var n = config.n_total()
    if len(tape[1]) != n or len(tape[2]) != n:
        raise Error(tag + ": gradient or parameter length differs from n_total")
    # 1. the stored digests
    var lt = bitcast[DType.uint32](tape[0])
    bad += check_stored(stored, tag, "loss", 1, String(lt))
    for j in range(config.n_tensors()):  # over registry tensors
        bad += check_stored(stored, tag, String("grad.") + names[j], offsets[j + 1] - offsets[j],
                            String(digest(tape[1], offsets[j], offsets[j + 1])))
    bad += check_stored(stored, tag, "params_after", n, String(digest(tape[2], 0, n)))
    # 2. the host oracle step from the same parameters, zero moments, step 0
    var p = params(config)
    var m0 = List[Float32](length=n, fill=Float32(0.0))
    var v0 = List[Float32](length=n, fill=Float32(0.0))
    var host = byte_host_train_step(p^, m0^, v0^, ids(config), config, optimizer(), 0)
    var lh = bitcast[DType.uint32](host.loss)
    print("LM_ORACLE", tag, "loss host", lh, "tape", lt, "SAME" if lh == lt else "DIFFER")
    if lh != lt:
        bad += 1
    if len(host.grad) != n or len(host.param) != n:
        raise Error(tag + ": host oracle gradient or parameter length differs from n_total")
    for j in range(config.n_tensors()):  # over registry tensors
        bad += compare_oracle(tag, String("grad.") + names[j], host.grad, tape[1], offsets[j], offsets[j + 1])
    bad += compare_oracle(tag, "params_after", host.param, tape[2], 0, n)
    return bad


def main() raises:
    var ctx = DeviceContext()
    var stored = load_digests()
    var bad = 0
    # batch, length, d_model, n_heads, n_kv, head_dim, intermediate, n_layers, vocab, chunked v2
    bad += run_case(ctx, stored, "v1", ByteConfig(2, 32, 64, 8, 4, 8, 128, 2, 256, False))
    bad += run_case(ctx, stored, "chunked_v2", ByteConfig(2, 32, 64, 8, 4, 8, 128, 2, 256, True))
    if bad != 0:
        raise Error("LM_TAPE_BACKWARD_FAIL " + String(bad) + " differing loss / gradient / parameter tensors")
    print("LM_TAPE_BACKWARD_PASS every digest equals the stored digests and every word equals the host oracle")
