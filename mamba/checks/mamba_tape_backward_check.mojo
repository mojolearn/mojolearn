# SPDX-License-Identifier: Apache-2.0
"""Tape-assembled Mamba-1 / Mamba-2 backward against stored digests and the
Mamba-1 host oracle.

For each fixture case, runs the tape assembly (autograd/rules_mamba.mojo) and
prints one digest per gradient (FNV-1a over the float32 words), then requires
every digest to equal the stored one in mamba_tape_backward_digests.tsv: the
digests of the word-for-word tape-vs-hand runs, which agreed on Apple, NVIDIA
and AMD. A two-block stack of each family is run and checked the same way.

Mamba-1 is also compared word for word with the host backward oracle
(mamba/checks/mamba_backward_oracle.mojo) on the oracle check's signed-dyadic
cotangent (mamba_backward_host_oracle_check.mojo spells it), for the eleven
public gradients. Mamba-2 has no host backward oracle; its digests are the
check. The lane that deleted the hand chain (del-mamba, 2026-10-19) recorded
the digests; docs/TOMBSTONES.md has the runs.

    mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . mamba/checks/mamba_tape_backward_check.mojo

MOJOLEARN_MAMBA_TAPE_DIGESTS overrides the digest file's path (default
mamba/checks/mamba_tape_backward_digests.tsv, relative to the repository root).
"""
from std.memory import bitcast
from std.os import getenv
from max.gpu.host import DeviceContext

from mamba.checks.mamba_fixture import MambaDims, MambaWeights, corpus_case, corpus_case_weights, corpus_case_x
from mamba.checks.mamba2_fixture import Mamba2Weights, m2_case_weights, m2_case_x, m2_corpus_case
from mamba.checks.mamba_oracle import MambaState, mamba_block_oracle
from mamba.checks.mamba_backward_oracle import mamba_block_backward_oracle
# TOMBSTONE: MOJOLEARN_AUTOGRAD_MAMBA_OFF (hand Mamba-1/2 prefill backward glue, superseded by the tape) deleted 2026-10-19 by 47582a7fb; code recoverable at 47582a7fb^.
# Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_MAMBA_OFF.patch
from autograd.rules_mamba import (
    mamba1_tape_backward,
    mamba1_tape_backward_stack,
    mamba2_tape_backward,
    mamba2_tape_backward_stack,
)

comptime DIGESTS_DEFAULT = "mamba/checks/mamba_tape_backward_digests.tsv"


def digest(values: List[Float32]) -> UInt64:
    var hash = UInt64(1469598103934665603)
    for i in range(len(values)):
        hash = (hash ^ UInt64(bitcast[DType.uint32](values[i]))) * UInt64(1099511628211)
    return hash


def load_stored() raises -> Dict[String, String]:
    """`tag name` -> `words digest` from the digest file (`#` lines are notes)."""
    var path = String(getenv("MOJOLEARN_MAMBA_TAPE_DIGESTS"))
    if path == "":
        path = String(DIGESTS_DEFAULT)
    var text: String
    with open(path, "r") as fh:
        text = fh.read()
    var out = Dict[String, String]()
    var lines = text.split("\n")
    for li in range(len(lines)):
        var line = String(lines[li])
        if line == "" or line.startswith("#"):
            continue
        var cols = line.split("\t")
        if len(cols) != 4:
            raise Error(path + ": bad line " + line)
        out[String(cols[0]) + " " + String(cols[1])] = String(cols[2]) + " " + String(cols[3])
    if len(out) == 0:
        raise Error(path + ": no digests")
    return out^


def check(tag: String, name: String, tape: List[Float32], stored: Dict[String, String]) raises -> Int:
    """1 when the tape's digest (or word count) differs from the stored row, else 0;
    a missing row raises."""
    var key = tag + " " + name
    if key not in stored:
        raise Error("mamba tape backward: no stored digest for " + key)
    var row = stored[key].copy()
    var parts = row.split(" ")
    var want_len = String(parts[0])
    var want = String(parts[1])
    var got = String(digest(tape))
    var same = got == want and (want_len == "-" or want_len == String(len(tape)))
    print("MAMBA_TAPE", tag, name, len(tape), "tape", got, "stored", want,
          "SAME" if same else "DIFFER stored_words=" + want_len)
    return 0 if same else 1


def compare_oracle(tag: String, name: String, oracle: List[Float32], tape: List[Float32]) raises -> Int:
    """Number of words that differ between the host oracle and the tape."""
    if len(oracle) != len(tape):
        raise Error(tag + " " + name + ": length oracle " + String(len(oracle)) + " vs tape " + String(len(tape)))
    var diff = 0
    var first = -1
    for i in range(len(oracle)):
        if bitcast[DType.uint32](oracle[i]) != bitcast[DType.uint32](tape[i]):
            diff += 1
            if first < 0:
                first = i
    print("MAMBA_TAPE_ORACLE", tag, name, len(tape), "oracle", digest(oracle), "tape", digest(tape),
          "SAME" if diff == 0 else "DIFFER words=" + String(diff) + " first=" + String(first))
    return diff


def cotangent(n: Int) -> List[Float32]:
    var c = List[Float32]()
    for i in range(n):
        c.append(Float32((i * 37 + 11) % 31 - 15) * Float32(0.0625))
    return c^


def oracle_cotangent(n: Int) -> List[Float32]:
    """mamba_backward_host_oracle_check.mojo::cot: the same signed sixteenths
    with the zero numerator replaced by 1."""
    var c = List[Float32]()
    for i in range(n):
        var numer = (i * 37 + 11) % 31 - 15
        if numer == 0:
            numer = 1
        c.append(Float32(numer) / Float32(16.0))
    return c^


def main() raises:
    var stored = load_stored()
    var ctx = DeviceContext()
    var bad = 0
    var oracle_bad = 0
    for case_k in [1, 2]:
        var c = corpus_case(case_k)
        var w = corpus_case_weights(case_k)
        var x = corpus_case_x(case_k)
        var dy = cotangent(len(x))
        var tag = "m1.case" + String(case_k)
        var t = mamba1_tape_backward(w, x, dy, c.b, c.l, ctx.copy())
        bad += check(tag, "x", t.x, stored)
        bad += check(tag, "norm_weight", t.norm_weight, stored)
        bad += check(tag, "in_proj_weight", t.in_proj_weight, stored)
        bad += check(tag, "conv1d_weight", t.conv1d_weight, stored)
        bad += check(tag, "conv1d_bias", t.conv1d_bias, stored)
        bad += check(tag, "x_proj_weight", t.x_proj_weight, stored)
        bad += check(tag, "dt_proj_weight", t.dt_proj_weight, stored)
        bad += check(tag, "dt_proj_bias", t.dt_proj_bias, stored)
        bad += check(tag, "A_log", t.A_log, stored)
        bad += check(tag, "D", t.D, stored)
        bad += check(tag, "out_proj_weight", t.out_proj_weight, stored)
        bad += check(tag, "stage_dB", t.stage_dB, stored)
        bad += check(tag, "stage_dC", t.stage_dC, stored)
        # two blocks, the second with the case's weights reused
        var stack = List[MambaWeights]()
        stack.append(w.copy())
        stack.append(w.copy())
        var s = mamba1_tape_backward_stack(stack, x, dy, c.b, c.l, ctx.copy())
        for k in range(len(s)):  # driver-loop: over stack blocks
            var stag = tag + ".stack.block" + String(k)
            bad += check(stag, "x", s[k].x, stored)
            bad += check(stag, "in_proj_weight", s[k].in_proj_weight, stored)
            bad += check(stag, "A_log", s[k].A_log, stored)
        # the host oracle, on its own cotangent, word for word
        var dims = MambaDims.of(c.d_model)
        var oy = oracle_cotangent(len(x))
        var s_in = MambaState(c.b, dims)
        var s_fw = MambaState(c.b, dims)
        var st = mamba_block_oracle(w, x, c.b, c.l, s_fw)
        var o = mamba_block_backward_oracle(w, x, oy, st, s_in, c.b, c.l)
        var to = mamba1_tape_backward(w, x, oy, c.b, c.l, ctx.copy())
        oracle_bad += compare_oracle(tag, "x", o.dx, to.x)
        oracle_bad += compare_oracle(tag, "norm_weight", o.dw_norm, to.norm_weight)
        oracle_bad += compare_oracle(tag, "in_proj_weight", o.dw_in, to.in_proj_weight)
        oracle_bad += compare_oracle(tag, "conv1d_weight", o.dcw, to.conv1d_weight)
        oracle_bad += compare_oracle(tag, "conv1d_bias", o.dcb, to.conv1d_bias)
        oracle_bad += compare_oracle(tag, "x_proj_weight", o.dw_x, to.x_proj_weight)
        oracle_bad += compare_oracle(tag, "dt_proj_weight", o.dw_dt, to.dt_proj_weight)
        oracle_bad += compare_oracle(tag, "dt_proj_bias", o.db_dt, to.dt_proj_bias)
        oracle_bad += compare_oracle(tag, "A_log", o.da_log, to.A_log)
        oracle_bad += compare_oracle(tag, "D", o.dd_skip, to.D)
        oracle_bad += compare_oracle(tag, "out_proj_weight", o.dw_out, to.out_proj_weight)
    for case_k in [1, 4]:
        var f = m2_corpus_case(case_k)
        var w = m2_case_weights(case_k)
        var x = m2_case_x(case_k)
        var dy = cotangent(len(x))
        var tag = "m2.case" + String(case_k)
        var t = mamba2_tape_backward(w, x, dy, f.b, f.l, f.dt_lo, f.dt_hi, ctx.copy())
        bad += check(tag, "x", t.x, stored)
        bad += check(tag, "block_norm_weight", t.block_norm_weight, stored)
        bad += check(tag, "in_proj_weight", t.in_proj_weight, stored)
        bad += check(tag, "conv1d_weight", t.conv1d_weight, stored)
        bad += check(tag, "conv1d_bias", t.conv1d_bias, stored)
        bad += check(tag, "dt_bias", t.dt_bias, stored)
        bad += check(tag, "A_log", t.A_log, stored)
        bad += check(tag, "D", t.D, stored)
        bad += check(tag, "norm_weight", t.norm_weight, stored)
        bad += check(tag, "out_proj_weight", t.out_proj_weight, stored)
        var stack = List[Mamba2Weights]()
        stack.append(w.copy())
        stack.append(w.copy())
        var s = mamba2_tape_backward_stack(stack, x, dy, f.b, f.l, f.dt_lo, f.dt_hi, ctx.copy())
        for k in range(len(s)):  # driver-loop: over stack blocks
            var stag = tag + ".stack.block" + String(k)
            bad += check(stag, "x", s[k].x, stored)
            bad += check(stag, "in_proj_weight", s[k].in_proj_weight, stored)
            bad += check(stag, "A_log", s[k].A_log, stored)
    if bad != 0 or oracle_bad != 0:
        raise Error("mamba tape backward: " + String(bad) + " digests differ from the stored record, "
                    + String(oracle_bad) + " Mamba-1 words differ from the host oracle")
    print("MAMBA_TAPE_BACKWARD_PASS every digest equals the stored record; Mamba-1 equals the host oracle word for word")
