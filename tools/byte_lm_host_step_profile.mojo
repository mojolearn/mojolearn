# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Where one byte LM host training step spends its wall (lane neural-pass6).

The board's `lm-host-train-step` cell runs `byte_host_train_step`
(`training/byte_lm_host_backward.mojo`) at batch 1, length 512, d_model 384,
6 heads, head_dim 64, intermediate 1024, 8 layers, vocab 8192. This program
runs THAT composition, stage by stage, with a wall clock between the stages,
so a reader learns which stages are serial and which already run through the
threaded GEMM (`gemm_host_rows`). It computes nothing the step does not; it
only reads the clock between the calls the step makes, in the step's order.

    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_CPU \\
        -I . -I bindings tools/byte_lm_host_step_profile.mojo [--small] [--layers N] [--repeat R]

Prints one `profile <stage> <ms>` line per stage (the best of `repeat` runs,
default 1): the serial oracle calls (`ce_forward`, `ce_backward`,
`optimizer_step`) AND their rows-over-host-tasks twins (`ce_rows`,
`optimizer_rows`, `training/loss_host_rows.mojo` and
`training/optimizer_host_rows.mojo`), then `profile total <ms>` for the whole
`byte_host_train_step` call under the process environment
(`MOJOLEARN_BYTE_LM_HOST_STEP_ROWS=0` selects the serial calls), which includes
the step's own copies. The block stages run the oracles, whose per-row stages
are over host tasks since lane neural-pass6."""

from std.sys import argv
from std.time import perf_counter_ns

from embedding.checks.embedding_oracle import (
    EmbConfig,
    emb_backward_oracle,
    emb_forward_oracle,
)
from gemm.host.gemm_host_rows import gemm_host_rows
from gemm.host.identical_gemm import OP_NT
from training.byte_lm_config import ByteConfig
from training.byte_lm_host import byte_host_block_weights, byte_host_dims
from training.byte_lm_host_backward import (
    byte_host_adamw,
    byte_host_split_ids,
    byte_host_train_step,
)
from training.checks.loss_oracle import (
    CeConfig,
    ce_backward_oracle,
    ce_forward_oracle,
)
from training.checks.optimizer_oracle import optimizer_step_oracle
from training.loss_host_rows import ce_host_rows
from training.optimizer_host_rows import adam_host_rows
from transformer.checks.transformer_backward_oracle import (
    _gemm_bwd_a,
    _gemm_bwd_b,
    transformer_block_backward_oracle,
)
from transformer.checks.transformer_fixture import ScorePlant
from transformer.checks.transformer_oracle import (
    TransformerKVCache,
    TransformerStages,
    build_rope_table,
    transformer_block_oracle,
)


def _ms(ns: Int) -> Float64:
    return Float64(ns) / 1.0e6


def _lcg_fill(n: Int, scale: Float32, seed: Int) -> List[Float32]:
    """Deterministic small values in (-scale, scale)."""
    var out = List[Float32](capacity=n)
    var state = UInt64(seed)
    for _ in range(n):
        state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var u = Float32(Int((state >> 33) & UInt64(0xFFFFFF))) / Float32(16777216.0)
        out.append((u * Float32(2.0) - Float32(1.0)) * scale)
    return out^


def _lcg_ids(n: Int, vocab: Int, seed: Int) -> List[Int32]:
    var out = List[Int32](capacity=n)
    var state = UInt64(seed)
    for _ in range(n):
        state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        out.append(Int32(Int((state >> 33) % UInt64(vocab))))
    return out^


def _slice(values: List[Float32], offsets: List[Int], j: Int) -> List[Float32]:
    var out = List[Float32](capacity=offsets[j + 1] - offsets[j])
    for i in range(offsets[j], offsets[j + 1]):
        out.append(values[i])
    return out^


def _extend(mut into: List[Float32], values: List[Float32]):
    for i in range(len(values)):
        into.append(values[i])


struct Walls(Movable):
    var names: List[String]
    var best: List[Int]

    def __init__(out self):
        self.names = List[String]()
        self.best = List[Int]()

    def add(mut self, name: String, ns: Int):
        for i in range(len(self.names)):
            if self.names[i] == name:
                if ns < self.best[i]:
                    self.best[i] = ns
                return
        self.names.append(name)
        self.best.append(ns)

    def report(self):
        for i in range(len(self.names)):
            print("profile", self.names[i], _ms(self.best[i]))


def _profile_once(params: List[Float32], m_state: List[Float32],
                  v_state: List[Float32], ids: List[Int32], config: ByteConfig,
                  mut walls: Walls) raises:
    """`byte_host_gradient` then the update, the clock read between the calls."""
    var offsets = config.offsets()
    var dims = byte_host_dims(config)
    var rope = build_rope_table(dims)
    var b = config.batch
    var l = config.length
    var m = b * l
    var dm = config.d_model
    var v = config.vocab_size
    var emb_cfg = EmbConfig.llama(v, dm)
    var ce_cfg = CeConfig.causal_lm(v)
    var split = byte_host_split_ids(ids, config)
    var inputs = split[0].copy()
    var targets = split[1].copy()

    var t0 = perf_counter_ns()
    var x = emb_forward_oracle(_slice(params, offsets, 0), inputs, emb_cfg)
    var t1 = perf_counter_ns()
    walls.add("emb_forward", t1 - t0)

    var inputs_of = List[List[Float32]]()
    var saved = List[TransformerStages]()
    var fwd_ns = 0
    var wslice_ns = 0
    for layer in range(config.n_layers):
        var ta = perf_counter_ns()
        var w = byte_host_block_weights(params, offsets, layer, dims)
        var tb = perf_counter_ns()
        wslice_ns += tb - ta
        var cache = TransformerKVCache(b, dims, l, 0)
        inputs_of.append(x.copy())
        var st = transformer_block_oracle(w, x, b, l, cache, rope, ScorePlant.none())
        x = st.residual2_out.copy()
        saved.append(st^)
        fwd_ns += perf_counter_ns() - tb
    walls.add("block_weights_fwd", wslice_ns)
    walls.add("blocks_forward", fwd_ns)

    var head_id = config.n_tensors() - 1
    var t2 = perf_counter_ns()
    var lm_w = _slice(params, offsets, head_id)
    var logits = gemm_host_rows(x, lm_w, OP_NT, m, v, dm)
    var t3 = perf_counter_ns()
    walls.add("head_gemm", t3 - t2)
    var ce = ce_forward_oracle(logits, targets, ce_cfg)
    var t4 = perf_counter_ns()
    walls.add("ce_forward", t4 - t3)
    ce_backward_oracle(ce, targets, ce_cfg)
    var t5 = perf_counter_ns()
    walls.add("ce_backward", t5 - t4)
    var ce_rows = ce_host_rows(logits, targets, ce_cfg)
    var t5b = perf_counter_ns()
    walls.add("ce_rows", t5b - t5)
    if ce_rows[0] != ce.loss[0]:
        raise Error("profile: ce_host_rows loss differs from the oracle")
    for i in range(len(ce.dlogits)):
        if ce_rows[1][i] != ce.dlogits[i]:
            raise Error("profile: ce_host_rows dlogits differ from the oracle at " + String(i))
    t5 = perf_counter_ns()
    var d_h = _gemm_bwd_a(ce.dlogits, lm_w, OP_NT, m, v, dm)
    var t6 = perf_counter_ns()
    walls.add("head_bwd_dA", t6 - t5)
    var dw_lm = _gemm_bwd_b(ce.dlogits, x, OP_NT, m, v, dm)
    var t7 = perf_counter_ns()
    walls.add("head_bwd_dB", t7 - t6)

    var per_layer = List[List[Float32]]()
    for _ in range(config.n_layers):
        per_layer.append(List[Float32]())
    var d_out = d_h.copy()
    var bwd_ns = 0
    var pack_ns = 0
    for layer in range(config.n_layers - 1, -1, -1):
        var ta = perf_counter_ns()
        var w = byte_host_block_weights(params, offsets, layer, dims)
        var bwd = transformer_block_backward_oracle(w, saved[layer], d_out, b, l, 0, rope)
        var tb = perf_counter_ns()
        bwd_ns += tb - ta
        var packed = List[Float32]()
        _extend(packed, bwd.dw_norm1)
        _extend(packed, bwd.dw_q)
        _extend(packed, bwd.dw_k)
        _extend(packed, bwd.dw_v)
        _extend(packed, bwd.dw_o)
        _extend(packed, bwd.dw_norm2)
        _extend(packed, bwd.dw_gate)
        _extend(packed, bwd.dw_up)
        _extend(packed, bwd.dw_down)
        per_layer[layer] = packed^
        d_out = bwd.d_x.copy()
        pack_ns += perf_counter_ns() - tb
    walls.add("blocks_backward", bwd_ns)
    walls.add("blocks_pack", pack_ns)

    var t8 = perf_counter_ns()
    var dw_emb = emb_backward_oracle(d_out, inputs, emb_cfg, List[Float32]())
    var t9 = perf_counter_ns()
    walls.add("emb_backward", t9 - t8)
    var grad = List[Float32](capacity=config.n_total())
    _extend(grad, dw_emb)
    for layer in range(config.n_layers):
        _extend(grad, per_layer[layer])
    _extend(grad, dw_lm)
    var t10 = perf_counter_ns()
    walls.add("grad_pack", t10 - t9)

    var p_out = params.copy()
    var g_out = grad.copy()
    var m_out = m_state.copy()
    var v_out = v_state.copy()
    var initialized = List[Bool]()
    for _ in range(config.n_tensors()):
        initialized.append(True)
    var t11 = perf_counter_ns()
    walls.add("optimizer_copies", t11 - t10)
    var opt = byte_host_adamw(Float32(1e-3), Float32(0.9), Float32(0.95),
                              Float32(1e-8), Float32(0.1))
    _ = optimizer_step_oracle(p_out, g_out, m_out, v_out, initialized, offsets, opt, 1)
    var t12 = perf_counter_ns()
    walls.add("optimizer_step", t12 - t11)
    walls.add("stages_sum", t12 - t0)
    var p_rows = params.copy()
    var m_rows = m_state.copy()
    var v_rows = v_state.copy()
    var t13 = perf_counter_ns()
    adam_host_rows(p_rows, grad, m_rows, v_rows, opt, 1)
    var t14 = perf_counter_ns()
    walls.add("optimizer_rows", t14 - t13)
    for i in range(len(p_out)):
        if p_rows[i] != p_out[i] or m_rows[i] != m_out[i] or v_rows[i] != v_out[i]:
            raise Error("profile: adam_host_rows differs from the oracle at " + String(i))


def main() raises:
    var small = False
    var layers = -1
    var repeat = 1
    var total_only = False
    var args = argv()
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "--small":
            small = True
        elif a == "--layers":
            i += 1
            layers = Int(String(args[i]))
        elif a == "--repeat":
            i += 1
            repeat = Int(String(args[i]))
        elif a == "--total-only":
            total_only = True
        else:
            raise Error("unknown argument " + a)
        i += 1

    var config: ByteConfig
    if small:
        config = ByteConfig(2, 64, 64, 4, 2, 16, 128, 2, 256)
    else:
        config = ByteConfig(1, 512, 384, 6, 6, 64, 1024, 8, 8192)
    if layers > 0:
        config.n_layers = layers
    config.validate()
    var n = config.n_total()
    print("profile shape batch", config.batch, "length", config.length,
          "d_model", config.d_model, "layers", config.n_layers,
          "vocab", config.vocab_size, "n_total", n)

    var params = _lcg_fill(n, Float32(0.02), 7)
    var m_state = _lcg_fill(n, Float32(0.001), 11)
    var v_state = _lcg_fill(n, Float32(0.0001), 13)
    for k in range(n):
        if v_state[k] < Float32(0.0):
            v_state[k] = -v_state[k]
    var ids = _lcg_ids(config.batch * (config.length + 1), config.vocab_size, 17)
    var opt = byte_host_adamw(Float32(1e-3), Float32(0.9), Float32(0.95),
                              Float32(1e-8), Float32(0.1))

    var walls = Walls()
    for _ in range(repeat):
        if not total_only:
            _profile_once(params, m_state, v_state, ids, config, walls)
        var t0 = perf_counter_ns()
        var step = byte_host_train_step(params, m_state, v_state, ids, config, opt, 0)
        var t1 = perf_counter_ns()
        walls.add("total", t1 - t0)
        print("profile loss", step.loss)
    walls.report()
