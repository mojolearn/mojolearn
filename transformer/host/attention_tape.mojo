# SPDX-License-Identifier: Apache-2.0
"""NN31/NN32 CPU tape: native immutable input/weight snapshots and real VJP.

Host uses the same retention policy and arithmetic profile as device sessions.
CPU storage differs, so a byte budget may select replay on only one column;
retention/replay both invoke that column's same forward and backward graph.
"""
from transformer.checks.transformer_fixture import TransformerWeights, ScorePlant
from transformer.checks.transformer_oracle import (
    TransformerStages, TransformerKVCache, RopeTable, build_rope_table,
    transformer_block_oracle,
)
from transformer.checks.transformer_backward_oracle import transformer_block_backward_oracle
from transformer.experiments.checkpoint_contract import attention_checkpoint_retain


def host_attention_stage_bytes(st: TransformerStages) -> Int:
    return 4 * (len(st.input_x)+len(st.norm1_sumsq)+len(st.norm1_out)
        +len(st.q_proj_out)+len(st.k_proj_out)+len(st.v_proj_out)
        +len(st.rope_inv_freq)+len(st.rope_cos)+len(st.rope_sin)
        +len(st.q_rope_out)+len(st.k_rope_out)+len(st.kv_k_cache)+len(st.kv_v_cache)
        +len(st.attn_scores)+len(st.attn_masked)+len(st.attn_max)+len(st.attn_exp)
        +len(st.attn_denom)+len(st.attn_weights)+len(st.attn_ctx)+len(st.o_proj_out)
        +len(st.residual1_out)+len(st.norm2_sumsq)+len(st.norm2_out)+len(st.gate_proj_out)
        +len(st.up_proj_out)+len(st.silu_out)+len(st.mlp_gated)
        +len(st.down_proj_out)+len(st.residual2_out))


struct HostAttentionTape(Movable):
    var weights: TransformerWeights
    var input: List[Float32]
    var rope: RopeTable
    var stages: Optional[TransformerStages]
    var b: Int
    var l: Int
    var window: Int
    var forward_generation: Int
    var input_generation: Int
    var weight_generation: Int
    var config_generation: Int
    var live: Bool
    var retained: Bool
    var retained_bytes: Int

    def __init__(out self,var weights: TransformerWeights,var input: List[Float32],
                 b: Int,l: Int,window: Int,generation: Int) raises:
        if not weights.opts.is_default() or b <= 0 or l <= 0 or window < 0 or generation <= 0:
            raise Error("transformer tape: default FP32 positive full-prefill shape required")
        if len(input) != b*l*weights.dims.d_model:
            raise Error("transformer tape: input shape mismatch")
        self.rope = build_rope_table(weights.dims)
        self.weights = weights^
        self.input = input^
        self.b = b
        self.l = l
        self.window = window
        self.forward_generation = generation
        self.input_generation = generation
        self.weight_generation = generation
        self.config_generation = generation
        self.live = True
        self.retained = False
        self.retained_bytes = 0
        self.stages = None
        self.replay()

    def replay(mut self) raises:
        var cache = TransformerKVCache(self.b,self.weights.dims,self.l,self.window)
        self.stages = transformer_block_oracle(self.weights,self.input,self.b,self.l,
            cache,self.rope,ScorePlant.none())

    def seal(mut self,budget: Int,minimum_ops_per_byte: Int) raises:
        if budget < 0 or minimum_ops_per_byte < 0:
            raise Error("transformer tape: negative activation budget/cost")
        self.retained_bytes = host_attention_stage_bytes(self.stages.value())
        var d = self.weights.dims.copy()
        var m = self.b*self.l
        var operations = 2*m*(d.d_model*(2*d.q_width()+2*d.kv_width())+3*d.d_model*d.intermediate)
        operations += 4*self.b*d.n_heads*self.l*self.l*d.head_dim
        self.retained = attention_checkpoint_retain(self.retained_bytes,budget,operations,minimum_ops_per_byte)
        if not self.retained:
            self.stages = None

    def backward(mut self,dy: List[Float32],generation: Int) raises -> List[List[Float32]]:
        if (not self.live or generation != self.forward_generation
                or generation != self.input_generation or generation != self.weight_generation
                or generation != self.config_generation):
            raise Error("transformer tape: stale, foreign or already consumed ticket")
        self.live = False
        if not self.stages:
            self.replay()
        var st = transformer_block_backward_oracle(self.weights,self.stages.value(),dy,
            self.b,self.l,0,self.rope,self.window)
        var gradients = List[List[Float32]]()
        gradients.append(st.d_x.copy())
        gradients.append(st.dw_norm1.copy())
        gradients.append(st.dw_norm2.copy())
        gradients.append(st.dw_q.copy())
        gradients.append(st.dw_k.copy())
        gradients.append(st.dw_v.copy())
        gradients.append(st.dw_o.copy())
        gradients.append(st.dw_gate.copy())
        gradients.append(st.dw_up.copy())
        gradients.append(st.dw_down.copy())
        self.stages = None
        return gradients^
