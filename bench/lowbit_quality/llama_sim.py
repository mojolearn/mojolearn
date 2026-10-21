# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""A Llama-architecture forward whose every matrix product is a named call
into `arith.product_prepared`, so an arm replaces the arithmetic of the
products and of nothing else.

Lane lane/lowbit-quality, 2026-09-29. The model is SmolLM2-360M from the R2
dataset store (`models/SmolLM2-360M`): 32 blocks, width 960, 15 query heads
over 5 key/value heads of width 64, SwiGLU of width 2560, RMSNorm epsilon
1e-5, RoPE theta 100000, the head tied to the embedding table. Its
safetensors are bf16; they widen to float32 by the shift, exactly.

THE PRODUCTS, all OP_NT (`A @ B^T`, both operands' rows along `k`):

  layers.<i>.q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
      A = the activation rows `[B*L, k]`, one row per token; B = the weight
      `[n, k]`, one row per output feature.
  lm_head
      A = the final normed rows; B = the embedding table `[49152, 960]`.
  layers.<i>.attn_qk    scores = Q @ K^T
      A = the rotated query rows `[B, H, L, 64]`; B = the rotated key rows,
      repeated to the 15 query heads. The scale by 1/sqrt(64) = 1/8 is
      applied to the dequantized float32 output, and is exact.
  layers.<i>.attn_pv    out = P @ V = P @ (V^T)^T
      A = the probability rows `[B, H, L, L]`; B = V^T `[B, H, 64, L]`,
      whose rows run along the positions. So a row scale of B is one scale
      per head channel over the whole window. In this batched window a
      channel's scale sees later positions than the row being computed; the
      values it multiplies do not (P is zero above the diagonal). A decode
      with a KV cache would scale each channel over the prefix only, which
      is never coarser.

EVERYTHING ELSE IS FLOAT32 AND UNCHANGED: the embedding lookup, RMSNorm,
RoPE, the causal mask and softmax, SiLU and the gate multiply, the residual
additions. An activation is quantized where it enters a product; the
product's output is dequantized to float32 before anything reads it.

`infer_eval.py --validate-hf` holds this forward to `transformers`' own
LlamaForCausalLM on the same ids, with an arm that must fail.
"""
import json
import math
import os

import torch

from arith import Operand, Spec, prepare, product_prepared

PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def load_config(model_dir):
    with open(os.path.join(model_dir, "config.json")) as fh:
        cfg = json.load(fh)
    want = dict(model_type="llama", hidden_act="silu", tie_word_embeddings=True,
                attention_bias=False, rope_scaling=None)
    for key, value in want.items():
        if cfg.get(key) != value:
            raise ValueError(f"config.json {key}={cfg.get(key)!r}; this forward was written for {value!r}")
    return cfg


def load_weights(model_dir, device):
    """Every tensor as float32 on `device`. bf16 widens exactly (L-1)."""
    from safetensors import safe_open
    out, dtypes = {}, set()
    with safe_open(os.path.join(model_dir, "model.safetensors"), framework="pt", device="cpu") as fh:
        for name in fh.keys():
            t = fh.get_tensor(name)
            dtypes.add(str(t.dtype))
            out[name] = t.to(torch.float32).to(device)
    return out, sorted(dtypes)


class LlamaSim:
    def __init__(self, weights, cfg, spec, device, rope_theta=None, head_rows=1024):
        self.w, self.cfg, self.spec, self.device = weights, cfg, spec, device
        self.d = cfg["hidden_size"]
        self.n_layers = cfg["num_hidden_layers"]
        self.heads = cfg["num_attention_heads"]
        self.kv = cfg["num_key_value_heads"]
        self.hd = self.d // self.heads
        self.eps = cfg["rms_norm_eps"]
        self.theta = float(cfg["rope_theta"] if rope_theta is None else rope_theta)
        self.head_rows = head_rows
        self._prepared = {}
        self.capture = None      # dict name -> tensor, filled when set to {}
        self.diag = None         # dict product -> [sum sq err, sum sq ref], filled when set to {}

    # ------------------------------------------------------------ products

    def _weight_operand(self, product, tensor_name, kind):
        key = (tensor_name, kind)
        if key not in self._prepared:
            self._prepared[key] = prepare(self.w[tensor_name], kind)
        return self._prepared[key]

    def _record(self, product, out, a_f32, b_f32):
        if self.diag is None:
            return
        ref = a_f32.to(torch.float64) @ b_f32.to(torch.float64).transpose(-1, -2)
        err = (out.to(torch.float64) - ref).square().sum().item()
        acc = self.diag.setdefault(product, [0.0, 0.0])
        acc[0] += err
        acc[1] += ref.square().sum().item()

    def _projection(self, x, product, tensor_name):
        """`x [B, L, k] @ W[n, k]^T`, the rows of `x` flattened to tokens."""
        ka, kb = self.spec.kinds(product)
        lead = x.shape[:-1]
        rows = x.reshape(-1, x.shape[-1])
        if self.capture is not None:
            self.capture[product + ".input"] = rows.detach().clone()
        b = self._weight_operand(product, tensor_name, kb)
        a = prepare(rows, ka)
        if product == "lm_head":
            chunks = []
            for lo in range(0, rows.shape[0], self.head_rows):
                part = a.rows(lo, lo + self.head_rows)
                out = product_prepared(part, b, acc64=self.spec.acc64)
                self._record(product, out, rows[lo:lo + self.head_rows], self.w[tensor_name])
                chunks.append(out)
            out = torch.cat(chunks, 0)
        else:
            out = product_prepared(a, b, acc64=self.spec.acc64)
            self._record(product, out, rows, self.w[tensor_name])
        return out.reshape(*lead, out.shape[-1])

    def _attention_product(self, A, B, product):
        ka, kb = self.spec.kinds(product)
        if self.capture is not None:
            self.capture[product + ".a"] = A.detach().clone()
            self.capture[product + ".b"] = B.detach().clone()
        out = product_prepared(prepare(A, ka), prepare(B, kb), acc64=self.spec.acc64)
        self._record(product, out, A, B)
        return out

    # ------------------------------------------------------------- forward

    def _norm(self, x, name):
        return self.w[name] * (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps))

    def _rope(self, length):
        inv = 1.0 / (self.theta ** (torch.arange(0, self.hd, 2, dtype=torch.float32, device=self.device) / self.hd))
        freqs = torch.arange(length, dtype=torch.float32, device=self.device)[:, None] * inv[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, None], emb.sin()[None, None]

    def _rotate(self, a, cos, sin):
        half = torch.cat((-a[..., self.hd // 2:], a[..., :self.hd // 2]), dim=-1)
        return a * cos + half * sin

    @torch.no_grad()
    def forward(self, ids):
        """`ids [B, L]` (int64) -> float32 logits `[B, L, vocab]`."""
        bsz, length = ids.shape
        h = self.w["model.embed_tokens.weight"][ids]
        cos, sin = self._rope(length)
        mask = torch.ones((length, length), dtype=torch.bool, device=self.device).triu(1)
        rep = self.heads // self.kv
        for i in range(self.n_layers):
            p = f"model.layers.{i}."
            n = f"layers.{i}."
            z = self._norm(h, p + "input_layernorm.weight")
            q = self._projection(z, n + "q_proj", p + "self_attn.q_proj.weight")
            k = self._projection(z, n + "k_proj", p + "self_attn.k_proj.weight")
            v = self._projection(z, n + "v_proj", p + "self_attn.v_proj.weight")
            q = q.reshape(bsz, length, self.heads, self.hd).transpose(1, 2)
            k = k.reshape(bsz, length, self.kv, self.hd).transpose(1, 2)
            v = v.reshape(bsz, length, self.kv, self.hd).transpose(1, 2)
            q, k = self._rotate(q, cos, sin), self._rotate(k, cos, sin)
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
            scores = self._attention_product(q, k, n + "attn_qk") * (1.0 / math.sqrt(self.hd))
            prob = scores.masked_fill(mask, float("-inf")).softmax(-1)
            att = self._attention_product(prob, v.transpose(-1, -2).contiguous(), n + "attn_pv")
            att = att.transpose(1, 2).reshape(bsz, length, self.d)
            h = h + self._projection(att, n + "o_proj", p + "self_attn.o_proj.weight")
            z = self._norm(h, p + "post_attention_layernorm.weight")
            gate = self._projection(z, n + "gate_proj", p + "mlp.gate_proj.weight")
            up = self._projection(z, n + "up_proj", p + "mlp.up_proj.weight")
            act = torch.nn.functional.silu(gate) * up
            h = h + self._projection(act, n + "down_proj", p + "mlp.down_proj.weight")
        z = self._norm(h, "model.norm.weight")
        return self._projection(z, "lm_head", "model.embed_tokens.weight")

    def release(self):
        self._prepared.clear()
