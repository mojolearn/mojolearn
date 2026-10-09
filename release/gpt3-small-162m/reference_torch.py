#!/usr/bin/env python3
"""Plain PyTorch reference forward pass for model.safetensors, for running anywhere.

This is not Mojolearn and makes no bitwise claim: PyTorch's kernels fold sums in
their own order. tools/compare_reference.py checks its logits against Mojolearn's
CPU forward pass on the same tokens (see results/reference-agreement.json).

Architecture (Mojolearn's byte LM, Llama-style layer order): token embedding, no
positional table; 12 pre-norm blocks of RMSNorm (eps 1e-6) -> multi-head causal
attention with rotary embeddings (theta 10,000, rotate-half pairing) -> residual,
RMSNorm -> SwiGLU MLP (down(silu(gate(x)) * up(x))) -> residual. No biases.
There is NO final norm: the untied lm_head reads the last residual stream directly.
Linear weights are stored [out_features, in_features].

    python3 reference_torch.py "The water cycle describes how water" --tokens 40

Requires torch, safetensors and tokenizers.
"""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent


class Model(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        self.c = config
        self.w = weights
        hd = config['head_dim']
        inv = 1.0 / (config['rope_theta'] ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
        self.register_buffer('inv_freq', inv, persistent=False)

    @staticmethod
    def rms(x, weight, eps):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight

    def rope(self, x, positions):
        freqs = torch.outer(positions.float(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos(), emb.sin()
        half = x.shape[-1] // 2
        rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
        return x * cos + rotated * sin

    def forward(self, ids):
        c, w = self.c, self.w
        b, t = ids.shape
        h, hd, eps = c['n_heads'], c['head_dim'], c['rms_norm_eps']
        positions = torch.arange(t, device=ids.device)
        x = w['embed'][ids]
        for i in range(c['n_layers']):
            p = f'block{i}.'
            y = self.rms(x, w[p + 'norm1_w'], eps)
            q = (y @ w[p + 'w_q'].T).view(b, t, h, hd).transpose(1, 2)
            k = (y @ w[p + 'w_k'].T).view(b, t, c['n_kv'], hd).transpose(1, 2)
            v = (y @ w[p + 'w_v'].T).view(b, t, c['n_kv'], hd).transpose(1, 2)
            q, k = self.rope(q, positions), self.rope(k, positions)
            a = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + a.transpose(1, 2).reshape(b, t, h * hd) @ w[p + 'w_o'].T
            y = self.rms(x, w[p + 'norm2_w'], eps)
            gate = torch.nn.functional.silu(y @ w[p + 'w_gate'].T)
            x = x + (gate * (y @ w[p + 'w_up'].T)) @ w[p + 'w_down'].T
        return x @ w['lm_head'].T


def load(device='cpu'):
    config = json.loads((ROOT / 'config.json').read_text())
    weights = {k: v.to(device) for k, v in load_file(str(ROOT / 'model.safetensors')).items()}
    return Model(config, weights).to(device)


@torch.no_grad()
def generate(model, ids, n_tokens, context):
    seq = list(ids)
    for _ in range(n_tokens):
        window = torch.tensor([seq[-context:]], dtype=torch.long)
        seq.append(int(model(window)[0, -1].argmax()))
    return seq


def main():
    from tokenizers import Tokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument('prompt')
    ap.add_argument('--tokens', type=int, default=40)
    args = ap.parse_args()
    tok = Tokenizer.from_file(str(ROOT / 'tokenizer.json'))
    model = load()
    seq = generate(model, tok.encode(args.prompt).ids, args.tokens, model.c['max_position_embeddings'])
    print(tok.decode(seq))


if __name__ == '__main__':
    main()
