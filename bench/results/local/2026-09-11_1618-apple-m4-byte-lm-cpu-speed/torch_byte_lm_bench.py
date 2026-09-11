"""PyTorch CPU FP32 reference timing for the byte LM forward.

usage: torch_byte_lm_bench.py CAPTURE_FULL128_DIR THREADS [OURS_2x32_LOGITS_F32] [--reps N]

Same parameters (step000128/post_p.f32), same architecture, eager PyTorch as a
user would write it: nn.functional.linear, RMSNorm eps 1e-6, rotate-half RoPE
theta 10000, GQA by repeat_interleave (head h reads kv head h // n_rep),
scaled_dot_product_attention with is_causal, SiLU gated MLP, no final norm.
NOT bit-identical and not meant to be; the optional logits file checks that
the two compute the same function (max abs difference). Prints one JSON line.
"""
import json
import statistics
import struct
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

cap = sys.argv[1]
threads = int(sys.argv[2])
assert 1 <= threads <= 3, 'at most 3 threads on the shared Mac'
ours = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith('--') else None
reps = int(sys.argv[sys.argv.index('--reps') + 1]) if '--reps' in sys.argv else 20
torch.set_num_threads(threads)
torch.set_num_interop_threads(1)

V, D, H, KV, HD, FF, LEN, LAYERS = 256, 32, 4, 2, 8, 64, 32, 2
p = np.fromfile(cap + '/step000128/post_p.f32', dtype='<f4')
assert p.size == 34944
offset = 0


def take(*shape):
    global offset
    n = int(np.prod(shape))
    t = torch.from_numpy(p[offset:offset + n].copy()).reshape(shape)
    offset += n
    return t


emb = take(V, D)
blocks = [dict(n1=take(D), q=take(H * HD, D), k=take(KV * HD, D), v=take(KV * HD, D), o=take(D, H * HD),
               n2=take(D), g=take(FF, D), u=take(FF, D), d=take(D, FF)) for _ in range(LAYERS)]
head = take(V, D)
assert offset == p.size

inv = 1.0 / (10000.0 ** (torch.arange(0, HD, 2, dtype=torch.float32) / HD))
angle = torch.arange(LEN, dtype=torch.float32)[:, None] * inv[None, :]
COS = torch.cat([angle.cos(), angle.cos()], -1)
SIN = torch.cat([angle.sin(), angle.sin()], -1)


def rms(x, w):
    return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6))


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], -1)


def forward(ids):
    b, l = ids.shape
    x = emb[ids]
    c, s = COS[:l], SIN[:l]
    for w in blocks:
        h = rms(x, w['n1'])
        q = F.linear(h, w['q']).view(b, l, H, HD).transpose(1, 2)
        k = F.linear(h, w['k']).view(b, l, KV, HD).transpose(1, 2)
        v = F.linear(h, w['v']).view(b, l, KV, HD).transpose(1, 2)
        q = q * c + rotate_half(q) * s
        k = k * c + rotate_half(k) * s
        k = k.repeat_interleave(H // KV, 1)
        v = v.repeat_interleave(H // KV, 1)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + F.linear(a.transpose(1, 2).reshape(b, l, H * HD), w['o'])
        h = rms(x, w['n2'])
        x = x + F.linear(F.silu(F.linear(h, w['g'])) * F.linear(h, w['u']), w['d'])
    return F.linear(x, head)


stream = []
for i in range(8):
    stream += struct.unpack('<66i', open(f'{cap}/heldout-final/batch{i:02d}.ids.i32', 'rb').read())
while len(stream) < 64 * 32:
    stream = stream + stream


def ids(batch, length):
    return torch.tensor(stream[:batch * length], dtype=torch.long).reshape(batch, length)


out = dict(label='torch-cpu-fp32-eager-sdpa', torch=torch.__version__, threads=threads, reps=reps)
with torch.inference_mode():
    for batch, length in ((1, 32), (2, 32), (8, 32), (32, 32)):
        x = ids(batch, length)
        for _ in range(3):
            forward(x)
        samples = []
        n = max(5, reps // max(1, batch // 2))
        for _ in range(n):
            t = time.perf_counter()
            forward(x)
            samples.append(time.perf_counter() - t)
        out[f'ms_{batch}x{length}'] = round(statistics.median(samples) * 1000, 3)
    if ours:
        mine = np.fromfile(ours, dtype='<f4').reshape(2, 32, V)
        theirs = forward(ids(2, 32)).numpy()
        out['max_abs_diff_vs_ours_2x32'] = float(np.abs(mine - theirs).max())
        out['argmax_agree_2x32'] = float((mine.argmax(-1) == theirs.argmax(-1)).mean())
print(json.dumps(out))
