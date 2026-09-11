#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The OPPONENT for our byte LM training step: torch, measured once per GPU.

bench/OPPONENT_REFERENCE.md "Rows that do not exist yet" item 5. One process
measures ONE column on ONE corpus at ONE shape and writes one JSON file:

  eager_fp32    torch eager, FP32, TF32 OFF (allow_tf32 False on matmul and
                cudnn, float32_matmul_precision 'highest'). THE ROW (the
                opponent's fast arm at our precision).
  eager_tf32    the same with TF32 ON. Extra column, labeled nondeterministic.
                TF32 is an NVIDIA CUDA matmul mode: on ROCm torch (and on CPU)
                the flag is accepted and does nothing, so this column writes a
                NOT APPLICABLE record (flags read back, no timing) and exits 4.
  compile_fp32  torch.compile (inductor, default mode) of the whole forward
                and loss, FP32, TF32 OFF. Extra column, labeled
                nondeterministic.
  eager_bf16    torch eager, bf16 MIXED PRECISION, the standard recipe:
                torch.autocast(device_type, dtype=torch.bfloat16) around the
                forward and the loss only; parameters, gradients and AdamW
                state stay float32; no GradScaler (bf16 needs none); TF32 OFF.
                Extra column, labeled nondeterministic. A probe on the device
                (a float32 linear under autocast must come out bfloat16 and
                its backward must run, and on the torch.cuda API
                is_bf16_supported() must answer True) decides whether the mode
                exists; when it does not, a NOT APPLICABLE record carrying the
                probe read back is written and the run exits 4.
  compile_tf32  compile_fp32 with TF32 ON (the eager_tf32 flags). Extra
                column, labeled nondeterministic; NOT APPLICABLE (exit 4)
                wherever eager_tf32 is.
  compile_bf16  compile_fp32 inside eager_bf16's autocast. Extra column,
                labeled nondeterministic; NOT APPLICABLE (exit 4) wherever
                eager_bf16 is.

No column asserts determinism: the fused SDPA backward is not a deterministic
kernel, and nothing here sets torch's deterministic switches.

SDPA BACKEND PER COLUMN: float32 columns (fp32 and tf32) pin ONE backend
(--sdpa-backend; auto probes efficient, then flash, then math). bf16 columns
under auto leave torch's own SDPA dispatch alone (switches at torch's
defaults, read back), because the kernel torch picks for bfloat16 (flash on
NVIDIA, where it exists) is part of their fast path; a named --sdpa-backend
pins it for bf16 too, probed in bfloat16. On EVERY column the last warmup
step runs under the CPU op profiler and the SDPA kernels that actually ran are
recorded (sdpa.observed); the timed steps are never profiled.

THE MODEL IS OURS, SHAPE FOR SHAPE (every item names the source it mirrors):

  shape fields  [batch, length, d_model, n_heads, n_kv, head_dim,
                intermediate, n_layers, vocab] (tools/lm_step_memory_probe.py
                CONTROL_SHAPE/TARGET_SHAPE, python/mojolearn/_byte_lm_config.py
                field order).
  registry      embed [V, DM]; per block norm1_w [DM], w_q [H*HD, DM],
                w_k [KV*HD, DM], w_v [KV*HD, DM], w_o [DM, H*HD], norm2_w [DM],
                w_gate [FF, DM], w_up [FF, DM], w_down [DM, FF]; lm_head [V, DM]
                (_byte_lm_config.py parameter_shapes). No biases, untied head,
                no final norm (training/byte_lm.mojo:9,
                tools/byte_lm_gradient_oracle.py:172).
  RMSNorm       w * x * rsqrt(mean(x^2) + eps), eps = float32(1e-6)
                (training/byte_lm.mojo:479; oracle :155).
  RoPE          theta 10000 (training/byte_lm.mojo:569), inv_freq =
                1 / theta^(2i/HD) in FP32, angle = position * inv_freq,
                cat(freqs, freqs), rotate_half = cat(-x2, x1)
                (transformer/impl/llama/modeling_llama.mojo:1437-1520 and
                :1640-1680), q and k at absolute positions 0..L-1.
  attention     causal, scale float32(1/sqrt(HD)) (modeling_llama.mojo:2634),
                k/v repeated H/KV times when grouped (oracle :162). Torch's
                fused scaled_dot_product_attention, is_causal=True, with ONE
                named backend (--sdpa-backend; auto probes efficient, then
                flash, then math on the device and records every probe): a
                different ALGORITHM from our eager softmax, which is the point.
  MLP           w_down(silu(w_gate z) * w_up z) (modeling_llama.mojo:2402-2450),
                F.silu.
  residuals     h + w_o(attn(norm1(h))), then r + mlp(norm2(r)) (oracle :157-171).
  loss          mean over batch*length next-byte cross entropies, no ignored
                positions, no smoothing (training/byte_lm.mojo:592-594,
                training/checks/loss_oracle.mojo:86-88).
  optimizer     AdamW on EVERY parameter (the flat buffer, training/byte_lm.mojo:
                964), trainer defaults lr 1e-3, betas (0.9, 0.999), eps 1e-8,
                weight_decay 0.01 (python/mojolearn/_byte_lm_impl.py:390-391),
                decoupled decay p*(1 - lr*wd), step_size = lr/bc1, denominator
                sqrt(v)/sqrt(bc2) + eps (training/checks/optimizer_oracle.mojo:
                136-166, training/checks/optimizer.mojo:554-660).
                torch.optim.AdamW is that formula; --adamw-impl picks its
                implementation (default: torch's own choice, recorded).
  init          numpy default_rng(93261).normal(0, .02, n_total) as float32,
                +1 on every norm (tools/lm_step_memory_probe.py:313-317), so
                step losses compare with the probe's on the same corpus. When
                numpy is absent a torch generator is used and the JSON says so.
  batches       step k (zero-based, counting warmups), row b reads bytes
                [(k*B*L + b*L) % (n - L - 1) : + L + 1], inputs [:, :L],
                targets [:, 1:] (tools/lm_step_memory_probe.py:215-261). ids
                sha256 is over the int32 [B, L+1] bytes, as the probe's.

THE STEP BOUNDARY is ours: synchronize, start the clock, host ids to the
device, forward, loss, backward, AdamW update, loss.item() to host,
synchronize, stop. Our boundary is the public train_step call, which
uploads ids, runs the step and reads the loss back (probe docstring :9-15).

Parameter count is computed from the registry AND from the torch module and
the run REFUSES (exit 3) unless both equal the pinned count for the shape
(target 162,147,840; control 20,453,376). The corpus sha256 and length are
checked against manifest.json beside it and the run REFUSES on mismatch.

No vendor branches in the step: the torch.cuda API serves CUDA and ROCm torch
alike. torch.version.cuda and torch.version.hip are recorded, and so is every
vendor tool that answers (nvidia-smi, rocm-smi, amd-smi, /opt/rocm/.info/
version). --device cpu is refused at the target shape (a smoke is the control
shape).

Exit codes: 0 measured, 3 refused, 4 not applicable (record written).
"""
import argparse
import contextlib
import hashlib
import json
import os
from array import array
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time

SCHEMA = 'mojolearn.torch-lm-step-opponent.v1'
SHAPES = {
    'control': [1, 2048, 384, 6, 6, 64, 1024, 8, 8192],
    'target': [1, 2048, 768, 12, 12, 64, 2048, 12, 50257],
}
SHAPE_FIELDS = ('batch', 'length', 'd_model', 'n_heads', 'n_kv', 'head_dim',
                'intermediate', 'n_layers', 'vocab_size')
EXPECTED_PARAMETERS = {'control': 20453376, 'target': 162147840}
# ENGINEERING_RULES 9: enwik8 and pile_github are the two corpora from
# 2026-09-11 night; tinyshakespeare and cpython312_lib are retired timing
# corpora, still accepted so older evidence can be re-read.
CORPORA = ('enwik8', 'pile_github', 'tinyshakespeare', 'cpython312_lib')
COLUMNS = {
    'eager_fp32': dict(tf32=False, compile=False, autocast=None, role='row', nondeterministic_label=False),
    'eager_tf32': dict(tf32=True, compile=False, autocast=None, role='extra', nondeterministic_label=True),
    'compile_fp32': dict(tf32=False, compile=True, autocast=None, role='extra', nondeterministic_label=True),
    'compile_tf32': dict(tf32=True, compile=True, autocast=None, role='extra', nondeterministic_label=True),
    'eager_bf16': dict(tf32=False, compile=False, autocast='bfloat16', role='extra',
                       nondeterministic_label=True),
    'compile_bf16': dict(tf32=False, compile=True, autocast='bfloat16', role='extra',
                         nondeterministic_label=True),
}
# Profiler event keys naming an SDPA kernel, most specific first.
SDPA_KERNEL_TAGS = (('cudnn_attention', 'cudnn'), ('efficient_attention', 'efficient'),
                    ('flash_attention_for_cpu', 'flash_cpu'), ('flash_attention', 'flash'),
                    ('attention_math', 'math'))
RMS_EPS = 1e-6          # training/byte_lm.mojo:479
ROPE_THETA = 10000.0    # training/byte_lm.mojo:569
LR, BETAS, ADAM_EPS, WEIGHT_DECAY = 1e-3, (0.9, 0.999), 1e-8, 0.01
INIT_SEED = 93261       # tools/lm_step_memory_probe.py --seed default
EXIT_REFUSED = 3
EXIT_NOT_APPLICABLE = 4
REPO = Path(__file__).resolve().parent.parent
SDPA_SWITCHES = dict(efficient=('enable_mem_efficient_sdp', 'mem_efficient_sdp_enabled'),
                     flash=('enable_flash_sdp', 'flash_sdp_enabled'),
                     math=('enable_math_sdp', 'math_sdp_enabled'),
                     cudnn=('enable_cudnn_sdp', 'cudnn_sdp_enabled'))


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def refuse(message):
    print('REFUSED: ' + message, file=sys.stderr, flush=True)
    raise SystemExit(EXIT_REFUSED)


def registry(dims):
    b, l, dm, h, kv, hd, ff, layers, vocab = dims
    shapes = [('embed', (vocab, dm))]
    for block in range(layers):
        shapes += [('block%d.%s' % (block, name), shape) for name, shape in (
            ('norm1_w', (dm,)), ('w_q', (h * hd, dm)), ('w_k', (kv * hd, dm)),
            ('w_v', (kv * hd, dm)), ('w_o', (dm, h * hd)), ('norm2_w', (dm,)),
            ('w_gate', (ff, dm)), ('w_up', (ff, dm)), ('w_down', (dm, ff)))]
    shapes.append(('lm_head', (vocab, dm)))
    return shapes


def _count(shape):
    n = 1
    for d in shape:
        n *= d
    return n


class Corpus:
    """The pinned byte corpus and the probe's schedule at this shape."""

    def __init__(self, name, batch, length):
        self.name = name
        self.path = REPO / 'training' / 'corpus' / name / 'input.txt'
        manifest_path = self.path.with_name('manifest.json')
        if not manifest_path.is_file():
            refuse('no manifest %s' % manifest_path)
        manifest_raw = manifest_path.read_bytes()
        if len(manifest_raw) > 65536:
            refuse('corpus manifest exceeds bound')
        self.manifest = json.loads(manifest_raw)
        if self.manifest.get('schema') != 'mojolearn.byte-lm.corpus.v1':
            refuse('corpus manifest schema is not mojolearn.byte-lm.corpus.v1')
        if not self.path.is_file():
            fetch = REPO / 'tools' / ('fetch_corpus_%s.sh' % name)
            hint = ' (run sh tools/%s)' % fetch.name if fetch.is_file() else ''
            refuse('no corpus %s%s' % (self.path, hint))
        raw = self.path.read_bytes()
        self.sha256 = _sha(raw)
        if self.sha256 != self.manifest.get('sha256') or len(raw) != self.manifest.get('bytes'):
            refuse('pinned corpus length/SHA mismatch for %s: sha256 %s bytes %d, manifest %s %s'
                   % (self.path, self.sha256, len(raw), self.manifest.get('sha256'),
                      self.manifest.get('bytes')))
        if len(raw) < length + 2:
            refuse('corpus shorter than one batch row')
        self.manifest_sha256 = _sha(manifest_raw)
        self.raw = raw
        self.batch = batch
        self.length = length
        self.modulus = len(raw) - length - 1

    def rows(self, step_index):
        """[B] byte strings of L+1 ids, exactly the probe's CorpusBatches.ids."""
        out = []
        for b in range(self.batch):
            start = (step_index * self.batch * self.length + b * self.length) % self.modulus
            out.append(self.raw[start:start + self.length + 1])
        return out

    @staticmethod
    def ids_sha256(rows):
        ints = array('i')
        if ints.itemsize != 4:
            raise RuntimeError('array int is not 4 bytes on this host')
        for row in rows:
            ints.extend(row)
        if sys.byteorder != 'little':
            ints.byteswap()
        return _sha(ints.tobytes())

    def describe(self):
        return dict(name=self.name, path=str(self.path.relative_to(REPO)), sha256=self.sha256,
                    bytes=len(self.raw), manifest_sha256=self.manifest_sha256,
                    source_url=self.manifest.get('source_url'),
                    manifest_train_batch_schedule=self.manifest.get('train_batch_schedule'),
                    schedule_used='step k (zero-based, warmups first) row b: bytes[(k*batch*length + '
                                  'b*length) % (bytes - length - 1) : +length+1]; inputs [:, :length], '
                                  'targets [:, 1:] (tools/lm_step_memory_probe.py CorpusBatches)')


def initial_flat(shapes, torch):
    """The probe's initial parameters (numpy), else a labeled torch fallback."""
    n_total = sum(_count(s) for _, s in shapes)
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        rng = np.random.default_rng(INIT_SEED)
        flat = rng.normal(0, .02, n_total).astype(np.float32)
        offset = 0
        for name, shape in shapes:
            size = _count(shape)
            if 'norm' in name:
                flat[offset:offset + size] += np.float32(1)
            offset += size
        source = ('numpy default_rng(%d).normal(0, .02, n_total) float32, +1 on norms '
                  '(tools/lm_step_memory_probe.py init)' % INIT_SEED)
        return torch.from_numpy(flat), source, _sha(flat.tobytes())
    gen = torch.Generator().manual_seed(INIT_SEED)
    flat = torch.empty(n_total, dtype=torch.float32).normal_(0, .02, generator=gen)
    offset = 0
    for name, shape in shapes:
        size = _count(shape)
        if 'norm' in name:
            flat[offset:offset + size] += 1
        offset += size
    return flat, 'torch Generator fallback (numpy absent); NOT the probe initialization', None


def build_model(torch, dims, shapes, flat, device):
    F = torch.nn.functional
    b, l, dm, h, kv, hd, ff, layers, vocab = dims
    n_rep = h // kv

    class ByteDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            tensors, offset = [], 0
            for _, shape in shapes:
                size = _count(shape)
                tensors.append(torch.nn.Parameter(
                    flat[offset:offset + size].reshape(shape).clone().to(device)))
                offset += size
            self.weights = torch.nn.ParameterList(tensors)
            exponent = torch.arange(0, hd // 2, dtype=torch.float32) * 2 / hd
            inv_freq = 1.0 / torch.pow(torch.tensor(ROPE_THETA, dtype=torch.float32), exponent)
            angle = torch.arange(l, dtype=torch.float32)[:, None] * inv_freq[None, :]
            angle = torch.cat((angle, angle), dim=-1)
            self.register_buffer('cos', angle.cos()[None, None].to(device), persistent=False)
            self.register_buffer('sin', angle.sin()[None, None].to(device), persistent=False)
            self.scale = attention_scale(torch, hd)
            self.eps = float(torch.tensor(RMS_EPS, dtype=torch.float32))

        def _norm(self, x, w):
            return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))

        def _rotate(self, x):
            x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
            return x * self.cos + torch.cat((-x2, x1), dim=-1) * self.sin

        def forward(self, ids):
            w = self.weights
            inputs, targets = ids[:, :-1], ids[:, 1:]
            hidden = F.embedding(inputs, w[0])
            for block in range(layers):
                base = 1 + 9 * block
                norm1, wq, wk, wv, wo, norm2, wg, wu, wd = [w[base + i] for i in range(9)]
                z = self._norm(hidden, norm1)
                q = F.linear(z, wq).view(b, l, h, hd).transpose(1, 2)
                k = F.linear(z, wk).view(b, l, kv, hd).transpose(1, 2)
                v = F.linear(z, wv).view(b, l, kv, hd).transpose(1, 2)
                q, k = self._rotate(q), self._rotate(k)
                if n_rep > 1:
                    k = k.repeat_interleave(n_rep, dim=1)
                    v = v.repeat_interleave(n_rep, dim=1)
                a = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale)
                residual = hidden + F.linear(a.transpose(1, 2).reshape(b, l, h * hd), wo)
                z = self._norm(residual, norm2)
                hidden = residual + F.linear(F.silu(F.linear(z, wg)) * F.linear(z, wu), wd)
            logits = F.linear(hidden, w[-1])
            return F.cross_entropy(logits.reshape(b * l, vocab), targets.reshape(b * l),
                                   reduction='mean')

    return ByteDecoder()


def attention_scale(torch, hd):
    return float(torch.tensor(hd, dtype=torch.float32).rsqrt())


def torch_build(torch):
    if getattr(torch.version, 'hip', None):
        return 'rocm'
    if torch.version.cuda:
        return 'cuda'
    return 'cpu-only'


def set_precision(torch, tf32):
    """Every TF32 switch this torch has, set explicitly, read back."""
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.set_float32_matmul_precision('high' if tf32 else 'highest')
    return dict(cuda_matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
                float32_matmul_precision=torch.get_float32_matmul_precision())


def _sdpa_switches(torch):
    cuda = torch.backends.cuda
    readback = {}
    for name, (_, query) in SDPA_SWITCHES.items():
        fn = getattr(cuda, query, None)
        readback[name] = bool(fn()) if fn is not None else None
    return readback


def choose_sdpa_backend(torch, requested, device, dims, sync, autocast_dtype=None):
    """ONE SDPA backend via the global switches (they hold for eager and for
    compiled graphs alike). A named backend must pass the probe or the run
    refuses; auto takes the first of efficient, flash, math (cpu: math) that
    passes. The probe is a small causal forward and backward at this head_dim
    on the device, in float32, or in bfloat16 for an autocast column; every
    attempt is recorded. EXCEPTION: auto on an autocast (bf16) column touches
    no switch and leaves the pick to torch's own dispatch, because that pick is
    part of their mixed precision fast path; sdpa.observed says what ran."""
    F = torch.nn.functional
    cuda = torch.backends.cuda
    hd = dims[5]
    if requested == 'auto' and autocast_dtype:
        return dict(requested=requested, backend='torch_default',
                    selection='auto on an autocast %s column: no switch set, torch picks per call '
                              '(switches read back below are torch defaults)' % autocast_dtype,
                    probes=[], enabled=_sdpa_switches(torch), probe_shape=None)
    probe_dtype = getattr(torch, autocast_dtype) if autocast_dtype else torch.float32
    if requested == 'auto':
        candidates = ['efficient', 'flash', 'math'] if device.type == 'cuda' else ['math']
    else:
        candidates = [requested]
    probes, chosen = [], None
    for backend in candidates:
        enable_name = SDPA_SWITCHES[backend][0]
        if getattr(cuda, enable_name, None) is None:
            probes.append(dict(backend=backend, ok=False, error='this torch has no %s' % enable_name))
            continue
        for name, (switch, _) in SDPA_SWITCHES.items():
            fn = getattr(cuda, switch, None)
            if fn is not None:
                fn(name == backend)
        try:
            q = torch.randn(1, 2, 16, hd, device=device, dtype=probe_dtype, requires_grad=True)
            out = F.scaled_dot_product_attention(q, q, q, is_causal=True, scale=attention_scale(torch, hd))
            out.float().sum().backward()
            sync()
            probes.append(dict(backend=backend, ok=True))
            chosen = backend
            break
        except Exception as exc:
            probes.append(dict(backend=backend, ok=False, error=repr(exc)[:600]))
    if chosen is None:
        refuse('no SDPA backend passed the probe: %s' % json.dumps(probes))
    return dict(requested=requested, backend=chosen,
                selection='global switches: only %s enabled' % chosen,
                probes=probes, enabled=_sdpa_switches(torch),
                probe_shape='[1, 2, 16, head_dim] %s, is_causal, forward and backward'
                            % str(probe_dtype).replace('torch.', ''))


def probe_autocast(torch, device, dtype_name, sync):
    """Does torch.autocast(device.type, dtype) exist on this device? A float32
    linear under autocast must come out in the autocast dtype, its backward
    must run and leave a float32 gradient, and on the torch.cuda API
    is_bf16_supported() must answer True. Everything is read back and
    recorded; an unsupported autocast dtype that torch silently disables is
    caught by the output dtype."""
    F = torch.nn.functional
    dtype = getattr(torch, dtype_name)
    info = dict(device_type=device.type, dtype=dtype_name)
    checker = getattr(torch.amp, 'is_autocast_available', None)
    try:
        info['amp_is_autocast_available'] = bool(checker(device.type)) if checker is not None else None
    except Exception as exc:
        info['amp_is_autocast_available'] = repr(exc)[:600]
    cuda_ok = True
    if device.type == 'cuda':
        try:
            info['cuda_is_bf16_supported'] = bool(torch.cuda.is_bf16_supported())
            cuda_ok = info['cuda_is_bf16_supported']
        except Exception as exc:
            info['cuda_is_bf16_supported'] = repr(exc)[:600]
            cuda_ok = False
        try:
            info['cuda_is_bf16_supported_native'] = bool(torch.cuda.is_bf16_supported(including_emulation=False))
        except TypeError:
            info['cuda_is_bf16_supported_native'] = 'this torch has no including_emulation argument'
        except Exception as exc:
            info['cuda_is_bf16_supported_native'] = repr(exc)[:600]
    try:
        weight = torch.randn(8, 8, device=device, dtype=torch.float32, requires_grad=True)
        x = torch.randn(2, 8, device=device, dtype=torch.float32)
        with torch.autocast(device_type=device.type, dtype=dtype):
            y = F.linear(x, weight)
            info['probe_output_dtype'] = str(y.dtype).replace('torch.', '')
        y.float().pow(2).mean().backward()
        sync()
        info['probe_grad_dtype'] = str(weight.grad.dtype).replace('torch.', '')
        info['probe_ok'] = y.dtype == dtype and weight.grad.dtype == torch.float32
    except Exception as exc:
        info['probe_ok'] = False
        info['probe_error'] = repr(exc)[:600]
    info['applicable'] = bool(info['probe_ok'] and cuda_ok)
    info['probe'] = ('F.linear float32 [2, 8] x [8, 8] under torch.autocast(%r, %s), then backward'
                     % (device.type, dtype_name))
    return info


def profiled(torch, fn):
    """fn() once under the CPU op profiler; returns (fn's value, the SDPA
    kernels that ran). A profiler that cannot start is recorded, never fatal;
    an exception from fn itself propagates."""
    try:
        from torch.profiler import ProfilerActivity, profile
        prof = profile(activities=[ProfilerActivity.CPU])
        prof.__enter__()
    except Exception as exc:
        return fn(), dict(error='profiler unavailable: %r' % exc)
    try:
        value = fn()
    finally:
        prof.__exit__(None, None, None)
    try:
        kernels = {}
        for event in prof.key_averages():
            key = str(event.key)
            if any(tag in key for tag in ('scaled_dot_product', 'flash_attention', 'efficient_attention',
                                          'cudnn_attention')):
                kernels[key] = int(event.count)
    except Exception as exc:
        return value, dict(error='profiler events unreadable: %r' % exc)
    backends = set()
    for key in kernels:
        for tag, name in SDPA_KERNEL_TAGS:
            if tag in key:
                backends.add(name)
                break
    backends = sorted(backends)
    observed = dict(kernels=kernels, backends=backends,
                    backend=backends[0] if len(backends) == 1 else ('mixed' if backends else None))
    if not backends:
        observed['note'] = ('no SDPA kernel appeared in the profiled step (a compiled graph can '
                            'decompose the math backend into plain ops)')
    return value, observed


def _bounded(text, limit=4000):
    text = (text or '').strip()
    return text[:limit] if text else None


def vendor_tools():
    """Every vendor tool that answers, raw and bounded; None when none does."""
    found = {}
    for key, cmd in (
            ('nvidia-smi', ['nvidia-smi', '--query-gpu=name,driver_version,compute_cap',
                            '--format=csv,noheader']),
            ('rocm-smi', ['rocm-smi', '--showproductname', '--showdriverversion']),
            ('amd-smi', ['amd-smi', 'version'])):
        if shutil.which(cmd[0]):
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                found[key] = dict(cmd=' '.join(cmd), exit=out.returncode, output=_bounded(out.stdout),
                                  stderr=_bounded(out.stderr, 600))
            except Exception as exc:  # recorded, never fatal
                found[key] = dict(cmd=' '.join(cmd), error=repr(exc))
    version_file = Path('/opt/rocm/.info/version')
    if version_file.is_file():
        found['rocm_version_file'] = dict(path=str(version_file),
                                          output=_bounded(version_file.read_text(errors='replace')))
    return found or None


def _commit():
    for env in ('MOJOLEARN_REPO_COMMIT', 'MOJOLEARN_COMMIT'):
        if os.environ.get(env):
            return os.environ[env], env
    if (REPO / '.git').exists() and shutil.which('git'):
        try:
            out = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'],
                                 capture_output=True, text=True, timeout=20)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip(), 'git rev-parse HEAD'
        except Exception:
            pass
    leg = Path('/root/gemm_leg_out/leg.txt')
    if leg.is_file():
        for line in leg.read_text(errors='replace').splitlines():
            if line.startswith('commit='):
                return line.split('=', 1)[1], str(leg)
    return None, 'unavailable'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--shape', choices=sorted(SHAPES), default='target')
    parser.add_argument('--corpus', choices=CORPORA, required=True)
    parser.add_argument('--column', choices=sorted(COLUMNS), required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda',
                        help='cuda = the torch.cuda API device (NVIDIA CUDA or AMD ROCm torch)')
    parser.add_argument('--warmup', type=int, default=2, help='untimed steps first (default 2)')
    parser.add_argument('--steps', type=int, default=7, help='timed steps (default 7)')
    parser.add_argument('--out', type=Path, required=True, help='new JSON file (refused if it exists)')
    parser.add_argument('--sdpa-backend', choices=('auto', 'efficient', 'flash', 'math', 'cudnn'),
                        default='auto', help='auto = first of efficient, flash, math passing the probe')
    parser.add_argument('--adamw-impl', choices=('default', 'foreach', 'fused', 'single'),
                        default='default', help="default = torch.optim.AdamW's own choice")
    args = parser.parse_args()
    if args.warmup < 0 or args.steps < 1:
        parser.error('need --warmup >= 0 and --steps >= 1')
    if args.device == 'cpu' and args.shape == 'target':
        parser.error('--device cpu is a smoke at --shape control only')
    if args.out.exists():
        parser.error('%s exists; evidence is never overwritten' % args.out)

    import torch
    column = COLUMNS[args.column]
    dims = SHAPES[args.shape]
    b, l, dm, h, kv, hd, ff, layers, vocab = dims
    if dm != h * hd or h % kv or hd % 2:
        refuse('shape violates DM = H*HD, H divisible by KV, even HD')
    shapes = registry(dims)
    formula = 2 * vocab * dm + layers * (2 * dm + 2 * dm * dm + 2 * dm * kv * hd + 3 * dm * ff)
    registry_count = sum(_count(s) for _, s in shapes)
    expected = EXPECTED_PARAMETERS[args.shape]
    if formula != expected or registry_count != expected:
        refuse('parameter count %d (registry) / %d (formula) is not the pinned %d'
               % (registry_count, formula, expected))
    if args.device == 'cuda' and not torch.cuda.is_available():
        refuse('--device cuda but torch.cuda.is_available() is False (torch %s, cuda %s, hip %s)'
               % (torch.__version__, torch.version.cuda, getattr(torch.version, 'hip', None)))
    corpus = Corpus(args.corpus, b, l)
    device = torch.device(args.device)
    on_gpu = device.type == 'cuda'
    build = torch_build(torch)
    # TF32 is a CUDA matmul mode (Ampere and newer). ROCm torch and CPU accept
    # the flag and ignore it: record the flag, never a TF32 number.
    tf32_applicable = on_gpu and build == 'cuda' and torch.cuda.get_device_capability(0)[0] >= 8
    precision = set_precision(torch, column['tf32'])
    commit, commit_source = _commit()
    sync = torch.cuda.synchronize if on_gpu else (lambda: None)
    autocast = column['autocast']
    # bf16 autocast: asked of the device, read back, never assumed. Only the
    # autocast columns probe, so the float32 columns run exactly as before.
    autocast_probe = probe_autocast(torch, device, autocast, sync) if autocast else None

    def base_record():
        return dict(
            schema=SCHEMA,
            opponent='torch byte LM training step (bench/OPPONENT_REFERENCE.md rows owed, item 5)',
            column=args.column, column_role=column['role'],
            column_flags=dict(tf32=column['tf32'], compile=column['compile'],
                              compile_backend='inductor' if column['compile'] else None,
                              compile_mode='default' if column['compile'] else None,
                              dtype='autocast_%s' % autocast if autocast else 'float32',
                              autocast_dtype=autocast,
                              autocast_device_type=device.type if autocast else None,
                              autocast_scope='forward and mean cross entropy; backward and AdamW outside'
                                             if autocast else None,
                              parameter_dtype='float32', gradient_dtype='float32',
                              optimizer_state_dtype='float32', grad_scaler=None,
                              autocast_applicable=autocast_probe['applicable'] if autocast else None,
                              autocast_probe=autocast_probe,
                              nondeterministic_label=column['nondeterministic_label'],
                              determinism_asserted=False, tf32_applicable=tf32_applicable,
                              tf32_note=None if tf32_applicable else
                              'TF32 is an NVIDIA CUDA (Ampere+) matmul mode; this torch/device accepts the '
                              'flag and does nothing with it', **precision),
            shape_name=args.shape, shape=dict(zip(SHAPE_FIELDS, dims)),
            parameters_expected=expected, tokens_per_step=b * l,
            corpus=corpus.describe(), device=args.device, torch_build=build,
            gpu_name=torch.cuda.get_device_name(0) if on_gpu else None,
            gpu_capability=list(torch.cuda.get_device_capability(0)) if on_gpu else None,
            driver=vendor_tools(),
            torch_version=str(torch.__version__), torch_cuda=torch.version.cuda,
            torch_hip=getattr(torch.version, 'hip', None),
            cudnn_or_miopen_version=torch.backends.cudnn.version() if on_gpu else None,
            python=platform.python_version(), python_executable=sys.executable,
            host=platform.node(), machine=platform.machine(),
            repo_commit=commit, repo_commit_source=commit_source,
            harness_sha256=_sha(Path(__file__).read_bytes()))

    reason = None
    if column['tf32'] and not tf32_applicable:
        reason = ('%s on torch build %r, device %r: no TF32 mode exists here, so no TF32 number is '
                  'invented' % (args.column, build, args.device))
    elif autocast and not autocast_probe['applicable']:
        reason = ('%s on torch build %r, device %r: torch.autocast(%r, %s) does not work here (probe read '
                  'back in column_flags.autocast_probe), so no %s number is invented'
                  % (args.column, build, args.device, device.type, autocast, autocast))
    if reason is not None:
        record = dict(base_record(), status='not_applicable', median_seconds=None, tokens_per_second=None,
                      reason=reason)
        with args.out.open('x') as handle:
            handle.write(json.dumps(record, indent=2, allow_nan=False) + '\n')
        print(json.dumps(dict(event='not_applicable', column=args.column, corpus=args.corpus,
                              torch_build=build, out=str(args.out))), flush=True)
        return EXIT_NOT_APPLICABLE

    sdpa = choose_sdpa_backend(torch, args.sdpa_backend, device, dims, sync, autocast)
    torch.manual_seed(INIT_SEED)

    setup_start = time.perf_counter()
    flat, init_source, init_sha = initial_flat(shapes, torch)
    model = build_model(torch, dims, shapes, flat, device)
    del flat
    module_count = sum(p.numel() for p in model.parameters())
    if module_count != expected:
        refuse('torch module holds %d parameters, pinned %d' % (module_count, expected))
    adam_kwargs = dict(lr=LR, betas=BETAS, eps=ADAM_EPS, weight_decay=WEIGHT_DECAY)
    if args.adamw_impl == 'foreach':
        adam_kwargs['foreach'] = True
    elif args.adamw_impl == 'fused':
        adam_kwargs['fused'] = True
    elif args.adamw_impl == 'single':
        adam_kwargs['foreach'] = False
    optimizer = torch.optim.AdamW(model.parameters(), **adam_kwargs)
    step_fn = torch.compile(model) if column['compile'] else model
    if autocast:
        autocast_dtype = getattr(torch, autocast)

        def autocast_scope():
            return torch.autocast(device_type=device.type, dtype=autocast_dtype)
    else:
        autocast_scope = contextlib.nullcontext
    sync()
    setup_seconds = time.perf_counter() - setup_start

    def one_step(step_index):
        rows = corpus.rows(step_index)
        host = torch.tensor([list(r) for r in rows], dtype=torch.long)
        if on_gpu:
            host = host.pin_memory()
        sync()
        start = time.perf_counter()
        ids = host.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_scope():
            loss = step_fn(ids)
        loss.backward()
        optimizer.step()
        value = loss.item()
        sync()
        seconds = time.perf_counter() - start
        if value != value or value in (float('inf'), float('-inf')):
            value = repr(value)  # a non-finite loss is a finding; keep the JSON valid
        return dict(step_index=step_index, seconds=seconds, loss=value, ids_sha256=corpus.ids_sha256(rows))

    if on_gpu:
        torch.cuda.reset_peak_memory_stats()
    warmups, timed = [], []
    observed = dict(note='not observed: --warmup 0 leaves no untimed step to profile')
    for k in range(args.warmup):
        if k == args.warmup - 1:
            # The last warmup (never a timed step) under the CPU op profiler:
            # which SDPA kernels this column actually ran.
            step, observed = profiled(torch, lambda: one_step(k))
            step['profiled'] = True
            observed['step_index'] = k
            warmups.append(step)
        else:
            warmups.append(one_step(k))
        print(json.dumps(dict(event='warmup', **warmups[-1])), flush=True)
    sdpa['observed'] = observed
    if sdpa['backend'] in SDPA_SWITCHES and observed.get('backends'):
        sdpa['observed_matches_selected'] = observed['backends'] == [sdpa['backend']]
    else:
        sdpa['observed_matches_selected'] = None
    for k in range(args.warmup, args.warmup + args.steps):
        timed.append(one_step(k))
        print(json.dumps(dict(event='step', **timed[-1])), flush=True)
    median = statistics.median(s['seconds'] for s in timed)

    result = dict(
        base_record(), status='measured',
        sdpa=sdpa,
        optimizer=dict(kind='torch.optim.AdamW', lr=LR, betas=list(BETAS), eps=ADAM_EPS,
                       weight_decay=WEIGHT_DECAY, impl_requested=args.adamw_impl,
                       constructor_kwargs={k: v for k, v in adam_kwargs.items() if k in ('foreach', 'fused')},
                       applies_to='every parameter tensor (embed, norms, projections, lm_head)'),
        parameters=module_count,
        init=dict(source=init_source, sha256=init_sha, seed=INIT_SEED),
        setup_seconds=setup_seconds,
        warmup_steps=warmups, timed_steps=timed,
        step_seconds=[s['seconds'] for s in timed],
        losses=[s['loss'] for s in warmups + timed],
        median_seconds=median, tokens_per_second=b * l / median,
        peak_memory_allocated_bytes=torch.cuda.max_memory_allocated() if on_gpu else None,
        timing_boundary='synchronize; clock; host ids (pinned) to device; zero_grad; forward + mean '
                        'cross entropy%s; backward; AdamW step; loss.item(); synchronize; clock. '
                        'Warmups (compilation included for compile columns) are untimed; the last '
                        'warmup runs under the CPU op profiler to record the SDPA kernels.'
                        % (' inside torch.autocast(%r, %s)' % (device.type, autocast) if autocast else ''),
        qualification='opponent measurement for our IDENTICAL step at the same shape and corpus; '
                      'extra columns are nondeterministic by label; no learning claim')
    with args.out.open('x') as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(event='result', column=args.column, corpus=args.corpus, shape=args.shape,
                          median_seconds=median, tokens_per_second=b * l / median,
                          sdpa_backend=sdpa['backend'], sdpa_observed=observed.get('backend'),
                          torch=str(torch.__version__),
                          torch_build=build, out=str(args.out))), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
