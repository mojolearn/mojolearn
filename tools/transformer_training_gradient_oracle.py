#!/usr/bin/env python3
"""Fixed-profile independent FP64 autograd gate. Root remote execution only.

This module's CaptureWriter is also called by the Mojo executable. Importing
it performs no model work. CLI intentionally admits only NVIDIA/AMD CUDA API
backends. Tolerances are declared here, never inferred from observed errors.
An oracle pass establishes tolerance correctness, NOT cross-vendor bit identity.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import struct

SCHEMA = 'mojolearn.training.gradient-capture.v1'
SHAPES = [('embed', (64, 32)), ('norm1_w', (32,)), ('w_q', (32, 32)),
          ('w_k', (16, 32)), ('w_v', (16, 32)), ('w_o', (32, 32)),
          ('norm2_w', (32,)), ('w_gate', (64, 32)), ('w_up', (64, 32)),
          ('w_down', (32, 64)), ('lm_head', (64, 32))]
PROFILE = dict(batch=2, length=8, d_model=32, vocab=64, n_heads=4,
               n_kv_heads=2, head_dim=8, intermediate=64, rms_eps=1e-6,
               rope_theta=10000, layers=1, causal=True, reduction='mean',
               tied_embeddings=False)
# FP32 reduction/elementwise error against independently evaluated FP64 math.
# Every cell must satisfy atol + rtol*abs(reference), including near-zero cells.
GRAD_ATOL, GRAD_RTOL = 2e-6, 2e-4
LOSS_ATOL, LOSS_RTOL = 2e-6, 2e-6
ARRAY_COUNTS = {k: 13376 for k in ('initial_params', 'initial_m', 'initial_v',
                'gradients', 'updated_params', 'updated_m', 'updated_v')}
ARRAY_COUNTS['loss'] = 1


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def registry():
    result, offset = [], 0
    for name, shape in SHAPES:
        count = math.prod(shape)
        result.append(dict(name=name, shape=list(shape), offset=offset, count=count))
        offset += count
    return result


def exclusive(path, raw):
    with Path(path).open('xb') as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def json_bytes(obj):
    return (json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def read_ids(path):
    with open(path, 'rb') as handle:
        raw = handle.read(4097)
    if len(raw) > 4096:
        raise ValueError('token fixture exceeds 4096-byte bound')
    ids = json.loads(raw)
    if not isinstance(ids, list) or len(ids) != 18 or any(type(x) is not int or not 0 <= x < 64 for x in ids):
        raise ValueError('expected flat list of exactly 18 integer IDs in [0,64)')
    return ids


class CaptureWriter:
    def __init__(self, output, vendor, seed):
        if not output or vendor not in ('cuda', 'hip'):
            raise ValueError('new output directory and native cuda/hip required')
        self.path = Path(output)
        self.path.mkdir(parents=False, exist_ok=False)
        self.arrays = {}
        self.meta = dict(schema=SCHEMA, numeric_mode='identical', vendor=vendor,
                         seed=seed, completed_steps=1, profile=PROFILE,
                         registry=registry(), optimizer=dict(kind='adamw', lr=1e-3,
                         beta1=.9, beta2=.999, eps=1e-8, weight_decay=.01,
                         max_norm=0), arrays=self.arrays)
        sources = {}
        for directory in ('checks', 'core', 'gemm', 'embedding', 'transformer', 'training'):
            for path in sorted(Path(directory).rglob('*.mojo')):
                sources[path.as_posix()] = sha(path.read_bytes())
        for name in ('pixi.toml', 'pixi.lock', 'tools/transformer_training_gradient_oracle.py'):
            sources[name] = sha(Path(name).read_bytes())
        self.meta['source_sha256'] = sources

    def require_registry(self, index, name, count):
        item = registry()[index]
        if item['name'] != name or item['count'] != count:
            raise ValueError('native parameter registry changed')

    def add_ids(self, words, supplied):
        ids = [int(x) for x in words.split(',')]
        if len(ids) != 18 or any(not 0 <= x < 64 for x in ids):
            raise ValueError('invalid IDs')
        raw = struct.pack('<18i', *ids)
        exclusive(self.path / 'token_ids.i32', raw)
        self.meta['token_ids'] = dict(file='token_ids.i32', sha256=sha(raw),
                                     shape=[2, 9], supplied=bool(supplied))

    def add_bits(self, name, words):
        if name not in ARRAY_COUNTS or name in self.arrays:
            raise ValueError('unknown or duplicate array')
        bits = [int(x) for x in words.split(',')]
        if len(bits) != ARRAY_COUNTS[name] or any(not 0 <= x <= 0xffffffff or x & 0x7f800000 == 0x7f800000 for x in bits):
            raise ValueError('wrong array length or nonfinite FP32')
        raw = struct.pack('<' + 'I' * len(bits), *bits)
        filename = name + '.f32'
        exclusive(self.path / filename, raw)
        self.arrays[name] = dict(file=filename, count=len(bits), sha256=sha(raw))

    def finish(self):
        if set(self.arrays) != set(ARRAY_COUNTS) or 'token_ids' not in self.meta:
            raise ValueError('incomplete capture')
        exclusive(self.path / 'capture.json', json_bytes(self.meta))


def load_capture(path):
    import numpy as np
    raw_manifest = (path / 'capture.json').read_bytes()
    meta = json.loads(raw_manifest)
    if (meta.get('schema') != SCHEMA or meta.get('profile') != PROFILE
            or meta.get('registry') != registry() or meta.get('numeric_mode') != 'identical'
            or meta.get('vendor') not in ('cuda', 'hip') or meta.get('completed_steps') != 1
            or set(meta.get('arrays', {})) != set(ARRAY_COUNTS)):
        raise ValueError('capture metadata differs from fixed admitted profile')
    arrays = {}
    for name, count in ARRAY_COUNTS.items():
        item = meta['arrays'][name]
        if item.get('file') != name + '.f32' or item.get('count') != count:
            raise ValueError('array descriptor mismatch')
        raw = (path / item['file']).read_bytes()
        if len(raw) != count * 4 or sha(raw) != item['sha256']:
            raise ValueError('array bytes/hash mismatch')
        arrays[name] = np.frombuffer(raw, dtype='<f4').copy()
        if not np.isfinite(arrays[name]).all():
            raise ValueError('nonfinite capture')
    item = meta['token_ids']
    if item['file'] != 'token_ids.i32' or item['shape'] != [2, 9]:
        raise ValueError('token descriptor mismatch')
    raw = (path / item['file']).read_bytes()
    if len(raw) != 72 or sha(raw) != item['sha256']:
        raise ValueError('token bytes/hash mismatch')
    ids = np.frombuffer(raw, dtype='<i4').copy().reshape(2, 9)
    if (ids < 0).any() or (ids >= 64).any():
        raise ValueError('invalid token')
    return meta, arrays, ids, sha(raw_manifest)


def reference(initial, ids, *, nonlinear_sabotage=False):
    """Independent math graph; no Mojo oracle or native arithmetic imports."""
    import torch
    import torch.nn.functional as F
    weights = {}
    for item in registry():
        a, b = item['offset'], item['offset'] + item['count']
        weights[item['name']] = torch.tensor(initial[a:b].reshape(item['shape']),
                            dtype=torch.float64, device='cuda', requires_grad=True)
    def norm(x, w):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * w
    def linear(x, name):
        return F.linear(x, weights[name])
    token = torch.tensor(ids, dtype=torch.long, device='cuda')
    x = F.embedding(token[:, :-1], weights['embed'])
    z = norm(x, weights['norm1_w'])
    q = linear(z, 'w_q').reshape(2, 8, 4, 8).transpose(1, 2)
    k = linear(z, 'w_k').reshape(2, 8, 2, 8).transpose(1, 2)
    v = linear(z, 'w_v').reshape(2, 8, 2, 8).transpose(1, 2)
    frequency = 10000.0 ** (-torch.arange(0, 8, 2, dtype=torch.float64, device='cuda') / 8)
    angles = torch.arange(8, dtype=torch.float64, device='cuda')[:, None] * frequency
    angles = torch.cat((angles, angles), dim=-1)[None, None]
    def rotate(a):
        half = torch.cat((-a[..., 4:], a[..., :4]), dim=-1)
        return a * angles.cos() + half * angles.sin()
    q, k = rotate(q), rotate(k)
    k, v = k.repeat_interleave(2, dim=1), v.repeat_interleave(2, dim=1)
    scores = q @ k.transpose(-1, -2) / math.sqrt(8)
    mask = torch.ones((8, 8), dtype=torch.bool, device='cuda').triu(1)
    probabilities = scores.masked_fill(mask, -torch.inf).softmax(-1)
    attention = (probabilities @ v).transpose(1, 2).reshape(2, 8, 32)
    residual = x + linear(attention, 'w_o')
    z = norm(residual, weights['norm2_w'])
    gate = linear(z, 'w_gate')
    # Effective derivative sabotage leaves forward SiLU values unchanged while
    # dropping sigmoid's derivative. Thus a forward-only check cannot pass it.
    activated = gate * gate.sigmoid().detach() if nonlinear_sabotage else F.silu(gate)
    hidden = residual + linear(activated * linear(z, 'w_up'), 'w_down')
    logits = linear(hidden, 'lm_head')
    loss = F.cross_entropy(logits.reshape(16, 64), token[:, 1:].reshape(16), reduction='mean')
    loss.backward()
    torch.cuda.synchronize()
    gradients = {name: weight.grad.detach().cpu().numpy().copy() for name, weight in weights.items()}
    return float(loss.detach().cpu()), gradients


def comparison(actual, expected, atol, rtol):
    import numpy as np
    actual, expected = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    error = np.abs(actual - expected)
    threshold = atol + rtol * np.abs(expected)
    bad = error > threshold
    return dict(passed=bool(np.isfinite(actual).all() and np.isfinite(expected).all() and not bad.any()),
                cells=int(error.size), failed_cells=int(bad.sum()),
                max_absolute_error=float(error.max()), max_excess=float((error-threshold).max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--expected-vendor', required=True, choices=('cuda', 'hip'))
    args = parser.parse_args()
    # Reserve the output before any model work; existing evidence is immutable.
    with args.output.open('xb') as output:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        vendor = 'hip' if torch.version.hip else 'cuda'
        if not torch.cuda.is_available() or vendor != args.expected_vendor:
            raise ValueError('expected remote NVIDIA/AMD GPU required')
        meta, arrays, ids, manifest_sha = load_capture(args.capture)
        if meta['vendor'] != vendor:
            raise ValueError('oracle must run on capture vendor')
        if arrays['initial_m'].any() or arrays['initial_v'].any():
            raise ValueError('single fresh step requires zero initial Adam moments')
        loss, grads = reference(arrays['initial_params'], ids)
        altered_loss, altered = reference(arrays['initial_params'], ids, nonlinear_sabotage=True)
        tensor_results, sign_results, nonlinear_results = {}, {}, {}
        for item in registry():
            name, start, count = item['name'], item['offset'], item['count']
            actual = arrays['gradients'][start:start+count].reshape(item['shape'])
            tensor_results[name] = comparison(actual, grads[name], GRAD_ATOL, GRAD_RTOL)
            sign_results[name] = comparison(-actual, grads[name], GRAD_ATOL, GRAD_RTOL)
            nonlinear_results[name] = comparison(altered[name], grads[name], GRAD_ATOL, GRAD_RTOL)
        loss_result = comparison([arrays['loss'][0]], [loss], LOSS_ATOL, LOSS_RTOL)
        sign_effective = any(not v['passed'] for v in sign_results.values())
        nonlinear_effective = any(not nonlinear_results[n]['passed'] for n in ('w_gate', 'norm2_w', 'embed'))
        # The nonlinear control must preserve forward values at the same gate.
        nonlinear_forward = comparison([altered_loss], [loss], LOSS_ATOL, LOSS_RTOL)
        moved = bool((arrays['initial_params'] != arrays['updated_params']).any())
        passed = (loss_result['passed'] and all(v['passed'] for v in tensor_results.values())
                  and sign_effective and nonlinear_effective and nonlinear_forward['passed'] and moved)
        import numpy as np
        reference_files = {}
        for label, values, value_loss in (('reference', grads, loss),
                                          ('nonlinear-control', altered, altered_loss)):
            buffer = io.BytesIO()
            np.savez(buffer, loss=np.asarray([value_loss], dtype='<f8'), **values)
            raw = buffer.getvalue()
            path = args.output.with_name(args.output.name + '.' + label + '.npz')
            exclusive(path, raw)
            reference_files[label] = dict(file=path.name, sha256=sha(raw))
        result = dict(schema='mojolearn.training.gradient-oracle.v1', passed=passed,
                      reference_files=reference_files,
                      scope='single fixed-profile step; FP64 tolerance correctness, not bitwise certification',
                      capture_manifest_sha256=manifest_sha, vendor=vendor,
                      torch_version=torch.__version__, cuda_version=torch.version.cuda,
                      hip_version=torch.version.hip, device=torch.cuda.get_device_name(0),
                      actual_loss=float(arrays['loss'][0]), reference_loss=loss,
                      tolerances=dict(grad_atol=GRAD_ATOL, grad_rtol=GRAD_RTOL,
                                      loss_atol=LOSS_ATOL, loss_rtol=LOSS_RTOL),
                      loss=loss_result, gradients=tensor_results, parameters_moved=moved,
                      controls=dict(sign_effective=sign_effective, sign=sign_results,
                          nonlinear_effective=nonlinear_effective, nonlinear=nonlinear_results,
                          nonlinear_forward=nonlinear_forward))
        output.write(json_bytes(result))
        output.flush()
        os.fsync(output.fileno())
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
