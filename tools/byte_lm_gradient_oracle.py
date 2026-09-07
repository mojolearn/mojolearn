#!/usr/bin/env python3
"""Independent two-block FP64 gradient and AdamW gate; ROOT EXECUTION ONLY.

Authored without executing tests, builds, models or measurements. Imports are
inert. Model evaluation requires a real CUDA/HIP device, never CPU or Metal.
External FP64 agreement is a tolerance-correctness claim, not bitwise identity.
No native model, gradient oracle or numerical arithmetic module is imported.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path

PROFILE = 'mojolearn.byte-lm.b2-l32-d32-h4-kv2-ff64-v256-blocks2.fp32.v1'
SCHEMA = 'mojolearn.byte-lm.gradient-capture.v1'
SHAPES = [('embed', (256, 32))]
for _block in range(2):
    SHAPES += [(f'block{_block}.{name}', shape) for name, shape in (
        ('norm1_w', (32,)), ('w_q', (32, 32)), ('w_k', (16, 32)),
        ('w_v', (16, 32)), ('w_o', (32, 32)), ('norm2_w', (32,)),
        ('w_gate', (64, 32)), ('w_up', (64, 32)), ('w_down', (32, 64)))]
SHAPES += [('lm_head', (256, 32))]
N = 34944
# Preset, provisional admission thresholds. Never inferred from observed errors.
TOLERANCES = {'gradient': (2e-6, 2e-4), 'loss': (2e-6, 2e-6),
              'post_p': (2e-6, 2e-5), 'post_m': (2e-7, 2e-5),
              'post_v': (2e-9, 2e-5)}
FLOAT_COUNTS = {name: N for name in ('initial_p', 'initial_m', 'initial_v',
                                    'grad', 'post_p', 'post_m', 'post_v')}
FLOAT_COUNTS['loss'] = 1
INT_COUNTS = {'ids': 66, 'initial_flags': 20, 'post_flags': 20}
OPT_FIELDS = {'kind', 'lr', 'beta1', 'beta2', 'eps', 'weight_decay',
              'momentum', 'dampening', 'nesterov', 'max_norm'}


def registry():
    entries, offset = [], 0
    for name, shape in SHAPES:
        size = math.prod(shape)
        entries.append(dict(name=name, shape=list(shape), offset=offset, count=size))
        offset += size
    return entries


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def _exclusive(path, raw):
    with Path(path).open('xb') as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _torch_device(expected=None):
    import torch
    vendor = 'hip' if torch.version.hip else 'cuda'
    if not torch.cuda.is_available() or (expected is not None and vendor != expected):
        raise ValueError('independent oracle requires expected remote NVIDIA/AMD GPU')
    return torch, vendor


def _validate_inputs(initial_params, ids):
    import numpy as np
    params, tokens = np.asarray(initial_params), np.asarray(ids)
    if params.dtype != np.dtype('float32') or params.shape != (N,) or not np.isfinite(params).all():
        raise ValueError('initial_params must be finite float32[34944]')
    if tokens.dtype != np.dtype('int32') or tokens.shape not in ((66,), (2, 33)):
        raise ValueError('ids must be int32[66] or int32[2,33]')
    tokens = tokens.reshape(2, 33)
    if (tokens < 0).any() or (tokens >= 256).any():
        raise ValueError('byte tokens must be in [0,256)')
    return params, tokens


def reference(initial_params, ids, *, wrong_silu_block=None):
    """Return (FP64 scalar loss, dict of 20 FP64 shaped parameter gradients).

    Mathematical transcription with independent PyTorch autograd. The optional
    control drops only the sigmoid derivative in one block while preserving
    that block's SiLU forward values; wrong_silu_block must be None, 0 or 1.
    """
    initial, ids = _validate_inputs(initial_params, ids)
    if wrong_silu_block not in (None, 0, 1):
        raise ValueError('invalid nonlinear control block')
    torch, _ = _torch_device()
    import torch.nn.functional as F
    weights = {}
    for entry in registry():
        start, end = entry['offset'], entry['offset'] + entry['count']
        weights[entry['name']] = torch.tensor(initial[start:end].reshape(entry['shape']),
            dtype=torch.float64, device='cuda', requires_grad=True)
    tokens = torch.tensor(ids, dtype=torch.long, device='cuda')
    h = F.embedding(tokens[:, :-1], weights['embed'])
    frequency = 10000.0 ** (-torch.arange(0, 8, 2, dtype=torch.float64, device='cuda') / 8)
    angle = torch.arange(32, dtype=torch.float64, device='cuda')[:, None] * frequency
    angle = torch.cat((angle, angle), dim=-1)[None, None]
    cosine, sine = angle.cos(), angle.sin()
    mask = torch.ones((32, 32), dtype=torch.bool, device='cuda').triu(1)
    def rotate(a):
        half = torch.cat((-a[..., 4:], a[..., :4]), dim=-1)
        return a * cosine + half * sine
    for block in range(2):
        prefix = f'block{block}.'
        def linear(x, name):
            return F.linear(x, weights[prefix + name])
        def norm(x, name):
            # Epsilon is the exact scalar that FP32 native configuration holds.
            eps = 9.999999974752427e-7
            return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weights[prefix + name]
        z = norm(h, 'norm1_w')
        q = linear(z, 'w_q').reshape(2, 32, 4, 8).transpose(1, 2)
        k = linear(z, 'w_k').reshape(2, 32, 2, 8).transpose(1, 2)
        v = linear(z, 'w_v').reshape(2, 32, 2, 8).transpose(1, 2)
        q, k = rotate(q), rotate(k)
        k, v = k.repeat_interleave(2, dim=1), v.repeat_interleave(2, dim=1)
        scores = q @ k.transpose(-1, -2) / math.sqrt(8)
        probability = scores.masked_fill(mask, -torch.inf).softmax(-1)
        attended = (probability @ v).transpose(1, 2).reshape(2, 32, 32)
        residual = h + linear(attended, 'w_o')
        z = norm(residual, 'norm2_w')
        gate = linear(z, 'w_gate')
        sigmoid = gate.sigmoid()
        activated = gate * (sigmoid.detach() if block == wrong_silu_block else sigmoid)
        h = residual + linear(activated * linear(z, 'w_up'), 'w_down')
    # No final RMSNorm, bias, tied head, dropout or cache carried between steps.
    logits = F.linear(h, weights['lm_head'])
    loss = F.cross_entropy(logits.reshape(64, 256), tokens[:, 1:].reshape(64), reduction='mean')
    loss.backward()
    torch.cuda.synchronize()
    value = float(loss.detach().cpu())
    gradients = {name: weight.grad.detach().cpu().numpy().copy() for name, weight in weights.items()}
    return value, gradients


def _validate_config(config):
    import numpy as np
    if (config.get('profile') != PROFILE or config.get('numeric_mode') != 'identical'
            or config.get('vendor') not in ('cuda', 'hip')):
        raise ValueError('wrong profile/mode/vendor')
    before, after = config.get('completed_steps'), config.get('post_completed_steps')
    if type(before) is not int or not 0 <= before < 999999 or type(after) is not int or after != before + 1:
        raise ValueError('capture must contain exactly one completed training step')
    opt = config.get('optimizer', {})
    if set(opt) != OPT_FIELDS or type(opt['kind']) is not int or opt['kind'] != 2 or opt['nesterov'] not in (False, 0):
        raise ValueError('expected explicit AdamW configuration')
    normalized = {'kind': 2, 'nesterov': False}
    for key in OPT_FIELDS - {'kind', 'nesterov'}:
        value = opt[key]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('nonfinite/non-numeric optimizer scalar')
        value = float(np.float32(value))
        if not math.isfinite(value):
            raise ValueError('optimizer scalar outside FP32 range')
        normalized[key] = value
    if (normalized['lr'] <= 0 or normalized['eps'] <= 0
            or not 0 <= normalized['beta1'] < 1 or not 0 <= normalized['beta2'] < 1
            or normalized['weight_decay'] < 0
            or any(normalized[key] != 0 for key in ('momentum', 'dampening', 'max_norm'))):
        raise ValueError('configuration outside fixed AdamW profile')
    return normalized


def adamw_reference(initial_p, initial_m, initial_v, actual_grad, optimizer, completed_steps):
    """Independent FP64 AdamW using ACTUAL captured FP32 gradients/prestate.

    Inputs/config must first pass evaluate_capture validation. FP64 mathematical
    operations intentionally do not reproduce native FP32 rounding or FMA seams.
    Decoupled decay, beta bias correction at completed_steps+1, eps outside sqrt.
    """
    torch, _ = _torch_device()
    p, m, v, g = [torch.tensor(x, dtype=torch.float64, device='cuda')
                   for x in (initial_p, initial_m, initial_v, actual_grad)]
    b1, b2, lr = optimizer['beta1'], optimizer['beta2'], optimizer['lr']
    step = completed_steps + 1
    new_m = b1 * m + (1.0 - b1) * g
    new_v = b2 * v + (1.0 - b2) * g.square()
    unbiased_m = new_m / (1.0 - b1 ** step)
    unbiased_v = new_v / (1.0 - b2 ** step)
    new_p = p * (1.0 - lr * optimizer['weight_decay']) - lr * unbiased_m / (unbiased_v.sqrt() + optimizer['eps'])
    torch.cuda.synchronize()
    return {key: value.detach().cpu().numpy().copy()
            for key, value in (('post_p', new_p), ('post_m', new_m), ('post_v', new_v))}


def _compare(actual, expected, tolerance):
    import numpy as np
    a, e = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    if a.shape != e.shape or not np.isfinite(a).all() or not np.isfinite(e).all():
        raise ValueError('reference/result shape mismatch or nonfinite value')
    atol, rtol = tolerance
    delta = np.abs(a - e)
    excess = delta - (atol + rtol * np.abs(e))
    failures = np.flatnonzero(excess.reshape(-1) > 0)
    return dict(passed=len(failures) == 0, cells=int(a.size), failed_cells=int(len(failures)),
                first_failure_flat=int(failures[0]) if len(failures) else None,
                max_absolute_error=float(delta.max()), max_excess=float(excess.max()))


def evaluate_capture(arrays, config, *, reference_sink=None):
    """Validate one full native step against independent FP64 math.

    arrays: float32 flat initial_p/initial_m/initial_v/grad/post_p/post_m/post_v
    (34944 each), float32 loss(1), int32 ids(66), initial_flags/post_flags(20).
    config: profile, numeric_mode, vendor, completed_steps, post_completed_steps,
    optimizer containing all OPT_FIELDS. Optional reference_sink is a NEW NPZ
    path for raw FP64 reference and control arrays. Returns a JSON-safe verdict.
    No subprocesses, provisioning, timing or cross-vendor comparisons occur here.
    """
    import numpy as np
    opt = _validate_config(config)
    if reference_sink is not None and os.path.lexists(reference_sink):
        raise FileExistsError('reference output already exists')
    if set(arrays) != set(FLOAT_COUNTS) | set(INT_COUNTS):
        raise ValueError('capture array set differs from schema')
    owned = {}
    for key, count in FLOAT_COUNTS.items():
        value = np.asarray(arrays[key])
        if value.dtype != np.dtype('float32') or value.shape != (count,) or not np.isfinite(value).all():
            raise ValueError(f'{key}: expected finite float32[{count}]')
        owned[key] = value.copy()
    for key, count in INT_COUNTS.items():
        value = np.asarray(arrays[key])
        if value.dtype != np.dtype('int32') or value.shape != (count,):
            raise ValueError(f'{key}: expected int32[{count}]')
        owned[key] = value.copy()
    _validate_inputs(owned['initial_p'], owned['ids'])
    if (owned['initial_v'] < 0).any() or (owned['post_v'] < 0).any():
        raise ValueError('negative Adam second moment')
    for key in ('initial_flags', 'post_flags'):
        if not np.isin(owned[key], (0, 1)).all():
            raise ValueError('flags must be exactly zero or one')
    torch, vendor = _torch_device(config['vendor'])
    loss, gradients = reference(owned['initial_p'], owned['ids'])
    gradient_results, sign_results = {}, {}
    reference_arrays = {'loss': np.asarray([loss], dtype='<f8')}
    for entry in registry():
        name, a, b = entry['name'], entry['offset'], entry['offset'] + entry['count']
        actual = owned['grad'][a:b].reshape(entry['shape'])
        gradient_results[name] = _compare(actual, gradients[name], TOLERANCES['gradient'])
        sign_results[name] = _compare(-actual, gradients[name], TOLERANCES['gradient'])
        reference_arrays['grad.' + name] = gradients[name]
    nonlinear_controls = {}
    for block in range(2):
        bad_loss, bad_gradients = reference(owned['initial_p'], owned['ids'], wrong_silu_block=block)
        forward = _compare(np.asarray([bad_loss]), np.asarray([loss]), TOLERANCES['loss'])
        details = {name: _compare(bad_gradients[name], gradients[name], TOLERANCES['gradient'])
                   for name, _ in SHAPES}
        # Require each altered block's own gate gradient to expose its missing
        # derivative, not merely an unrelated downstream difference.
        effective = not details[f'block{block}.w_gate']['passed']
        nonlinear_controls[str(block)] = dict(effective=effective, forward_unchanged=forward, gradients=details)
        for name, values in bad_gradients.items():
            reference_arrays[f'control{block}.' + name] = values
    updates = adamw_reference(owned['initial_p'], owned['initial_m'], owned['initial_v'],
                             owned['grad'], opt, config['completed_steps'])
    update_results = {}
    for key, values in updates.items():
        per_tensor = {}
        for entry in registry():
            a, b = entry['offset'], entry['offset'] + entry['count']
            per_tensor[entry['name']] = _compare(owned[key][a:b], values[a:b], TOLERANCES[key])
        update_results[key] = per_tensor
        reference_arrays[key] = values
    loss_result = _compare(owned['loss'], np.asarray([loss]), TOLERANCES['loss'])
    sign_effective = any(not value['passed'] for value in sign_results.values())
    nonlinear_effective = all(value['effective'] and value['forward_unchanged']['passed']
                              for value in nonlinear_controls.values())
    flags_same = bool(np.array_equal(owned['initial_flags'], owned['post_flags']))
    moved = bool(np.any(owned['initial_p'].view(np.uint32) != owned['post_p'].view(np.uint32)))
    passed = (loss_result['passed'] and all(v['passed'] for v in gradient_results.values())
              and all(v['passed'] for group in update_results.values() for v in group.values())
              and flags_same and moved and sign_effective and nonlinear_effective)
    result = dict(schema='mojolearn.byte-lm.gradient-oracle.v1', passed=passed,
        claim='one-step FP64 tolerance correctness; no external bitwise or learning claim',
        profile=PROFILE, vendor=vendor, torch_version=str(torch.__version__),
        input_array_sha256={key: _sha(value.astype('<f4' if key in FLOAT_COUNTS else '<i4').tobytes())
                            for key, value in owned.items()},
        oracle_source_sha256=_sha(Path(__file__).read_bytes()),
        cuda_version=torch.version.cuda, hip_version=torch.version.hip,
        device=torch.cuda.get_device_name(0), config=config, normalized_optimizer=opt,
        tolerances={k: dict(atol=v[0], rtol=v[1]) for k, v in TOLERANCES.items()},
        loss=loss_result, actual_loss=float(owned['loss'][0]), reference_loss=loss,
        gradients=gradient_results, optimizer_updates=update_results,
        flags_preserved=flags_same, parameters_moved=moved,
        controls=dict(sign_effective=sign_effective, sign=sign_results,
                      nonlinear_effective=nonlinear_effective, nonlinear=nonlinear_controls))
    if reference_sink is not None:
        stream = io.BytesIO()
        np.savez(stream, **reference_arrays)
        raw = stream.getvalue()
        _exclusive(reference_sink, raw)
        result['reference_file'] = dict(file=Path(reference_sink).name, sha256=_sha(raw))
    return result


def load_capture(directory):
    """Read fixed bounded files and manifest; never infer a missing descriptor."""
    import numpy as np
    directory = Path(directory)
    with (directory / 'capture.json').open('rb') as handle:
        raw = handle.read(1048577)
    if len(raw) > 1048576:
        raise ValueError('capture manifest exceeds one MiB')
    manifest = json.loads(raw)
    if manifest.get('schema') != SCHEMA or manifest.get('registry') != registry():
        raise ValueError('wrong capture schema/registry')
    arrays = {}
    expected = set(FLOAT_COUNTS) | set(INT_COUNTS)
    if set(manifest.get('arrays', {})) != expected:
        raise ValueError('wrong capture array set')
    for key in expected:
        floating = key in FLOAT_COUNTS
        count = FLOAT_COUNTS[key] if floating else INT_COUNTS[key]
        suffix, dtype = ('f32', '<f4') if floating else ('i32', '<i4')
        descriptor = manifest['arrays'][key]
        filename = key + '.' + suffix
        if descriptor.get('file') != filename or descriptor.get('count') != count:
            raise ValueError('wrong array descriptor')
        with (directory / filename).open('rb') as handle:
            body = handle.read(count * 4 + 1)
        if len(body) != count * 4 or _sha(body) != descriptor.get('sha256'):
            raise ValueError('wrong raw array length/hash')
        arrays[key] = np.frombuffer(body, dtype=dtype).copy()
    return arrays, manifest['config'], _sha(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--expected-vendor', choices=('cuda', 'hip'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Reserve primary output before model work. Failure leaves a new incomplete
    # artifact; it can never replace previous evidence or be read as a verdict.
    with args.output.open('xb') as output:
        arrays, config, capture_sha = load_capture(args.capture)
        if config.get('vendor') != args.expected_vendor:
            raise ValueError('capture vendor differs from expected vendor')
        torch, _ = _torch_device(args.expected_vendor)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        result = evaluate_capture(arrays, config,
                                  reference_sink=args.output.with_name(args.output.name + '.reference.npz'))
        result['capture_manifest_sha256'] = capture_sha
        result['oracle_source_sha256'] = _sha(Path(__file__).read_bytes())
        output.write(_json_bytes(result))
        output.flush()
        os.fsync(output.fileno())
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
