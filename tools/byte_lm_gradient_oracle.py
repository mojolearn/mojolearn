#!/usr/bin/env python3
"""Independent two-block FP64 gradient and AdamW gate; ROOT EXECUTION ONLY.

Imports are inert. The bounded runtime-shape gate records actual native Metal
and independent CPU FP64 checks in bench/results/byte_runtime_numerical_2026-09-10. Existing capture evaluation requires CUDA/HIP. Explicit oracle_device=cpu
on reference/adamw_reference supports FP64 comparison of native Metal captures.
External FP64 agreement is a tolerance-correctness claim, not bitwise identity.
No native model, gradient oracle or numerical arithmetic module is imported.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path

# DEVIATION 2682. The shape is no longer literals here, because identity is
# claimed per shape and a shape spelled out in two places is eventually two
# shapes. Loaded by path so this file never depends on tools/ being on sys.path,
# the way the sibling comparator loads its receipt policy. The helper is pure
# Python and imports nothing native, so the independence claimed below stands.
_shape_spec = importlib.util.spec_from_file_location(
    '_byte_lm_shape', Path(__file__).with_name('byte_lm_shape.py'))
byte_lm_shape = importlib.util.module_from_spec(_shape_spec)
_shape_spec.loader.exec_module(byte_lm_shape)

#: The certified default. A capture written before DEVIATION 2682 records no
#: shape and is this one by construction, since it was the only one that existed.
DEFAULT_SHAPE = byte_lm_shape.Shape()
PROFILE = DEFAULT_SHAPE.profile
SCHEMA = 'mojolearn.byte-lm.gradient-capture.v1'
SHAPES = [(entry['name'], tuple(entry['shape'])) for entry in DEFAULT_SHAPE.registry()]
N = DEFAULT_SHAPE.n_total
# Preset, provisional admission thresholds. Never inferred from observed errors.
TOLERANCES = {'gradient': (2e-6, 2e-4), 'loss': (2e-6, 2e-6),
              'post_p': (2e-6, 2e-5), 'post_m': (2e-7, 2e-5),
              'post_v': (2e-9, 2e-5)}
#: The recorded arrays by dtype, in the order this file has always read them.
FLOAT_KEYS = ('initial_p', 'initial_m', 'initial_v', 'grad', 'post_p', 'post_m',
              'post_v', 'loss')
INT_KEYS = ('ids', 'initial_flags', 'post_flags')


def _split_counts(shape):
    """One step's array element counts, split the way they are read."""
    counts = shape.counts()
    return ({key: counts[key] for key in FLOAT_KEYS},
            {key: counts[key] for key in INT_KEYS})


FLOAT_COUNTS, INT_COUNTS = _split_counts(DEFAULT_SHAPE)
OPT_FIELDS = {'kind', 'lr', 'beta1', 'beta2', 'eps', 'weight_decay',
              'momentum', 'dampening', 'nesterov', 'max_norm'}


def model_dimensions(model_shape=None):
    """Independent shape admission; deliberately imports no native registry."""
    fields = (2, 32, 32, 4, 2, 8, 64) if model_shape is None else tuple(model_shape)
    if len(fields) == 7:
        fields += (2, 256)
    if len(fields) != 9 or any(type(x) is not int or not 0 < x <= 1048576 for x in fields):
        raise ValueError('expected nine positive bounded integer model dimensions')
    batch, length, dm, heads, kv, hd, ff, layers, vocab = fields
    if length > 8192 or dm != heads * hd or heads % kv or hd % 2:
        raise ValueError('invalid length, head dimensions or GQA ratio')
    if max(batch * length * vocab, batch * length * dm,
           batch * length * ff, batch * heads * length * length) >= 2**31:
        raise ValueError('model extent exceeds int32 indexing')
    total = 2 * vocab * dm + layers * (2 * dm + 2 * dm * dm + 2 * dm * kv * hd + 3 * dm * ff)
    if total >= 2**31:
        raise ValueError('parameter registry exceeds int32 indexing')
    return fields


def registry(model_shape=None):
    fields = model_dimensions(model_shape)
    _, _, dm, heads, kv, hd, ff, layers, vocab = fields
    shapes = [('embed', (vocab, dm))]
    for block in range(layers):
        shapes += [(f'block{block}.{name}', shape) for name, shape in (
            ('norm1_w', (dm,)), ('w_q', (heads * hd, dm)),
            ('w_k', (kv * hd, dm)), ('w_v', (kv * hd, dm)),
            ('w_o', (dm, heads * hd)), ('norm2_w', (dm,)),
            ('w_gate', (ff, dm)), ('w_up', (ff, dm)), ('w_down', (dm, ff)))]
    shapes += [('lm_head', (vocab, dm))]
    entries, offset = [], 0
    for name, shape in shapes:
        size = math.prod(shape)
        entries.append(dict(name=name, shape=list(shape), offset=offset, count=size))
        offset += size
    # DEVIATION 2682. The transcription stays local because a capture's own
    # registry is admitted against it, and deriving both sides of that check
    # from one module would only confirm that module. Agreement with the shared
    # shape is still required, so the two cannot drift apart in silence.
    if entries != byte_lm_shape.Shape(fields).registry():
        raise ValueError('independent registry differs from the shared shape registry')
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


def _torch_device(expected=None, device="cuda"):
    import torch
    if device == "cpu" and expected is None:
        return torch, "cpu"
    if device != "cuda":
        raise ValueError("oracle device must be cpu or cuda")
    vendor = 'hip' if torch.version.hip else 'cuda'
    if not torch.cuda.is_available() or (expected is not None and vendor != expected):
        raise ValueError('independent oracle requires expected remote NVIDIA/AMD GPU')
    return torch, vendor


def _validate_inputs(initial_params, ids, model_shape=None):
    import numpy as np
    batch, length, *_ = model_dimensions(model_shape)
    entries = registry(model_shape)
    count = entries[-1]["offset"] + entries[-1]["count"]
    params, tokens = np.asarray(initial_params), np.asarray(ids)
    if params.dtype != np.dtype('float32') or params.shape != (count,) or not np.isfinite(params).all():
        raise ValueError(f'initial_params must be finite float32[{count}]')
    if tokens.dtype != np.dtype('int32') or tokens.shape not in ((batch * (length + 1),), (batch, length + 1)):
        raise ValueError('ids must match configured int32[batch,length+1]')
    tokens = tokens.reshape(batch, length + 1)
    if (tokens < 0).any() or (tokens >= model_dimensions(model_shape)[8]).any():
        raise ValueError('byte tokens must be in [0,256)')
    return params, tokens


def reference(initial_params, ids, *, wrong_silu_block=None, model_shape=None, oracle_device="cuda"):
    """Return (FP64 scalar loss, dict of configured FP64 shaped parameter gradients).

    Mathematical transcription with independent PyTorch autograd. The optional
    control drops only the sigmoid derivative in one block while preserving
    that block's SiLU forward values; wrong_silu_block must be None or an existing layer index.
    """
    batch, length, dm, heads, kv, hd, _, layers, vocab = model_dimensions(model_shape)
    initial, ids = _validate_inputs(initial_params, ids, model_shape)
    if wrong_silu_block is not None and (type(wrong_silu_block) is not int or not 0 <= wrong_silu_block < layers):
        raise ValueError('invalid nonlinear control block')
    torch, _ = _torch_device(device=oracle_device)
    import torch.nn.functional as F
    weights = {}
    for entry in registry(model_shape):
        start, end = entry['offset'], entry['offset'] + entry['count']
        weights[entry['name']] = torch.tensor(initial[start:end].reshape(entry['shape']),
            dtype=torch.float64, device=oracle_device, requires_grad=True)
    tokens = torch.tensor(ids, dtype=torch.long, device=oracle_device)
    h = F.embedding(tokens[:, :-1], weights['embed'])
    frequency = 10000.0 ** (-torch.arange(0, hd, 2, dtype=torch.float64, device=oracle_device) / hd)
    angle = torch.arange(length, dtype=torch.float64, device=oracle_device)[:, None] * frequency
    angle = torch.cat((angle, angle), dim=-1)[None, None]
    cosine, sine = angle.cos(), angle.sin()
    mask = torch.ones((length, length), dtype=torch.bool, device=oracle_device).triu(1)
    def rotate(a):
        half = torch.cat((-a[..., hd // 2:], a[..., :hd // 2]), dim=-1)
        return a * cosine + half * sine
    for block in range(layers):
        prefix = f'block{block}.'
        def linear(x, name):
            return F.linear(x, weights[prefix + name])
        def norm(x, name):
            # Epsilon is the exact scalar that FP32 native configuration holds.
            eps = 9.999999974752427e-7
            return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weights[prefix + name]
        z = norm(h, 'norm1_w')
        q = linear(z, 'w_q').reshape(batch, length, heads, hd).transpose(1, 2)
        k = linear(z, 'w_k').reshape(batch, length, kv, hd).transpose(1, 2)
        v = linear(z, 'w_v').reshape(batch, length, kv, hd).transpose(1, 2)
        q, k = rotate(q), rotate(k)
        k, v = k.repeat_interleave(heads // kv, dim=1), v.repeat_interleave(heads // kv, dim=1)
        scores = q @ k.transpose(-1, -2) / math.sqrt(hd)
        probability = scores.masked_fill(mask, -torch.inf).softmax(-1)
        attended = (probability @ v).transpose(1, 2).reshape(batch, length, dm)
        residual = h + linear(attended, 'w_o')
        z = norm(residual, 'norm2_w')
        gate = linear(z, 'w_gate')
        sigmoid = gate.sigmoid()
        activated = gate * (sigmoid.detach() if block == wrong_silu_block else sigmoid)
        h = residual + linear(activated * linear(z, 'w_up'), 'w_down')
    # No final RMSNorm, bias, tied head, dropout or cache carried between steps.
    logits = F.linear(h, weights['lm_head'])
    loss = F.cross_entropy(logits.reshape(batch * length, vocab), tokens[:, 1:].reshape(batch * length), reduction='mean')
    loss.backward()
    if oracle_device == "cuda":
        torch.cuda.synchronize()
    value = float(loss.detach().cpu())
    gradients = {name: weight.grad.detach().cpu().numpy().copy() for name, weight in weights.items()}
    return value, gradients


def _validate_config(config, shape=None):
    import numpy as np
    shape = DEFAULT_SHAPE if shape is None else shape
    if (config.get('profile') != shape.profile or config.get('numeric_mode') != 'identical'
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


def adamw_reference(initial_p, initial_m, initial_v, actual_grad, optimizer, completed_steps, *, oracle_device="cuda"):
    """Independent FP64 AdamW using ACTUAL captured FP32 gradients/prestate.

    Inputs/config must first pass evaluate_capture validation. FP64 mathematical
    operations intentionally do not reproduce native FP32 rounding or FMA seams.
    Decoupled decay, beta bias correction at completed_steps+1, eps outside sqrt.
    """
    torch, _ = _torch_device(device=oracle_device)
    p, m, v, g = [torch.tensor(x, dtype=torch.float64, device=oracle_device)
                   for x in (initial_p, initial_m, initial_v, actual_grad)]
    b1, b2, lr = optimizer['beta1'], optimizer['beta2'], optimizer['lr']
    step = completed_steps + 1
    new_m = b1 * m + (1.0 - b1) * g
    new_v = b2 * v + (1.0 - b2) * g.square()
    unbiased_m = new_m / (1.0 - b1 ** step)
    unbiased_v = new_v / (1.0 - b2 ** step)
    new_p = p * (1.0 - lr * optimizer['weight_decay']) - lr * unbiased_m / (unbiased_v.sqrt() + optimizer['eps'])
    if oracle_device == "cuda":
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

    arrays: float32 flat initial_p/initial_m/initial_v/grad/post_p/post_m/post_v,
    float32 loss(1), int32 ids, initial_flags/post_flags(20). Every count comes
    from the shape the capture's own config records.
    config: profile, numeric_mode, vendor, completed_steps, post_completed_steps,
    optimizer containing all OPT_FIELDS, and since DEVIATION 2682 an optional
    model_shape. Optional reference_sink is a NEW NPZ path for raw FP64
    reference and control arrays. Returns a JSON-safe verdict.
    No subprocesses, provisioning, timing or cross-vendor comparisons occur here.
    """
    import numpy as np
    # The shape is read from the capture under evaluation, never assumed, so the
    # model checked here is the one the capture says it holds.
    shape = byte_lm_shape.from_capture(config)
    float_counts, int_counts = _split_counts(shape)
    opt = _validate_config(config, shape)
    if reference_sink is not None and os.path.lexists(reference_sink):
        raise FileExistsError('reference output already exists')
    if set(arrays) != set(float_counts) | set(int_counts):
        raise ValueError('capture array set differs from schema')
    owned = {}
    for key, count in float_counts.items():
        value = np.asarray(arrays[key])
        if value.dtype != np.dtype('float32') or value.shape != (count,) or not np.isfinite(value).all():
            raise ValueError(f'{key}: expected finite float32[{count}]')
        owned[key] = value.copy()
    for key, count in int_counts.items():
        value = np.asarray(arrays[key])
        if value.dtype != np.dtype('int32') or value.shape != (count,):
            raise ValueError(f'{key}: expected int32[{count}]')
        owned[key] = value.copy()
    _validate_inputs(owned['initial_p'], owned['ids'], shape.fields)
    if (owned['initial_v'] < 0).any() or (owned['post_v'] < 0).any():
        raise ValueError('negative Adam second moment')
    for key in ('initial_flags', 'post_flags'):
        if not np.isin(owned[key], (0, 1)).all():
            raise ValueError('flags must be exactly zero or one')
    torch, vendor = _torch_device(config['vendor'])
    loss, gradients = reference(owned['initial_p'], owned['ids'], model_shape=shape.fields)
    gradient_results, sign_results = {}, {}
    reference_arrays = {'loss': np.asarray([loss], dtype='<f8')}
    entries = registry(shape.fields)
    for entry in entries:
        name, a, b = entry['name'], entry['offset'], entry['offset'] + entry['count']
        actual = owned['grad'][a:b].reshape(entry['shape'])
        gradient_results[name] = _compare(actual, gradients[name], TOLERANCES['gradient'])
        sign_results[name] = _compare(-actual, gradients[name], TOLERANCES['gradient'])
        reference_arrays['grad.' + name] = gradients[name]
    nonlinear_controls = {}
    for block in range(shape.n_layers):
        bad_loss, bad_gradients = reference(owned['initial_p'], owned['ids'],
                                            wrong_silu_block=block, model_shape=shape.fields)
        forward = _compare(np.asarray([bad_loss]), np.asarray([loss]), TOLERANCES['loss'])
        details = {entry['name']: _compare(bad_gradients[entry['name']], gradients[entry['name']],
                                           TOLERANCES['gradient']) for entry in entries}
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
        for entry in entries:
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
        profile=shape.profile, vendor=vendor, torch_version=str(torch.__version__),
        input_array_sha256={key: _sha(value.astype('<f4' if key in float_counts else '<i4').tobytes())
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
    if manifest.get('schema') != SCHEMA:
        raise ValueError('wrong capture schema/registry')
    # The registry and the array lengths follow from the shape the capture
    # records, so a second shape is never read with the first one's lengths.
    shape = byte_lm_shape.from_capture(manifest.get('config'))
    float_counts, int_counts = _split_counts(shape)
    if manifest.get('registry') != registry(shape.fields):
        raise ValueError('wrong capture schema/registry')
    arrays = {}
    expected = set(float_counts) | set(int_counts)
    if set(manifest.get('arrays', {})) != expected:
        raise ValueError('wrong capture array set')
    for key in expected:
        floating = key in float_counts
        count = float_counts[key] if floating else int_counts[key]
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
    parser.add_argument('--shape', default=None,
                        help='batch,length or the nine dimensions; must agree with the capture')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Reserve primary output before model work. Failure leaves a new incomplete
    # artifact; it can never replace previous evidence or be read as a verdict.
    with args.output.open('xb') as output:
        arrays, config, capture_sha = load_capture(args.capture)
        if config.get('vendor') != args.expected_vendor:
            raise ValueError('capture vendor differs from expected vendor')
        # DEVIATION 2682. --shape states which shape the caller believes it
        # fetched. It never reinterprets the capture, so a disagreement is
        # refused by name rather than resolved in either direction.
        if args.shape is not None:
            wanted = byte_lm_shape.parse(args.shape)
            actual = byte_lm_shape.from_capture(config)
            if wanted != actual:
                raise ValueError(f'--shape {args.shape} is {wanted.profile}; '
                                 f'the capture is {actual.profile}')
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
