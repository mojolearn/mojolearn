#!/usr/bin/env python3
"""Root-only B2/L8/D32 Mamba2/3 VJP fixture, authored but not executed.

Requires MOJOLEARN_MAMBA23_SHARED_SHAPE=1 and IDENTICAL process mode on Linux
NVIDIA/AMD. Uses the nine existing constructor weight files per family from
mamba/corpus/mamba{2,3}/m{2,3}_base_b2_l4_d32; no generated ref32/ref64 corpus
or full generation run is needed. This is a NEW L8 case, not promotion of L4
proofs. All20 leaves (input+9weights per family) are independently gated.
No provisioning, subprocess, compiler, timing or automatic comparison job.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ATOL, RTOL = 1e-6, 1e-5  # Same declared FP64 thresholds as the existing L4 gate.
PROFILE = 'mojolearn.mamba23.shared-b2-l8-d32.backward.v1'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def exclusive(path, raw):
    with Path(path).open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def weight_shapes(family):
    # The two families have DIFFERENT projections and normalization layouts.
    common = [('block_norm.weight', (32,))]
    if family == 2:
        return common + [('in_proj.weight', (385, 32)), ('conv1d.weight', (320, 1, 4)),
            ('conv1d.bias', (320,)), ('dt_bias', (1,)), ('A_log', (1,)),
            ('D', (1,)), ('norm.weight', (64,)), ('out_proj.weight', (32, 64))]
    if family == 3:
        return common + [('in_proj.weight', (419, 32)), ('dt_bias', (1,)),
            ('B_norm.weight', (128,)), ('C_norm.weight', (128,)),
            ('B_bias', (1, 128)), ('C_bias', (1, 128)), ('D', (1,)),
            ('out_proj.weight', (32, 64))]
    raise ValueError('unknown Mamba family')


def fixture(family, corpus):
    import numpy as np
    directory = corpus / f'mamba{family}' / f'm{family}_base_b2_l4_d32'
    weights, witnesses = {}, {}
    for name, shape in weight_shapes(family):
        path = directory / (name + '.f32')
        count = math.prod(shape)
        with path.open('rb') as stream:
            raw = stream.read(count * 4 + 1)
        if len(raw) != count * 4:
            raise ValueError(f'wrong fixture byte count: {path}')
        value = np.frombuffer(raw, dtype='<f4').reshape(shape).copy()
        if not np.isfinite(value).all():
            raise ValueError('nonfinite constructor fixture: ' + name)
        weights[name] = value
        witnesses[name] = dict(source=str(path), shape=list(shape), sha256=sha(raw))
    index = np.arange(512, dtype=np.int32).reshape(2, 8, 32)
    x = (((index * 17 + 23) % 127 - 63) / 64.0).astype(np.float32)
    dy = (((index * 19 + 5) % 43 - 21) / 32.0).astype(np.float32)
    return weights, x, dy, witnesses


def retain(directory, name, value, dtype):
    import numpy as np
    value = np.asarray(value)
    raw = np.asarray(value, dtype=dtype, order='C').tobytes()
    path = directory / name
    exclusive(path, raw)
    return dict(file=name, dtype=dtype, shape=list(value.shape), cells=int(value.size), sha256=sha(raw))


def comparison(actual, expected):
    import numpy as np
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('nonfinite/incorrectly shaped gradient')
    error = np.abs(actual.astype(np.float64) - expected)
    excess = error - (ATOL + RTOL * np.abs(expected))
    bad = np.flatnonzero(excess.reshape(-1) > 0)
    return dict(passed=len(bad) == 0, cells=int(actual.size), failed_cells=int(len(bad)),
        first_failure_flat=int(bad[0]) if len(bad) else None,
        max_absolute_error=float(error.max()), max_excess=float(excess.max()))


def family_capture(family, cls, torch, gen, corpus, output, vendor):
    import numpy as np
    directory = output / f'mamba{family}'
    directory.mkdir()
    weights, x, dy, fixture_files = fixture(family, corpus)
    expected_names = ('x', *weights)
    block = cls(weights, numeric_mode='identical')
    native = block._extension()
    if str(native.mamba_vendor()) != vendor or int(native.mamba_numeric_mode()) != 1:
        raise ValueError('loaded native Mamba binary mode/vendor mismatch')
    if tuple(block._W_NAMES) != tuple(weights):
        raise ValueError('public constructor order differs from fixed family registry')
    binding_sha = file_sha(native.__file__)
    input_bytes = {'x': x.tobytes(), 'dy': dy.tobytes(), **{k: v.tobytes() for k, v in weights.items()}}
    inputs = {name: retain(directory, 'input.' + name + '.f32', value, '<f4')
              for name, value in {'x': x, 'dy': dy, **weights}.items()}
    # Independent external FP64 graph is evaluated entirely on the requested
    # CUDA/HIP accelerator. The transcription creates some states without an
    # explicit device argument, so the default device is scoped as well.
    with torch.device('cuda'):
        tx = torch.tensor(x, dtype=torch.float64, device='cuda', requires_grad=True)
        params = {name: torch.tensor(value, dtype=torch.float64, device='cuda', requires_grad=True)
                  for name, value in weights.items()}
        forward = gen.m2_forward if family == 2 else gen.m3_forward
        stages = forward(params, tx, torch.float64)
        y = stages['residual.out'].reshape(tx.shape)
        if y.device.type != 'cuda' or y.dtype != torch.float64:
            raise ValueError('external reference did not stay FP64 on CUDA/HIP')
        objective = (y * torch.tensor(dy, dtype=torch.float64, device='cuda')).sum()
        tensor_grads = torch.autograd.grad(objective, [tx] + list(params.values()))
    torch.cuda.synchronize()
    objective_value = float(objective.detach().cpu())
    refs = {name: value.detach().cpu().numpy().copy() for name, value in zip(expected_names, tensor_grads)}
    if not math.isfinite(objective_value):
        raise ValueError('nonfinite FP64 objective')
    # Drop external graph/activations before native calls; no concurrent model
    # execution and no retained GPU reference graph while Mojo allocates state.
    del tensor_grads, stages, params, tx, y, objective
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    actual = block.backward(x, dy)
    repeated = block.backward(x, dy)
    zero = block.backward(x, np.zeros_like(dy))
    if tuple(actual) != expected_names or tuple(repeated) != expected_names or tuple(zero) != expected_names:
        raise ValueError('native result has missing/reordered/extra gradient leaves')
    after = {'x': x.tobytes(), 'dy': dy.tobytes(), **{k: v.tobytes() for k, v in weights.items()}}
    if after != input_bytes:
        raise ValueError('native call changed a caller-owned input')
    leaves, sign_effective = {}, False
    for name in expected_names:
        value, again, expected = actual[name], repeated[name], refs[name]
        template = x if name == 'x' else weights[name]
        if value.dtype != np.float32 or again.dtype != np.float32 or zero[name].dtype != np.float32:
            raise ValueError('native gradient dtype differs from FP32')
        if not value.flags.c_contiguous or not again.flags.c_contiguous or not zero[name].flags.c_contiguous:
            raise ValueError('native gradients must be contiguous owned arrays')
        if value.shape != template.shape or again.shape != template.shape or zero[name].shape != template.shape:
            raise ValueError('native gradient layout differs from its public parameter/input')
        gate = comparison(value, expected)
        sign = comparison(-value, expected)
        sign_effective |= not sign['passed']
        repeat_equal = value.tobytes() == again.tobytes()
        zero_cotangent_zero = bool(np.isfinite(zero[name]).all() and np.all(zero[name] == 0))
        leaves[name] = dict(reference_gate=gate, repeat_bits_equal=repeat_equal,
            zero_cotangent_zero=zero_cotangent_zero, sign_control=sign,
            actual=retain(directory, 'grad.' + name + '.f32', value, '<f4'),
            repeated=retain(directory, 'repeat.' + name + '.f32', again, '<f4'),
            zero=retain(directory, 'zero.' + name + '.f32', zero[name], '<f4'),
            reference=retain(directory, 'reference.' + name + '.f64', expected, '<f8'))
    passed = (len(leaves) == 10 and sign_effective and all(
        item['reference_gate']['passed'] and item['repeat_bits_equal'] and item['zero_cotangent_zero']
        for item in leaves.values()))
    if file_sha(native.__file__) != binding_sha:
        raise ValueError('native binding artifact changed during capture')
    result = dict(schema='mojolearn.mamba.shared-shape-backward.family.v1', passed=passed,
        family=family, profile=PROFILE, shape=[2, 8, 32], constructor_weight_count=9,
        gradient_leaf_count=10, native_mode='identical', native_vendor=vendor,
        binding_file=native.__file__, binding_sha256=binding_sha,
        fixture_weight_sources=fixture_files, inputs=inputs, leaves=leaves,
        dt_limit=[0.0, 'infinity'] if family == 2 else None,
        objective='sum(residual.out * retained arbitrary cotangent)', objective_fp64=objective_value,
        state_scope='zero-state prefill; no recurrent-state cotangent or decode backward',
        tolerances=dict(atol=ATOL, rtol=RTOL), sign_control_effective=sign_effective)
    exclusive(directory / 'capture.json', json_bytes(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-vendor', choices=('cuda', 'hip'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, default=ROOT / 'mamba/corpus')
    args = parser.parse_args()
    if (sys.platform != 'linux' or os.environ.get('MOJOLEARN_MAMBA23_SHARED_SHAPE') != '1'
            or os.environ.get('MOJOLEARN_NUMERIC_MODE') != 'identical'):
        raise ValueError('explicit shared-shape opt-in, IDENTICAL, and remote Linux required')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '2'
    args.output.mkdir(parents=False, exist_ok=False)
    import torch
    if not torch.cuda.is_available():
        raise ValueError('CUDA/HIP GPU required; CPU/Metal oracle execution is forbidden')
    vendor = 'hip' if torch.version.hip is not None else 'cuda'
    if vendor != args.expected_vendor:
        raise ValueError('PyTorch vendor differs from expected remote vendor')
    from mojolearn import Mamba2Block, Mamba3Block
    generator = ROOT / 'mamba/corpus/gen_corpus.py'
    original_determinism = torch.are_deterministic_algorithms_enabled()
    spec = importlib.util.spec_from_file_location('mamba23_shared_shape_reference', generator)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    # External FP64 tolerance math need not imitate native pinned reduction
    # order; CUDA cumsum is refused by PyTorch's deterministic=True mode.
    torch.use_deterministic_algorithms(False)
    sources = {str(path.relative_to(ROOT)): file_sha(path) for path in (
        Path(__file__).resolve(), generator, ROOT / 'python/mojolearn/_mamba_impl.py',
        ROOT / 'bindings/_mojolearn_mamba.mojo')}
    try:
        results = [family_capture(family, cls, torch, gen, args.corpus, args.output, vendor)
                   for family, cls in ((2, Mamba2Block), (3, Mamba3Block))]
    finally:
        torch.use_deterministic_algorithms(original_determinism)
    if any(file_sha(ROOT / name) != digest for name, digest in sources.items()):
        raise ValueError('reference/wrapper source changed during capture')
    passed = all(result['passed'] for result in results)
    summary = dict(schema='mojolearn.mamba23.shared-shape-backward.v1', passed=passed,
        profile=PROFILE, shape=[2, 8, 32], gradient_leaf_count=20, vendor=vendor,
        torch_version=str(torch.__version__), cuda_version=torch.version.cuda, hip_version=torch.version.hip,
        device=torch.cuda.get_device_name(0), cpu_threads=2, source_sha256=sources,
        families=[dict(family=r['family'], passed=r['passed'],
                       capture=f"mamba{r['family']}/capture.json",
                       capture_sha256=sha(json_bytes(r))) for r in results],
        scope='new B2/L8/D32 zero-state public VJPs; native repeat bits and external FP64 tolerance correctness',
        external_bitwise_claim=False, cross_vendor_bitwise_claim='NOT_ADMITTED; root must compare retained raw arrays',
        guard_exit_evidence='REQUIRED_EXTERNALLY; summary alone does not prove successful teardown')
    exclusive(args.output / 'summary.json', json_bytes(summary))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
