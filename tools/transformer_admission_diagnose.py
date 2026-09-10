#!/usr/bin/env python3
"""Numerical-only original-grid comparison in separate MAX/Torch processes.

Pass the original mojolearn-grid/tools/speed_torch_seq.py as --spec.
This script records no timings and does not change benchmark tolerances.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np


def load_spec(path):
    module_spec = importlib.util.spec_from_file_location('admission_original_spec', path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def report(label, ours, ref):
    if ours.shape != ref.shape or ours.ndim != 3 or ours.size == 0:
        raise ValueError('admission report requires equal nonempty [B,L,D] arrays')
    if not (np.isfinite(ours).all() and np.isfinite(ref).all()):
        raise ValueError('admission report refuses nonfinite outputs')
    diff = np.abs(ours.astype(np.float64) - ref.astype(np.float64))
    allowed = 1e-5 + 5e-4 * np.abs(ref.astype(np.float64))
    where = np.unravel_index(np.argmax(diff), diff.shape)
    out = {'variant': label, 'passed': bool(np.all(diff <= allowed)),
           'outside': int(np.count_nonzero(diff > allowed)), 'total': int(diff.size),
           'rtol': 5e-4, 'atol': 1e-5,
           'max_tolerance_multiple': float(np.max(diff / allowed)),
           'max_abs': float(diff.max()), 'rms_abs': float(np.sqrt(np.mean(diff * diff))),
           'max_index': list(map(int, where)), 'ours_at_max': float(ours[where]),
           'ref_at_max': float(ref[where]),
           'position_max_abs': {str(p): float(diff[:, p, :].max())
                                for p in [0, 1, 127, 511, 1023, 2047, 4095]
                                if p < ours.shape[1]}}
    print(json.dumps(out, sort_keys=True), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', required=True)
    ap.add_argument('--shape', choices=['narrow', 'wide'], required=True)
    ap.add_argument('--arm', choices=['ours', 'reference'], required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--rope-log', help='production inverse-frequency bit export')
    ap.add_argument('--reference64', action='store_true')
    ap.add_argument('--full-rope-log', help='production full-table export for positions0..L-1')
    ap.add_argument('--stage-errors', action='store_true',
                    help='numerical-only cumulative and same-input local FP64 stage errors; requires --reference64')
    args = ap.parse_args()
    if args.stage_errors and (args.arm != 'reference' or not args.reference64):
        ap.error('--stage-errors requires --arm reference --reference64')
    spec = load_spec(args.spec)
    row = next(r for r in spec.py_rows('llama') if r['name'].startswith(args.shape + '.'))
    b, l, dm = row['b'], row['l'], row['d_model']
    weights = {name: spec.hashed_tensor_numpy(row['seed'], spec.LLAMA_TIDS[name], n, lo, hi).reshape(shape)
               for name, n, lo, hi, shape in spec.llama_spec(row)}
    x = spec.hashed_tensor_numpy(row['seed'], spec.LLAMA_TIDS['x'], b*l*dm, -2.0, 2.0).reshape(b, l, dm)
    print(json.dumps({'spec_sha256': hashlib.sha256(Path(args.spec).read_bytes()).hexdigest(),
                      'shape': row, 'input_sha256': hashlib.sha256(x.tobytes()).hexdigest(),
                      'weights_sha256': {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in weights.items()}}), flush=True)
    if args.arm == 'ours':
        from mojolearn.transformer import TransformerBlock
        api_names = {'norm1.weight': 'input_layernorm.weight', 'norm2.weight': 'post_attention_layernorm.weight'}
        block = TransformerBlock({api_names.get(k, k): v for k, v in weights.items()},
                                 n_heads=row['n_heads'], n_kv_heads=row['n_kv'], head_dim=row['head_dim'])
        result = block.forward(x)
        digest = hashlib.sha256(result.tobytes()).hexdigest()
        expected = {'narrow': '645a35c194d82beb8a49ff50050a67d7492af34de27cc99d8077a39a516bc37d',
                    'wide': '6fe08dee3ef2a0ad769fde1803098789abd0ca4b112af93599e86e109a144532'}
        assert digest == expected[args.shape], ('own output moved from stored baseline', digest)
        np.save(args.output, result)
        print(json.dumps({'output_sha256': digest, 'stored_own_baseline_equal': True}), flush=True)
        return
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device('cuda')
    print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda,
                      'gpu': torch.cuda.get_device_name(), 'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
                      'cudnn_tf32': torch.backends.cudnn.allow_tf32,
                      'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                      'float32_matmul_precision': torch.get_float32_matmul_precision()}), flush=True)
    ours = np.load(args.output)
    W = {k: torch.from_numpy(v).to(dev) for k, v in weights.items()}
    xt = torch.from_numpy(x.reshape(b*l, dm)).to(dev)
    with torch.inference_mode():
        model = spec.LlamaEager(torch, dev, row, W, torch.float32)
        original = model.block(xt, None, b, l)[0].reshape(b, l, dm).cpu().numpy()
        report('original_fp32', ours, original)
        if args.rope_log:
            inv_words = {}
            for line in Path(args.rope_log).read_text().splitlines():
                if line.startswith('ROPE '):
                    hd, p, i, inv, _, _ = map(int, line.split()[1:])
                    if hd == row['head_dim'] and p == 0:
                        inv_words[i] = inv
            assert sorted(inv_words) == list(range(row['head_dim']//2)), 'incomplete inverse-frequency export'
            inv = np.asarray([inv_words[i] for i in sorted(inv_words)], dtype=np.uint32).view(np.float32)
            angle = np.arange(l, dtype=np.float32)[:, None] * inv[None, :]
            for name, fn in [('cos', np.cos), ('sin', np.sin)]:
                t = torch.from_numpy(fn(angle)).to(dev)
                setattr(model, name, torch.cat((t, t), dim=-1))
            aligned = model.block(xt, None, b, l)[0].reshape(b, l, dm).cpu().numpy()
            report('production_inv_fp32', ours, aligned)
            report('inverse_change_vs_original', aligned, original)
        if args.full_rope_log:
            table = np.empty((l, row['head_dim']//2, 2), dtype=np.uint32)
            seen = np.zeros(table.shape[:2], dtype=bool)
            for line in Path(args.full_rope_log).read_text().splitlines():
                if line.startswith('ROPE '):
                    hd, p, i, inv, cos, sin = map(int, line.split()[1:])
                    if hd == row['head_dim'] and p < l:
                        table[p, i] = cos, sin
                        seen[p, i] = True
            assert np.all(seen), 'incomplete production trig table'
            floats = table.view(np.float32)
            for i, name in enumerate(['cos', 'sin']):
                t = torch.from_numpy(floats[:, :, i].copy()).to(dev)
                setattr(model, name, torch.cat((t, t), dim=-1))
            aligned = model.block(xt, None, b, l)[0].reshape(b, l, dm).cpu().numpy()
            report('production_tables_fp32', ours, aligned)
        if args.reference64:
            model64 = spec.LlamaEager(torch, dev, row, W, torch.float64)
            if args.rope_log or args.full_rope_log:
                model64.cos, model64.sin = model.cos.double(), model.sin.double()
            ref64 = model64.block(xt.double(), None, b, l)[0].reshape(b, l, dm).cpu().numpy()
            report('matched_constants_fp64', ours, ref64)
            report('torch_fp32_vs_matched_fp64', aligned if args.rope_log or args.full_rope_log else original, ref64)
            if args.stage_errors:
                from transformer_admission_stages import stage_errors
                stage_errors(model, model64, xt, b, l, report)


if __name__ == '__main__':
    main()
