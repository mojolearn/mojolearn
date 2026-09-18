# SPDX-License-Identifier: Apache-2.0
"""Whole loaded-model capture. Numerical captures are not reference admission.

Run CPU and GPU separately and compare complete captures. Fault controls below
alter composition, not native arithmetic, and are explicitly labelled as such.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import sys
import platform
import subprocess
import re

from . import Array, _backend
from .models import CausalLM
from . import _causal_lm_fixtures as fixtures

__all__ = ['capture', 'compare']

PROFILE = 'loaded-causal-lm-v2'
ARCHITECTURES = (('llama', False), ('llama', True), ('mistral', False),
                 ('qwen2', False), ('qwen3', False), ('phi3', False),
                 ('mamba', True), ('mamba2', True))
PARTS = frozenset(('logits', 'prefill', 'decode_1', 'decode_2', 'greedy', 'state'))
CHECKS = frozenset(('prefill_step', 'batch', 'reset', 'reload', 'greedy_repeat',
                    'state_positions', 'tensor_mapping', 'composition_fault_detected'))


def state_digest(state):
    parts = []
    for layer in state.layers:
        if hasattr(layer, 'snapshot'):
            layer = layer.snapshot()
        for name, value in sorted(vars(layer).items()):
            if hasattr(value, 'tobytes'):
                parts.append((name, digest(value)))
            elif isinstance(value, (int, float, str)):
                parts.append((name, value))
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()


def digest(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def capture_case(root, architecture, tied, weight_format, device, layer_devices=None):
    cfg, tensors = fixtures.family_fixture(architecture, tied)
    fixtures._write_checkpoint(root, cfg, tensors)
    models = []
    def load():
        if layer_devices is None:
            model = CausalLM.load(root, device=device, weight_format=weight_format)
        else:
            from .models.parallel_causal_lm import ParallelCausalLM
            model = ParallelCausalLM.load(root, layer_devices=layer_devices, weight_format=weight_format)
        models.append(model)
        return model
    try:
        lm = load()
        ids = Array.from_list([[1, 7, 3, 11, 5], [2, 9, 2, 4, 8]], '<i4')
        full = lm.forward(ids)
        state = lm.allocate_state(2, 8)
        prefill = lm.forward(ids[:, :3], state)
        steps = [lm.step(ids[:, i:i+1], state) for i in (3, 4)]
        joined = Array.from_list([prefill.tolist()[b] + [x.tolist()[b] for x in steps]
                                  for b in range(2)], '<f4')
        batches = Array.from_list([lm.forward(ids[b:b+1]).tolist()[0] for b in range(2)], '<f4')
        lm.reset_state(state)
        reset = lm.forward(ids, state)
        reloaded = load().forward(ids)
        greedy = lm.generate(ids[:, :3], 2)
        repeated = lm.generate(ids[:, :3], 2)
        checks = {'prefill_step': digest(full) == digest(joined),
                  'batch': digest(full) == digest(batches),
                  'reset': digest(full) == digest(reset),
                  'reload': digest(full) == digest(reloaded),
                  'greedy_repeat': digest(greedy) == digest(repeated),
                  'state_positions': state.positions == 5,
                  'tensor_mapping': not lm.unused_names}
        # Bypass a whole nontrivial block. This exercises the loader/composition
        # comparator, and makes no claim about a compiled native fault variant.
        original = lm._blocks
        lm._blocks = original[1:]
        try:
            fault = lm.forward(ids)
        finally:
            lm._blocks = original
        checks['composition_fault_detected'] = digest(fault) != digest(full)
        return {'architecture': architecture, 'tied': tied, 'weight_format': weight_format,
                'checks': checks, 'parts': {name: digest(value) for name, value in
                    [('logits', full), ('prefill', prefill), ('decode_1', steps[0]),
                     ('decode_2', steps[1]), ('greedy', greedy)]},
                'state_sha256': state_digest(state),
                'checkpoint_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in sorted(Path(root).iterdir())}}
    finally:
        for model in models:
            close = getattr(model, 'close', None)
            if close is not None:
                close()


def capture(device='cpu', formats=('float32',), *, layer_devices=None):
    if _backend.numeric_mode() != 'identical':
        raise ValueError('loaded-model proof requires numeric_mode=identical')
    if layer_devices is not None and device != 'gpu':
        raise ValueError('layer_devices requires device=gpu')
    formats = tuple(formats)
    if not formats or len(set(formats)) != len(formats) or set(formats) - {'float32', 'bfloat16', 'int8'}:
        raise ValueError('formats must be distinct supported formats')
    result = {'formats': list(formats), 'repeats': 2,
              'platform': platform.platform(), 'backend': _backend.vendor(), 'layer_devices': layer_devices,
              'profile': PROFILE, 'device': device, 'status': 'CAPTURED_UNQUALIFIED',
              'native_fault': 'OWED', 'physical_execution': 'OWED', 'cases': []}
    with tempfile.TemporaryDirectory() as root:
        for architecture, tied in ARCHITECTURES:
            for fmt in formats:
                path = Path(root) / f'{architecture}-{tied}-{fmt}'
                result['cases'].append(capture_case(path, architecture, tied, fmt, device, layer_devices))
    for c in result['cases']:
        c['parts']['state'] = c.pop('state_sha256')
    result['bindings'] = {name: hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest()
                          for name, mod in list(sys.modules.items())
                          if '_mojolearn' in name and getattr(mod, '__file__', None)
                          and Path(mod.__file__).is_file()}
    if device == 'cpu':
        name = '_mojolearn_neural_host'
        binding = _backend.load_host_module(name)
        path = Path(_backend.host_module_path(name))
        result['bindings'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        result['native_fault'] = bool(binding.neural_host_sabotage())
    result['source_sha256'] = hashlib.sha256(b''.join(path.read_bytes() for path in
        (Path(__file__), Path(fixtures.__file__), Path(__file__).parent / 'models' / 'causal_lm.py',
         Path(__file__).parent / 'models' / 'parallel_causal_lm.py',
         Path(__file__).parent / '_causal_lm_worker.py'))).hexdigest()
    try:
        result['capture_commit'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).parent, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        result['capture_commit'] = None
    if not all(all(c['checks'].values()) for c in result['cases']):
        result['status'] = 'PROPERTY_FAILURE'
    return result


def compare(left, right):
    if (left.get('profile') != PROFILE or right.get('profile') != PROFILE
            or not left.get('source_sha256')
            or left.get('source_sha256') != right.get('source_sha256')):
        raise ValueError('profile mismatch')
    def cells(report):
        if report.get('status') != 'CAPTURED_UNQUALIFIED' or not report.get('cases'):
            raise ValueError('incomplete or failed capture')
        formats = report.get('formats', [])
        if (not formats or len(formats) != len(set(formats))
                or set(formats) - {'float32', 'bfloat16', 'int8'} or report.get('repeats') != 2):
            raise ValueError('invalid capture scope or repetitions')
        expected = {(arch, tied, fmt) for arch, tied in ARCHITECTURES
                    for fmt in formats}
        out = {}
        for c in report['cases']:
            key = (c['architecture'], c['tied'], c['weight_format'])
            if (key in out or set(c['checks']) != CHECKS or not all(c['checks'].values())
                    or set(c['parts']) != PARTS or not c['checkpoint_sha256']):
                raise ValueError('duplicate or failed case')
            if any(not isinstance(h, str) or not re.fullmatch('[0-9a-f]{64}', h)
                   for h in [*c['checkpoint_sha256'].values(), *c['parts'].values()]):
                raise ValueError('invalid numerical or checkpoint digest')
            out[key] = (c['checkpoint_sha256'], c['parts'])
        if set(out) != expected:
            raise ValueError('capture missing requested architecture/format cases')
        return out
    return cells(left) == cells(right)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'gpu'), default='cpu')
    parser.add_argument('--formats', nargs='+', choices=('float32', 'bfloat16', 'int8'), default=['float32'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--layer-devices', nargs='+', type=int)
    args = parser.parse_args()
    result = capture(args.device, args.formats, layer_devices=args.layer_devices)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    return 0 if result['status'] == 'CAPTURED_UNQUALIFIED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
