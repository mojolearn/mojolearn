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

from . import Array, _backend
from .models import CausalLM
from . import _causal_lm_fixtures as fixtures

__all__ = ['capture', 'compare']

PROFILE = 'loaded-causal-lm-v1'
PARTS = frozenset(('logits', 'prefill', 'decode_1', 'decode_2', 'greedy', 'state'))
CHECKS = frozenset(('prefill_step', 'batch', 'reset', 'reload', 'greedy_repeat',
                    'state_positions', 'tensor_mapping', 'composition_fault_detected'))


def state_digest(state):
    parts = []
    for layer in state.layers:
        for name, value in sorted(vars(layer).items()):
            if hasattr(value, 'tobytes'):
                parts.append((name, digest(value)))
            elif isinstance(value, (int, float, str)):
                parts.append((name, value))
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()


def digest(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def capture_case(root, architecture, tied, weight_format, device):
    if architecture == 'llama':
        cfg = fixtures._llama_config(tie_word_embeddings=tied)
        tensors = fixtures._llama_tensors(cfg)
    else:
        cfg = fixtures._mamba_config()
        tensors = fixtures._mamba_tensors(cfg)
    fixtures._write_checkpoint(root, cfg, tensors)
    load = lambda: CausalLM.load(root, device=device, weight_format=weight_format)
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


def capture(device='cpu', formats=('float32',)):
    if _backend.numeric_mode() != 'identical':
        raise ValueError('loaded-model proof requires numeric_mode=identical')
    result = {'profile': PROFILE, 'device': device, 'status': 'CAPTURED_UNQUALIFIED',
              'native_fault': 'OWED', 'physical_execution': 'OWED', 'cases': []}
    with tempfile.TemporaryDirectory() as root:
        for architecture, tied in [('llama', False), ('llama', True), ('mamba', True)]:
            for fmt in formats:
                path = Path(root) / f'{architecture}-{tied}-{fmt}'
                result['cases'].append(capture_case(path, architecture, tied, fmt, device))
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
        (Path(__file__), Path(fixtures.__file__), Path(__file__).parent / 'models' / 'causal_lm.py'))).hexdigest()
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
        out = {}
        for c in report['cases']:
            key = (c['architecture'], c['tied'], c['weight_format'])
            if (key in out or set(c['checks']) != CHECKS or not all(c['checks'].values())
                    or set(c['parts']) != PARTS or not c['checkpoint_sha256']):
                raise ValueError('duplicate or failed case')
            out[key] = (c['checkpoint_sha256'], c['parts'])
        return out
    return cells(left) == cells(right)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'gpu'), default='cpu')
    parser.add_argument('--formats', nargs='+', choices=('float32', 'bfloat16', 'int8'), default=['float32'])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = capture(args.device, args.formats)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    return 0 if result['status'] == 'CAPTURED_UNQUALIFIED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
