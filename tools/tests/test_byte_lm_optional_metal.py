"""Authored file-only orchestration fixtures; no execution by subagents."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_spec = importlib.util.spec_from_file_location('optional_metal_compare',
    Path(__file__).resolve().parents[1] / 'byte_lm_state_compare.py')
compare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compare)


def setup_comparison(monkeypatch, metal=True):
    names = ['cuda', 'hip', 'head', 'resume', 'control'] + (['metal'] if metal else [])
    args = SimpleNamespace(cuda=Path('cuda'), hip=Path('hip'), metal=Path('metal') if metal else None,
        head=Path('head'), resume=Path('resume'), control=Path('control'),
        guard=[name + '=' + name + '.guard' for name in names],
        oracle=['cuda=cuda.oracle', 'hip=hip.oracle'],
        oracle_guard=['cuda=cuda.oracle.guard', 'hip=hip.oracle.guard'])
    captures = {}
    for name in names:
        vendor = 'cuda' if name == 'head' else 'hip' if name in ('resume', 'control') else name
        captures[name] = dict(runtime={'native_vendor': vendor}, source={}, ratio=.5,
                              summary_sha256=name, root=Path(name))
    loads, pairs, receipts, oracles = [], [], [], []
    def load(path, action, vendor=None):
        loads.append((str(path), action, vendor))
        return captures[str(path)]
    monkeypatch.setattr(compare, 'load_capture', load)
    monkeypatch.setattr(compare, 'compare_continuous', lambda a, b: pairs.append(
        (a['runtime']['native_vendor'], b['runtime']['native_vendor'])) or 128)
    monkeypatch.setattr(compare, 'compatible', lambda a, b: None)
    monkeypatch.setattr(compare, 'resume_control', lambda *args: {'effective': True})
    monkeypatch.setattr(compare, 'receipt', lambda path, digest, vendor: receipts.append(vendor) or {})
    monkeypatch.setattr(compare, 'oracle', lambda path, capture, guard: oracles.append(
        capture['runtime']['native_vendor']) or {})
    return args, loads, pairs, receipts, oracles


def test_optional_metal_adds_continuous_comparison_and_guard_not_oracle(monkeypatch):
    args, loads, pairs, receipts, oracles = setup_comparison(monkeypatch)
    result = compare.compare(args)
    assert result['identity_admitted'] and result['learning_admitted']
    assert pairs == [('cuda', 'hip'), ('cuda', 'metal')]
    assert ('metal', 'continuous', 'metal') in loads and 'metal' in receipts
    assert oracles == ['cuda', 'hip']
    assert result['continuous_vendors'] == ['cuda', 'hip', 'metal']
    assert result['resume_direction'] == {'source': 'cuda', 'destination': 'hip'}
    assert result['metal_checkpoint_resume_admitted'] is False


def test_no_metal_preserves_two_vendor_admission(monkeypatch):
    args, _, pairs, receipts, oracles = setup_comparison(monkeypatch, False)
    result = compare.compare(args)
    assert result['status'] == 'QUALIFIED_BOUNDED_CAPTURE'
    assert result['continuous_vendors'] == ['cuda', 'hip'] and pairs == [('cuda', 'hip')]
    assert 'metal' not in receipts and oracles == ['cuda', 'hip']


@pytest.mark.parametrize('missing', ['metal_guard', 'hip_oracle', 'control'])
def test_optional_metal_cannot_replace_existing_or_new_prerequisites(monkeypatch, missing):
    args, *_ = setup_comparison(monkeypatch)
    if missing == 'metal_guard':
        args.guard.remove('metal=metal.guard')
    elif missing == 'hip_oracle':
        args.oracle.remove('hip=hip.oracle')
    else:
        args.control = None
        args.guard.remove('control=control.guard')
    result = compare.compare(args)
    assert not result['identity_admitted'] and result['prerequisites_missing']


def test_metal_oracle_mapping_is_still_refused(monkeypatch):
    args, *_ = setup_comparison(monkeypatch)
    args.oracle.append('metal=metal.oracle')
    with pytest.raises(ValueError, match='mapping'):
        compare.compare(args)


def test_metal_requires_complete_darwin_arm64_runtime():
    host = dict(system='Darwin', machine='arm64', release='24.0', python='3.14', macos_version='15.0')
    compare.validate_metal_host_runtime({'host_runtime': host})
    for field, value in [('system', 'Linux'), ('machine', 'x86_64'), ('macos_version', '')]:
        with pytest.raises(ValueError, match='Darwin/arm64'):
            compare.validate_metal_host_runtime({'host_runtime': dict(host, **{field: value})})
    with pytest.raises(ValueError):
        compare.validate_metal_host_runtime({})


def test_continuous_comparison_rejects_changed_source_inventory():
    with pytest.raises(ValueError, match='source inventories'):
        compare.compare_continuous({'source': {'kernel': 'a'}}, {'source': {'kernel': 'b'}})


def test_metal_last_heldout_loss_is_compared(monkeypatch):
    monkeypatch.setattr(compare, 'same_steps', lambda a, b: 128)
    monkeypatch.setattr(compare, 'read', lambda path: b'changed' if str(path) ==
        'metal/heldout-final/batch07.loss.f32' else b'same')
    with pytest.raises(ValueError, match='heldout'):
        compare.compare_continuous(dict(root=Path('cuda'), checkpoint=b'checkpoint'),
                                   dict(root=Path('metal'), checkpoint=b'checkpoint'))


@pytest.mark.parametrize('count,left,right', [(127, b'a', b'a'), (128, b'a', b'b')])
def test_continuous_requires_all_steps_and_exact_checkpoint(monkeypatch, count, left, right):
    monkeypatch.setattr(compare, 'same_steps', lambda a, b: count)
    with pytest.raises(ValueError, match='checkpoint comparison'):
        compare.compare_continuous({'checkpoint': left}, {'checkpoint': right})
