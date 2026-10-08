"""Bundled kernel models retain the exact independently recorded GPU bytes."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_all_six_kernel_variants_ship_all_recorded_fixtures():
    base = ROOT / 'python/mojolearn/verify_reference'
    table = json.loads((base / 'table.json').read_text())
    models = json.loads((base / 'models/models.json').read_text())['models']
    kernels = {f'{family}-{kernel}' for family in ('kernel-ridge', 'nystroem')
               for kernel in ('poly', 'sigmoid', 'laplacian')}
    selected = [m for m in models if m['lane'] in kernels]
    assert len(selected) == len(kernels) * len(table['fixtures'])
    assert {(m['lane'], m['fixture']) for m in selected} == {
        (lane, fixture) for lane in kernels for fixture in table['fixtures']}
    for model in selected:
        path = base / 'models' / model['file']
        cell = table['cells'][model['lane'] + '/' + model['fixture']]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest[:16] == model['model_hash'] == cell['model']['ref']
        assert model['batch_hash'] == cell['batch']['ref']
        assert path.stat().st_size == model['bytes']
        # bba034bbe regenerated the bundle on one NVIDIA L40S: `trained_on`
        # names the vendor that trained it and the classes whose table
        # column agrees, no per-model record directory any more; the model's
        # bytes are held to the table's model hash above.
        assert model['trained_on']['vendor'] in ('cuda', 'hip', 'metal', 'cpu')
        agreeing = {c for c, ref in cell['model']['cols'].items() if isinstance(ref, int)}
        assert set(model['trained_on']['classes_agreeing']) <= agreeing
        assert {'cpu', 'apple', 'amd'} <= agreeing
