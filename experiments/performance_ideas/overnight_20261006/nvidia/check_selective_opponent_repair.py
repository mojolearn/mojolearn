"""Focused orchestration checks; never imports or runs an algorithm."""
import importlib.util
import json
import pathlib
import tempfile

path = pathlib.Path(__file__).with_name('selective-opponent-repair.py')
spec = importlib.util.spec_from_file_location('repair', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
with tempfile.TemporaryDirectory() as folder:
    old = {'status': 'failed', 'started': 'a', 'finished': 'b', 'cells': [
        {'arm': 'catboost-gpu', 'status': 'ok', 'median_ms': 3, 'source': 'original'},
        {'arm': 'xgboost-gpu', 'status': 'ok', 'median_ms': 4},
        {'arm': 'lightgbm-cuda', 'status': 'REFUSED'}],
        'infer_cells': [{'arm': 'catboost-gpu', 'status': 'ok', 'median_ms': 2}]}
    pathlib.Path(folder, 'board.json').write_text(json.dumps({'races': {'race': old}}))
    calls = []
    def run(ctx, race):
        calls.append((ctx['python'], race['arms'], ctx.get('infer')))
        return {'cells': [{'arm': 'lightgbm-cuda', 'status': 'ok', 'median_ms': 1}],
                'infer_cells': [], 'status': 'done', 'rc': 0, 'started': 'c', 'finished': 'd'}
    arms = ['catboost-gpu', 'xgboost-gpu', 'lightgbm-cuda']
    race = {'id': 'race', 'family': 'trees', 'opponents': arms, 'arms': arms}
    out = module.repair_tree({'out': folder, 'python': 'main', 'infer': True}, race, run,
                             {'arm': 'lightgbm-cuda', 'python': 'isolated'})
    assert calls == [('isolated', ['lightgbm-cuda'], False)]
    assert out['cells'][:2] == old['cells'][:2]
    assert out['infer_cells'][0] == old['infer_cells'][0]
    assert all(c['median_ms'] is None and c['rounds'] == 0 and
               'GPU-INFERENCE-NOT-SUPPORTED' in c['status'] for c in out['infer_cells'][1:])
    assert out['status'] == 'done' and old['cells'][2]['status'] == 'REFUSED'
    old['infer_cells'][0]['status'] = 'REFUSED(categorical GPU inference unsupported)'
    pathlib.Path(folder, 'board.json').write_text(json.dumps({'races': {'race': old}}))
    out = module.repair_tree({'out': folder, 'python': 'main', 'infer': True}, race, run,
                             {'arm': 'lightgbm-cuda', 'python': 'isolated'})
    assert out['status'] == 'failed' and out['cells'][-1]['status'] == 'ok'
print('PASS target-only fit, untouched successful fit/infer, no CPU inference, retained unrelated refusal')
