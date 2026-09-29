"""Contract additions must invalidate saved evidence, including physical UMAP."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def test_added_model_and_train_fields_reject_old_references():
    tree = ast.parse((ROOT / 'tools/identity_break.py').read_text())
    revisions = next(ast.literal_eval(node.value) for node in tree.body
                     if isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'LANE_REVISIONS'
                             for t in node.targets))
    spec = importlib.util.spec_from_file_location(
        'reference_contract_test', ROOT / 'python/mojolearn/_verify_reference.py')
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    lanes = ['tsvd', 'umap', 'par-graph-umap']
    table = {'cells': {f'{lane}/base': {'model': 'unchanged-test-hash'}
                       for lane in lanes + ['ols']},
             'lane_revisions': {'umap': 'transform-row-separable-1',
                                'par-graph-umap': 'transform-row-separable-1'}}
    harness = SimpleNamespace(LANE_REVISIONS=revisions)
    assert reference.stale_reference_lanes(table, harness) == sorted(lanes)
    table['lane_revisions'].update({lane: revisions[lane] for lane in lanes})
    assert reference.stale_reference_lanes(table, harness) == []
    # Recording single-device UMAP never admits an obsolete physical-par column.
    table['lane_revisions']['par-graph-umap'] = 'transform-row-separable-1'
    assert reference.stale_reference_lanes(table, harness) == ['par-graph-umap']
