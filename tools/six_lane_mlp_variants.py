"""Reviewed full MLP inputs: metadata admission, no model execution or preparation."""
import copy
import functools
import json
from pathlib import Path

from six_lane_classification_variants import canonical, digest, retained

ROOT = Path(__file__).resolve().parents[1]
VARIANT = 'mlp-full-v1'
SUFFIX = '@input=' + VARIANT
CONTRACT = 'experiments/six_lane_integration/mlp_full_contracts.json'


@functools.lru_cache(maxsize=1)
def contracts():
    doc = json.loads((ROOT / CONTRACT).read_text())
    if doc.get('schema') != 'mojolearn.mlp-full-contracts/1' or doc.get('variant') != VARIANT:
        raise ValueError('Unknown full MLP contract')
    for path, expected in doc['source_hashes'].items():
        if digest(ROOT / path) != expected:
            raise ValueError('Reviewed MLP recipe source changed: ' + path)
    return doc


def profile(cell):
    rows = [r for r in contracts()['rows'] if r['vendor'] == cell['vendor']
            and r['configuration'] == cell['configuration']
            and r['original_workload_id'] == cell['workload_id']]
    if len(rows) > 1:
        raise ValueError('Ambiguous full MLP contract')
    return rows[0] if rows else None


def variant_cell(scope, row):
    if (scope['key'] != row['scope_cell_key'] or scope['workload_id'] != row['scope_workload_id']
            or any(scope[k] != row[k] for k in ('mode', 'vendor', 'configuration'))):
        raise ValueError('Unreviewed MLP candidate scope')
    result = copy.deepcopy(scope)
    wid = row['variant_workload_id']
    result.update(key=canonical([scope['configuration'], scope['vendor'], wid])[:20],
                  workload_id=wid, original_workload_id=row['original_workload_id'],
                  original_cell_key=scope['key'], input_variant=VARIANT)
    result['workload'] = dict(id=wid, input_variant=VARIANT, harness=row['harness'],
        lane=row['lane'], original_workload_id=row['original_workload_id'],
        recipe_status='Explicit full saved MLP recipe; distinct from linked neural component scope')
    return result


def append_registered_cells(cells):
    bykey = {c['key']: c for c in cells}
    return cells + [variant_cell(bykey[r['scope_cell_key']], r) for r in contracts()['rows']
                    if r['scope_cell_key'] in bykey]


def workload(row, directory):
    directory = Path(directory).resolve()
    inputs = [dict(path=str(directory / item['name']), sha256=item['sha256'])
              for item in row['input_files']]
    return dict(harness=row['harness'], harness_sha256=row['harness_sha256'],
        input_variant=VARIANT, lane=row['lane'], dataset=row['dataset'],
        data_directory=str(directory), input_files=inputs,
        actual_shapes=row['actual_shapes'], estimator_settings_record=row['estimator_settings_record'],
        inference='separate', dataset_version=row['dataset_version'], split=row['split'],
        seed=7, output_paths=row['output_paths'], output_schema=row['output_schema'],
        repeated_operations=1, runtime_vendor=row['runtime_vendor'],
        quality_gate_source=row['quality_gate_source'], capture_limitation=row['limitations'],
        intrinsic_cap_audit=dict(reviewed=True, unresolved=[], evidence=str(ROOT / CONTRACT)))


def validate_variant(facts, cell=None):
    reg = facts['registered_input_variant']
    if reg.get('variant') != VARIANT or facts.get('changes_frozen_race') is not True:
        raise ValueError('MLP change marker erased')
    matrix = json.loads((ROOT / 'experiments/six_lane_integration/matrix.json').read_text())
    rows = [r for r in contracts()['rows'] if r['original_workload_id'] == reg['original_workload_id']
            and r['scope_cell_key'] == reg['original_cell_key']]
    if len(rows) != 1:
        raise ValueError('Missing reviewed full MLP recipe')
    row = rows[0]
    originals = [c for c in matrix['cells'] if c['key'] == row['scope_cell_key']]
    if len(originals) != 1:
        raise ValueError('Missing original MLP candidate scope')
    original = originals[0]
    expected_cell = variant_cell(original, row)
    if reg['contract_sha256'] != digest(ROOT / CONTRACT):
        raise ValueError('MLP source contract changed')
    if (reg['variant_workload_id'] != expected_cell['workload_id']
            or reg['variant_cell_key'] != expected_cell['key']
            or cell and (cell['key'] != expected_cell['key']
                         or cell['workload_id'] != expected_cell['workload_id']
                         or any(cell[k] != original[k] for k in ('vendor','mode','configuration')))):
        raise ValueError('MLP variant identity differs')
    job = facts.get('job')
    if 'job' in facts:
        if (not isinstance(job, dict) or job.get('key') != expected_cell['key']
                or job.get('workload_id') != expected_cell['workload_id']
                or facts.get('vendor') != original['vendor'] or job.get('mode') != original['mode']
                or job.get('master_selection', {}).get('id') != original['configuration']):
            raise ValueError('MLP worker identity differs')
        # The worker executes this job, not the predeployment top-level facts.
        provenance = job.get('artifact_provenance')
        if not provenance:
            raise ValueError('Full MLP worker artifact provenance missing')
    else:
        provenance = facts.get('artifact_provenance')
    if facts.get('source_sha', reg['measurement_source_sha']) != reg['measurement_source_sha']:
        raise ValueError('Wrong MLP measurement freeze')
    if provenance is not None:
        for arm in ('A', 'B'):
            items = provenance.get(arm)
            if not items:
                raise ValueError('Full MLP binding dependency missing: ' + arm)
            bindings = {json.loads(Path(item['receipt']).read_text())['binding']
                        for item in items}
            if not set(row['required_bindings']).issubset(bindings):
                raise ValueError('Full MLP binding dependency missing: ' + arm)
    actual = facts['workload']
    expected = workload(row, actual['data_directory'])
    if actual != expected:
        raise ValueError('Full MLP input/settings/operation/output contract differs')
    sidecar = next(f for f in actual['input_files'] if f['path'].endswith('.json'))
    meta = retained(sidecar)
    if meta != row['sidecar_metadata']:
        raise ValueError('MLP full saved population differs')
    if (facts['dimensions'] != row['actual_shapes']
            or facts['estimator_settings'] != row['estimator_settings_record']
            or facts['dataset_sha256'] != canonical(row['arrays'])
            or facts['timed_boundary'] != row['timed_boundary']
            or facts.get('intrinsic_caps') != [] or facts.get('full_dataset_coverage') is not True):
        raise ValueError('MLP full operation contract differs')


def validate_output_scope(work, output):
    if work.get('input_variant') != VARIANT:
        return
    import numpy as np
    actual = {item['path']: item for item in output['manifest']}
    for path, expected in work['output_schema'].items():
        item = actual.get(path)
        if (item is None or np.dtype(item['dtype']) != np.dtype(expected['dtype'])
                or item['shape'] != expected['shape']):
            raise ValueError('Full MLP output dtype/shape differs: ' + path)
