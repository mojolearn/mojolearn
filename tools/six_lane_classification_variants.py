"""Strict registration of reviewed uncapped classification input recipes.

This is metadata admission, never an estimator or a numerical implementation.
Preparation and measurement may use different commits only when the exact
reviewed helper and preprocessing source bytes remain unchanged.
"""
import copy
import functools
import hashlib
import json
import math
import struct
from pathlib import Path
from six_lane_matrix_io import read_matrix

ROOT = Path(__file__).resolve().parents[1]
VARIANT = 'classification-full-v1'
SUFFIX = '@input=' + VARIANT
CONTRACT = 'experiments/six_lane_integration/classification_full_contracts.json'
NAN_PARAMETER = {'nonfinite_parameter': 'NaN'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def retained(ref):
    if digest(ref['path']) != ref['sha256']:
        raise ValueError('Classification evidence hash changed: ' + ref['path'])
    return json.loads(Path(ref['path']).read_text())


@functools.lru_cache(maxsize=1)
def contracts():
    doc = json.loads((ROOT / CONTRACT).read_text())
    if doc['schema'] != 'mojolearn.classification-full-contracts/1' or doc['variant'] != VARIANT:
        raise ValueError('Unknown classification contract schema')
    for path, expected in doc['source_hashes'].items():
        if digest(ROOT / path) != expected:
            raise ValueError('Reviewed classification source changed: ' + path)
    return doc


def profile(cell):
    rows = [r for r in contracts()['rows'] if r['vendor'] == cell['vendor']
            and r['configuration'] == cell['configuration']
            and r['original_workload_id'] == cell['workload_id']]
    if len(rows) > 1:
        raise ValueError('Ambiguous classification source contract')
    return rows[0] if rows else None


def variant_cell(cell):
    row = profile(cell)
    if row is None:
        raise ValueError('Unreviewed classification cell')
    result = copy.deepcopy(cell)
    wid = cell['workload_id'] + SUFFIX
    if wid != row['variant_workload_id'] or cell['mode'] != row['mode']:
        raise ValueError('Classification contract identity differs')
    result.update(key=canonical([cell['configuration'], cell['vendor'], wid])[:20],
                  workload_id=wid, original_workload_id=cell['workload_id'],
                  original_cell_key=cell['key'], input_variant=VARIANT)
    result['workload'] = dict(result['workload'], id=wid, input_variant=VARIANT,
                              original_workload_id=cell['workload_id'])
    return result


def append_registered_cells(cells):
    return cells + [variant_cell(c) for c in cells if profile(c) is not None]


def parameter_record(row, meta):
    if row['estimator_settings_record'] is not None:
        return copy.deepcopy(row['estimator_settings_record'])
    params = copy.deepcopy(row['partial_or_complete_params'])
    if row['lane'] == 'categorical-nb':
        item = meta['derived_parameters']['categorical-nb']['min_categories']
        values = item['values']
        if (item['dtype'] != 'int64' or item['shape'] != [meta['arrays']['X']['shape'][1]]
                or item['original_representation'] != 'list[int]'
                or any(type(x) is not int or x < 1 for x in values)
                or hashlib.sha256(b'int64' + str((len(values),)).encode()
                    + struct.pack('<' + 'q' * len(values), *values)).hexdigest() != item['sha256']):
            raise ValueError('Invalid retained categorical min_categories')
        params['min_categories'] = values
    elif row['lane'] == 'simple-imputer':
        params['missing_values'] = copy.deepcopy(NAN_PARAMETER)
    else:
        raise ValueError('Unresolved original estimator parameters')
    return dict(__record__=True, library='mojolearn', source='get_params', params=params)


def normalize_parameter_record(work, record):
    """Tag only the intentional SimpleImputer parameter NaN, never output data.

    This is metadata representation after the operation. Constructors, metrics,
    output/state values, timings and the strict finite comparator are untouched.
    """
    if work.get('input_variant') != VARIANT or work.get('lane') != 'simple-imputer':
        return record
    result = copy.deepcopy(record)
    value = result.get('params', {}).get('missing_values')
    if not isinstance(value, float) or not math.isnan(value):
        raise ValueError('SimpleImputer original missing_values must be NaN')
    result['params']['missing_values'] = copy.deepcopy(NAN_PARAMETER)
    return result


def validate_output_scope(work, output):
    if work.get('input_variant') != VARIANT:
        return
    import numpy as np
    actual = {item['path']:item for item in output['manifest']}
    for path, expected in work['output_schema'].items():
        item = actual.get(path)
        if item is None or np.dtype(item['dtype']) != np.dtype(expected['dtype']):
            raise ValueError('Classification output dtype/scope differs: ' + path)
        shape = expected['shape']
        if (len(item['shape']) != len(shape)
                or any(want is not None and want != got for want,got in zip(shape,item['shape']))):
            raise ValueError('Classification output shape differs: ' + path)


@functools.lru_cache(maxsize=1)
def original_matrix():
    return read_matrix(ROOT / 'experiments/six_lane_integration/matrix.json.gz')


def validate_variant(facts, cell=None):
    reg = facts['registered_input_variant']
    if reg.get('variant') != VARIANT or facts.get('changes_frozen_race') is not True:
        raise ValueError('Classification change marker erased')
    # Recreate the source matrix without registering variants recursively.
    matrix = original_matrix()
    originals = [c for c in matrix['cells'] if c['key'] == reg['original_cell_key']
                 and c['workload_id'] == reg['original_workload_id']]
    if len(originals) != 1:
        raise ValueError('Missing original classification matrix cell')
    original = originals[0]; row = profile(original); expected = variant_cell(original)
    if reg['contract_sha256'] != digest(ROOT / CONTRACT):
        raise ValueError('Classification source contract replaced')
    if (reg['variant_workload_id'] != expected['workload_id'] or reg['variant_cell_key'] != expected['key']
            or (cell and (cell['key'] != expected['key'] or cell['workload_id'] != expected['workload_id']))):
        raise ValueError('Full classification cannot execute under another cell identity')
    job = facts.get('job')
    if job and (job['key'] != expected['key'] or job['workload_id'] != expected['workload_id']
                or facts['vendor'] != original['vendor']):
        raise ValueError('Worker classification identity differs')
    if facts.get('source_sha', reg['measurement_source_sha']) != reg['measurement_source_sha']:
        raise ValueError('Wrong measurement source freeze')
    receipt = retained(reg['preparation_receipt']); plan = retained(reg['preparation_plan'])
    if (receipt.get('schema') != 'mojolearn.full-classification-preparation/1'
            or receipt.get('status') != 'PREPARED_NOT_MEASURED_NOT_ADMITTED'
            or receipt.get('variant') != VARIANT or plan.get('variant') != VARIANT
            or receipt.get('model_executions') != 0 or receipt.get('builds') != 0
            or receipt['source_sha'] != reg['preparation_source_sha']
            or plan['source_sha'] != receipt['source_sha']
            or receipt['plan_sha256'] != reg['preparation_plan']['sha256']
            or receipt['plan']['sha256'] != receipt['plan_sha256']
            or receipt['helper_sha256'] != digest(ROOT / 'tools/six_lane_prepare_full_classification.py')
            or plan['helper_sha256'] != receipt['helper_sha256']
            or receipt['source_hashes'] != plan['source_hashes']):
        raise ValueError('Unreviewed or failed full classification preparation')
    for path, expected_hash in receipt['source_hashes'].items():
        if digest(ROOT / path) != expected_hash:
            raise ValueError('Preparation/measurement preprocessing closure differs: ' + path)
    work = facts['workload']; name = row['block'] + '-' + row['dataset']
    blocks = [b for b in receipt['blocks'] if b['name'] == name]
    if len(blocks) != 1:
        raise ValueError('Missing prepared classification block')
    block = blocks[0]
    inputs = work['input_files']
    if len(inputs) != 2 or {Path(f['path']).name for f in inputs} != {name+'.npz',name+'.json'}:
        raise ValueError('Expected exact full classification archive and sidecar')
    # Relocation is permitted only with byte-identical archived data and metadata.
    by_suffix = {Path(f['path']).suffix:f for f in inputs}
    if (by_suffix['.npz']['sha256'] != block['npz_sha256']
            or by_suffix['.json']['sha256'] != block['sidecar_sha256']
            or any(Path(f['path']).resolve().parent != Path(work['data_directory']).resolve() for f in inputs)):
        raise ValueError('Unattested classification input relocation')
    meta = retained(by_suffix['.json'])
    expected_shapes = row['expected_shapes_from_population_metadata']
    raw_shapes = {k:v for k,v in expected_shapes.items() if k not in ('X_true','Xq_true')}
    if (meta['variant'] != VARIANT or meta['changes_frozen_race'] is not True
            or meta['source_sha'] != receipt['source_sha'] or meta['dataset'] != row['dataset']
            or meta['block'] != row['block'] or meta['arrays'] != block['arrays']
            or {k:v['shape'] for k,v in meta['arrays'].items()} != raw_shapes
            or any(v['dtype'] != 'float32' for v in meta['arrays'].values())
            or meta['preparation_caps'] != {'query':None,'train':None}
            or meta['original_preparation_caps'] != {'query':100000,'train':1000000}
            or meta['full_dataset_coverage'] is not True or meta['seed'] != 7
            or meta['population'] != plan['population'][row['dataset']]['rule']
            or meta['source_archive'] not in receipt['source_archives']):
        raise ValueError('Not the exact original full classification population/transforms')
    params = parameter_record(row, meta)
    split = dict(population=meta['population'],fit=meta['fit_rows'],query=meta['eval_rows'],original_preparation_caps=meta['original_preparation_caps'],preparation_caps=meta['preparation_caps'])
    if (work['input_variant'] != VARIANT or work['lane'] != row['lane'] or work['dataset'] != row['dataset']
            or work['harness'] != row['harness'] or work['harness_sha256'] != row['harness_sha256']
            or work['actual_shapes'] != expected_shapes or facts['dimensions'] != expected_shapes
            or work['estimator_settings_record'] != params or facts['estimator_settings'] != params
            or work['inference'] != row['inference'] or work['output_paths'] != row['output_paths']
            or work['output_schema'] != row['output_schema'] or work['split'] != split
            or work['seed'] != row['seed'] or work['repeated_operations'] != row['repeated_operations']
            or facts['dataset_sha256'] != canonical(meta['arrays'])
            or facts['timed_boundary'] != row['timed_boundary']
            or work.get('overrides') or work.get('adapter')):
        raise ValueError('Classification input/settings/operation/output contract differs')
