"""Explicit, source-reviewed input variants; unknown frozen-race edits stay refused.

This registers only the full raw X-only TSVD corpus projection. It does not
change constructor settings, numerical code, original races, or qualification.
"""
import copy
import functools
import hashlib
import json
from pathlib import Path
from six_lane_matrix_io import read_matrix
from six_lane_prepare_full_tsvd import VARIANT, SOURCES, canonical_hash, settings

ROOT = Path(__file__).resolve().parents[1]
SUFFIX = '@input=' + VARIANT
LANES = ('tsvd', 'incremental-pca', 'randomized-svd', 'gaussian-rp', 'sparse-rp', 'fastica')


def original_id(workload_id):
    from six_lane_mlp_variants import SUFFIX as mlp_suffix
    if workload_id.endswith(mlp_suffix):
        return workload_id[:-len(mlp_suffix)]
    from six_lane_classification_variants import SUFFIX as classification_suffix
    if workload_id.endswith(classification_suffix):
        return workload_id[:-len(classification_suffix)]
    return workload_id[:-len(SUFFIX)] if workload_id.endswith(SUFFIX) else workload_id


def eligible(cell):
    vendor, cfg = cell['vendor'], cell['configuration']
    if vendor in ('nvidia', 'amd') and cfg == 'I.X.complete-proposed':
        prefixes = ['more:tsvd'] + ['expanded:' + x for x in LANES if x != 'tsvd']
    elif vendor == 'apple' and cfg == 'AF.X.complete-proposed':
        prefixes = ['algos/randomized-svd']
    else:
        return False
    return cell['workload_id'] in {p + '@dataset=' + d for p in prefixes for d in ('taxi', 'istella')}


def variant_cell(cell):
    if not eligible(cell):
        raise ValueError('Original cell is outside the reviewed full-TSVD registration')
    result = copy.deepcopy(cell)
    wid = cell['workload_id'] + SUFFIX
    result.update(key=canonical_hash([cell['configuration'], cell['vendor'], wid])[:20],
                  workload_id=wid, original_workload_id=cell['workload_id'],
                  original_cell_key=cell['key'], input_variant=VARIANT)
    result['workload'] = dict(result['workload'], id=wid, input_variant=VARIANT,
                              original_workload_id=cell['workload_id'])
    return result


def append_registered_cells(cells):
    # Original capped cells remain untouched and independently pending/measured.
    from six_lane_classification_variants import append_registered_cells as append_classification
    from six_lane_mlp_variants import append_registered_cells as append_mlp
    return append_mlp(append_classification(cells)) + [variant_cell(cell) for cell in cells if eligible(cell)]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def retained(reference):
    path = Path(reference['path'])
    if digest(path) != reference['sha256']:
        raise ValueError('Registered variant evidence changed: ' + str(path))
    return json.loads(path.read_text())


@functools.lru_cache(maxsize=1)
def source_contract():
    sources = {p: (ROOT / p).read_text() for p in SOURCES}
    matrix = read_matrix(ROOT / 'experiments/six_lane_integration/matrix.json.gz')
    return sources, matrix


def validate_variant(facts, cell=None):
    """Validate at materialization, queue creation and inside each actual worker.

    A proposal or a truthy registration field alone never authorizes a changed
    race. This rederives the sole reviewed variant contract from frozen source
    and the projector's retained full input receipt.
    """
    if facts.get('input_variant_control_transfer'):
        from six_lane_targeted_variants import validate_transfer
        return validate_transfer(facts, cell, validate_variant)
    reg = facts.get('registered_input_variant')
    if reg and reg.get('variant') == 'mlp-full-v1':
        from six_lane_mlp_variants import validate_variant as validate_mlp
        return validate_mlp(facts, cell)
    if reg and reg.get('variant') == 'classification-full-v1':
        from six_lane_classification_variants import validate_variant as validate_classification
        return validate_classification(facts, cell)
    if not reg:
        if facts.get('changes_frozen_race') or '@input=' in facts.get('job', {}).get('workload_id', ''):
            raise ValueError('Unregistered frozen-race change')
        if cell and cell.get('input_variant'):
            raise ValueError('Registered variant facts are missing')
        return
    if reg.get('variant') != VARIANT or facts.get('changes_frozen_race') is not True:
        raise ValueError('Unknown variant or erased frozen-race change marker')
    sources, matrix = source_contract()
    originals = [c for c in matrix['cells'] if c['key'] == reg['original_cell_key']
                 and c['workload_id'] == reg['original_workload_id'] and eligible(c)]
    if len(originals) != 1:
        raise ValueError('Registered variant lacks its exact original matrix cell')
    original = originals[0]
    expected = variant_cell(original)
    if reg['variant_workload_id'] != expected['workload_id'] or reg['variant_cell_key'] != expected['key']:
        raise ValueError('Wrong distinct variant identity')
    if cell and (cell['key'] != expected['key'] or cell['workload_id'] != expected['workload_id']):
        raise ValueError('Full variant cannot execute as another matrix cell')
    job = facts.get('job')
    if job and (job['key'] != expected['key'] or job['workload_id'] != expected['workload_id']
                or facts['vendor'] != original['vendor']):
        raise ValueError('Worker variant identity differs')
    proposal = retained(reg['proposal'])
    receipt = retained(reg['projection_receipt'])
    if proposal['schema'] != 'mojolearn.full-tsvd-recipes/1' or proposal.get('execution_authorized') is not False:
        raise ValueError('Not an unmodified retained projector proposal')
    rows = [r for r in proposal['recipes'] if r['variant_cell_key'] == expected['key']]
    if len(rows) != 1:
        raise ValueError('Missing exact projected variant recipe')
    row = rows[0]
    work = facts['workload']
    expected_lane = original['workload_id'].split('@dataset=')[0].rsplit(':',1)[-1].rsplit('/',1)[-1]
    expected_dataset = original['workload_id'].split('@dataset=')[1]
    configurations = {c['id']: c for c in matrix['configurations']}
    if row['configuration_sha256'] != canonical_hash(configurations[original['configuration']]):
        raise ValueError('Registered candidate configuration changed')
    if receipt.get('model_executions') != 0 or receipt.get('compilations') != 0:
        raise ValueError('Projection was not data-only')
    if receipt.get('status') != 'PROJECTED_NOT_ADMITTED' or receipt['vendor'] != original['vendor'] or receipt['variant'] != VARIANT:
        raise ValueError('Projection failed or belongs to another vendor/variant')
    if receipt['source_sha'] != proposal['source_sha'] or receipt['source_sha'] != reg['source_sha']:
        raise ValueError('Projection and registration source freezes differ')
    if facts.get('source_sha', reg['source_sha']) != reg['source_sha']:
        raise ValueError('Variant recipe source differs')
    if receipt['projector_sha256'] != digest(ROOT / 'tools/six_lane_prepare_full_tsvd.py'):
        raise ValueError('Unreviewed projector')
    if row['input_files'] != receipt['output_files'] or work['input_files'] != row['input_files']:
        raise ValueError('Projected inputs were replaced or relocated without a new receipt')
    metadata_files = [f for f in work['input_files'] if Path(f['path']).suffix == '.json']
    if len(metadata_files) != 1 or len(work['input_files']) != 2:
        raise ValueError('Expected exact X-only archive and metadata')
    meta = retained(metadata_files[0]);source = meta['source_metadata'];arr = meta['arrays']['X'];n,d = arr['shape']
    plan = retained(receipt['plan'])
    if (plan['variant'] != VARIANT or plan['source_sha'] != reg['source_sha']
            or plan['projector_sha256'] != receipt['projector_sha256']
            or source != plan['inputs'][expected_dataset]['source_metadata']
            or arr != {k: receipt['X'][k] for k in ('dtype','shape','sha256')}
            or row['projected_array'] != receipt['X'] or meta['source_sha'] != reg['source_sha']):
        raise ValueError('Full projection provenance chain differs')
    if (set(meta['arrays']) != {'X'} or meta['variant'] != VARIANT or meta['block'] != 'tsvd'
            or meta['scaling'] != 'none' or arr['dtype'] != 'float32' or meta['intrinsic_caps']
            or meta['full_dataset_coverage'] is not True or meta['fit_rows'] != [0,n]
            or meta['fit_rows_available'] != n or meta['original_preparation_cap'] != 1_000_000
            or source['block'] != 'big' or source['scaling'] != 'none'
            or source['fit_rows'] != [0,n] or source['fit_rows_available'] != n
            or source['arrays']['X'] != arr or 'regression=True' not in source['loader']):
        raise ValueError('Not the reviewed complete raw regression-population X-only input')
    lane = row['lane'];more = lane == 'tsvd';ds = meta['dataset'];cut = max(n - n//10,1)
    shapes = {'X':[n,d]} if more else {'X':[cut,d],'Xq':[n//10,d]}
    split = dict(fit=[0,n],held_out='none; original loader test split not used') if more else dict(fit=[0,cut],query=[cut,n],rule='Original cut=max(N-N//10,1), X-only archive; no loader Xq; randomized-svd does not use query arrays')
    params = settings(sources, lane)
    paths = ['$.components'] if more or lane == 'randomized-svd' else ['$.pred']
    if lane == 'incremental-pca':paths += ['$.components','$.mean']
    harness = 'tools/bench_board_more.py' if more else 'tools/bench_board_algos.py'
    inference = 'not_applicable' if more or lane == 'randomized-svd' else 'separate'
    if (lane not in LANES or lane != expected_lane or ds != expected_dataset or ds != row['dataset'] or work['dataset'] != ds or work['lane'] != lane
            or row['original_workload_id'] != original['workload_id']
            or row['configuration'] != original['configuration'] or row['vendor'] != original['vendor']
            or row['changes_frozen_race'] is not True or row['new_preparation_cap'] is not None
            or work['actual_shapes'] != shapes or facts['dimensions'] != shapes
            or work['split'] != split or work['seed'] != 7 or work['inference'] != inference
            or work['harness'] != harness or work['harness_sha256'] != digest(ROOT/harness)
            or work['estimator_settings_record'] != params or facts['estimator_settings'] != params
            or work['output_paths'] != paths or facts['dataset_sha256'] != canonical_hash(meta['arrays'])
            or work.get('overrides') or work.get('adapter')
            or work.get('capture_limitation') != row['capture_limitation']):
        raise ValueError('Full variant differs from the registered input/split/settings/output contract')
