"""Reuse an unchanged reviewed full-input recipe with explicit new A/B controls.

This is benchmark metadata admission, not product execution. The old registration
remains immutable. Its existing validator still checks every input, shape, cap,
setting and boundary; only the experiment cell and measurement freeze differ.
Numerical binary/source/define admission remains the materializer's responsibility.
"""
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'mojolearn.full-input-control-transfer/1'


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def reference_identities(value):
    """Permit path-only transport of SHA-bound metadata, never changed bytes."""
    if isinstance(value, dict):
        return {k: ('SHA_BOUND_RELOCATION' if k == 'path' and 'sha256' in value
                    else reference_identities(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [reference_identities(v) for v in value]
    return value


def original_variant(reg):
    """Derive the reviewed variant; generated variants need not be serialized."""
    from six_lane_matrix_io import read_matrix
    matrix = read_matrix(ROOT / 'experiments/six_lane_integration/matrix.json.gz')
    matches = [c for c in matrix['cells'] if c['key'] == reg.get('original_cell_key')
               and c['workload_id'] == reg.get('original_workload_id')]
    if len(matches) != 1:
        raise ValueError('Transfer lacks exact original matrix cell')
    original = matches[0]
    if reg['variant'] == 'classification-full-v1':
        from six_lane_classification_variants import variant_cell
        result = variant_cell(original)
    elif reg['variant'] == 'mlp-full-v1':
        from six_lane_mlp_variants import contracts, variant_cell
        rows = [r for r in contracts()['rows'] if r['scope_cell_key'] == original['key']
                and r['original_workload_id'] == original['workload_id']]
        if len(rows) != 1:
            raise ValueError('Transfer lacks reviewed MLP variant contract')
        result = variant_cell(original, rows[0])
    else:
        from six_lane_full_variants import variant_cell
        result = variant_cell(original)
    if (result['key'] != reg.get('variant_cell_key')
            or result['workload_id'] != reg.get('variant_workload_id')):
        raise ValueError('Original registered full-input variant identity changed')
    return result


def validate_transfer(facts, cell, validate_original):
    transfer = facts['input_variant_control_transfer']
    reg = facts.get('registered_input_variant', {})
    if (transfer.get('schema') != SCHEMA or facts.get('changes_frozen_race') is not True
            or reg.get('variant') not in ('classification-full-v1', 'tsvd-full-v1', 'mlp-full-v1')):
        raise ValueError('Unknown targeted full-input transfer')
    original = original_variant(reg)
    target = transfer.get('target', {})
    required = ('configuration', 'vendor', 'mode', 'workload_id', 'key')
    if any(k not in target for k in required):
        raise ValueError('Incomplete targeted cell identity')
    if (target['vendor'] != original['vendor'] or target['mode'] != original['mode']
            or target['mode'] != 'identical' or target['workload_id'] != original['workload_id']
            or target['configuration'] == original['configuration']
            or target['key'] != canonical([target['configuration'], target['vendor'], target['workload_id']])[:20]):
        raise ValueError('Targeted transfer changed vendor, mode or reviewed full workload')
    if (not isinstance(transfer.get('measurement_source_sha'), str)
            or len(transfer['measurement_source_sha']) != 40
            or facts.get('source_sha') != transfer['measurement_source_sha']):
        raise ValueError('Targeted measurement freeze differs')
    if cell and any(cell.get(k) != target[k] for k in required):
        raise ValueError('Targeted materialization cell differs')
    job = facts.get('job')
    if job and (job.get('key') != target['key'] or job.get('workload_id') != target['workload_id']
                or job.get('mode') != target['mode'] or facts.get('vendor') != target['vendor']
                or job.get('master_selection', {}).get('id') != target['configuration']):
        raise ValueError('Targeted worker identity differs')
    reference = transfer.get('original_facts', {})
    path = Path(reference.get('path', ''))
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != reference.get('sha256'):
        raise ValueError('Original full-input facts missing or changed')
    old = json.loads(path.read_text())
    if reference_identities(old.get('registered_input_variant')) != reference_identities(reg):
        raise ValueError('Original full-input registration changed')
    # Capture-only extensions do not alter numerical execution or old recipes.
    old_work = copy.deepcopy(old['workload']); new_work = copy.deepcopy(facts['workload'])
    for work in (old_work, new_work):
        work.pop('model_state_paths', None)
        work.pop('required_model_state_contract', None)
        work.pop('data_directory', None)
        work.pop('intrinsic_cap_audit', None)
        inputs = work.pop('input_files')
        work['input_identities'] = sorted((Path(x['path']).name, x['sha256']) for x in inputs)
    if old_work != new_work:
        raise ValueError('Targeted transfer altered reviewed workload/settings/output contract')
    for key in ('dataset_sha256', 'dimensions', 'estimator_settings', 'timed_boundary',
                'intrinsic_caps', 'full_dataset_coverage'):
        if old.get(key) != facts.get(key):
            raise ValueError('Targeted transfer altered reviewed full recipe: ' + key)
    audit = facts['workload'].get('intrinsic_cap_audit', {})
    if not audit.get('reviewed') or audit.get('unresolved'):
        raise ValueError('Targeted full-input cap audit incomplete')
    # Validate the reference identity as a reference, not as current numerical
    # provenance. Current artifact flags/source/target are checked independently.
    probe = copy.deepcopy(facts)
    probe.pop('input_variant_control_transfer')
    probe['source_sha'] = reg.get('measurement_source_sha', reg.get('source_sha'))
    if reg['variant'] == 'classification-full-v1':
        from six_lane_classification_variants import CONTRACT, contracts
        current_path = ROOT / CONTRACT
        current_hash = hashlib.sha256(current_path.read_bytes()).hexdigest()
        if reg.get('contract_sha256') != current_hash:
            # Later contracts may append unrelated reviewed algorithms. Retain
            # and compare the old selected row; a new file hash is not evidence
            # that an old input recipe or its preprocessing may be changed.
            reference = transfer.get('original_contract', {})
            path = Path(reference.get('path', ''))
            if (not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()
                    != reg.get('contract_sha256') or reference.get('sha256') != reg.get('contract_sha256')):
                raise ValueError('Original classification contract unavailable')
            previous = json.loads(path.read_text()); current = contracts()
            def selected(doc):
                return [r for r in doc['rows'] if r['vendor'] == original['vendor']
                        and r['configuration'] == original['configuration']
                        and r['original_workload_id'] == reg['original_workload_id']]
            if (previous.get('schema') != current.get('schema')
                    or previous.get('variant') != current.get('variant')
                    or len(selected(previous)) != 1 or selected(previous) != selected(current)
                    or any(current['source_hashes'].get(p) != h for p, h in previous['source_hashes'].items())):
                raise ValueError('Previously reviewed classification contract row/source changed')
            probe['registered_input_variant']['contract_sha256'] = current_hash
    probe['workload'].pop('required_model_state_contract', None)
    if reg['variant'] == 'mlp-full-v1':
        # Its validator requires exact original capture metadata as well.
        if 'model_state_paths' in old['workload']:
            probe['workload']['model_state_paths'] = old['workload']['model_state_paths']
    if job:
        probe['job'].update(key=original['key'], workload_id=original['workload_id'],
                            mode=original['mode'], master_selection={'id': original['configuration']})
    validate_original(probe, original)
