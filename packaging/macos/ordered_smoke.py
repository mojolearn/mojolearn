"""Installed OrderedRMSE release gate. Imported by smoke.py; no work on import."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import runpy
import sys


MODE_CODES = {'fast': 0, 'identical': 1}


def admit_output(output, mode, vendor):
    lines = output.splitlines()
    records = [line.removeprefix('ORDERED_PYTHON_JSON ') for line in lines
               if line.startswith('ORDERED_PYTHON_JSON ')]
    if len(records) != 1 or lines.count('ORDERED PYTHON SURFACE PASS') != 1:
        raise AssertionError('Missing or duplicated OrderedRMSE completion evidence')
    record = json.loads(records[0])
    if (record.get('schema') != 'mojolearn.ordered_python.v1' or record.get('status') != 'PASS'
            or record.get('requested_mode') != mode or record.get('native_vendor') != vendor
            or record.get('native_mode_code') != MODE_CODES[mode]):
        raise AssertionError('OrderedRMSE installed mode/vendor/status mismatch')
    if (record.get('rows') != 32 or record.get('query_rows') != 8 or record.get('trees') != 3
            or len(record.get('prediction_bits', [])) != 32
            or len(record.get('query_prediction_bits', [])) != 8
            or len(record.get('repeat_prediction_bits', [])) != 32):
        raise AssertionError('Incomplete OrderedRMSE fixture evidence')
    if set(record.get('refusals', [])) != {
            'depth', 'objective', 'categorical', 'duplicate_permutation', 'zero_weight_mass', 'eval_set'}:
        raise AssertionError('Incomplete OrderedRMSE refusal evidence')
    model = record.get('model_text', '')
    if not model or hashlib.sha256(model.encode()).hexdigest() != record.get('model_sha256'):
        raise AssertionError('OrderedRMSE model bytes/hash disagree')
    if mode != 'fast' and (record.get('repeat_exact') is not True or
                           record['prediction_bits'] != record['repeat_prediction_bits']):
        raise AssertionError('OrderedRMSE pinned-mode repeat evidence differs')
    serialization = record.get('serialization', {})
    if (serialization.get('saved_mode') != mode or serialization.get('restored_mode') != mode
            or serialization.get('restored_native_mode_code') != MODE_CODES[mode]
            or serialization.get('restored_native_vendor') != vendor
            or any(serialization.get(key) is not True for key in
                   ('model_text_equal', 'training_predictions_equal', 'query_predictions_equal'))):
        raise AssertionError('Incomplete OrderedRMSE saved-mode restoration evidence')
    selected = serialization.get('selected_modes', [])
    installed = serialization.get('installed_gbdt_tiers', [])
    for values in (selected, installed):
        if (not isinstance(values, list) or not values or any(type(v) is not str for v in values)
                or len(values) != len(set(values)) or not set(values) <= set(MODE_CODES)
                or mode not in values):
            raise AssertionError('Invalid OrderedRMSE tier coverage')
    alternate = 'fast' if mode == 'identical' else 'identical'
    reason = ('alternate_tier_not_installed' if alternate not in installed else
              'alternate_tier_not_selected' if alternate not in selected else None)
    expected = {'status': 'NOT_EXERCISED' if reason else 'PASS', 'reason': reason}
    if (serialization.get('alternate_mode') != alternate
            or serialization.get('alternate_default_contrast') != expected
            or serialization.get('process_default_before_load') != (mode if reason else alternate)):
        raise AssertionError('OrderedRMSE alternate-default coverage is mislabeled')
    return record


def run_installed_ordered(root, package, mode, vendor):
    installed = Path(package.__file__).resolve()
    if not installed.is_relative_to(Path(sys.prefix).resolve()) or 'site-packages' not in installed.parts:
        raise AssertionError('OrderedRMSE release gate requires an isolated installed package')
    binding = package.OrderedRMSE(numeric_mode=mode)._bind('_mojolearn_gbdt')
    binary = Path(binding.__file__).resolve()
    if not binary.is_relative_to(installed.parent):
        raise AssertionError('OrderedRMSE binding is outside the installed package')
    if int(binding.gbdt_numeric_mode()) != MODE_CODES[mode] or str(binding.gbdt_vendor()) != vendor:
        raise AssertionError('OrderedRMSE binary mode/vendor differs from release request')
    target = Path(root) / 'tools/ordered_rmse_surface_check.py'
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            try:
                runpy.run_path(str(target), run_name='__main__')
            except SystemExit as exc:
                raise AssertionError('OrderedRMSE gate exited before returning complete evidence') from exc
    finally:
        # Preserve the full native transcript on failures as well as success.
        print(captured.getvalue(), end='', flush=True)
    admit_output(captured.getvalue(), mode, vendor)
    print('ORDERED_INSTALLED_JSON ' + json.dumps({
        'package': str(installed), 'version': package.__version__, 'mode': mode, 'vendor': vendor,
        'binding': str(binary), 'binding_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
        'gate_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
    }, sort_keys=True), flush=True)
