# SPDX-License-Identifier: Apache-2.0
"""Non-building checks for neural metadata admission and native dependencies."""
from pathlib import Path
import re

import pytest

from boundary_keys_plain import plain
from check_neural_native_boundary import audit, imports

ROOT = Path(__file__).resolve().parents[1]


def function(path, name):
    # Doors that read through held name keys are read in their plain (by-name) form.
    source = plain(path, (ROOT / path).read_text())
    return re.search(rf'^def {name}\(.*?(?=^def |^struct |\Z)', source,
                     flags=re.M | re.S).group()


def test_native_execution_closures_have_no_provider_dependencies():
    result = audit()
    assert result['passed'], result['failures']
    assert len(result['source_modules']) >= len(result['roots'])


@pytest.mark.parametrize('dependency', ['std.python', 'bindings.py_balanced',
                                       'providers.python.autograd_program_arrays_pyapi_door', 'adapters.torch',
                                       'providers.python.buffers'])
def test_transitive_provider_dependency_is_rejected(tmp_path, dependency):
    (tmp_path / 'core').mkdir()
    (tmp_path / 'core/entry.mojo').write_text('from core.inner import run\n')
    (tmp_path / 'core/inner.mojo').write_text(f'from {dependency} import boundary\n')
    result = audit(tmp_path, ('core.entry',))
    assert not result['passed']
    assert result['failures'][0]['chain'] == ['core.entry', 'core.inner', dependency]


def test_dependency_parser_ignores_comments_and_documentation():
    assert imports('''# from std.python import Python
"""from bindings.py_balanced import py_call"""
from core.cast_in import (
    CastIn,
)
''') == ['core.cast_in']


def test_sequential_snapshot_is_reused_after_admission():
    source = (ROOT / 'bindings/sequential_execution_pack.mojo').read_text()
    assert 'sequential_descriptor(x,' not in source
    assert 'sequential_descriptor(vx,' not in source
    assert 'sequential_sample(model,' not in source
    assert not re.search(r"\b(?:xd|vxd)\[", source)
    assert source.count('sequential_x_snapshot(') == 3  # fit, validation, prediction
    snapshot = function('bindings/sequential_arrays_door.mojo', '_sequential_snapshot')
    for field in ('shape', 'strides', 'dtype'):
        assert snapshot.count('value.' + field) == 1
    assert 'return SequentialInputSnapshot(value, source,' in snapshot
    assert 'CastIn(' in snapshot
    # No model/global cache can survive this new input's operation.
    assert 'py_setitem' not in snapshot and '_Global' not in snapshot


def test_program_input_reads_provider_fields_before_native_rank_loop():
    source = function('providers/python/autograd_program_pyapi_door.mojo', 'array_input')
    for field in ('shape', 'strides', 'dtype'):
        assert source.count('a.' + field) == 1
    assert 'return (src, sh)' in source


def test_samba_dropout_consumes_native_metadata_in_both_directions():
    source = function('bindings/samba_training_native_door.mojo', '_samba_dropout')
    for token in ('samba_fields(', 'config[', 'value.shape', 'value.size', 'fields['):
        assert token not in source
    # The public samba_gradients is a thin wrapper; the body is _samba_gradients (admitted flag).
    whole = function('bindings/samba_training_native_door.mojo', '_samba_gradients')
    assert whole.count('SambaDropoutMetadata(fields, dropout_probability, stream)') == 1
    assert whole.count('dropout_metadata.value()') == 2
    assert 'var i = layer_count - reverse - 1' in whole


def test_mlp_packing_reuses_destination_address():
    # Packing moved to _mlp_pack: the destination address is read once and every layer is
    # copied by one native pack call (no per-layer Python-side destination arithmetic).
    pack = function('bindings/xseq_neural_native_door.mojo', '_mlp_pack')
    assert pack.count('xseq_addr(params)') == 1
    assert pack.count('xseq_run_mlp_pack(') == 1 and 'dest=' not in pack
    # The read is admitted through xseq_private_input first (cf832cdf7d).
    assert 'admit_nd_f32(xseq_private_input(coefs[l], context, "coefs_"), "coefs_", False, context)' in pack
    # Current public weights are read each prediction; the packed record is reused only for the
    # same coefs_/intercepts_ arrays while they stay read-only (p3-cnn-sequence-tsa), so a
    # changed or writable weight never serves a stale pack.
    source = function('bindings/xseq_neural_native_door.mojo', '_mlp_forward')
    assert 'var coefs = fields["coefs_"]' in source
    assert '_record_get(mods, _MLP_PACKED_SLOT, model, coefs, intercepts)' in source
    assert '_xseq_params_frozen(coefs, intercepts, False, context) == 0' in source
    assert source.index('_xseq_params_frozen(coefs, intercepts, False, context) == 0') < source.index('_mlp_pack(')


def test_inventory_covers_every_current_gpu_neural_recipe():
    import json
    import bench_board_neural as neural

    inventory = json.loads((ROOT / 'experiments/neural_metadata_once_20261027/inventory.json').read_text())
    expected = {lane for lane in neural.LANES if neural.DEVICE_OF[lane] == 'gpu'}
    rows = inventory['rows']
    assert inventory['recipe_count'] == len(rows) == len(expected) == 29
    assert {row['lane'] for row in rows} == expected
    # The inventory is the dated 2026-10-27 record: its paths predate fence-flip-2 (fed1b4e384,
    # boundary files renamed to *_door/*_pack/*_unpack); the neural_gemm export's boundary was
    # deleted whole by bb0483fd27. Every other source must still exist under one of those names.
    deleted = {'bindings/neural_gemm_boundary.mojo', 'bindings/neural_gemm_boundary_common.mojo'}
    for row in rows:
        for source in row['sources']:
            if source in deleted:
                continue
            stem = source[:-len('.mojo')] if source.endswith('.mojo') else source
            candidates = [source] + [stem + suffix + '.mojo' for suffix in ('_door', '_pack', '_unpack')]
            assert any((ROOT / name).is_file() for name in candidates), source
        assert row['public_runtime_validated'] is False
