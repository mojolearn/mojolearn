# SPDX-License-Identifier: Apache-2.0
"""Source contracts for generation admission; not compiled/runtime evidence."""
from pathlib import Path
import re

from check_neural_native_boundary import NATIVE_ROOTS, audit

ROOT = Path(__file__).resolve().parents[1]
BINDING = (ROOT / 'bindings/causal_lm_public_door.mojo').read_text()
PROVIDER = (ROOT / 'providers/python/generation_metadata_pyapi_door.mojo').read_text()
NATIVE = (ROOT / 'core/generation_metadata.mojo').read_text()


def function(name):
    return re.search(rf'^def {name}\(.*?(?=^def |^struct |\Z)', BINDING, re.M | re.S).group()


def test_native_metadata_dependency_closure_is_enforced():
    assert 'core.generation_metadata' in NATIVE_ROOTS
    result = audit(roots=('core.generation_metadata',))
    assert result['passed'], result['failures']
    assert 'PythonObject' not in NATIVE


def test_layer_request_and_route_types_are_native():
    # dw-neural-models (2026-11-21): the typed layer request and route import no Python.
    result = audit(roots=('core.causal_layer_run',))
    assert result['passed'], result['failures']
    source = (ROOT / 'core/causal_layer_run.mojo').read_text()
    assert 'PythonObject' not in source and 'std.python' not in source


def test_layer_inner_execution_uses_native_dimensions_and_weight_addresses():
    body = function('_causal_layer_run_bound')
    assert 'return _causal_layer_run_bound(' in function('_causal_layer_run_prepared')
    for field in ('d_model', 'n_heads', 'n_kv_heads', 'head_dim', 'intermediate',
                  'window', '_opts_tail', '_w', '_wopt', '_int15', 'dt_limit',
                  'd_inner', 'nheads', 'conv_dim'):
        assert f'blk["{field}"]' not in body
    assert '_transformer_param_words(' not in body
    assert '_transformer_arg_words(' not in body
    assert 'layer.weights.resolve(keep)' in body
    assert 'layer.words.transformer_params(b, l, capacity, Int(py=st.cached_tokens))' in body
    assert 'Int(py=st.buffered_tokens)' in body
    assert '_mutable(st.k_cache' in body and '_mutable(st.h' in body
    assert body.index('_ = layer') > body.index('_writeback(copies)')


def test_native_template_keeps_mutable_progress_out_of_snapshot():
    for assignment in ('result[0] = batch', 'result[1] = length',
                       'result[7] = capacity', 'result[8] = cached'):
        assert assignment in NATIVE
    # Immutable weight/dimension snapshots exclude progress. The separate
    # operation-local engine admission reads progress once into native state.
    snapshots = PROVIDER.split('struct GenerationEngineAdmission')[0]
    assert 'st.cached_tokens' not in snapshots
    assert 'st.buffered_tokens' not in snapshots
    assert 'st.positions' not in snapshots


def test_snapshot_is_lazy_and_scoped_to_one_public_operation():
    run = function('_run_prepared')
    capture = run.index('snapshot = GenerationOperationSnapshot(')
    assert capture > run.index('py_call(parallel,')
    assert capture > run.index('return logits')
    assert run.index('_state(mods, model, state, b, l)') < capture
    read = function('_read')
    declaration = 'var snapshot: Optional[GenerationOperationSnapshot] = None'
    assert read.index('try:') < read.index(declaration) < read.index('while wrote')
    assert declaration in function('_run')
    handle = re.search(r'^struct GenerationHandle.*?(?=^def |\Z)', BINDING, re.M | re.S).group()
    assert 'GenerationOperationSnapshot' not in handle
    assert 's.usable = False' in read.split('except error:', 1)[1]
    assert 's.busy = False' in read.rsplit('finally:', 1)[1]


def test_exact_array_owners_pin_addresses_and_foreign_copies_still_refresh():
    assert 'elif py_type(value) is cls:' in PROVIDER
    assert 'var captured = generation_array_snapshot(value)' in PROVIDER
    assert 'self.owners.append(captured)' in PROVIDER
    assert 'self.addresses.append(model_integer(captured._addr))' in PROVIDER
    assert 'view = generation_buffer_view(value)' in PROVIDER
    assert 'self.owners.append(view)' in PROVIDER
    assert 'self.layouts.append(layout^)' in PROVIDER
    assert 'self.formats.append(format)' in PROVIDER
    assert 'self.refresh.append(j)' in PROVIDER
    resolve = PROVIDER.split('def resolve(', 1)[1].split('struct GenerationLayerSnapshot', 1)[0]
    assert 'layout.copy_to(address)' in resolve
    assert 'keep.append(packed)' in resolve
    assert 'generation_address(' not in resolve and 'generation_buffer_view(' not in resolve
    assert 'self.failures[k]' in resolve  # foreign refusals retain the original use position
    assert '_ = weights' in function('_causal_model_embedding_prepared')
    assert '_ = head' in function('_causal_model_head_prepared')
    assert '_ = snapshot' in function('_run_snapshot')


def test_prepared_embedding_head_and_layer_loop_do_not_read_model_metadata():
    for name in ('_causal_model_embedding_prepared', '_causal_model_head_prepared', '_run_snapshot'):
        body = function(name)
        assert 'model.' not in body
        assert 'x.shape' not in body and 'ids.shape' not in body
    assert 'for i in range(len(snapshot.layers))' in function('_run_snapshot')


def test_existing_standalone_and_profile_routes_remain_available():
    # dw-neural-models (2026-11-21): the SimpleNamespace-owner layer run (causal_layer_run) had no caller and is
    # deleted; the standalone door is causal_layer_run_fields with the plan's typed route.
    assert not re.search(r'^def causal_layer_run\(', BINDING, re.M)
    layer = function('causal_layer_run_fields')
    assert 'GenerationLayerSnapshot(fields, kind, planes)' in layer
    assert 'py_contains(entries' not in layer and 'entries[' not in layer
    body = function('_causal_layer_run_bound')
    run = function('_causal_layer_request_run')
    assert 'PythonObject' not in run and 'py_' not in run
    assert 'transformer_block_session_run_words(request.entry, request.handle, request.operation,' in run
    assert '_causal_table_call(request.entry, request.operation, request.addresses, request.words, request.floats)' in run
    # the layer run reads no entry word by name per call: the arms test the typed route
    for name in ('e["cpu"]', 'py_contains(e, "block_session")', 'e["mamba1_profile"]', 'causal_entry_key(e)'):
        assert name not in body
    assert body.index('\n    # RUN\n') < body.index('result = _causal_layer_request_run(request)') < body.index('\n    # PACK\n')
    assert 'return _causal_layer_run_bound(' in function('_causal_layer_run_prepared')
    # 3daf242df3: the CPU block runs through the CausalLM entry table (plain and fixed15 ops); the GPU block runs
    # its native session (plain and fixed15 ops); the bound Python forward/decode entries remain the fallback.
    for route in ('CAUSAL_BLOCK_HOST_FORWARD_INT15 if head_profile else CAUSAL_BLOCK_HOST_FORWARD',
                  'request.handle = transformer_block_session_handle(session)',
                  'NEURAL_BLOCK_SESSION_FORWARD_INT15 if device15 else NEURAL_BLOCK_SESSION_FORWARD',
                  'e["decode" if step else "forward"]'):
        assert route in body
    # Resident session setup retains its original, separate metadata admission.
    assert '_transformer_params(' in BINDING
    assert 'self.planes = self.head.packed and py_contains(entries, "forward15")' in PROVIDER
