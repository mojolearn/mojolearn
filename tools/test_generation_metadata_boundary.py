# SPDX-License-Identifier: Apache-2.0
"""Source contracts for generation admission; not compiled/runtime evidence."""
from pathlib import Path
import re

from check_neural_native_boundary import NATIVE_ROOTS, audit

ROOT = Path(__file__).resolve().parents[1]
BINDING = (ROOT / 'bindings/causal_lm_public_door.mojo').read_text()
PROVIDER = (ROOT / 'providers/python/generation_metadata_pyapi_door.mojo').read_text()
OWNER = (ROOT / 'bindings/causal_operation_owner_door.mojo').read_text()
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


def test_model_owner_words_are_native_and_decide_the_forward_route():
    # dw-causal (2026-11-21): the model owner's words and rules import no Python; forward, step and the
    # per-token legacy read decide the route and check the state on them, not on the model record.
    result = audit(roots=('core.causal_owner',))
    assert result['passed'], result['failures']
    source = (ROOT / 'core/causal_owner.mojo').read_text()
    assert 'PythonObject' not in source and 'std.python' not in source
    for name in ('_run', '_run_prepared', '_use_native_generation_engine', '_resident_forward_ok', '_positions'):
        body = function(name)
        for read in ('model.kind', 'model._head_int15', 'model.max_positions', 'model.vocab_size', 'len(model._blocks)',
                     '"parallel_call") is not None', '"native_primitives"'):
            assert read not in body, (name, read)
    forward = function('causal_lm_forward_binding')
    assert forward.index('causal_model_owner(mods, model)') < forward.index('_run(mods, owner.words, model_held, model, ids, state, one)')
    assert 'model.vocab_size' not in forward
    read = function('_read')
    assert read.index('causal_model_owner(mods, s.model)') < read.index('while wrote')
    owner = (ROOT / 'bindings/causal_operation_owner_door.mojo').read_text()
    template = owner.index('var template = py_call(bi.dict, fields)')
    assert owner.index('native_owner = causal_model_owner_build(fields)') < template
    assert 'add_type[CausalModelOwner]("_CausalModelOwner")' in (ROOT / 'bindings/_mojolearn_tokenizer_host.mojo').read_text()


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


def test_snapshot_is_lazy_and_scoped_to_one_model_generation():
    # dw-causal (2026-11-21): the weight snapshot is the model owner's, built once per model generation at the
    # first local forward; one operation keeps only its block captures (GenerationCallLayers), once per read.
    run = function('_run_prepared')
    capture = run.index('call_layers = _owner_call_layers(mods, held, model, True)')
    build = function('_owner_call_layers')
    assert build.index('if not owner[].snapshot:') < build.index('GenerationOperationSnapshot(model, model._native_entries, blocks)')
    assert 'causal_model_snapshot_keep(held, built^)' in build
    keep = OWNER.split('def causal_model_snapshot_keep(', 1)[1].split('\ndef ', 1)[0]
    assert 'if not owner[].snapshot:' in keep and 'built.detach()' in keep  # written once, never replaced
    snapshot = PROVIDER.split('struct GenerationOperationSnapshot', 1)[1].split('\ndef causal_layer_route', 1)[0]
    assert 'var model:' not in snapshot and 'self.model' not in snapshot  # an owner-held snapshot keeps no capture
    assert 'self.snapshot' not in PROVIDER.split('struct GenerationEngineAdmission', 1)[1]  # no per-call deep copy
    assert capture > run.index('py_call(parallel,')
    assert capture > run.index('return logits')
    # dw-causal (2026-11-21): the state check reads the model owner's typed words
    assert run.index('_state(mods, words, model, state, b, l)') < capture
    assert 'model.max_positions' not in function('_positions')
    read = function('_read')
    declaration = 'var call_layers: Optional[GenerationCallLayers] = None'
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
    assert 'var held: PythonObject' in PROVIDER.split('struct GenerationEngineAdmission', 1)[1]  # the owner pins the weights


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


def test_decode_state_owner_is_native_and_the_layer_run_reads_its_words():
    """Lane dw-causal, 2026-11-21 (design 1.2 B): the decode state has a
    native owner behind a handle with a generation check; a whole-model
    layer call takes its state buffers from the owner's words."""
    core = (ROOT / 'core/causal_owner.mojo').read_text()
    assert 'from std.python' not in core and 'PythonObject' not in core
    for name in ('struct CausalStateWords', 'struct CausalLayerStateWords', 'struct CausalBufferWords',
                 'def causal_state_admit(', 'def causal_state_buffer_address('):
        assert name in core
    owner = (ROOT / 'bindings/causal_operation_owner_door.mojo').read_text()
    assert 'struct CausalStateOwner(Movable, Writable):' in owner
    bind = owner.split('def causal_state_owner_bind(', 1)[1].split('\ndef ', 1)[0]
    # the handle: a weakly keyed record; the generation check before reuse
    assert bind.index('prepared_get(mods, _CAUSAL_STATE_SLOT, state)') < bind.index('_causal_state_owner_kept(record, generation, layers)')
    assert 'prepared_put(mods, _CAUSAL_STATE_SLOT, state, held)' in bind
    kept = owner.split('def _causal_state_owner_kept(', 1)[1].split('\ndef ', 1)[0]
    assert 'words.model_generation != generation' in kept and 'originals[i] is not layers[i]' in kept
    capture = owner.split('def causal_state_capture(', 1)[1].split('\ndef ', 1)[0]
    assert 'causal_state_owner_bind(mods, state, fields, layers,' in capture
    unpack = owner.split('def causal_state_layer_unpack(', 1)[1].split('\ndef ', 1)[0]
    # a buffer is read only when its storage pin changed; a foreign exporter takes the object route
    assert 'value is owner[].pins[base + j]' in unpack and 'generation_array_snapshot' not in unpack
    assert 'if py_type(value) is not g[].array_cls:\n                return PythonObject(None)' in unpack
    assert 'add_type[CausalStateOwner]("_CausalStateOwner")' in (ROOT / 'bindings/_mojolearn_tokenizer_host.mojo').read_text()
    public = (ROOT / 'bindings/causal_lm_public_door.mojo').read_text()
    owned = public.split('def _causal_layer_run_owned(', 1)[1].split('\ndef ', 1)[0]
    # no state buffer object, no per-call metadata snapshot of state, no write-back
    for gone in ('_mutable(', '_writeback(', 'st.k_cache', 'st.conv_window', 'py_setattr(', 'getattr'):
        assert gone not in owned, gone
    assert owned.count('causal_state_buffer_address(state.buffers[') == 11
    # the arms of the bound run are intact in the owned one
    assert 'comptime if not IDN_MAMBA12_FORWARD_COPIES_OLD:' in owned
    assert 'not TRANSFORMER_LEGACY_SETUP and route.block_session' in owned
    run = public.split('def _run_snapshot(', 1)[1].split('\ndef ', 1)[0]
    assert run.index('causal_state_owner(mods, state)') < run.index('for i in range(len(snapshot.layers))')
    assert '_causal_layer_run_owned(' in run and '_causal_layer_run_bound(' in run
    state = public.split('def _state(mut mods: NativeModules, words: CausalModelWords', 1)[1].split('\ndef ', 1)[0]
    assert 'causal_state_admit(words, state_owner[].words, b, l)' in state


def test_stream_reads_hold_the_owners_and_take_state_words():
    # Lane dw-causal, continuation 4: a read holds the model and state owners
    # on its stream handle; the program read, the resident load / export and
    # the per-layer route take state buffer addresses from the owner's words.
    public = (ROOT / 'bindings/causal_lm_public_door.mojo').read_text()
    handle = public.split('struct GenerationHandle(', 1)[1].split('def _options(', 1)[0]
    assert 'var model_held: PythonObject' in handle and 'var state_held: PythonObject' in handle
    assert 'stream[].state_held = causal_state_owner(mods, state)' in public
    program = public.split('def _read_program(', 1)[1].split('def _resident_read_disjoint_state(', 1)[0]
    assert 'causal_state_layer_unpack(mods, state_held, i, layer_state, kind)' in program
    assert 'var family = mw.program_read' in program
    refusal = public.split('def _program_refusal(', 1)[1].split('def _program_state_pieces(', 1)[0]
    assert '_native_entries' not in refusal and 'words.norm_eps' in refusal
    load = public.split('def _resident_load_states(', 1)[1].split('def ', 1)[0]
    assert 'causal_state_layer_unpack(mods, state_held, i, st, CAUSAL_LAYER_TRANSFORMER)' in load
    export = public.split('def _resident_export_states(', 1)[1].split('def _legacy_resident_export(', 1)[0]
    assert 'causal_state_buffer_address(' in export
    plan = public.split('struct LegacyGenerationReadPlan(', 1)[1].split('def _resident_export_states(', 1)[0]
    assert 'k_positions' in plan and 'Int(py=state.positions)' not in plan
    assert 'GenerationHandle' not in PROVIDER
