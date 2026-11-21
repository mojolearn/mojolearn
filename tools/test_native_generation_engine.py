"""Source architecture/ownership regressions; public runtime qualification separate."""
from pathlib import Path
from check_generation_native_boundary import audit

ROOT = Path(__file__).resolve().parents[1]


def text(path):
    return (ROOT / path).read_text()


def test_native_engine_transitive_dependencies_resolve():
    _, problems = audit(ROOT, ('transformer.native_generation_engine',))
    assert not problems, problems


def test_engine_has_no_token_loop():
    # Lane x45c: generation reads run as the decode program; the engine keeps
    # only CausalLM.forward's layer walk.
    src = text('transformer/native_generation_engine.mojo')
    assert 'def native_generation_read(' not in src
    for forbidden in ('PythonObject', 'py_call', 'py_get', '._addr', '.shape', 'getenv', 'GILReleased'):
        assert forbidden not in src


def test_numerical_order_and_native_cached_token_publication_are_explicit():
    src = text('transformer/native_generation_engine.mojo')
    body = src.split('def forward(mut self,')[1].split('struct NativeGenerationProgress')[0]
    assert body.index('self.primitive_call(0,') < body.index('layer.endpoint.call(')
    assert body.index('layer.endpoint.call(') < body.index('self.primitive_call(2,') < body.index('self.primitive_call(4,')
    assert 'ref layer = self.layers[i]' in body
    assert 'layer.parameters[8] = layer.cached' in body
    assert 'var result = layer.endpoint.call' in body
    assert body.index('self.position += length') > body.index('self.primitive_call(4,')


def test_reservations_and_outputs_remain_at_checked_boundary():
    provider = text('providers/python/generation_metadata_pyapi_door.mojo').split('struct GenerationEngineAdmission')[1]
    assert '_generation_engine_state(key, [cache_count]' in provider
    assert 'self.owners.append(AdmittedNativeExecution(self.provider, session, handle,' in provider
    # Lane dw-causal (continuation 4): a layer whose state the state owner's
    # words describe takes its buffer addresses from them; the object
    # admission stays for any other layer.
    assert 'causal_state_buffer_address(words.buffers[CAUSAL_STATE_K], cache_shape, "k_cache")' in provider
    assert 'def admit_layer(mut self, snapshot: GenerationOperationSnapshot' in provider
    public = text('bindings/causal_lm_public_door.mojo').split('def _engine_admission(')[1].split('def _run_engine(')[0]
    assert public.index('causal_state_owner(mods, state)') < public.index('causal_state_layer_unpack(mods, state_held, i, st, kind)')
    assert 'windowed_result5.state_owner = state_held' in public
    assert 'OperationLockLease(lock_owner)' in provider
    assert '-1 if cpu else 0, -1 if cpu else 1, lock_owner, lock.handle)' in provider
    assert 'self.reserved' not in provider
    assert 'native_layer.failure = String(problem)' in provider
    assert 'counter_value = self.native.state_counters[identity]' in provider
    assert 'py_setitem(self.states[i], counter, PythonObject(counter_value))' in provider
    assert 'py_setattr(self.states[i]' not in provider
    assert '"_usable", PythonObject(False)' in provider
    assert 'if self.published:' in provider


def test_experiment_never_silently_relabels_unsupported_routes():
    public = text('bindings/causal_lm_public_door.mojo')
    setup = public.split('def _engine_admission(')[1].split('def _run_engine(')[0]
    assert 'distributed execution is pending' in setup
    assert 'does not replace the existing resident route' in setup
    assert 'MOJOLEARN_GENERATION_NATIVE_ENGINE' in public
    provider = text('providers/python/generation_metadata_pyapi_door.mojo')
    assert '_generation_engine_weights(snapshot.embedding' in provider
    state = provider.split('def _generation_engine_state(', 1)[1].split('struct GenerationEngineAdmission', 1)[0]
    assert state.index('generation_buffer_view(value)') < state.index('AdmittedBorrowedBuffer(view, True)')
    assert 'keep.append(view)' in state
    assert 'foreign-weight descriptor admission is pending' not in provider


def test_deferred_layer_refusals_keep_earlier_progress_and_profile_publication():
    src = text('transformer/native_generation_engine.mojo')
    forward = src.split('def forward(mut self,')[1].split('struct NativeGenerationProgress')[0]
    assert forward.index('self.primitive_call(0,') < forward.index('layer.visited = True')
    assert forward.index('raise Error(layer.failure)') < forward.index('layer.endpoint.call(')
    provider = text('providers/python/generation_metadata_pyapi_door.mojo')
    assert 'native_layer.failure = String(problem)' in provider
    assert 'self.native.layers[i].visited and self.native.layers[i].profile_admitted' in provider


def test_delayed_foreign_copies_preserve_layer_order_and_alias_counters():
    native=text('transformer/native_generation_engine.mojo')
    body=native.split('def forward(mut self,')[1].split('struct NativeGenerationProgress')[0]
    assert body.index('refresh_layouts[j].copy_to(') < body.index('layer.endpoint.call(')
    assert body.index('layer.endpoint.call(') < body.index('writeback_layouts[j].copy_from(')
    assert 'self.state_counters[layer.state_index] = layer.cached' in body
    assert body.index('embedding_refresh[i].copy_to(') < body.index('self.primitive_call(0,')
    assert body.index('head_refresh[i].copy_to(') < body.index('self.primitive_call(2,')
    provider=text('providers/python/generation_metadata_pyapi_door.mojo')
    assert 'identity = py_get(fields, "_native_state_identity", py_get(fields, "_causal_original", st))' in provider
    assert 'var identity_key = Int(identity._obj_ptr)' in provider
    assert 'state_indices.get(identity_key, -1)' in provider
    assert 'state_indices[identity_key] = state_index' in provider
    assert 'self.state_keys.append(identity)' in provider
    weights = provider.split('def _generation_engine_weights(', 1)[1].split('def _generation_engine_state(', 1)[0]
    assert 'snapshot.layouts[k].copy()' in weights
    assert 'AdmittedBorrowedBuffer(' not in weights and 'generation_buffer_view(' not in weights
    assert 'snapshot.failures[k]' in weights
    assert 'var pins: List[AdmittedBorrowedBuffer]' in provider
    assert 'var storage: List[List[Float32]]' in provider


def test_mamba_engine_keeps_local_forward_route_report_and_counter_semantics():
    native=text('transformer/native_generation_engine.mojo')
    assert 'def forward(mut self, ids: Int, length: Int)' in native
    assert 'operation += 1' not in native
    assert 'parameters = decode' not in native
    public=text('bindings/causal_lm_public_door.mojo')
    # f6e85dffcb: the route is decided once per forward and the resident verdict is passed down
    # dw-causal (2026-11-21): the route decision reads the model owner's typed words
    assert 'return _run_engine(mods, words, held, model, ids, state, verdict == 0)' in public
    assert 'if not causal_engine_candidate(words):' in public
    # 1ce1204524 (row N199): the head writes straight into the caller's result Array
    assert 'owner.native.forward_to(address, length, out_address, owner.native.batch * length * owner.native.vocab)' in public
    assert 'def forward_to(mut self, ids: Int, length: Int, address: Int, count: Int)' in native
    original=public.split('def _run_snapshot(')[1].split('\ndef ')[0]
    # the original per-layer route (stateful, not a step); fe44134875 dropped the deleted mamba_workspace argument
    # dw-neural-models (2026-11-21): the snapshot's typed layer route replaces layer_entry / layer_vendor
    # dw-causal (2026-11-21): the owner's snapshot with this operation's block captures and profile words
    assert '_causal_layer_run_bound(mods, snapshot.layer_route, snapshot.entries, snapshot.width, b, l,' in original
    # dw-causal continuation 3: the layer list is read once; a layer with current owner words takes the owned run
    assert 'x, st, False,' in original and 'var layer_states = state.layers' in original
    assert '_causal_layer_run_owned(mods, snapshot.layer_route, snapshot.entries, snapshot.width, b, l,' in original
    assert 'call.blocks[i], call.profiles[i], call.changed[i], False)' in original
    assert 'addresses[layer.output_slot]' in native
    assert 'layer.addresses[15] = _generation_zero_workspace(layer.report_workspace, layer.report_size)' in native
    assert 'layer.cached + length if kind == 1 else result' in native
    provider=text('providers/python/generation_metadata_pyapi_door.mojo')
    assert 'native_layer.floats = [layer.words.dt_low, layer.words.dt_high]' in provider
    assert '"absolute_position" if self.native.layers[i].kind == 1 else "buffered_tokens"' in provider
    assert 'py_setitem(self.call.blocks[i], "h_last_", report)' in provider
