"""Source architecture/ownership regressions; public runtime qualification separate."""
from pathlib import Path
from check_generation_native_boundary import audit

ROOT = Path(__file__).resolve().parents[1]


def text(path):
    return (ROOT / path).read_text()


def test_native_engine_transitive_closure_has_no_provider_or_python():
    _, problems = audit(ROOT, ('transformer.native_generation_engine',))
    assert not problems, problems


def test_token_loop_uses_native_state_and_original_sampling_policy():
    src = text('transformer/native_generation_engine.mojo')
    read = src.split('def native_generation_read(')[1]
    assert 'model.forward_into(Int(progress.current.unsafe_ptr()), progress.length, progress.logits)' in read
    assert 'causal_lm_pick_step(' in read
    for forbidden in ('PythonObject', 'py_call', 'py_get', '._addr', '.shape', 'getenv', 'GILReleased'):
        assert forbidden not in src
    assert 'swap(progress.current, progress.next)' in read
    assert 'progress.produced += 1' in read and 'progress.wrote += 1' in read


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
    provider = text('providers/python/generation_metadata_pyapi.mojo').split('struct GenerationEngineAdmission')[1]
    assert '_generation_engine_state(key, [cache_count]' in provider
    assert 'self.owners.append(AdmittedNativeExecution(provider, session, handle,' in provider
    assert 'OperationLockLease(lock_owner)' in provider
    assert '-1 if cpu else 0, -1 if cpu else 1, lock_owner, lock.handle)' in provider
    assert 'self.reserved' not in provider
    assert 'native_layer.failure = String(problem)' in provider
    assert 'counter_value = self.native.state_counters[identity]' in provider
    assert 'py_setitem(self.states[i], counter, PythonObject(counter_value))' in provider
    assert 'py_setattr(self.states[i]' not in provider
    assert '"_usable", PythonObject(False)' in provider
    assert 'if self.published:' in provider
    public = text('bindings/causal_lm_public.mojo')
    read = public.split('def _read_engine(')[1].split('def _read(')[0]
    assert 'owner.reserve()' not in read and 'GILReleased' not in read
    assert read.count('owner.publish()') == 2
    assert read.count('_engine_publish_stream(s, progress, batch)') == 2


def test_experiment_never_silently_relabels_unsupported_routes():
    public = text('bindings/causal_lm_public.mojo')
    setup = public.split('def _engine_admission(')[1].split('def _run_engine(')[0]
    assert 'distributed execution is pending' in setup
    assert 'does not replace the existing resident route' in setup
    assert 'MOJOLEARN_GENERATION_NATIVE_ENGINE' in public
    provider = text('providers/python/generation_metadata_pyapi.mojo')
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
    provider = text('providers/python/generation_metadata_pyapi.mojo')
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
    provider=text('providers/python/generation_metadata_pyapi.mojo')
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
    public=text('bindings/causal_lm_public.mojo')
    assert 'return _run_engine(mods, model, ids, state)' in public
    assert 'logits = owner.native.forward(address, length)' in public
    original=public.split('def _run_snapshot(')[1].split('\ndef ')[0]
    assert 'x, state.layers[i], False, PythonObject(None))' in original
    assert 'addresses[layer.output_slot]' in native
    assert 'layer.addresses[15] = _generation_zero_workspace(layer.report_workspace, layer.report_size)' in native
    assert 'layer.cached + length if kind == 1 else result' in native
    provider=text('providers/python/generation_metadata_pyapi.mojo')
    assert 'native_layer.floats = [layer.words.dt_low, layer.words.dt_high]' in provider
    assert '"absolute_position" if self.native.layers[i].kind == 1 else "buffered_tokens"' in provider
    assert 'py_setitem(self.snapshot.layers[i].block, "h_last_", report)' in provider
