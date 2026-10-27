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
    assert 'model.forward(Int(progress.current.unsafe_ptr()), progress.length)' in read
    assert 'causal_lm_pick_step(' in read
    for forbidden in ('PythonObject', 'py_call', 'py_get', '._addr', '.shape', 'getenv', 'GILReleased'):
        assert forbidden not in src
    assert 'progress.current = next^' in read
    assert 'progress.produced += 1' in read and 'progress.wrote += 1' in read


def test_numerical_order_and_native_cached_token_publication_are_explicit():
    src = text('transformer/native_generation_engine.mojo')
    body = src.split('def forward(mut self,')[1].split('struct NativeGenerationProgress')[0]
    assert body.index('self.primitive_call(0,') < body.index('generation_session_forward(1,')
    assert body.index('generation_session_forward(1,') < body.index('self.primitive_call(2,') < body.index('self.primitive_call(4,')
    assert 'parameters[8] = self.layers[i].cached' in body
    assert 'self.layers[i].cached = generation_session_forward' in body
    assert body.index('self.position += length') > body.index('self.primitive_call(4,')


def test_reservations_and_outputs_remain_at_checked_boundary():
    provider = text('providers/python/generation_metadata_pyapi.mojo').split('struct GenerationEngineAdmission')[1]
    assert 'generation_mutable(key, [cache_count]' in provider
    assert 'self.links.append(linked)' in provider
    assert 'self.reserved.append(i)' in provider and 'while len(self.reserved) > 0' in provider
    assert 'if already_reserved:' in provider
    assert 'self.native.layers[i].failure = String(problem)' in provider
    assert 'Never unlock another owner' in provider
    assert '"cached_tokens", PythonObject(self.native.layers[i].cached)' in provider
    assert '"_usable", PythonObject(False)' in provider
    assert 'if self.published:' in provider
    public = text('bindings/causal_lm_public.mojo')
    read = public.split('def _read_engine(')[1].split('def _read(')[0]
    assert read.index('owner.reserve()') < read.index('with GILReleased(Python())')
    assert read.count('owner.release()') == 2 and read.count('owner.publish()') == 2
    assert read.count('_engine_publish_stream(s, progress, batch)') == 2


def test_experiment_never_silently_relabels_unsupported_routes():
    public = text('bindings/causal_lm_public.mojo')
    setup = public.split('def _engine_admission(')[1].split('def _run_engine(')[0]
    assert 'distributed execution is pending' in setup
    assert 'does not replace the existing resident route' in setup
    assert 'MOJOLEARN_GENERATION_NATIVE_ENGINE' in public
    provider = text('providers/python/generation_metadata_pyapi.mojo')
    assert 'foreign-weight descriptor admission is pending' in provider
    assert 'foreign mutable-state descriptor admission is pending' in provider


def test_deferred_layer_refusals_keep_earlier_progress_and_profile_publication():
    src = text('transformer/native_generation_engine.mojo')
    forward = src.split('def forward(mut self,')[1].split('struct NativeGenerationProgress')[0]
    assert forward.index('self.primitive_call(0,') < forward.index('self.layers[i].visited = True')
    assert forward.index('raise Error(self.layers[i].failure)') < forward.index('generation_session_forward(1,')
    provider = text('providers/python/generation_metadata_pyapi.mojo')
    assert 'native_layer.failure = String(problem)' in provider
    assert 'self.native.layers[i].visited and self.native.layers[i].profile_admitted' in provider
