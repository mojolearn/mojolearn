"""Source contracts, not compiled ABI/runtime qualification."""
from pathlib import Path
import re
from check_generation_native_boundary import audit

ROOT = Path(__file__).resolve().parents[1]
NATIVE = (ROOT / 'transformer/native_block_dispatch.mojo').read_text()
PROVIDER = (ROOT / 'providers/python/neural_descriptor_pyapi_door.mojo').read_text()
PUBLIC = (ROOT / 'bindings/neural_block_public_door.mojo').read_text()


def test_native_block_dependencies_resolve():
    _, problems = audit(ROOT, ('transformer.native_block_dispatch',))
    assert not problems, problems
    assert 'PythonObject' not in NATIVE


def test_exact_fresh_and_backward_numerical_owners_are_preserved():
    for call in ('_transformer_run_session[True](owner[], expanded,',
                 '_transformer_run[True](expanded,',
                 '_transformer_run_session_backward(owner[], a,',
                 '_transformer_backward_run(a,'):
        assert call in NATIVE
    # Fresh and backward session failures clear the owner (other ops clear too:
    # close and the int15 path, so pin the two sites rather than a total count).
    fresh = NATIVE.split('_transformer_run_session[True](owner[], expanded,', 1)[1].split('_transformer_run[True](expanded,', 1)[0]
    clears = re.compile(r'except problem:\n\s+owner\[\]\.clear\(\)\n\s+raise problem')
    assert clears.search(fresh)
    backward = NATIVE.split('_transformer_run_session_backward(owner[], a,', 1)[1].split('_transformer_backward_run(a,', 1)[0]
    assert clears.search(backward)
    assert 'p[6], p[1], 0, p[7], opts)' in NATIVE
    assert 'owner[].busy or owner[].closed' in NATIVE
    assert 'if not owner[].busy or owner[].closed:' in NATIVE


def test_common_descriptor_table_not_new_provider_or_extension_unwrap():
    assert 'List[NativeDescriptorCall]' in NATIVE
    assert 'native_descriptor_report(problem, context, sink)' in NATIVE
    body = PROVIDER.split('def neural_descriptor_reserved_execute(')[1].split('def neural_descriptor_info(')[0]
    # The provider is resolved to its typed id once; reserve runs under the GIL, the operation
    # without it, and release on both the success and the failure path.
    assert 'var provider_id = neural_descriptor_provider_id(provider)' in body
    assert body.index('neural_descriptor_reserved_begin(provider_id, handle, reserve, release,') < body.index('with GILReleased(Python())')
    assert body.count('neural_descriptor_reserved_end(provider_id, loan, release)') == 2
    assert '_ = session_owner' in body and '_ = provider' in body
    for forbidden in ('getattr(', '__call__', 'PyObject_Call', 'unsafe_get_as_pointer'):
        assert forbidden not in body


def test_public_fresh_session_ephemeral_and_backward_are_hooked():
    # Fresh (session op 2 / ephemeral op 4) and backward (session op 3 / ephemeral op 5) run
    # through the image's neural table entry by named operation (core/neural_entry_table.mojo).
    flat = re.sub(r'\s+', ' ', PUBLIC)
    assert '_neural_table_reserved(forward, handle, NEURAL_BLOCK_RESERVE, NEURAL_BLOCK_RELEASE, NEURAL_BLOCK_SESSION_FORWARD_FRESH, addresses, params,' in flat
    assert '_neural_table_call(forward, NEURAL_BLOCK_FORWARD_FRESH, addresses, params,' in flat
    assert '_neural_table_reserved(train_entry, handle, NEURAL_BLOCK_RESERVE, NEURAL_BLOCK_RELEASE, NEURAL_BLOCK_SESSION_BACKWARD, addresses, params,' in flat
    assert '_neural_table_call(train_entry, NEURAL_BLOCK_BACKWARD, addresses, params,' in flat
    # The route choice still keys on the image exporting each supported entry.
    # dw-neural-models (2026-11-21): read once at plan construction (NativeBlockPlan words), not per forward.
    assert 'self.session_fresh = py_contains(self.entries, "transformer_session_forward_fresh")' in PUBLIC
    assert 'self.forward_fresh = py_contains(self.entries, "transformer_forward_fresh")' in PUBLIC
    assert 'py_contains(e, ' not in PUBLIC
    assert '_native_has_entry(mods, entries, "transformer_session_backward")' in PUBLIC


def test_connection_carries_named_factory_and_checked_session_admission():
    bridge = (ROOT / 'python/mojolearn/_native_neural_bridge.py').read_text()
    assert '"transformer_block_api_v1",' in bridge and '"transformer_native_session_handle_v1"' not in bridge
    shell = (ROOT / 'bindings/_mojolearn_transformer.mojo').read_text()
    assert 'm.def_function[transformer_block_api_binding]("transformer_block_api_v1")' in shell
    assert 'NATIVE_TRANSFORMER_BLOCK_API.get_or_create_ptr()' in shell
    causal = (ROOT / 'bindings/causal_lm_public_door.mojo').read_text()
    assert 'py_setitem(entries, "native_block_operations", block_provider)' in causal
    # The provider connects in identical mode on every vendor, CPU included since b1c97898f5.
    # Native dispatch is the one form (lane ab-prep-neural, 2026-11-23): the build constant
    # NEURAL_NATIVE_DISPATCH and its by-name B arm -D MOJOLEARN_NEURAL_NATIVE_DISPATCH_OFF are
    # deleted, and the define is refused at build time; no neural door reads it from the environment.
    assert PUBLIC.count('if code == 1:') == 2
    guards = (ROOT / 'core/experiment_guards_neural.mojo').read_text()
    assert 'comptime assert not is_defined["MOJOLEARN_NEURAL_NATIVE_DISPATCH_OFF"]()' in guards
    for door in ('bindings/neural_block_public_door.mojo', 'bindings/mamba_public_native_door.mojo',
                 'bindings/causal_lm_public_door.mojo', 'bindings/causal_lm_construct_door.mojo',
                 'providers/python/transformer_decode_pyapi_door.mojo',
                 'providers/python/neural_descriptor_pyapi_door.mojo'):
        code = '\n'.join(line.split('#', 1)[0] for line in (ROOT / door).read_text().splitlines())
        assert 'getenv("MOJOLEARN_NEURAL_NATIVE_DISPATCH' not in code, door  # comments may name the retired read
