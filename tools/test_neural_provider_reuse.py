"""Source admission regressions; compiled public probes qualify runtime separately."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def text(path):
    return (ROOT / path).read_text()


def test_reuse_checks_exact_factory_module_and_session_admission_identity():
    gen = text('providers/python/generation_session_pyapi.mojo').split('def generation_reuse_native_provider(')[1].split('def generation_connect_native_session')[0]
    assert 'held[].module is module' in gen
    assert 'held[].factory is factory' in gen
    assert 'held[].image is factory.__self__' in gen
    assert 'held[].admit_session is admit_session' in gen
    assert 'return generation_connect_native_provider(module, factory, admit_session)' in gen
    neural = text('providers/python/neural_descriptor_pyapi.mojo').split('def neural_descriptor_reuse(')[1].split('def neural_descriptor_cached')[0]
    assert 'held[].factory is factory' in neural
    assert 'held[].module is factory.__self__' in neural
    assert 'held[].name == name' in neural
    assert 'return neural_descriptor_connect(factory, name)' in neural


def test_existing_mode_vendor_and_factory_admission_precedes_reuse():
    src = text('bindings/causal_lm_public.mojo').split('def causal_lm_admit_binding(')[1].split('def _block(')[0]
    assert src.index('primitive_vendor != vendor or primitive_mode != mode') < src.index('generation_reuse_native_provider(')
    assert '_native_function(block_binding, "transformer_native_api_v1")' in src
    assert '_native_function(block_binding, "transformer_native_session_handle_v1")' in src
    assert '_native_function(primitives, "neural_primitives_api_v1")' in src
    assert '_native_function(block_binding, "mamba_state_api_v1")' in src
    assert 'model._blocks[0], "__dict__"' in src
    for name in ('_native_session_provider_cache', '_native_primitives_provider_cache', '_native_mamba_provider_cache'):
        assert f'py_get(owner_fields, "{name}")' in src
        assert f'py_setitem(owner_fields, "{name}"' in src


def test_default_recipe_does_not_require_owned_snapshot_or_dropout_abi():
    import bench_board_neural as board
    assert board.SAMBA_DROPOUT == 0.0
    src = text('mamba/impl/mamba_shell_impl.mojo')
    assert 'is_defined["MOJOLEARN_NN40_OWNED_WEIGHTS"]()' in src.split('comptime NN40_OWNED_WEIGHTS =')[1].split('\n')[0]
