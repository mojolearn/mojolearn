"""Source admission regressions; compiled public probes qualify runtime separately."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def text(path):
    return (ROOT / path).read_text()


def test_reuse_checks_exact_factory_module_and_session_admission_identity():
    # The generation session provider is deleted (lane transformer-session-native).
    neural = text('providers/python/neural_descriptor_pyapi.mojo').split('def neural_descriptor_reuse(')[1].split('def neural_descriptor_cached')[0]
    assert 'held[].factory is factory' in neural
    # The owner check reads PyCFunction_GetSelf directly instead of resolving __self__.
    assert 'held[].module._obj_ptr == _neural_descriptor_function_owner(factory)' in neural
    assert 'held[].name == name' in neural
    assert 'return neural_descriptor_connect(factory, name, operation_count)' in neural
    assert 'len(held[].calls) == operation_count' in neural
    assert 'operation_count < 1' in neural


def test_existing_mode_vendor_and_factory_admission_precedes_reuse():
    src = text('bindings/causal_lm_public_door.mojo').split('def causal_lm_admit_binding(')[1].split('def _block(')[0]
    assert src.index('primitive_vendor != vendor or primitive_mode != mode') < src.index('neural_table_entry[NEURAL_TABLE_BLOCK_FORWARD]')
    assert 'transformer_native_session_handle_v1")' not in src
    assert '_native_function(mods, primitives, "neural_primitives_api_v1")' in src
    assert '_native_function(mods, block_binding, "mamba_state_api_v1")' in src
    # Owner caches live in the admitted block record (22f3790f7c), not a raw __dict__ read.
    assert 'neural_block_fields(mods, model._blocks[0], _causal_context_for(mods, model))' in src
    for name in ('_native_primitives_provider_cache', '_native_mamba_provider_cache'):
        assert f'py_get(owner_fields, "{name}")' in src
        assert f'py_setitem(owner_fields, "{name}"' in src


def test_default_recipe_does_not_require_owned_snapshot_or_dropout_abi():
    import bench_board_neural as board
    assert board.SAMBA_DROPOUT == 0.0
    src = text('mamba/impl/mamba_shell_impl.mojo')
    assert 'is_defined["MOJOLEARN_NN40_OWNED_WEIGHTS"]()' in src.split('comptime NN40_OWNED_WEIGHTS =')[1].split('\n')[0]
