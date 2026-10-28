"""CPU provider source contracts; compilation and public runtime stay separate."""
from pathlib import Path
from check_generation_native_boundary import audit
ROOT=Path(__file__).resolve().parents[1]


def test_cpu_descriptor_closure_has_no_python_or_device_backedge():
    seen, failures=audit(ROOT,('core.neural_host_dispatch',))
    assert not failures, failures
    assert 'core.mojolearn_neural_host_impl' in seen
    assert 'training.native_primitive_dispatch' not in seen
    assert 'transformer.transformer_binding_impl' not in seen
    assert 'mamba.impl.mamba_shell_impl' not in seen


def test_host_binding_and_provider_share_relocated_numerical_functions():
    binding=(ROOT/'bindings/_mojolearn_neural_host.mojo').read_text()
    core=(ROOT/'core/mojolearn_neural_host_impl.mojo').read_text()
    provider=(ROOT/'core/neural_host_dispatch.mojo').read_text()
    for name in ('_run_forward','_mamba1_run','_mamba2_run','_mamba3_run'):
        assert f'def {name}(' not in binding
        assert f'def {name}(' in core
        assert name+'(' in provider
    for name in ('neural_primitives_api','mamba_state_api','transformer_block_api','mamba_block_api'):
        assert f'module.def_function[{name}_binding]("{name}_v1")' in binding
    assert 'Float32(f[0]), Float32(f[1])' in provider
    assert 'opts, packed)' in provider


def test_cpu_session_and_causal_frontdoors_use_the_common_provider():
    causal=(ROOT/'bindings/causal_lm_public.mojo').read_text()
    assert 'String(model.device) == "cpu" and block_provider is not None' in causal
    assert '34 if head_profile else 33, a_words, p_words' in causal
    mamba=(ROOT/'bindings/mamba_public_native.mojo').read_text()
    assert '2 * kind - 1, _neural_int_words(addresses), p, f' in mamba
    transformer=(ROOT/'providers/python/transformer_decode_pyapi.mojo').read_text()
    assert 'neural_descriptor_execute(h[].provider, 33, a, p, [])' in transformer
