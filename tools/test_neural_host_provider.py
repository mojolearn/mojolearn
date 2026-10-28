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
    assert 'Bool(py=e["cpu"]) and block_provider is not None' in causal
    assert '34 if head_profile else 33, a_words, p_words' in causal
    mamba=(ROOT/'bindings/mamba_public_native.mojo').read_text()
    assert '2 * kind - 1, _neural_int_words(addresses), p, f' in mamba
    transformer=(ROOT/'providers/python/transformer_decode_pyapi.mojo').read_text()
    assert 'neural_descriptor_execute(h[].provider, 33, a, p, [])' in transformer


def test_training_hosts_use_their_original_numerical_owners_and_common_transport():
    roots=('mamba.host.native_block_dispatch','transformer.host.native_block_dispatch')
    seen, failures=audit(ROOT,roots)
    assert not failures, failures
    assert 'mamba.host.mamba_host_shell_impl' in seen
    assert 'transformer.transformer_host_binding_impl' in seen
    assert 'mamba.impl.mamba_shell_impl' not in seen
    assert 'transformer.transformer_binding_impl' not in seen
    for family in ('mamba','transformer'):
        binding=(ROOT/f'bindings/_mojolearn_{family}_host.mojo').read_text()
        assert f'[{family}_block_api_binding]("{family}_block_api_v1")' in binding
        assert 'neural_descriptor_capsule(api[].name, api[].entries)' in binding
    boundary=(ROOT/'bindings/neural_block_public.mojo').read_text()
    admission=boundary.split('def neural_training_entries(')[1].split('def _neural_int_words(')[0]
    assert 'vendor != "cpu" and getenv' not in admission


def test_host_owned_operations_preserve_failure_state_and_original_counters():
    source=(ROOT/'mamba/host/native_block_dispatch.mojo').read_text()
    for helper in ('_m3_owned_install_run','_m3_owned_forward_run','_m3_owned_backward_run',
                   '_m3_owned_state_run','_m3_owned_export_run'):
        assert helper+'(owner[],' in source
    assert 'except problem:\n        owner[].usable = False' in source
    assert 'owner[].weight_reuses += 1' in source
    assert 'owner[].backward_recomputes += 1' in source
    assert 'Float32(f[0]), Float32(f[1])' in source
    original=(ROOT/'mamba/host/mamba_host_shell_impl.mojo').read_text()
    assert 'generation: Int, reserved: Bool = False)' in original
    assert 's.busy != reserved' in original
    for text in ('len(a) != want_a', 'len(p) != want_p', 'op >= 44', 'result == 0'):
        assert text in source


def test_cpu_mamba_public_admission_refreshes_both_tables():
    public=(ROOT/'bindings/mamba_public_native.mojo').read_text()
    admission=public.split('def _m3_entries(')[1].split('def _m3_canonical(')[0]
    for factory in ('mamba_block_api_v1','mamba_state_api_v1'):
        assert factory in admission
    assert 'py_setitem(fields, "_native_mamba_state", state_provider)' in admission
    native=(ROOT/'mamba/host/native_block_dispatch.mojo').read_text()
    state=native.split('def mamba_host_state_operation(')[1].split('def mamba_host_state_v1(')[0]
    assert '_mamba1_run(a, p[0], length, width, 0)' in state
    assert '_mamba3_run(a, p[0], length, width, p[tail], p[tail + 1], False)' in state
    binding=(ROOT/'bindings/_mojolearn_mamba_host.mojo').read_text()
    assert '[mamba_state_api_binding]("mamba_state_api_v1")' in binding


def test_cpu_training_primitives_share_relocated_public_helpers():
    _, failures=audit(ROOT,('training.host.native_primitive_dispatch',))
    assert not failures, failures
    binding=(ROOT/'bindings/_mojolearn_training_host.mojo').read_text()
    core=(ROOT/'core/mojolearn_training_host_impl.mojo').read_text()
    provider=(ROOT/'training/host/native_primitive_dispatch.mojo').read_text()
    for name in ('embedding_forward','embedding_backward','rms_norm_forward','rms_norm_backward',
                 'linear_forward','linear_backward','accumulate','chunked_lm_head_v2_loss','chunked_lm_head_v2_train'):
        assert f'def {name}_host_into(' in core
        assert f'count = {name}_host_into(a,' in binding
        assert f'return {name}_host_into(a,' in provider
    assert 'return optimizer_step_host_into(' in provider
    assert 'return ce_loss_host_into(' in provider
    assert 'op != 25' in provider
    assert '[neural_primitives_api_binding]("neural_primitives_api_v1")' in binding
    samba=(ROOT/'providers/python/samba_execution_pyapi.mojo').read_text()
    assert 'var use_session = not cpu and' in samba
    assert 'operation = 33 if cpu else (35 if packed else 11)' in samba
    assert '12 if use_session or layer_state is not None else 10' in samba
