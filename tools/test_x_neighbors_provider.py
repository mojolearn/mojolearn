"""Shared packed CPU/GPU transport admission; source-only, no runtime claim."""
import ast
import re
from pathlib import Path
from gen_x_neighbors_provider import generated, unpack
from check_generation_native_boundary import audit
ROOT = Path(__file__).resolve().parents[1]


def test_generated_tables_preserve_current_wrapper_schemas_and_native_calls():
    for path, expected in generated(ROOT).items():
        assert (ROOT / path).read_text() == expected, path
    for suffix, target in [('', 'device'), ('_host', 'host')]:
        old = (ROOT / f'bindings/_mojolearn_x_neighbors{suffix}.mojo').read_text()
        new = (ROOT / f'x_neighbors/native_descriptor_{target}.mojo').read_text()
        rows = unpack(old)
        assert len(rows) == 77
        for name, _, body in rows:
            new_body = re.search(rf'^def xn_native_{name}\(.*?(?=^def |\Z)', new, re.M | re.S).group()
            assert re.findall(r'\bop_\w+\(', body) == re.findall(r'\bop_\w+\(', new_body), (target,name)
        assert 'np < xn_min_ints(op)' in new  # optional nan_cells flag retained
        assert new.index('na != shape[0]') < new.index('native_descriptor_words(aa, na)')
    _, violations = audit(ROOT, ('x_neighbors.native_descriptor_device','x_neighbors.native_descriptor_host'))
    assert not violations, violations


def test_existing_provider_is_admitted_outside_packed_operation_loops():
    text = (ROOT / 'providers/python/x_neighbors_descriptor_pyapi.mojo').read_text()
    assert 'struct ' not in text
    assert 'downcast_value_ptr[CompiledBindingSnapshot]' in text
    assert 'neural_descriptor_connect(factory, String(XN_ABI_NAME), XN_OPERATION_COUNT)' in text
    execute = text[text.index('def xn_execute'):]
    assert 'neural_descriptor_connect' not in execute and 'xn_admit(' not in execute
    assert 'neural_descriptor_execute(snapshot[].provider, xn_operation_id(name), a, p, f)' in execute
    for filename in ('neighbors_local_native','neighbors_semi_native','neighbors_graph_native','svc_ocsvm_native'):
        source = (ROOT / 'bindings' / (filename+'.mojo')).read_text()
        assert source.count('args = xn_admit_arguments(args, context)') == 1
        assert 'xn_execute(binding, name, addrs, ints, floats)' in source


def test_selected_public_method_bodies_have_no_python_binding_lookup():
    tree = ast.parse((ROOT / 'python/mojolearn/_expansion_neighbors.py').read_text())
    seen = 0
    for n in ast.walk(tree):
        if not (isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_NATIVE' and len(n.args)>=3):
            continue
        op=n.args[1]
        if not (isinstance(op,ast.Constant) and isinstance(op.value,str) and op.value.startswith(('family_xn_loc_','family_xn_semi_','family_xn_graph_','family_svc_ocsvm_'))):
            continue
        if isinstance(n.args[0],ast.Name) and n.args[0].id=='self':
            assert isinstance(n.args[2].elts[0],ast.Constant) and n.args[2].elts[0].value is None
            seen += 1
    assert seen >= 30


def test_capabilities_are_operation_local_and_do_not_mutate_cached_provider():
    snapshot = (ROOT/'providers/python/compiled_binding_snapshot_pyapi.mojo').read_text()
    assert 'var provider: PythonObject' in snapshot and 'var capabilities: List[Int]' in snapshot
    assert 'def execute' not in snapshot and 'neural_descriptor_execute' not in snapshot
    source = (ROOT/'providers/python/x_neighbors_descriptor_pyapi.mojo').read_text()
    assert 'var capabilities = neural_descriptor_execute(provider, xn_operation_id("capabilities"), [], [], [])' in source
    assert 'CompiledBindingSnapshot(provider, binding, String(XN_ABI_NAME), mode, [capabilities])' in source
    assert 'capabilities[0]' in source
