"""Public wrapper and admission guards; arithmetic runtime remains separately checked."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_layernorm_public_operations_enter_native_before_validation():
    tree = ast.parse((ROOT/'python/mojolearn/_x_sequence_norm.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LayerNorm')
    for name in ('__init__', 'to_device', 'forward', 'backward', '_device_io'):
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
        body = [n for n in method.body if not isinstance(n, ast.Expr)]
        assert len(body) == 1 and isinstance(body[0], ast.Return)
        assert isinstance(body[0].value, ast.Call) and body[0].value.func.id == '_SEQ_NATIVE'


def test_native_layernorm_keeps_validation_and_finished_gradients():
    source = (ROOT/'bindings/sequence_layers_native.mojo').read_text()
    assert 'var fields = _ln_fields(owner, context)' in source
    assert 'LayerNorm.backward: call forward first' in source
    assert 'py_setitem(fields, "weight_grad"' in source
    assert 'py_setitem(fields, "bias_grad"' in source
    assert 'LayerNorm refuses replaced Python storage hooks' in source
    assert 'stats require two native tensor handles' in source
