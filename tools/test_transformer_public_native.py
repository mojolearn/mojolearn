"""Source boundary contracts only; compile and public runtime qualification are separate."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "python/mojolearn/_transformer_impl.py"
NATIVE = (ROOT / "bindings/transformer_public_native.mojo").read_text()


def test_public_state_and_tape_methods_are_one_native_call():
    tree = ast.parse(SOURCE.read_text())
    expected = {
        "TransformerState": ("__init__", "capacity", "keys", "values"),
        "TransformerForwardTape": ("__init__", "__reduce__", "close"),
        "TransformerBlock": ("forward_with_tape", "backward_from_tape"),
    }
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef) or cls.name not in expected:
            continue
        methods = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}
        for name in expected[cls.name]:
            body = methods[name].body
            if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                body = body[1:]
            assert len(body) == 1 and isinstance(body[0], ast.Return), (cls.name, name)
            assert isinstance(body[0].value, ast.Call)
            assert ast.unparse(body[0].value.func) == "_native_neural_bridge.RUNTIME.transformer_public_call"
            assert not any(isinstance(node, (ast.For, ast.While, ast.ListComp)) for node in ast.walk(methods[name]))


def test_state_admits_storage_before_buffer_export_and_preserves_view_copy_split():
    constructor = NATIVE.split("struct TransformerBuffer(Movable):", 1)[1].split("def _transformer_state_words", 1)[0]
    assert constructor.index("admit_native_buffer_exporter(store)") < constructor.index("AdmittedBorrowedBuffer(store)")
    assert "self.owner = value" in constructor and "self.store = store" in constructor
    assert "_transformer_array_slots(context)" in constructor
    state = NATIVE.split("def transformer_state_read(", 1)[1].split("struct TransformerPublicTapeOwner", 1)[0]
    assert 'if window == 0:\n        return _transformer_cache_view(buffer, dims, context)' in state
    assert "var start = (cached - held) % window" in state
    assert "count=first * hd" in state and "count=(held - first) * hd" in state
    view = NATIVE.split("def _transformer_cache_view(", 1)[1].split("def transformer_state_read", 1)[0]
    assert 'py_setitem(fields, "_store", buffer.store)' in view
    assert 'py_setitem(fields, "_base", buffer.owner)' in view
    assert 'py_setitem(fields, "_pin", view)' in view


def test_tape_owns_native_metadata_and_uses_original_budget_aware_session_route():
    owner = NATIVE.split("struct TransformerPublicTapeOwner", 1)[1].split("def _transformer_configuration", 1)[0]
    for field in ("block: PythonObject", "session: PythonObject", "provider: PythonObject", "handle: Int", "token: List[Int]", "configuration: List[Int]", "weight_shapes: List[List[Int]]", "consumed: Bool"):
        assert field in owner
    forward = NATIVE.split("def transformer_forward_with_tape(", 1)[1].split("def transformer_tape_init", 1)[0]
    assert "transformer_session_tape_forward(provider, session, handle, addresses, params)" in forward
    assert "configuration[5], budget, cost]" in forward
    assert '"transformer_forward_tape"' not in forward
    backward = NATIVE.split("def transformer_backward_from_tape(", 1)[1].split("def transformer_tape_close", 1)[0]
    assert backward.index("owner[].consumed = True") < backward.index("transformer_session_tape_backward(owner[].provider")
    assert "owner[].session is not fields[\"_native_session\"]" in backward
    assert "_transformer_same_words(owner[].configuration" in backward
    assert backward.count("lease.release()") == 2
    assert "_ = dy^" in backward and "_ = hobj" in backward
