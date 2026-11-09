"""Source boundary contracts only; compile and public runtime qualification are separate."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "python/mojolearn/_transformer_impl.py"
NATIVE = (ROOT / "bindings/transformer_public_native.mojo").read_text()


def test_public_state_methods_are_one_native_call():
    tree = ast.parse(SOURCE.read_text())
    expected = {
        "TransformerState": ("__init__", "capacity", "keys", "values"),
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
    assert "_transformer_array_slots(mods, context)" in constructor
    state = NATIVE.split("def transformer_state_read(", 1)[1].split("def _transformer_configuration", 1)[0]
    assert 'if window == 0:\n        return _transformer_cache_view(mods, buffer, dims, context)' in state
    assert "var start = (cached - held) % window" in state
    assert "count=first * hd" in state and "count=(held - first) * hd" in state
    view = NATIVE.split("def _transformer_cache_view(", 1)[1].split("def transformer_state_read", 1)[0]
    assert 'py_setitem(fields, "_store", buffer.store)' in view
    assert 'py_setitem(fields, "_base", buffer.owner)' in view
    assert 'py_setitem(fields, "_pin", view)' in view


def test_tape_api_is_gone():
    # TRANSFORMER_FORWARD_TAPE_API (docs/TOMBSTONES.md): no tape ticket, owner or door.
    tree = ast.parse(SOURCE.read_text())
    names = {node.name for node in ast.walk(tree) if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    assert not names & {"TransformerForwardTape", "forward_with_tape", "backward_from_tape"}
    live = "\n".join(line for line in NATIVE.splitlines() if not line.lstrip().startswith("#"))
    for spelling in ("TransformerPublicTapeOwner", '"forward_tape"', '"backward_tape"', '"tape_init"',
                     '"tape_close"', "transformer_session_tape_"):
        assert spelling not in live


def test_public_state_uses_connected_modules_without_import_callbacks():
    assert 'Python.import_module(' not in NATIVE
    assert 'from bindings.native_loaded_module import native_loaded_module' in NATIVE
    # Per-operation NativeModules.module() resolves through native_loaded_module.
    assert 'mods.module("builtins")' in NATIVE
    loaded = (ROOT / "bindings/native_loaded_module.mojo").read_text()
    assert 'var module = native_loaded_module(name)' in loaded.split("struct NativeModules(Movable):", 1)[1]
    allocation = (ROOT / "bindings/generation_buffers.mojo").read_text()
    initialization = allocation.split("def generation_initialize_binding", 1)[1].split("def py_ints", 1)[0]
    assert 'Python.import_module(' not in initialization
    assert 'native_loaded_module("builtins")' in initialization
    assert 'native_loaded_module("array")' in initialization
