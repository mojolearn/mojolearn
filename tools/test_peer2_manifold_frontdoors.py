"""Source-only shell regression; native/runtime qualification is separate."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "python/mojolearn/_expansion_decomp.py"
METHODS = {
    "Isomap": {"fit", "fit_transform", "transform", "reconstruction_error"},
    "ClassicalMDS": {"fit", "fit_transform"},
    "MDS": {"fit", "fit_transform"},
    "LocallyLinearEmbedding": {"fit", "fit_transform", "transform"},
}


def test_manifold_shells_pass_inputs_without_inspection_or_binding_selection():
    tree = ast.parse(SOURCE.read_text())
    checked = set()
    for cls in (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in METHODS):
        for method in (n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in METHODS[cls.name]):
            calls = [n for n in ast.walk(method) if isinstance(n, ast.Call)]
            assert len(calls) == 1, (cls.name, method.name)
            call = calls[0]
            assert isinstance(call.func, ast.Name) and call.func.id == "_NATIVE"
            assert all(isinstance(v, ast.Constant) and v.value is None for v in call.args[2].elts[:2])
            payload = call.args[2].elts
            if method.name in {"fit", "fit_transform"}:
                expected = ["X", "init", "y"] if cls.name == "MDS" else ["X", "y"]
            elif method.name == "transform":
                expected = ["X"]
            else:
                expected = []
            assert [v.id for v in payload[2:]] == expected, (cls.name, method.name)
            checked.add((cls.name, method.name))
    assert len(checked) == 11
