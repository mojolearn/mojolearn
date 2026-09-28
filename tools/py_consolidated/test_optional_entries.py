"""Exercise optional capability probing without loading native code."""
import ast
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[2]


def helper(filename, name):
    module = ast.parse((ROOT / "python/mojolearn" / filename).read_text())
    node = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"), ns)
    return ns[name]


@pytest.mark.parametrize("filename,name,mandatory,optional", [
    ("_expansion_metrics.py", "_optional_metrics_entry", "x_metrics_run", "x_metrics_run_ranges"),
    ("_training_impl.py", "_optional_samba_head", "linear_forward", "samba_head_loss"),
])
def test_optional_host_facade_preserves_mandatory_failures(filename, name, mandatory, optional):
    fn = helper(filename, name)
    def call(binding):
        return fn(binding, optional) if name == "_optional_metrics_entry" else fn(binding)
    class Facade:
        def __getattr__(self, attr):
            raise ImportError("missing export " + attr)
    b = Facade()
    with pytest.raises(ImportError, match=mandatory):
        call(b)
    setattr(b, mandatory, lambda: None)
    assert call(b) is None
    optimized = lambda: "native"
    setattr(b, optional, optimized)
    assert call(b) is optimized
