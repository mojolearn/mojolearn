"""Regression for native factories exposed at import-time connection."""
from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]

def text(path):
    return (ROOT / path).read_text()


def test_import_connection_lists_expose_new_factories_and_canonical_owner():
    for file, name in [('python/mojolearn/_native_neural_bridge.py', 'mamba_state_api_v1'),
                       ('python/mojolearn/_native_samba_bridge.py', 'neural_primitives_api_v1')]:
        tree = ast.parse(text(file))
        assert any(isinstance(n, ast.Constant) and n.value == name for n in ast.walk(tree))
    assert '"entries": SimpleNamespace(canonical=binding, **entries)' in text('python/mojolearn/_native_samba_bridge.py')

