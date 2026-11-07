"""Lane ne1-h-labels: every `_OP.<name>` a shell passes to the classical
door is a name in the one native operation table
(bindings/native_operation_codes.mojo), and the table holds no name twice.
Source-only check: no binding is needed."""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[3]
SHELLS = ("_expansion_neighbors.py", "_expansion_trees.py", "_expansion_metrics.py",
          "_metrics_impl.py", "_forest_host.py", "_gbdt_host.py")


def _table():
    source = (ROOT / "bindings" / "native_operation_codes.mojo").read_text()
    return re.findall(r'names\.append\("([^"]+)"\)', source)


def test_table_sorted_and_unique():
    names = _table()
    assert names == sorted(set(names))


def test_every_coded_operation_is_in_the_table():
    names = set(_table())
    for shell in SHELLS:
        source = (ROOT / "python" / "mojolearn" / shell).read_text()
        used = set(re.findall(r"_OP\.(\w+)", source))
        assert used <= names, (shell, sorted(used - names))
        # no operation left as a str literal at a door call
        assert not re.search(r'\b(_NATIVE|_TREES|_native_host_archive)\(\s*[\w.]+,\s*"(family|trees|host_tree)_',
                             source), shell


def test_routes_cover_the_prefixes():
    for name in _table():
        assert name.startswith(("family_", "trees_", "host_tree_")), name
