"""Source controls for the one CPython function table (bindings/cpython_table.mojo).

Compilation and runtime checks are separate; these read the source only.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "bindings" / "cpython_table.mojo"
# Version-dependent symbols resolved by existence check, never on a hot path.
EXTERNAL_ALLOWED = {
    "bindings/cpython_table.mojo",
    "bindings/native_loaded_module.mojo",  # Py_IsFinalizing / _Py_IsFinalizing
}


def _mojo_sources():
    for top in ("bindings", "providers", "core", "sequence", "training", "neighbors", "cluster",
                "kernel_methods", "tokenizer", "x_cnn", "glm", "trees", "linalg"):
        base = ROOT / top
        if base.is_dir():
            yield from sorted(base.rglob("*.mojo"))


def _rel(path):
    return path.relative_to(ROOT).as_posix()


def _table_fields():
    text = TABLE.read_text()
    fields = re.findall(r"^    var (\w+): _(\w+)\.type$", text, re.M)
    loads = re.findall(r"^        self\.(\w+) = _(\w+)\.load\(lib\)$", text, re.M)
    protos = re.findall(r'^comptime _(\w+) = ExternalFunction\[\s*"(\w+)"', text, re.M)
    return text, fields, loads, protos


def test_table_fields_loads_and_prototypes_agree():
    text, fields, loads, protos = _table_fields()
    assert fields and all(a == b for a, b in fields)
    assert all(a == b for a, b in loads)
    assert all(a == b for a, b in protos)
    names = [a for a, _ in fields]
    assert len(names) == len(set(names))
    assert names == [a for a, _ in loads]
    assert sorted(names) == sorted(a for a, _ in protos)
    assert 'name="MojolearnCPythonTableV6"' in text


def test_absorbed_tables_are_gone():
    assert not (ROOT / "bindings" / "perf2_shared_cpython_table.mojo").exists()
    banned = re.compile(r"\b(perf2_cpython_table|Perf2SharedCPythonTable|lm_cpython_table|"
                        r"P3LmTokenizerCPythonTable|p3_table|P3Table|ClusterCPython|ManifoldCalls|"
                        r"manifold_calls)\b")
    offenders = []
    for path in _mojo_sources():
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if banned.search(line):
                offenders.append(f"{_rel(path)}:{number}")
    assert offenders == []


def test_no_second_cpython_symbol_table():
    offenders = [_rel(p) for p in _mojo_sources()
                 if "ExternalFunction[" in p.read_text() and _rel(p) not in EXTERNAL_ALLOWED]
    assert offenders == []


def test_one_form_for_the_plan_item_a_calls():
    # PyModule_GetDict, PyObject_GetIter, PyIter_Next and PyErr_Clear go
    # through the table only: no std-handle method form, no by-name lookup.
    pattern = re.compile(r'cpy\.(PyModule_GetDict|PyObject_GetIter|PyIter_Next|PyErr_Clear)\(|'
                         r'get_function\[[^\]]*\]\(\s*"(PyModule_GetDict|PyObject_GetIter|PyIter_Next|PyErr_Clear)"')
    offenders = []
    for path in _mojo_sources():
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{_rel(path)}:{number}")
    assert offenders == []


def test_every_table_call_names_a_table_field():
    _, fields, _, _ = _table_fields()
    names = {a for a, _ in fields}
    used = re.compile(r"(?:cpython_table\(\)|\b\w+)\[\]\.(Py\w+|_Py\w+)\b")
    missing = []
    for path in _mojo_sources():
        text = path.read_text()
        if "cpython_table" not in text:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            for name in used.findall(line):
                if name not in names:
                    missing.append(f"{_rel(path)}:{number}:{name}")
    assert missing == []
