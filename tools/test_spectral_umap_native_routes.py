"""Connection and public-boundary tests without a Mojo build or GPU."""
import ast
import importlib.util
from pathlib import Path
import re
import sys
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SURFACE = ROOT / "python/mojolearn"


@pytest.fixture
def bridge(monkeypatch):
    package = ModuleType("spectral_route_fixture")
    package.__path__ = [str(SURFACE)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    spec = importlib.util.spec_from_file_location(
        package.__name__ + "._native_spectral_umap_bridge", SURFACE / "_native_spectral_umap_bridge.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return package, module


def test_connection_detaches_proxy_and_preserves_optional_absence(bridge):
    _, module = bridge
    seen = []

    class Proxy:
        def __getattr__(self, name):
            seen.append(name)
            if name == "native":
                return len
            raise ImportError(name)

    proxy = Proxy()
    captured = module._capture(proxy, "compiled", ("native", "missing"))
    assert type(captured) is ModuleType
    assert vars(captured)["native"] is len
    assert "missing" not in vars(captured) and "__getattr__" not in vars(captured)
    assert captured.native([1, 2]) == 2
    assert seen == ["native", "missing"]
    assert proxy not in vars(captured).values()


def test_python_callable_cannot_enter_native_entry_table(bridge):
    _, module = bridge
    candidate = ModuleType("callback")
    candidate.native = lambda value: value
    with pytest.raises(TypeError, match="must be a compiled entry"):
        module._capture(candidate, "callback", ("native",))


def test_modes_are_connected_once_and_missing_tier_is_retained(bridge, monkeypatch):
    package, module = bridge
    calls = []
    backend = ModuleType(package.__name__ + "._backend")
    metrics = ModuleType(package.__name__ + "._metrics_impl")
    backend._DEFAULT_MODE = "identical"
    backend.default_mode = lambda: backend._DEFAULT_MODE

    class Proxy:
        def __getattr__(self, name):
            return len

    def binding(name, mode):
        calls.append((name, mode))
        if mode == "fast":
            raise ValueError("no fast tier")
        return Proxy()

    backend.binding = binding
    metrics._get_binding = lambda mode: binding("_mojolearn_metrics", mode)
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    monkeypatch.setitem(sys.modules, metrics.__name__, metrics)
    routes = module.connect()
    count = len(calls)
    backend._DEFAULT_MODE = "fast"
    assert module.connect() is routes
    assert len(calls) == count
    for group, choices in routes.items():
        assert type(choices["identical"]) is ModuleType
        if group != "base":
            assert choices["fast"] == "ValueError: no fast tier"
    assert ("_mojolearn", "fast") not in calls


def test_umap_initializer_retains_its_transitive_decomp_symbols(bridge):
    # Prevent losing a supported PCA/random initializer entry when connecting
    # only the more visible spectral-affinity decomp cells.
    _, module = bridge
    source = (ROOT / "bindings/mani_kit.mojo").read_text()
    definitions = list(re.finditer(r"^    def (\w+)\(", source, re.M))
    methods = {match[1]: source[match.start():definitions[i + 1].start() if i + 1 < len(definitions) else len(source)]
               for i, match in enumerate(definitions)}
    initializer = (ROOT / "bindings/mani_umap_native.mojo").read_text().split("def _umap_initial(", 1)[1].split("\ndef ", 1)[0]
    initializer = "\n".join(line.split("#", 1)[0] for line in initializer.splitlines())
    pending = re.findall(r"\b(?:k|hk)\.(\w+)\(", initializer)
    visited, symbols = set(), set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        body = methods[name]
        symbols.update(re.findall(r'"(x_decomp_[^"]+)"', body))
        pending.extend(re.findall(r"self\.(\w+)\(", body))
    assert symbols <= set(module._SYMBOLS["decomp"][1])
    assert "family_mani_binding" not in initializer


def _shell(filename, cls, name):
    tree = ast.parse((SURFACE / filename).read_text())
    container = tree if cls is None else next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == cls)
    node = next(node for node in container.body if isinstance(node, ast.FunctionDef) and node.name == name)
    node.decorator_list = []
    return ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))


@pytest.mark.parametrize("filename,cls,name", [
    ("_spectral_impl.py", "SpectralClustering", "fit"),
    ("_spectral_impl.py", "SpectralClustering", "fit_predict"),
    ("_spectral_impl.py", "SpectralClustering", "predict"),
    ("_spectral_impl.py", "SpectralClustering", "_predict_embedding"),
    ("_spectral_impl.py", "SpectralEmbedding", "fit"),
    ("_spectral_impl.py", "SpectralEmbedding", "fit_transform"),
    ("_spectral_impl.py", "SpectralEmbedding", "_precomputed_knn_affinity"),
    ("_spectral_impl.py", None, "_coo_triples"),
    ("_spectral_impl.py", None, "spectral_embedding"),
    ("_umap_impl.py", "UMAP", "fit"),
    ("_umap_impl.py", "UMAP", "fit_transform"),
    ("_umap_impl.py", "UMAP", "transform"),
])
def test_public_operation_hands_inputs_directly_to_one_owner(filename, cls, name):
    calls = []
    result, source, context = object(), object(), object()

    class NoPythonLookup:
        def __getattribute__(self, name):
            raise AssertionError("Python runtime attribute lookup: " + name)

    def native(*args):
        calls.append(args)
        return result

    namespace = dict(_NATIVE=native, _NATIVE_CONTEXT=context, SpectralEmbedding=object,
                     _backend=NoPythonLookup())
    exec(compile(_shell(filename, cls, name), filename, "exec"), namespace)
    args = (source,) if cls is None else (NoPythonLookup(), source)
    assert namespace[name](*args) is result
    assert len(calls) == 1 and calls[0][3] is context
    assert source in calls[0][2]
