"""Source contracts for manifold route migration; no runtime qualification."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _source(name):
    return (ROOT / name).read_text()


def test_route_connections_are_import_time_and_cover_both_families():
    tree = ast.parse(_source("python/mojolearn/_native_runtime_manifold_bridge.py"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "connect_bindings"]
    assert len(calls) == 1
    assert {ast.literal_eval(arg) for arg in calls[0].args} == {
        "_mojolearn_x_decomp", "_mojolearn_x_ann", "_mojolearn_estimators"}


def test_native_manifold_selects_fitted_and_default_modes():
    source = _source("bindings/mani_decomp_native.mojo")
    selection = source.split("def _mani_arguments(", 1)[1].split("def mani_decomp_operation(", 1)[0]
    assert 'operation.endswith("_fit_transform")' in selection
    assert 'PythonObject("numeric_mode_")' in selection
    assert 'estimator_binding_mode(requested, "_mojolearn_x_decomp", context)' in selection
    assert 'estimator_binding_mode(PythonObject(None), "_mojolearn_x_decomp", context)' in selection
    assert 'args = _mani_arguments(model, operation, args, context)' in source
    assert '.hasattr' not in source.split("def _mani_check(", 1)[1].split("def _mani_seed(", 1)[0]


def test_native_tsne_resolves_both_optional_binding_slots():
    source = _source("bindings/mani_tsne_native.mojo")
    assert 'estimator_binding(model, "_mojolearn_x_ann", context) if args[0] is None else args[0]' in source
    assert 'estimator_binding(model, "_mojolearn_estimators", context) if args[1] is None else args[1]' in source
    assert 'spectral_umap_optional_entry(estimators, "pca_fit_owned", context)' in source


def test_manifold_kit_refuses_proxies_before_looking_up_cells():
    source = _source("bindings/mani_kit.mojo")
    kit = source.split("struct MKit(", 1)[1].split("    def res(", 1)[0]
    assert 'raise Error("TypeError: manifold kit requires an exact native module")' in kit
    assert "object.__getattribute__" not in kit
    assert ".getattr" not in kit
    assert 'return spectral_umap_entry(self.b, name, self.ctx)' in kit
    assert 'spectral_umap_optional_entry(self.raw, name, self.ctx)' in kit


def test_manifold_input_copy_uses_snapshot_words_and_keeps_original_owner():
    source = _source("bindings/mani_kit.mojo")
    capture = source.split("def mk_from_input(", 1)[1]
    assert capture.count("family_f32_snapshot(") == 1
    assert "mk_from_snapshot(snapshot, name, ctx)" in capture
    assert "_ = source" in capture
    copy = source.split("def mk_from_snapshot(", 1)[1].split("def mk_from_input(", 1)[0]
    for forbidden in (".shape[", ".ndim", "classical_address(value)"):
        assert forbidden not in copy
    assert "value.source.addr" in copy
    assert "family_all_finite_f32(src, r * c)" in copy


def test_tsne_pca_and_optimizer_share_one_x_snapshot():
    source = _source("bindings/mani_tsne_native.mojo")
    assert source.count('family_f32_snapshot(args[2], 2, "X", context)') == 1
    assert "_tsne_pca_start(estimators, x_address, n, d, k, context)" in source
    assert "family_call(fit, Python.list(PythonObject(x_address)," in source
    assert "x.shape" not in source
    assert "classical_address(x)" not in source
    assert "_ = snapshot" in source
