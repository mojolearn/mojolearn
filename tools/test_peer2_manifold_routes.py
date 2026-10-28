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
    assert 'estimator_binding_mode(requested, "_mojolearn_x_decomp", context)' in selection
    assert 'estimator_binding_mode(settings.default_mode, "_mojolearn_x_decomp", context)' in selection
    assert 'args = _mani_arguments(settings, operation, args, context)' in source
    assert 'var requested = settings.mode' in selection
    assert '.hasattr' not in source.split("def _mani_check(", 1)[1].split("def _mani_seed(", 1)[0]


def test_native_tsne_resolves_both_optional_binding_slots():
    source = _source("bindings/mani_tsne_native.mojo")
    assert 'estimator_binding_mode(settings.mode, "_mojolearn_x_ann", context) if args[0] is None else args[0]' in source
    assert 'estimator_binding_mode(settings.mode, "_mojolearn_estimators", context) if args[1] is None else args[1]' in source
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
    entry = _source("bindings/manifold_native.mojo")
    assert entry.count('manifold_input(source, name, not tsne, context)') == 1
    assert 'snapshot: ArrayMetadata' in source
    assert 'family_f32_snapshot(args[2]' not in source
    assert "_tsne_pca_start(estimators, x_address, n, d, k, context)" in source
    assert "family_call(fit, Python.list(PythonObject(x_address)," in source
    assert "x.shape" not in source
    assert "classical_address(x)" not in source
    assert "_ = snapshot" in source


def test_settings_admission_precedes_snapshot_and_is_shared_by_every_owner():
    source = _source("bindings/manifold_settings_native.mojo")
    assert source.index('self.fields = classical_family_admit(model)') < source.index('self.values = py_call(b.dict, self.fields)')
    assert source.count('context["backend_fields"]["_DEFAULT_MODE"]') == 1
    assert 'operation.endswith("_fit_transform")' in source
    assert 'self.fitted = py_get(self.values, "numeric_mode_")' in source
    assert 'if not operation.startswith("family_mani_umap_")' in source
    assert '_setting(flat[i], context, depth + 1, freeze_buffers)' in source
    assert '_setting(value[keys[i]], context, depth + 1, freeze_buffers)' in source
    assert 'native_array_from_bytes(String("B"), py_call(buffer.tobytes, "C"))' in source
    assert 'py_setitem(output, i, item)' in source
    assert 'return py_call(cls, shape, dtype=dtype, buffer=store)' in source
    dispatch = _source("bindings/manifold_native.mojo")
    assert dispatch.count('ManifoldSettings(model, operation, context)') == 1
    for owner in ('mani_decomp_native', 'mani_tsne_native', 'mani_umap_native'):
        implementation = _source(f"bindings/{owner}.mojo")
        for forbidden in ('py_fields(model)', '_fields(model)', '_DEFAULT_MODE', 'family_raw_get(model', '_mani_mode('):
            assert forbidden not in implementation, (owner, forbidden)
        assert 'settings.values' in implementation
        assert 'py_setitem(settings.fields,' in implementation


def test_native_lifecycle_owns_checks_and_finished_results():
    entry = _source("bindings/manifold_native.mojo")
    assert entry.index('manifold_require_fitted(') < entry.index('var x = manifold_input(')
    assert entry.index('manifold_ignored_target(') < entry.index('result = tsne_operation(')
    assert 'return manifold_finish(settings, result, method, names, context)' in entry
    source = _source("bindings/manifold_lifecycle_native.mojo")
    conversion = source.split('def _cast(', 1)[1].split('def _target_finite(', 1)[0]
    assert conversion.index('if not snapshot.castable:') < conversion.index('family_cast_rows(')
    assert 'snapshot.source.kind' not in source.split('def _target_finite(', 1)[1].split('def manifold_input(', 1)[0]
    assert 'parallel_public_result(result, method, PythonObject(None), context)' in source
    for forbidden in ('before_fit(', 'before_call(', '._mojolearn_after_fit(', '.warnings.warn('):
        assert forbidden not in source
