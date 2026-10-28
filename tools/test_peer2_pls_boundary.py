"""PLS boundary source regressions. No product imports or runtime claims."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHELL = ROOT / "python/mojolearn/_expansion_decomp.py"


def _score():
    tree = ast.parse(SHELL.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_PLS")
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "score")


def test_score_passes_original_arguments_and_returns_finished_native_result():
    calls = []
    context, result = object(), object()

    class Trap:
        def __getattribute__(self, name):
            raise AssertionError("score must not inspect inputs/settings before native entry")

    def entry(*args):
        calls.append(args)
        return result

    namespace = dict(_PLS_CALL=entry, _PLS_CONTEXT=context)
    exec(compile(ast.Module(body=[_score()], type_ignores=[]), str(SHELL), "exec"), namespace)
    model, x, y, weight = Trap(), Trap(), Trap(), Trap()
    assert namespace["score"](model, x, y, weight) is result
    assert calls == [(model, "score", (x, y, weight), context)]
    assert len([n for n in ast.walk(_score()) if isinstance(n, ast.Call)]) == 1


def test_native_score_uses_connected_binding_and_direct_native_metric_owner():
    owner = (ROOT / "bindings/pls_native.mojo").read_text()
    assert 'estimator_binding_mode(mode, "_mojolearn_metrics", context, False)' in owner
    assert 'metrics_cls_operation(PythonObject(None), "family_mcls_r2"' in owner
    assert 'py_call(owner[0]' not in owner
    assert '_native_family_bridge.connect_bindings("_mojolearn_metrics")' in SHELL.read_text()


def test_native_target_uses_captured_metadata_without_array_or_reshape_protocol():
    owner = (ROOT / "bindings/pls_native.mojo").read_text()
    target = owner[owner.index("def _target("):owner.index("def _center_scale(")]
    assert 'decomp_input_snapshot(source, "Y", context, -1)' in target
    assert 'kit.input_snapshot(input, 1, input.shape[0])' in target
    assert 'return (kit.transpose(row), True)' in target
    code = '\n'.join(line.split('#', 1)[0] for line in target.splitlines())
    assert not any(token in code for token in ('py_call(', 'getattr', 'hasattr', 'source.shape', 'source.reshape'))
    provider = (ROOT / "providers/python/decomp_input_pyapi.mojo").read_text()
    assert provider.count('array_shape_once(view, context)') == 1
    assert provider.count('array_metadata_once(view, shape^, name, context, complex_error=complex_error)') == 1
    assert 'snapshot.owner = Python.tuple(original_owner, snapshot.owner)' in provider
    assert 'family_all_finite_f32(' in provider


def test_kit_retains_snapshot_owner_and_drops_address_when_device_writes():
    kit = (ROOT / "bindings/decomp_kit_native.mojo").read_text()
    assert 'self.input_owners.append(snapshot.owner)' in kit
    assert 'self.matrices[index].host_address = snapshot.source.addr' in kit
    invalidation = kit[kit.index('    def wrote_device('):kit.index('    def wrote_host(')]
    assert 'self.matrices[a].host_address = -1' in invalidation


def test_public_lifecycle_replaces_decorators_only_for_pls_family():
    tree = ast.parse(SHELL.read_text())
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    flags = [node for node in classes["_PLS"].body if isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id == "_mojolearn_native_public" for target in node.targets)]
    assert len(flags) == 1 and ast.literal_eval(flags[0].value) is True
    assert not any(isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "_mojolearn_native_public" for target in node.targets) for node in classes["_Base"].body)
    owner = (ROOT / "bindings/pls_native.mojo").read_text()
    for step in ("_pls_require_fitted(fields, operation, context)",
                 "classical_check_width(x.shape[1], Int(py=features), name)",
                 "estimator_feature_check(model, fields, args[0], context)",
                 'py_setitem(state, "_mojolearn_fitting", PythonObject(depth + 1))',
                 'py_setitem(state, "_mojolearn_fitted", PythonObject(True))',
                 'py_setitem(state, "feature_names_in_", feature_names)',
                 'py_call(state.pop, "_mojolearn_fitting", PythonObject(None))',
                 'parallel_public_result(result, operation, PythonObject("regressor"), context)'):
        assert step in owner
    assert 'var result = pls_public_operation(call[0]' in owner
    assert 'classical_python_error(error^, args_ptr)' in owner


def test_snapshot_addresses_survive_shared_host_views():
    kit = (ROOT / "bindings/decomp_kit_native.mojo").read_text()
    views = kit[kit.index("    def transpose("):kit.index("    def hstack(")]
    for line in views.splitlines():
        if "self.add(shared," in line or "self.add(store," in line:
            assert "self.matrices[a].host_address" in line
    assert 'Int(py=classical_address(source))' not in kit
    score = (ROOT / "bindings/pls_native.mojo").read_text()
    default = score[score.index('mode = context["backend_fields"]["_DEFAULT_MODE"]'):]
    assert default.index('if py_type(mode) is not b.str:') < default.index('mode.strip')


def test_scalar_admission_never_converts_object_dtype_payloads():
    owner = (ROOT / "bindings/pls_native.mojo").read_text()
    scalar = owner[owner.index("def _pls_scalar("):owner.index("def _pls_settings(")]
    assert 'cls is py_get(context, "ndarray")' in scalar
    assert 'not _one_of(cls, context["scalar_types"])' in scalar
    assert 'Int(py=value.ndim) != 0' in scalar
    assert 'var scalar = native_scalar_buffer(value)' in scalar
    code = '\n'.join(line.split('#', 1)[0] for line in scalar.splitlines())
    assert 'py_call(' not in code and 'Bool(py=value)' not in code
    assert 'decomp_int(value)' not in code and 'decomp_float(value)' not in code


def test_execution_uses_one_private_settings_snapshot_and_separate_state():
    owner = (ROOT / "bindings/pls_native.mojo").read_text()
    assert owner.count('classical_family_admit(model)') == 1
    assert owner.count('var fields = py_call(b.dict.copy, state)') == 1
    assert 'py_fields(' not in owner
    assert 'pls_operation(model, operation, args, context, fields, state, x, y)' in owner
    assert 'pls_operation(model, "predict", Python.tuple(args[0]), context, fields, state, captured_x)' in owner
    assert 'estimator_binding(model,' not in owner
    assert 'var modes = Python.tuple("numeric_mode") if fitting' in owner


def test_completed_iteration_counts_are_published_from_native_words():
    owner = (ROOT / "bindings/pls_native.mojo").read_text()
    assert 'var its = native_array_zeros("i", nc)' in owner
    assert 'py_call(b.list, its)' not in owner
    assert 'py_attr(all_iterations' not in owner
    assert 'PythonObject(Int(iterations[index]))' in owner


def test_fitted_check_precedes_defaults_and_fit_transform_uses_private_outputs():
    owner = (ROOT / "bindings/pls_native.mojo").read_text()
    assert owner.index('_pls_require_fitted(fields, operation, context)') < owner.index('var default_mode = PythonObject(None)')
    assert '_transform(kit, held[0], held[1], fields, context)' in owner
    assert '_transform(kit, held[0], held[1], state, context)' not in owner
    assert '_pls_store(fields, state, "x_rotations_"' in owner
    assert '_pls_store(fields, state, "_x_scores"' in owner
