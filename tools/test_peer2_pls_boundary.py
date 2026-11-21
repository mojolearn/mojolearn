"""PLS boundary source regressions. No product imports or runtime claims."""
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from boundary_keys_plain import plain  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SHELL = ROOT / "python/mojolearn/_expansion_decomp.py"


def _owner():
    # The door reads held keys and operation codes (bindings/pls_keys_unpack.mojo);
    # plain() gives the by-name form these source checks were written against.
    # The owner is three files (UNPACK, PACK and the door that sequences them); read together.
    return "\n".join(plain(path, (ROOT / path).read_text()) for path in (
        "bindings/pls_boundary_unpack.mojo", "bindings/pls_boundary_pack.mojo", "bindings/pls_native_door.mojo"))


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
    owner = _owner()
    assert 'estimator_binding_routed(model, mode, "_mojolearn_metrics", context, False)' in owner
    assert 'metrics_cls_operation(PythonObject(None), "family_mcls_r2"' in owner
    assert 'py_call(owner[0]' not in owner
    assert '_native_family_bridge.connect_bindings("_mojolearn_metrics")' in SHELL.read_text()


def test_native_target_uses_captured_metadata_without_array_or_reshape_protocol():
    owner = _owner()
    target = owner[owner.index("def _target("):owner.index("def pls_store(")]
    assert '_pls_input(source, "Y", context, pin, -1)' in target
    assert 'decomp_door_snapshot(door, kit, input, pin, 1, input.shape[0])' in target
    assert 'return (kit.transpose(row), True)' in target
    code = '\n'.join(line.split('#', 1)[0] for line in target.splitlines())
    assert not any(token in code for token in ('py_call(', 'getattr', 'hasattr', 'source.shape', 'source.reshape'))
    # The operand admission is the one path (lane a1b: decomp_input_snapshot
    # deleted): admit_x_unrecorded / admit_f32, then the captured metadata.
    helper = owner[owner.index("def _pls_input("):owner.index("def _matrix(")]
    assert 'admit_x_unrecorded(value, public_name, context, defer_finite=defer)' in helper
    assert 'admit_f32(view, nd, name, context, admitted=True)' in helper
    assert 'array_metadata_once(array, array_shape_once(array, context), name, context, pin)' in helper
    assert 'pin.owner = Python.tuple(source, pin.owner)' in helper


def test_kit_retains_snapshot_owner_and_drops_address_when_device_writes():
    # The door keeps the pin's owners for the call (lane entangled-decompkit:
    # the typed kit holds addresses only); a device write drops the host words.
    door = (ROOT / "bindings/decomp_kit_native_door.mojo").read_text()
    snapshot = door[door.index("def decomp_door_snapshot("):door.index("def decomp_door_tested(")]
    assert 'door.keep(pin.owner)' in snapshot
    assert 'kit.borrow(snapshot.source.addr, rows, cols)' in snapshot
    kit = (ROOT / "bindings/decomp_kit_run.mojo").read_text()
    invalidation = kit[kit.index('    def wrote_device('):kit.index('    def wrote_host(')]
    assert 'self.matrices[a].store = -1' in invalidation
    assert 'self.matrices[a].borrowed = 0' in invalidation


def test_public_lifecycle_replaces_decorators_only_for_pls_family():
    tree = ast.parse(SHELL.read_text())
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    flags = [node for node in classes["_PLS"].body if isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id == "_mojolearn_native_public" for target in node.targets)]
    assert len(flags) == 1 and ast.literal_eval(flags[0].value) is True
    assert not any(isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "_mojolearn_native_public" for target in node.targets) for node in classes["_Base"].body)
    owner = _owner()
    for step in ("_pls_require_fitted(fields, operation_name, context)",
                 "classical_check_width(x.shape[1], Int(py=features), name)",
                 "estimator_feature_check(model, fields, args[0], context)",
                 'py_setitem(state, "_mojolearn_fitting", PythonObject(depth + 1))',
                 'py_setitem(state, "_mojolearn_fitted", PythonObject(True))',
                 'py_setitem(state, "feature_names_in_", feature_names)',
                 'py_call(state.pop, "_mojolearn_fitting", PythonObject(None))',
                 'parallel_public_result(result, String(py=operation_name), PythonObject("regressor"), context)'):
        assert step in owner
    assert 'var result = pls_public_operation(call[0], operation, call[1], call[2], context)' in owner
    assert 'classical_python_error(error^, args_ptr)' in owner


def test_snapshot_addresses_survive_shared_host_views():
    # The typed kit RUN (bindings/decomp_kit_run.mojo) gives a view the parent's store and its
    # borrowed flag, so a view never outlives or re-addresses the snapshot it shares.
    kit = (ROOT / "bindings/decomp_kit_run.mojo").read_text()
    views = kit[kit.index("    def transpose("):kit.index("    def hstack(")]
    adds = [line for line in views.splitlines() if "self.add(" in line]
    assert adds
    for line in adds:
        assert "self.add(self.matrices[a].store," in line and "self.matrices[a].borrowed)" in line
    # Addresses of Python sources are read only in the UNPACK door at upload; RUN holds none.
    assert 'classical_address(' not in kit
    score = _owner()
    default = score[score.index('mode = context["backend_fields"]["_DEFAULT_MODE"]'):]
    assert default.index('if py_type(mode) is not b.str:') < default.index('mode.strip')


def test_scalar_admission_never_converts_object_dtype_payloads():
    owner = _owner()
    scalar = owner[owner.index("def _pls_scalar("):owner.index("def _pls_settings(")]
    assert 'cls is py_get(context, "ndarray")' in scalar
    assert 'not _one_of(cls, context["scalar_types"])' in scalar
    assert 'Int(py=value.ndim) != 0' in scalar
    assert 'var scalar = native_scalar_buffer(value)' in scalar
    code = '\n'.join(line.split('#', 1)[0] for line in scalar.splitlines())
    assert 'py_call(' not in code and 'Bool(py=value)' not in code
    assert 'decomp_int(value)' not in code and 'decomp_float(value)' not in code


def test_execution_uses_one_private_settings_snapshot_and_separate_state():
    owner = _owner()
    assert owner.count('classical_family_admit(model)') == 1
    assert owner.count('var fields = py_call(py_type(Python.dict()).__dict__["copy"], state)') == 1
    assert 'py_fields(' not in owner
    # Both calls also carry the deferred-finite flag (lane a8c) after the operands.
    assert 'pls_operation(model, operation, args, context, admitted.fields, admitted.state, admitted.x.copy(),' in owner
    assert 'pls_operation(model, "predict", Python.tuple(args[0]), context, fields, state, captured_x,' in owner
    assert 'estimator_binding(model,' not in owner
    assert 'var modes = Python.tuple("numeric_mode") if fitting' in owner


def test_completed_iteration_counts_are_published_from_native_words():
    owner = _owner()
    # The counts are the typed fit result's native words (PlsFit.iterations), published once.
    assert 'py_call(b.list, its)' not in owner
    assert 'py_attr(all_iterations' not in owner
    assert 'PythonObject(Int(fit.iterations[index]))' in owner
    assert 'pls_store(fields, state, "n_iter_", all_iterations)' in owner


def test_fitted_check_precedes_defaults_and_fit_transform_uses_private_outputs():
    owner = _owner()
    assert owner.index('_pls_require_fitted(fields, operation_name, context)') < owner.index('var default_mode = Optional[PythonObject]()')
    # fit_transform's projections come from the same typed RUN (pls_fit_run, transform=True) over
    # the X and Y the fit held, never re-read from the published state.
    assert 'fit = pls_fit_run(kit, unpacked.x, unpacked.ym, unpacked.words, transform)' in owner
    assert 'return Python.tuple(decomp_kit_publish(door, kit, fit.tx), decomp_kit_publish(door, kit, fit.ty))' in owner
    assert '_transform(kit, held[0], held[1], state, context)' not in owner
    assert 'pls_store(fields, state, "x_rotations_"' in owner
    assert 'pls_store(fields, state, "_x_scores"' in owner
