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
    assert 'estimator_binding(model, "_mojolearn_metrics", context, False)' in owner
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
    assert provider.count('array_metadata_once(view, shape^, name, context)') == 1
    assert 'snapshot.owner = Python.tuple(original_owner, snapshot.owner)' in provider
    assert 'family_all_finite_f32(' in provider


def test_kit_retains_snapshot_owner_and_drops_address_when_device_writes():
    kit = (ROOT / "bindings/decomp_kit_native.mojo").read_text()
    assert 'self.input_owners.append(snapshot.owner)' in kit
    assert 'self.matrices[index].host_address = snapshot.source.addr' in kit
    invalidation = kit[kit.index('    def wrote_device('):kit.index('    def wrote_host(')]
    assert 'self.matrices[a].host_address = -1' in invalidation
