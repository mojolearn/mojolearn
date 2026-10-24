# SPDX-License-Identifier: Apache-2.0
"""The multi-GPU kernel-method drivers are one native call each.

docs/plans/NATIVE_PARALLEL_DRIVER_20261024.md: `fit_kernel_method` and
`apply_kernel_method` hand the estimator, the data and the devices to the
native door (`family_km_par_fit` / `family_km_par_apply`), which binds a
thread-scoped device plan in the kernel-methods binding around the plain
owners. These are source contracts, read without importing the package or
loading a binding; the numerical contract (equal bytes to the one-device fit
at one and two devices) is test_parallel_public_drivers.py and the
par-kernel-ridge / par-nystroem identity lanes.
"""
import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PKG = REPO / "python" / "mojolearn"


def _read(relative):
    return (REPO / relative).read_text(encoding="utf-8")


def _function(source, name):
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(name + " not found")


def _native_ops(function):
    """The operation names of every `<module>._NATIVE(...)` call in `function`."""
    ops = []
    for node in ast.walk(function):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_NATIVE"):
            assert isinstance(node.args[1], ast.Constant), ast.dump(node)
            ops.append(node.args[1].value)
    return ops


def test_drivers_make_exactly_one_native_call_and_open_no_pool():
    source = (PKG / "parallel_classical.py").read_text(encoding="utf-8")
    for name, op in (("fit_kernel_method", "family_km_par_fit"),
                     ("apply_kernel_method", "family_km_par_apply")):
        function = _function(source, name)
        assert _native_ops(function) == [op], name
        text = ast.get_source_segment(source, function)
        assert "DevicePool" not in text and "pool" not in text.split('"""')[-1], name
        loops = [n for n in ast.walk(function) if isinstance(n, (ast.For, ast.While, ast.comprehension))]
        assert not loops, name
    assert "_admit_kernel_method" not in source


def test_worker_no_longer_serves_kernel_methods():
    source = (PKG / "_parallel_worker.py").read_text(encoding="utf-8")
    assert "'km_fit'" not in source.replace("# 'km_fit' / 'km_apply' moved", "")
    assert "operation in ('km_fit', 'km_apply')" not in source


def test_door_dispatches_and_scopes_the_device_plan():
    door = _read("bindings/kernel_methods_native.mojo")
    assert 'operation == "family_km_par_fit"' in door
    assert 'operation == "family_km_par_apply"' in door
    for name in ("_km_par_fit", "_km_par_apply"):
        body = door[door.index("def " + name + "("):]
        body = body[:body.index("\n\ndef ")] if "\n\ndef " in body else body
        # enter before the owners, exit in a finally that always runs
        enter = body.index("family_call(scope[0], devices)")
        guarded = body.index("try:")
        final = body.index("finally:")
        leave = body.index("family_call(scope[1])")
        assert enter < guarded < final < leave, name
    # the parallel predict bypasses the resident device copy
    assert "_krr_predict(model, Python.tuple(binding, args[0]), context, False)" in door
    # the classes the door admits are connected at kernel_methods' import
    assert "family_km_par_classes=(KernelRidge, Nystroem)" in (PKG / "kernel_methods.py").read_text(encoding="utf-8")


def test_gpu_binding_registers_the_plan_entries_and_the_host_binding_does_not():
    gpu = _read("bindings/_mojolearn_kernel_methods.mojo")
    host = _read("bindings/_mojolearn_kernel_methods_host.mojo")
    for entry in ("kernel_methods_device_plan_enter", "kernel_methods_device_plan_exit"):
        assert '("' + entry + '")' in gpu, entry
        assert entry not in host, entry


def test_owner_loops_take_devices_and_counts_from_the_plan():
    for relative, variable in (("svm/impl/distance/kernel_matrices.mojo", "MOJOLEARN_SVM_DEVICE_COUNT"),
                               ("cholesky/multi_gpu.mojo", "MOJOLEARN_CHOLESKY_DEVICE_COUNT"),
                               ("cholesky/checks/trsm.mojo", None),
                               ("kernel_methods/checks/kernel_matrix.mojo", "MOJOLEARN_SVM_DEVICE_COUNT")):
        source = _read(relative)
        assert "DeviceContext(device_id=rank)" not in source, relative
        if variable is not None:
            assert 'getenv("' + variable + '")' not in source, relative
            assert 'device_plan_setting("' + variable + '")' in source, relative


def test_device_plan_falls_back_to_the_environment_and_the_rank():
    plan = _read("core/device_plan.mojo")
    setting = plan[plan.index("def device_plan_setting("):]
    assert "return String(getenv(name))" in setting
    device = plan[plan.index("def device_plan_device("):plan.index("def device_plan_root(")]
    assert re.search(r"if slot < 0:\n\s+return rank", device)
    # the plan is keyed by the calling thread, never process-wide
    assert 'external_call["pthread_self"' in plan


def test_root_context_follows_the_plan():
    estimator = _read("kernel_methods/estimator.mojo")
    body = estimator[estimator.index("def _family_ctx()"):]
    body = body[:body.index("\n\nfrom ")]
    assert "device_plan_root()" in body
    assert "process_ctx_on[_FAMILY_CTX, _FAMILY_DEVICES_CTX](root)" in body
