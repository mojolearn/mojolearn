"""Source-only regression checks; native execution still requires the frozen build."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_function(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text())
    function = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class NativeExecutionConnectionTests(unittest.TestCase):
    def test_connection_resolves_lazy_proxy_only_once(self):
        reads = []

        class Proxy:
            def __getattr__(self, name):
                reads.append(name)
                return len

        connect = load_function("python/mojolearn/_native_parallel_shard.py", "_connect_binding",
                                {"_backend": SimpleNamespace(binding=lambda *_: Proxy())})
        connected = connect("binding", ("fit", "predict"))
        self.assertEqual(reads, ["fit", "predict"])
        self.assertIs(connected.fit, len)
        self.assertIs(connected.predict, len)
        self.assertEqual(reads, ["fit", "predict"])
        with self.assertRaises(AttributeError):
            _ = connected.missing
        self.assertEqual(reads, ["fit", "predict"])

    def test_connection_rejects_python_entries(self):
        connect = load_function("python/mojolearn/_native_parallel_shard.py", "_connect_binding",
                                {"_backend": SimpleNamespace(binding=lambda *_: SimpleNamespace(fit=lambda: None))})
        with self.assertRaisesRegex(TypeError, "compiled"):
            connect("binding", ("fit",))

    def test_missing_binding_is_deferred_without_runtime_loader(self):
        def missing(*_):
            raise ImportError("binding not installed")
        connect = load_function("python/mojolearn/_native_parallel_shard.py", "_connect_binding",
                                {"_backend": SimpleNamespace(binding=missing)})
        self.assertEqual(connect("binding", ("fit",)), "binding not installed")

    def test_shard_wrappers_forward_without_touching_user_objects(self):
        class Untouchable:
            def __getattribute__(self, name):
                raise AssertionError(f"Python wrapper touched {name}")

        cases = (
            ("parallel_gaussian_process.py", "fit_gaussian_process_classifier", 3),
            ("parallel_gaussian_process.py", "predict_gaussian_process_classifier", 2),
            ("parallel_ensemble.py", "fit_forest", 3),
        )
        for filename, name, arity in cases:
            with self.subTest(name=name):
                calls = []
                def native(operation, args):
                    calls.append((operation, args))
                    return "forwarded"
                function = load_function("python/mojolearn/" + filename, name,
                                         {"_native_parallel_shard": native})
                self.assertEqual(function(*(Untouchable() for _ in range(arity))), "forwarded")
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0][0], name)

    def test_torch_connection_requires_native_factories(self):
        connect = load_function("python/mojolearn/adapters/_torch_compile.py", "_connect_torch_execution", {})
        program = SimpleNamespace(_s=SimpleNamespace(_b=SimpleNamespace(ops_torch_program_run=len)))
        torch = SimpleNamespace(_C=SimpleNamespace(_VariableFunctions=SimpleNamespace(empty=len)),
                                device=lambda _: "cpu", float32="f32", int32="i32")
        entry, context = connect(torch, program, (1,), 2, True)
        self.assertIs(entry, len)
        self.assertEqual(context["free"], (1,))
        self.assertEqual(context["n_args"], 2)
        torch._C._VariableFunctions.empty = lambda: None
        with self.assertRaisesRegex(TypeError, "compiled empty"):
            connect(torch, program, (1,), 2, True)

    def test_torch_connection_refuses_a_python_execution_entry(self):
        connect = load_function("python/mojolearn/adapters/_torch_compile.py", "_connect_torch_execution", {})
        program = SimpleNamespace(_s=SimpleNamespace(_b=SimpleNamespace(ops_torch_program_run=lambda: None)))
        torch = SimpleNamespace(_C=SimpleNamespace(_VariableFunctions=SimpleNamespace(empty=len)))
        with self.assertRaisesRegex(TypeError, "compiled program entry"):
            connect(torch, program, (), 0, True)


if __name__ == "__main__":
    unittest.main()
