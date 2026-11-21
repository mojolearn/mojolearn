"""Source controls only; compiled model/checkpoint execution remains unqualified."""
import ast
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from boundary_keys_plain import plain  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class ModelMetadataSource(unittest.TestCase):
    def test_config_and_checkpoint_public_methods_are_single_native_calls(self):
        count = 0
        for name in ("config", "safetensors"):
            tree = ast.parse((ROOT / f"python/mojolearn/models/{name}.py").read_text())
            for cls in tree.body:
                if not isinstance(cls, ast.ClassDef):
                    continue
                for method in cls.body:
                    if not isinstance(method, ast.FunctionDef):
                        continue
                    body = method.body
                    if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                        body = body[1:]
                    self.assertEqual(len(body), 1, (name, cls.name, method.name))
                    self.assertIsInstance(body[0], ast.Return)
                    calls = [n for n in ast.walk(body[0]) if isinstance(n, ast.Call)]
                    self.assertEqual(len(calls), 1)
                    self.assertIsInstance(calls[0].func.value, ast.Name)
                    self.assertEqual(calls[0].func.value.id, "_NATIVE")
                    count += 1
        self.assertGreaterEqual(count, 35)

    def test_configuration_snapshots_precede_mutable_state_reads(self):
        source = plain("bindings/causal_model_config_door.mojo", (ROOT / "bindings/causal_model_config_door.mojo").read_text())
        for old in ("config._d", "plan.__dict__", "plan._layer_map", "plan.n_layers", "Python.import_module"):
            self.assertNotIn(old, source)
        for guard in ("model_record_fields", "model_plain", "model_plan_snapshot",
                      "model_checkpoint_names_snapshot", "_model_error_class"):
            self.assertIn("def " + guard + "(", source)
        owner = source.split("def model_record_fields(", 1)[1].split("def _model_context(", 1)[0]
        # 406368bd93: a class-watcher record of the ancestry verdict may skip the walk; without it the hook walk
        # runs before the storage read.
        self.assertLess(owner.index('class_record(mods, _RECORD_FIELDS_SLOT, py_type(value))'), owner.index("_fields(mods, value)"))
        self.assertLess(owner.index('"__getattribute__"'), owner.rindex("_fields(mods, value)"))
        self.assertLess(owner.index('"__getattribute__"'), owner.index("class_admit(mods, _RECORD_FIELDS_SLOT"))

    def test_checkpoint_slots_and_native_exports_are_admitted(self):
        source = (ROOT / "bindings/causal_checkpoint_door.mojo").read_text()
        for old in ("value._infos", "value._mm", "info.shape", "info.begin", "plan.__dict__",
                    'Python.import_module("array")', "bi.open", "file.close()"):
            self.assertNotIn(old, source)
        for guard in ("_info_slots", "_info_snapshot", "_file_fields", "_checkpoint_fields"):
            self.assertIn("def " + guard + "(", source)
        # 6dcf21b621 (row N204): the pinned bytes go straight into the store; every pin is admitted first.
        for capture in (source.split("def _checkpoint_store(", 1)[1].split("\ndef ", 1)[0],
                        source.split("def checkpoint_widen_f16(", 1)[1].split("\ndef ", 1)[0]):
            self.assertIn("AdmittedBorrowedBuffer(", capture)
            self.assertLess(capture.index("admit_native_buffer_exporter("), capture.index("AdmittedBorrowedBuffer("))
        self.assertEqual(source.count("admit_native_buffer_exporter("), source.count("AdmittedBorrowedBuffer("))
        product = source.split("def _file_fields(", 1)[1]
        self.assertNotIn("Python.import_module", product)

    def test_existing_half_conversion_routes_and_native_storage_remain(self):
        source = (ROOT / "bindings/causal_checkpoint_door.mojo").read_text()
        self.assertIn("checkpoint_widen_halves(source, dest, count, dtype)", source)
        self.assertIn("_lin_from_f16(Python.tuple(binding, source), linalg)", source)
        self.assertIn("_checkpoint_store_pinned(pin, ", source)   # one copy into the store (6dcf21b621)
        self.assertIn("neural_array_snapshot(captured, context)", source)
        tree = ast.parse((ROOT / "python/mojolearn/models/safetensors.py").read_text())
        widen = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "widen_f16")
        self.assertEqual(sum(isinstance(n, ast.Call) for n in ast.walk(widen)), 1)

    def test_catalog_public_paths_use_exported_artifacts_and_native_inputs(self):
        source = (ROOT / "python/mojolearn/models/_catalog.py").read_text()
        tree = ast.parse(source)
        for method in tree.body:
            if not isinstance(method, ast.FunctionDef):
                continue
            body = method.body[1:]
            self.assertEqual(len(body), 1, method.name)
            self.assertIsInstance(body[0], ast.Return)
            self.assertEqual(sum(isinstance(n, ast.Call) for n in ast.walk(body[0])), 1)
        native = (ROOT / "bindings/model_catalog_native_door.mojo").read_text()
        for old in ("torch.onnx", "default_rng", "urlopen", "Python.import_module", "prog.run", "hashlib"):
            self.assertNotIn(old, native)
        for expected in ("_import(source,", "ops_imported_run_py(program, inputs, Python.dict())",
                         "input_hash.hexdigest()", "output_hash.hexdigest()", "neural_array_slots(context)",
                         "admit_native_buffer_exporter(value)", "pin.release()"):
            self.assertIn(expected, native)

    def test_catalog_imported_output_and_error_helpers_use_connected_builtins(self):
        source = (ROOT / "providers/python/autograd_program_pyapi_door.mojo").read_text()
        for name in ("_result", "_imported_repr", "_imported_tuple", "ops_imported_run_py"):
            body = source.split("def " + name + "(", 1)[1].split("\n\ndef ", 1)[0]
            self.assertNotIn('Python.import_module("builtins")', body, name)


    def test_causal_door_runs_are_typed(self):
        # The config, construct and checkpoint doors are UNPACK / PACK; their
        # decisions run in typed RUN files that hold no Python object.
        for run, door, entry in (
            ("bindings/causal_model_config_run.mojo", "bindings/causal_model_config_door.mojo", "causal_config_plan("),
        ):
            body = (ROOT / run).read_text()
            for word in ("PythonObject", "std.python", "py_call", "py_get", "GIL"):
                self.assertNotIn(word, body.split('"""', 2)[2], run)
            self.assertIn(entry, (ROOT / door).read_text(), door)

if __name__ == "__main__":
    unittest.main()
