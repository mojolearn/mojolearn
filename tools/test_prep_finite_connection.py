"""Exercise import-time prep connection without importing product binaries."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class PrepFiniteConnection(unittest.TestCase):
    def connect(self, *, finite32=abs, finite64=ord, missing=None):
        package = "_prep_connection_fixture"
        modules = {}

        def module(name, **values):
            obj = ModuleType(package + name)
            vars(obj).update(values)
            modules[obj.__name__] = obj
            return obj

        root = module("", __path__=[])
        calls = []
        base = SimpleNamespace(gather_rows_bytes=len, all_finite_f32=finite32,
                               all_finite_f64=finite64)
        if missing:
            delattr(base, missing)
        prep = SimpleNamespace(x_prep_run=id, x_prep_numeric_mode=hash)

        def binding(name, mode):
            calls.append((name, mode))
            return base if name == "_mojolearn" else prep

        root._backend = module("._backend", binding=binding, default_mode=lambda: "identical")
        # Real instance dictionaries keep the connector's descriptor handling
        # exercised; no public product class or native operation is imported.
        class Rows:
            pass

        class Cache:
            pass

        class Scope:
            pass

        class RFE:
            _parameters = ("estimator", "step")

        module("._arena_io", DeviceRows=Rows, DeviceCache=Cache, _Scope=Scope, _LOCAL=object())
        state = {"existing_admission": object()}
        output = {"array_class": object()}
        module("._native_classical_bridge", _state_context=lambda: state,
               _numpy_free_context=lambda: output)
        module("._native_labels_bridge", label_context=lambda: {"labels": object()})
        door = SimpleNamespace(_module=SimpleNamespace(prep_encode_native_call=repr))
        module(".tokenizer", _binding=lambda: door)
        path = ROOT / "python/mojolearn/_native_prep_encode_bridge.py"
        spec = importlib.util.spec_from_file_location(package + ".connector", path)
        connector = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(connector)
            call, context = connector.connect({}, False, {}, {RFE: "rfe"})
        return call, context, calls, state, output, RFE

    def test_actual_connector_retains_both_native_widths_and_existing_context(self):
        call, context, calls, state, output, rfe = self.connect()
        self.assertIs(call, repr)
        self.assertIs(context, state)
        self.assertIs(context["output_context"], output)
        self.assertIs(context["maxabs_gather_rows"], len)
        self.assertIs(context["maxabs_finite_f32"], abs)
        self.assertIs(context["maxabs_finite_f64"], ord)
        self.assertEqual(calls.count(("_mojolearn", "identical")), 1)
        self.assertIn("existing_admission", context)
        self.assertIn("labels", context)
        self.assertEqual(context["prep_composition_plans"][rfe]["parameters"], rfe._parameters)
        instance = rfe()
        self.assertIs(context["prep_composition_plans"][rfe]["fields_get"](instance, rfe),
                      instance.__dict__)

    def test_python_finite_callbacks_are_rejected_at_connection(self):
        for key in ("finite32", "finite64"):
            with self.subTest(key=key), self.assertRaisesRegex(TypeError, "finite-check"):
                self.connect(**{key: lambda *args: self.fail("callback executed")})

    def test_missing_finite_entry_cannot_make_an_incomplete_context(self):
        for key in ("all_finite_f32", "all_finite_f64"):
            with self.subTest(key=key), self.assertRaisesRegex(AttributeError, key):
                self.connect(missing=key)


if __name__ == "__main__":
    unittest.main()
