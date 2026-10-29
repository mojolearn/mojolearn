#!/usr/bin/env python3
"""Class-definition checks only: no package/native binding import or computation."""
import ast
import functools
import inspect
from pathlib import Path
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "python/mojolearn"


def mode_declarations():
    source = ast.parse((PACKAGE / "_mode.py").read_text())
    constants = {"_FITS", "_LOADS", "_NEEDS_FIT", "_OUTPUTS"}
    nodes = [node for node in source.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
             or (isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id in constants for target in node.targets))]
    namespace = {"__name__": "mojolearn._mode", "functools": functools, "inspect": inspect,
                 "_PARAMS_METADATA_CONTEXT": None,
                 # Validation is never executed: only wrapper creation/identity
                 # is checked, with inert argument-slot admission metadata.
                 "_sk_validate": types.SimpleNamespace(FIT_METHODS={"fit"},
                     arg_slots=lambda method: (None, None, None))}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "_mode setup declarations", "exec"), namespace)
    # The real import installs a native generic slot here. Its behavior is not
    # under test; this sentinel proves the selected classes own their final slot.
    def generic_slot(self, name):
        return object.__getattribute__(self, name)
    namespace["PublicOutputsMixin"].__getattribute__ = generic_slot
    return namespace


def neural_classes(namespace):
    result = {}
    for filename, names in (("_mamba_impl.py", {"_MambaBase", "Mamba1Block", "Mamba2Block", "Mamba3Block"}),
                            ("_transformer_impl.py", {"TransformerBlock"})):
        parsed = ast.parse((PACKAGE / filename).read_text())
        classes = [node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name in names]
        namespace["__name__"] = "mojolearn." + filename[:-3]
        exec(compile(ast.Module(body=classes, type_ignores=[]), filename + " class declarations", "exec"), namespace)
        result.update({name: namespace[name] for name in names if not name.startswith("_")})
    return result


class ImportSetup(unittest.TestCase):
    def setUp(self):
        self.namespace = mode_declarations()
        self.classes = neural_classes(self.namespace)

    def test_four_final_constructors_are_never_wrapped(self):
        for cls in self.classes.values():
            with self.subTest(cls=cls.__name__):
                initializer = vars(cls)["__init__"]
                self.assertFalse(hasattr(initializer, "__wrapped__"))
                self.assertFalse(hasattr(initializer, "__signature__"))
                mode = inspect.signature(initializer).parameters["numeric_mode"]
                self.assertEqual(mode.kind, inspect.Parameter.KEYWORD_ONLY)
                self.assertIsNone(mode.default)
                self.assertIs(vars(cls)["__getattribute__"], object.__getattribute__)
                self.assertNotIn("_mojolearn_native_public", vars(cls))
                self.assertNotIn("_native_constructor", vars(cls))

    def test_passive_subclass_preserves_final_identity(self):
        for cls in self.classes.values():
            child = types.new_class("Passive", (cls,))
            self.assertIs(child.__init__, cls.__init__)
            self.assertIs(child.__getattribute__, object.__getattribute__)

    def test_application_constructor_still_gets_mode_injection(self):
        for cls in self.classes.values():
            def initialize(self, value=7):
                self.value = value
            child = types.new_class("Application", (cls,), {},
                                    lambda fields: fields.update(__module__="application", __init__=initialize))
            self.assertIs(child.__init__.__wrapped__, initialize)
            self.assertIn("numeric_mode", inspect.signature(child).parameters)
            instance = child(11, numeric_mode="identical")
            self.assertEqual(instance.value, 11)
            self.assertEqual(instance.numeric_mode, "identical")

    def test_application_own_methods_keep_fit_guards_and_slots(self):
        for cls in self.classes.values():
            def fit(self, value):
                return self
            def predict(self, value):
                return value
            def own_slot(self, name):
                return object.__getattribute__(self, name)
            child = types.new_class("Application", (cls,), {}, lambda fields: fields.update(
                __module__="application", fit=fit, predict=predict, __getattribute__=own_slot))
            self.assertTrue(child.fit._mojolearn_tracks_fit)
            self.assertTrue(child.predict._mojolearn_needs_fit)
            self.assertIs(child.__getattribute__, own_slot)

    def test_constructor_opt_in_does_not_skip_fit_guards(self):
        base = self.namespace["NumericModeMixin"]
        def initialize(self, *, numeric_mode=None):
            pass
        def fit(self, value):
            return self
        child = types.new_class("NativeConstructor", (base,), {"_native_constructor": True},
                                lambda fields: fields.update(__module__="application", __init__=initialize, fit=fit))
        self.assertIs(child.__init__, initialize)
        self.assertTrue(child.fit._mojolearn_tracks_fit)

    def test_constructor_opt_in_requires_explicit_mode(self):
        base = self.namespace["NumericModeMixin"]
        def initialize(self):
            pass
        with self.assertRaisesRegex(TypeError, "explicitly accept numeric_mode"):
            types.new_class("Invalid", (base,), {"_native_constructor": True},
                            lambda fields: fields.update(__init__=initialize))

    def test_no_module_tail_constructor_or_slot_repair(self):
        for filename in ("_mamba_impl.py", "_transformer_impl.py"):
            parsed = ast.parse((PACKAGE / filename).read_text())
            for node in parsed.body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                            self.assertFalse(target.value.id in self.classes
                                             and target.attr in ("__init__", "__getattribute__"))


if __name__ == "__main__":
    unittest.main()
