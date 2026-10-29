#!/usr/bin/env python3
"""Source/class-definition checks; no native binding or numerical execution."""
import ast
import inspect
import types
import unittest
from test_neural_import_setup import PACKAGE, mode_declarations


def manifold_declarations():
    bridge = {"__name__": "mojolearn._native_runtime_manifold_bridge"}
    exec(compile((PACKAGE / "_native_runtime_manifold_bridge.py").read_text(), "manifold bridge declarations", "exec"), bridge)
    original = {}
    def declaration(method):
        result = bridge["native_lifecycle"](method)
        original[(method.__module__, method.__qualname__)] = result
        return result
    namespace = mode_declarations()
    namespace["_native_runtime_manifold_bridge"] = types.SimpleNamespace(native_lifecycle=declaration)
    classes = {}
    for filename, names in (("_expansion_decomp.py", {"_Base", "Isomap", "ClassicalMDS", "MDS", "LocallyLinearEmbedding"}),
                            ("_expansion_ann.py", {"TSNE"})):
        namespace["__name__"] = "mojolearn." + filename[:-3]
        nodes = [node for node in ast.parse((PACKAGE / filename).read_text()).body
                 if isinstance(node, ast.ClassDef) and node.name in names]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), filename + " class declarations", "exec"), namespace)
        classes.update({name: namespace[name] for name in names if not name.startswith("_")})
    return classes, original, bridge


class ManifoldSetup(unittest.TestCase):
    def setUp(self):
        self.classes, self.original, self.bridge = manifold_declarations()

    def test_twelve_methods_keep_their_declaration_identity(self):
        self.assertEqual(len(self.original), 12)
        for cls in self.classes.values():
            names = self.bridge["_LIFECYCLE_METHODS"][(cls.__module__, cls.__name__)]
            for name in names:
                with self.subTest(cls=cls.__name__, method=name):
                    method = vars(cls)[name]
                    self.assertIs(method, self.original[(cls.__module__, cls.__name__ + "." + name)])
                    self.assertFalse(hasattr(method, "__wrapped__"))
                    self.assertTrue(method._mojolearn_native_public_operation)
                    self.assertTrue(method._mojolearn_public_out)
                    self.assertTrue(method._mojolearn_validates)

    def test_existing_constructor_mode_defaults_and_injection_survive(self):
        for cls in self.classes.values():
            self.assertIsNone(inspect.signature(cls).parameters["numeric_mode"].default)
            self.assertNotIn("_mojolearn_native_public", vars(cls))
        # TSNE still needs the existing injected constructor keyword. This is
        # Python class setup only; none of its fitted operations are called.
        tsne = self.classes["TSNE"]
        self.assertTrue(hasattr(tsne.__init__, "__wrapped__"))
        self.assertNotIn("numeric_mode", inspect.signature(tsne.__init__.__wrapped__).parameters)
        self.assertEqual(tsne(numeric_mode="identical").numeric_mode, "identical")

    def test_passive_subclasses_keep_native_methods(self):
        for cls in self.classes.values():
            child = types.new_class("Passive", (cls,))
            self.assertIs(child.fit, cls.fit)
            self.assertIs(child.fit_transform, cls.fit_transform)

    def test_application_own_overrides_keep_fit_tracking_and_slot(self):
        for cls in self.classes.values():
            def fit(self, value):
                return self
            def transform(self, value):
                return value
            def slot(self, name):
                return object.__getattribute__(self, name)
            child = types.new_class("Application", (cls,), {}, lambda fields: fields.update(
                __module__="application", fit=fit, transform=transform, __getattribute__=slot))
            self.assertIsNot(child.fit, fit)
            self.assertTrue(child.fit._mojolearn_tracks_fit)
            self.assertTrue(child.transform._mojolearn_needs_fit)
            self.assertIs(child.__getattribute__, slot)

    def test_declaration_refuses_nonpackage_methods(self):
        def fit(self, value):
            return self
        with self.assertRaisesRegex(TypeError, "original public method"):
            self.bridge["native_lifecycle"](fit)
        with self.assertRaisesRegex(TypeError, "original public method"):
            self.bridge["native_lifecycle"](object())

    def test_no_late_lifecycle_repair_remains(self):
        for name in ("_native_runtime_manifold_bridge.py", "_expansion_decomp.py", "_expansion_ann.py"):
            parsed = ast.parse((PACKAGE / name).read_text())
            for node in ast.walk(parsed):
                if isinstance(node, ast.Name):
                    self.assertNotIn(node.id, ("connect_lifecycle", "_connect_manifold_lifecycle"))
                if name == "_native_runtime_manifold_bridge.py" and isinstance(node, ast.Attribute):
                    self.assertNotEqual(node.attr, "__wrapped__")


if __name__ == "__main__":
    unittest.main()
