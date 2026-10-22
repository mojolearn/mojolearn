# SPDX-License-Identifier: Apache-2.0
"""Storage can be narrow; executing or publishing changed metadata cannot."""

import ast
import sys
from pathlib import Path
from textwrap import dedent

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lane_select as ls
from import_metadata_effects import metadata_only_call

CALLER = """
from . import connector
class Model:
    def changed(self):
        return 1
FORMATS = {"format": {"Model": Model}}
CONTEXT = {}
connector.connect(CONTEXT, FORMATS)
"""


def proof(body, caller=CALLER, prefix=""):
    modules = {
        "python/mojolearn/caller.py": ast.parse(caller),
        "python/mojolearn/connector.py": ast.parse(
            prefix
            + "\ndef connect(context, formats):\n"
            + "\n".join("    " + line for line in dedent(body).splitlines())
        ),
    }
    call = modules["python/mojolearn/caller.py"].body[-1].value
    return metadata_only_call(
        "python/mojolearn/caller.py", call, {"FORMATS"}, modules.get, lambda *args: False
    )


def test_plain_class_registry_storage_has_no_model_execution():
    assert proof("""
        copied = {fmt: dict(classes) for fmt, classes in formats.items()}
        context['formats'] = copied
    """)


@pytest.mark.parametrize(
    "body",
    [
        "for classes in formats.values():\n    for cls in classes.values():\n        cls.changed()",
        "unknown_callback(formats)",
        "for classes in formats.values():\n    for cls in classes.values():\n        context['flag'] = cls.custom_descriptor",
        "for classes in formats.values():\n    for cls in classes.values():\n        context['flag'] = len(cls.__dict__)",
        "for classes in formats.values():\n    for cls in classes.values():\n        context['flag'] = hasattr(cls, 'new_method')",
        "for classes in formats.values():\n    for cls in classes.values():\n        if len(cls.__dict__) > 2:\n            context['flag'] = True",
        "for classes in formats.values():\n    for cls in classes.values():\n        if len(cls.__dict__) > 2:\n            publish_flag(True)",
        "for classes in formats.values():\n    context['selected'] = tuple(name for name, cls in classes.items() if len(cls.__dict__) > 2)",
        "for classes in formats.values():\n    for cls in classes.values():\n        context['mixed'] = {'size': len(cls.__dict__), 'model': cls}",
        "for classes in formats.values():\n    for cls in classes.values():\n        context['flag'] = cls.__dict__['changed']",
        "return formats",
    ],
)
def test_calls_hooks_and_metadata_dependent_global_state_stay_wide(body):
    assert not proof(body)


def test_custom_metaclasses_are_not_builtin_class_metadata():
    caller = CALLER.replace("class Model:", "class Meta(type):\n    pass\nclass Model(metaclass=Meta):")
    assert not proof(
        "for classes in formats.values():\n    for cls in classes.values():\n        context['name'] = cls.__name__",
        caller,
    )


def test_registry_copy_does_not_assume_custom_keys_have_no_hash_hooks():
    caller = CALLER.replace('{"format": {"Model": Model}}', '{custom_key(): {"Model": Model}}')
    assert not proof("context['formats'] = {key: dict(classes) for key, classes in formats.items()}", caller)


def test_mapping_update_does_not_call_an_unknown_supplied_protocol():
    caller = CALLER.replace('{"format": {"Model": Model}}', "unknown_mapping()")
    assert not proof("context.update(formats)", caller)


@pytest.mark.parametrize(
    "prefix",
    [
        "def dict(values):\n    return unknown_callback(values)\n",
        "dict = unknown_callback\n",
        "dict = unknown_callback\ndict = another_callback\n",
    ],
)
def test_shadowed_builtins_do_not_acquire_storage_permission(prefix):
    assert not proof("context['formats'] = dict(formats)", prefix=prefix)


def test_rebound_callee_does_not_acquire_old_function_permission():
    caller = CALLER.replace(
        "connector.connect(CONTEXT, FORMATS)",
        "connector = unknown_provider()\nconnector.connect(CONTEXT, FORMATS)",
    )
    assert not proof("context['formats'] = dict(formats)", caller)


def real_connector_call():
    rel = "python/mojolearn/_classical_host.py"
    calls = [
        node
        for node in ast.walk(ls._parse(rel))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "connect"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "_native_host_archive_bridge"
    ]
    assert len(calls) == 1
    return rel, calls[0]


def test_actual_archive_connector_is_storage_and_native_descriptor_installation():
    rel, call = real_connector_call()
    assert metadata_only_call(
        rel, call, {"_FORMATS"}, ls._parse, ls._native_metadata_factory, ls._metadata_registry_guard
    )
    # Erasing the proof of the native descriptor factory must remove admission.
    assert not metadata_only_call(
        rel, call, {"_FORMATS"}, ls._parse, lambda *args: False, ls._metadata_registry_guard
    )


@pytest.mark.parametrize(
    "extra",
    [
        "    for classes in formats.values():\n        for cls in classes.values():\n            cls.changed()\n",
        "    for classes in formats.values():\n        for cls in classes.values():\n            context['flag'] = len(cls.__dict__)\n",
        "    unknown_callback(formats)\n",
    ],
)
def test_actual_connector_perturbations_require_wide_selection(extra):
    rel, call = real_connector_call()
    bridge = "python/mojolearn/_native_host_archive_bridge.py"
    changed = ast.parse(ls._read(bridge) + extra)
    parse = lambda path: changed if path == bridge else ls._parse(path)
    assert not metadata_only_call(
        rel, call, {"_FORMATS"}, parse, ls._native_metadata_factory, ls._metadata_registry_guard
    )


def test_class_layout_edits_do_not_enable_the_method_only_exception():
    rel, call = real_connector_call()
    assert ls._import_time_hits(ast.Expr(value=call), {"_FORMATS"}, set(), set()) == "_FORMATS"
    assert ls._import_time_hits(ast.Expr(value=call), {"_FORMATS"}, set(), set(), rel=rel) is None


def test_a_native_factory_that_executes_its_argument_is_not_storage(monkeypatch):
    real = ls._MojoDefGraph.text

    def changed(graph, node):
        text = real(graph, node)
        if node[-1] == "classical_attribute_descriptor_c":
            text = text.replace(
                "var descriptor = make_descriptor(function)",
                'make_descriptor = cpy.lib.get_function[PyObjectPtr]("PyObject_CallObject")\n'
                "        var descriptor = make_descriptor(function)",
            )
        return text

    monkeypatch.setattr(ls._MojoDefGraph, "text", changed)
    rel, call = real_connector_call()
    assert not metadata_only_call(
        rel, call, {"_FORMATS"}, ls._parse, ls._native_metadata_factory, ls._metadata_registry_guard
    )


@pytest.mark.parametrize("mutation", ["FORMATS.update(unknown_mapping())", "unknown_callback(FORMATS)"])
def test_a_mutated_registry_is_not_proved_by_its_initial_literal(mutation):
    caller = CALLER.replace(
        "connector.connect(CONTEXT, FORMATS)", mutation + "\nconnector.connect(CONTEXT, FORMATS)"
    )
    assert not proof("context['formats'] = dict(formats)", caller)


@pytest.mark.parametrize("mutation", ["provider", "getter"])
def test_new_expansion_providers_and_module_hooks_invalidate_the_registry_proof(monkeypatch, mutation):
    original = ls._parse
    rel, call = real_connector_call()
    path = "python/mojolearn/_expansion_sequence.py"
    text = ls._read(path)
    if mutation == "provider":
        text += "\ndef classical_host_formats():\n    return unknown_mapping()\n"
    else:
        text = text.replace("def __getattr__(name):", "def __getattr__(name):\n    unknown_callback(name)")
    changed = ast.parse(text)
    monkeypatch.setattr(ls, "_parse", lambda p: changed if p == path else original(p))
    assert not metadata_only_call(
        rel, call, {"_FORMATS"}, ls._parse, ls._native_metadata_factory, ls._metadata_registry_guard
    )


def test_foreign_registry_attributes_do_not_taint_the_local_registry(monkeypatch):
    rel = "python/mojolearn/local.py"
    modules = {
        rel: ast.parse(
            "from . import foreign\n_FORMATS = {}\n"
            "def other():\n    return foreign._FORMATS\n"
            "def local():\n    return _FORMATS\n"
            "def unknown(obj):\n    return obj._FORMATS\n"
        ),
        "python/mojolearn/foreign.py": ast.parse("_FORMATS = {}\n"),
    }
    monkeypatch.setattr(ls, "_parse", modules.get)
    names, _ = ls._module_closure(rel, modules[rel], {"_FORMATS"}, runs=set())
    assert "local" in names and "unknown" in names
    assert "other" not in names


def test_import_alias_used_in_a_subscript_index_is_not_rebound(monkeypatch):
    from import_metadata_effects import Source

    rel = "python/mojolearn/local.py"
    tree = ast.parse(
        "def connect():\n    from . import foreign\n    rows = {}\n"
        "    rows[foreign.SUFFIX] = 1\n    return foreign._FORMATS\n"
    )
    foreign = ast.parse("SUFFIX = 'suffix'\n_FORMATS = {}\n")
    modules = {rel: tree, "python/mojolearn/foreign.py": foreign}
    source = Source(modules.get, lambda *args: False)
    target = source.target(rel, tree.body[0].body[-1].value, tree.body[0].body)
    assert target == ("python/mojolearn/foreign.py", foreign.body[-1].value)
