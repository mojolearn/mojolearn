#!/usr/bin/env python3
"""Collect registered native attribute slots from an actual development activation.

Run in a fresh Python process after the frozen build, with the same interpreter
and environment as runtime tests. This metadata witness does not run or qualify
constructors, numerical operations, output handling or transitive callees.
"""
from __future__ import annotations

import argparse
import ctypes
import importlib
import json
from pathlib import Path
import sys
from types import BuiltinFunctionType, FunctionType, GetSetDescriptorType, MethodType

import installed_attribute_slots as witness


def native_origin(entry, name):
    witness.require(type(entry) is BuiltinFunctionType and entry.__name__ == name,
                    "native entry/factory was replaced: " + name)
    # Public, read-only CPython/dynamic-loader APIs. No object-layout offsets,
    # pointer writes, binary edits or guessed __module__ provenance.
    function = ctypes.pythonapi.PyCFunction_GetFunction
    function.argtypes, function.restype = [ctypes.py_object], ctypes.c_void_p
    pointer = function(entry)

    class DlInfo(ctypes.Structure):
        _fields_ = [("filename", ctypes.c_char_p), ("base", ctypes.c_void_p),
                    ("symbol", ctypes.c_char_p), ("symbol_address", ctypes.c_void_p)]

    info = DlInfo()
    loader = ctypes.CDLL(None).dladdr
    loader.argtypes, loader.restype = [ctypes.c_void_p, ctypes.POINTER(DlInfo)], ctypes.c_int
    witness.require(pointer and loader(pointer, ctypes.byref(info)) and info.filename and info.base,
                    "native function has no resolvable loaded binary: " + name)
    return {"name": name, "path": str(Path(info.filename.decode()).resolve()),
            "offset": pointer - info.base}


def inspect_registered(cls, context, descriptor, entry, descriptor_type):
    """Require current live identities, not collector-authored boolean claims."""
    raw = type.__dict__["__dict__"].__get__
    mro = type.__dict__["__mro__"].__get__(cls)
    fields = raw(cls)
    witness.require(fields.get("__getattribute__") is descriptor, "installed getter was replaced")
    witness.require(type(descriptor) is descriptor_type, "installed descriptor type differs from native factory")
    bound = descriptor.__func__
    witness.require(type(bound) is MethodType and bound.__func__ is entry and bound.__self__ is context,
                    "installed descriptor entry/context was replaced")
    plan = context["plans"].get(cls)
    witness.require(type(plan) is dict and cls not in context["pending"], "native plan is missing or pending")
    witness.require(type(plan.get("mro")) is tuple and len(plan["mro"]) == len(mro)
                    and all(a is b for a, b in zip(plan["mro"], mro)), "captured class ancestry changed")
    names = {}
    for base in mro:
        witness.require(type(base) is type, "Python metaclass cannot be witnessed")
        namespace = raw(base)
        witness.require("__getattr__" not in namespace, "Python __getattr__ cannot be witnessed")
        for name, value in namespace.items():
            names.setdefault(name, value)
    witness.require(plan.get("fields_descriptor") is names.get("__dict__"), "instance dictionary descriptor changed")
    for field in ("bindings", "properties"):
        witness.require(type(plan.get(field)) is dict, "captured descriptor plan is malformed")
        for name, pair in plan[field].items():
            # The compiled owner intentionally binds current exact FunctionType
            # values with PyMethod_New, including later native setter installers.
            # That says nothing about whether invoking that method is native.
            if field == "bindings" and name != "__getattribute__" and type(names.get(name)) is FunctionType:
                continue
            witness.require(type(pair) is tuple and pair and names.get(name) is pair[0],
                            "captured " + field + " identity changed: " + name)
    witness.require(type(plan.get("availability")) is dict, "captured availability plan is malformed")
    for name, saved in plan["availability"].items():
        witness.require(type(saved) is dict and names.get(name) is saved.get("member"),
                        "captured availability identity changed: " + name)
        member = saved["member"]
        ancestry = type.__dict__["__mro__"].__get__(type(member))
        witness.require(len(ancestry) == len(saved["mro"]) and all(a is b for a, b in zip(ancestry, saved["mro"])),
                        "availability descriptor ancestry changed")
        members = {}
        for base in ancestry:
            for key, value in raw(base).items():
                members.setdefault(key, value)
        for key, value in saved["hooks"]:
            witness.require(members.get(key, context["missing"]) is value, "availability descriptor hook changed")
        fields_member = members.get("__dict__")
        witness.require(type(fields_member) is GetSetDescriptorType, "availability dictionary descriptor changed")
        state = fields_member.__get__(member, type(member))
        witness.require(type(saved.get("check")) is FunctionType and type(saved.get("function")) is FunctionType,
                        "availability predicate/function was replaced")
        witness.require(type(state) is dict and state.get("check") is saved["check"]
                        and state.get("fn") is saved["function"]
                        and state.get("attribute_name", context["missing"]) is saved["attribute_name"]
                        and saved["check"].__code__ is saved["code"], "availability descriptor data changed")
    witness.require(plan["bindings"].get("__getattribute__", (None,))[0] is descriptor,
                    "captured getter binding changed")
    return list(witness.CHECKS)


def collect(root, cache, output):
    root, cache, output = Path(root).resolve(), Path(cache).resolve(), Path(output).resolve()
    raw = output.with_name(output.stem + ".observations.json")
    witness.require(not output.exists() and not raw.exists(), "use a new evidence path; retain earlier observations")
    before = witness.source_snapshot(root)
    witness.require("mojolearn" not in sys.modules, "collector requires a fresh product import")
    sys.path.insert(0, str(root / "python"))
    package = importlib.import_module("mojolearn")
    witness.require(Path(package.__file__).resolve() == root / "python/mojolearn/__init__.py",
                    "imported package is outside frozen source")
    owner = importlib.import_module(witness.OWNER)
    context = vars(owner).get("_CONTEXT")
    witness.require(type(context) is dict and type(context.get("plans")) is dict
                    and type(context.get("pending")) is dict, "native attribute owner is not connected")
    module = importlib.import_module("mojolearn.tokenizer")._binding()._module
    entry, factory = getattr(module, witness.ENTRY), getattr(module, witness.FACTORY)
    origins = {name: native_origin(value, name) for name, value in ((witness.ENTRY, entry), (witness.FACTORY, factory))}
    witness.require(origins[witness.ENTRY]["path"] == origins[witness.FACTORY]["path"],
                    "attribute entry and factory came from different binaries")
    binary = Path(origins[witness.ENTRY]["path"])
    index = root / "python/mojolearn/host/_mojolearn_merged.json"
    activation = json.loads(index.read_text())
    key = activation.get("provenance", {}).get(binary.stem)
    witness.require(type(key) is str and len(key) == 64, "loaded owner has no activation key")
    proof = {"binary": witness.reference(binary), "index": witness.reference(index), "origins": origins,
             "receipt": witness.reference(cache / "artifacts" / key / "receipt.json")}
    witness.validate_build(root, proof)
    # The known compiled factory provides its exact native descriptor type.
    descriptor = context.get("descriptor")
    native_type = type(factory(MethodType(entry, context)))
    exports = {}
    for name, cls in vars(package).items():
        if not name.startswith("_") and type(cls) is type and cls.__module__.startswith("mojolearn."):
            exports.setdefault(cls, []).append("mojolearn." + name)
    rows = []
    for cls, names in exports.items():
        row = {"class": cls.__module__ + "." + cls.__qualname__, "exports": sorted(names),
               "method": "__getattribute__"}
        if cls in context["plans"]:
            row.update(status="witnessed", checks=inspect_registered(cls, context, descriptor, entry, native_type))
        elif cls in context["pending"]:
            row.update(status="pending", reason=str(context["pending"][cls]))
        else:
            row.update(status="unmodeled", reason="no registered public-attribute owner plan")
        rows.append(row)
    witness.require(before == witness.source_snapshot(root), "source changed during slot collection")
    data = {"schema": witness.OBSERVATIONS, "owner": witness.OWNER, "source": before,
            "provenance": proof, "runtime_qualified": False, "rows": sorted(rows, key=lambda r: r["class"])}
    output.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(json.dumps(data, indent=2) + "\n")
    output.write_text(json.dumps({"schema": witness.SCHEMA, "observations": witness.reference(raw)}, indent=2) + "\n")
    admitted = witness.validate(output, root)
    return {"witness": str(output), "observed_classes": len(rows), "admitted_getters": len(admitted),
            "pending": sum(r["status"] == "pending" for r in rows),
            "unmodeled": sum(r["status"] == "unmodeled" for r in rows), "runtime_qualified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache/mojolearn-development")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(collect(args.root, args.cache, args.out), sort_keys=True))
    except (witness.WitnessError, OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print("installed-slot collection refused: " + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
