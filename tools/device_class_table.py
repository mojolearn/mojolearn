#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A static estimate of which estimator classes take `device=` and what
`device="cpu"` answers beside a GPU set, read from the package SOURCE (no
import, no binary): `python3 tools/device_class_table.py [--tsv PATH]`.
`--write-routes` regenerates `CPU_TRAINING_ROUTES` in python/mojolearn/_device_table.py
from the host surface manifest (the package never imports the manifest).

The package's own answer is `mojolearn._mode.device_report()` on an installed
build (it knows which constructors the compiled installers took). This tool
approximates it from source for a report written where nothing is installed:
it follows class bases by name across python/mojolearn, resolves `_BINDING`
and `_mojolearn_device` through that ancestry, and reads the three fixed
tables that take a class away from the shared constructor wrapper (the
full-reset event table, the mode-reconstruction recipes, `_connect_native_init`).
One process, one pass over the files; nothing here is product code.
"""
import ast
import importlib.util
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python", "mojolearn")


def _host_surface():
    spec = importlib.util.spec_from_file_location("_hs", os.path.join(ROOT, "host_surface.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules["_hs"] = module
    spec.loader.exec_module(module)
    return module


def _scan():
    classes = {}
    text = {}
    for name in sorted(os.listdir(ROOT)):
        if not name.endswith(".py"):
            continue
        source = open(os.path.join(ROOT, name), encoding="utf-8").read()
        text[name] = source
        tree = ast.parse(source)
        alias = {a.asname: a.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                 for a in node.names if a.asname}
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            named = [b.attr if isinstance(b, ast.Attribute) else getattr(b, "id", None) for b in node.bases]
            row = dict(module=name[:-3], bases=[alias.get(b, b) for b in named],
                       native_ctor=any(k.arg == "_native_constructor" for k in node.keywords), fields={}, init=None)
            for item in node.body:
                if isinstance(item, ast.Assign) and len(item.targets) == 1 and isinstance(item.targets[0], ast.Name):
                    if isinstance(item.value, ast.Constant):
                        row["fields"][item.targets[0].id] = item.value.value
                elif isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    row["init"] = [a.arg for a in item.args.args + item.args.kwonlyargs]
                elif isinstance(item, ast.FunctionDef) and item.name in ("get_params", "set_params", "_get_param_names"):
                    row["fields"]["own_" + item.name] = True
            classes.setdefault(node.name, row)
    return classes, text


def _ancestry(classes, name, seen=None):
    seen = seen or []
    if name in seen or name not in classes:
        return seen
    seen.append(name)
    for base in classes[name]["bases"]:
        if base:
            _ancestry(classes, base, seen)
    return seen


DEVICE_TABLE = os.path.join(ROOT, "_device_table.py")
BEGIN = "# BEGIN generated: tools/device_class_table.py --write-routes\n"
END = "# END generated\n"


def routes_block():
    """The generated `CPU_TRAINING_ROUTES` of python/mojolearn/_device_table.py:
    the keys of `host_surface.cpu_training_routes()`, so the package reads the
    manifest's answer without importing the manifest."""
    names = sorted(_host_surface().cpu_training_routes())
    return "CPU_TRAINING_ROUTES = frozenset((\n" + "".join(f"    {n!r},\n" for n in names) + "))\n"


def write_routes():
    source = open(DEVICE_TABLE, encoding="utf-8").read()
    head, rest = source.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    open(DEVICE_TABLE, "w", encoding="utf-8").write(head + BEGIN + routes_block() + END + tail)


def main(argv):
    if "--write-routes" in argv:
        write_routes()
        return 0
    classes, text = _scan()
    routes = _host_surface().cpu_training_routes()
    compiled = set(re.findall(r"^\s+'(\w+)':", text.get("_native_full_reset_params.py", ""), re.M))
    recipes = set(m.split(".")[-1] for m in re.findall(r"^\s+'([\w.]+)': \(", text.get("_native_mode_reconstruction.py", ""), re.M))
    native_inits = set(re.findall(r"_connect_native_init\((\w+),", "\n".join(text.values())))
    rows = []
    for name, row in sorted(classes.items()):
        line = _ancestry(classes, name)
        if "NumericModeMixin" not in line or name == "NumericModeMixin" or name.startswith("_"):
            continue

        def inherited(field):
            return next((classes[c]["fields"][field] for c in line if field in classes[c]["fields"]), None)

        init = next((classes[c]["init"] for c in line if classes[c]["init"] is not None), None)
        owner = next((c for c in line if classes[c]["init"] is not None), None)
        if inherited("_HOST_INFERENCE_ONLY") or row["module"] in ("_classical_host",):
            continue
        if init is not None and "device" in init:
            takes = "own constructor"
        elif name in compiled:
            # python/mojolearn/_native_full_reset_params.py `_install` adds the keyword and the
            # compiled constructor and setter check it (lane dw-core round 2).
            takes = "shared constructor door"
        elif inherited("_mojolearn_native_public") or (owner and classes[owner]["native_ctor"]):
            takes = "NO: compiled constructor (its family names device)"
        elif any(c in compiled for c in line) or any(c in recipes for c in line) or any(c in native_inits for c in line):
            takes = "NO: compiled constructor or recipe (its family names device)"
        elif any(inherited("own_" + m) and "ParamsMixin" not in [c for c in line if classes[c]["fields"].get("own_" + m)]
                 for m in ("get_params", "set_params", "_get_param_names")):
            takes = "NO: declared parameter list (its family names device)"
        elif init is None:
            takes = "NO: no constructor"
        else:
            takes = "shared wrapper"
        binding = inherited("_BINDING") or "_mojolearn"
        if inherited("_mojolearn_device") is True:
            cpu = "runs"
        elif binding not in routes:
            cpu = "refused: its family has no CPU implementation"
        else:
            cpu = "refused: its CPU implementation is not connected to device= yet"
        rows.append((name, row["module"], binding, takes, cpu))
    out = "\n".join("\t".join(r) for r in [("class", "module", "binding", "takes device=", "device='cpu' beside a GPU set")] + rows)
    if "--tsv" in argv:
        open(argv[argv.index("--tsv") + 1], "w", encoding="utf-8").write(out + "\n")
    takes = [r for r in rows if not r[3].startswith("NO")]
    print("estimator classes:", len(rows))
    print("take device=:", len(takes), "(shared wrapper:", sum(r[3] == "shared wrapper" for r in rows),
          "shared constructor door:", sum(r[3] == "shared constructor door" for r in rows),
          "own constructor:", sum(r[3] == "own constructor" for r in rows), ")")
    print("of those, device='cpu' beside a GPU set runs:", sum(r[4] == "runs" for r in takes))
    print("refused, no CPU implementation:", sum("no CPU" in r[4] for r in takes))
    print("refused, not connected yet:", sum("not connected" in r[4] for r in takes))
    print("do not take device= yet:", len(rows) - len(takes))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
