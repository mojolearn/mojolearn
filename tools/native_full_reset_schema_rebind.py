#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Rebind tools/native_full_reset_constructor_schemas.json to the current source.

The proof pins, by AST digest (docstring and positions excluded), the
NumericModeMixin class hook and its helpers in python/mojolearn/_mode.py, and
each schema class's constructor functions. This tool recomputes the hook and
helper digests from the source and writes them, and records the one write the
installer adds to every class of the door (`device`, last, default None:
python/mojolearn/_native_full_reset_params.py `_install`). It REFUSES to
rebind a class constructor: a changed constructor body changes the audited
write events, which a person reads and re-derives.

    python3 tools/native_full_reset_schema_rebind.py --note "why the helpers changed"
    python3 tools/native_full_reset_schema_rebind.py --check    # exit 1 when the proof is stale
    python3 tools/native_full_reset_schema_rebind.py --rederive IsolationForest
        # one row from a one-function, field-only constructor; prints its _EVENTS tuple

One process, static analysis only; nothing is imported from the package.
"""
import ast
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROOF = ROOT / "tools" / "native_full_reset_constructor_schemas.json"


def ast_sha(node):
    body = node.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        node = copy.copy(node)
        node.body = body[1:] or [ast.Pass()]
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def rederive(proof, name):
    """Re-derive one row whose constructor is ONE function whose body is a
    docstring plus `self.<field> = <parameter or literal>` assignments (the
    only shape this tool reads; anything else is refused). The
    NumericModeMixin writes after the body are kept from the old row."""
    row = next(r for r in proof["classes"] if r["declared_class"] == name)
    if len(row["source_functions"]) != 1 or len(row["trace"]) != 1:
        raise SystemExit("REFUSED: " + name + " has a constructor chain; re-derive it by reading it")
    path = row["source_functions"][0]["path"]
    tree = ast.parse((ROOT / path).read_text())
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == name)
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    arguments = init.args
    if arguments.vararg or arguments.kwarg or arguments.posonlyargs:
        raise SystemExit("REFUSED: " + name + " takes variadic or positional-only arguments")
    parameters = [a.arg for a in arguments.args[1:] + arguments.kwonlyargs]
    where = path + ":" + str(init.lineno)
    writes = []
    for statement in init.body:
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
            continue
        ok = (isinstance(statement, ast.Assign) and len(statement.targets) == 1
              and isinstance(statement.targets[0], ast.Attribute)
              and isinstance(statement.targets[0].value, ast.Name) and statement.targets[0].value.id == "self")
        if not ok:
            raise SystemExit("REFUSED: " + name + ".__init__ is not field-only: " + ast.unparse(statement))
        value = statement.value
        if isinstance(value, ast.Name) and value.id in parameters:
            source = {"parameter": value.id}
        elif isinstance(value, ast.Constant):
            source = {"literal": value.value}
        else:
            raise SystemExit("REFUSED: " + name + " writes a computed value: " + ast.unparse(statement))
        writes.append({"field": statement.targets[0].attr, "value": source, "source": where})
    for write in row["writes"]:
        if write["source"].startswith("NumericModeMixin"):
            writes.append({"field": write["field"], "value": write["value"], "source": "NumericModeMixin after " + where})
    defaults = dict(zip([a.arg for a in arguments.args][len(arguments.args) - len(arguments.defaults):],
                        [ast.unparse(d) for d in arguments.defaults]))
    defaults.update({a.arg: ast.unparse(d) for a, d in zip(arguments.kwonlyargs, arguments.kw_defaults) if d is not None})
    final = {}
    for write in writes:
        if write["field"] == "numeric_mode" and write["source"].startswith("NumericModeMixin") and write["field"] in final:
            continue
        final[write["field"]] = write["value"]
    signature = copy.copy(init)
    row.update(writes=writes, trace=[where], constructor=where, parameters=parameters,
               signature_source="def __init__(" + ast.unparse(arguments) + ")",
               constructor_ast_sha256=ast_sha(init), field_order=list(dict.fromkeys(w["field"] for w in writes)),
               final_fields=final, signature_defaults={k: defaults[k] for k in parameters if k in defaults},
               source_functions=[{"path": path, "name": "__init__", "line": init.lineno, "ast_sha256": ast_sha(init)}])
    events = tuple((w["field"], "parameter" if "parameter" in w["value"] else "constant",
                    next(iter(w["value"].values())), w["source"].startswith("NumericModeMixin")) for w in writes)
    print("_EVENTS[" + repr(name) + "] = " + repr(events))
    return events


def main(argv):
    proof = json.loads(PROOF.read_text())
    if "--rederive" in argv:
        rederive(proof, argv[argv.index("--rederive") + 1])
        PROOF.write_text(json.dumps(proof, indent=2, ensure_ascii=False) + "\n")
        return 0
    wrapper = proof["mode_wrapper_source"]
    source = ast.parse((ROOT / wrapper["path"]).read_text())
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == wrapper["class"])
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == wrapper["method"])
    current = {"ast_sha256": ast_sha(method), "helpers": {
        name: ast_sha(next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == name))
        for name in wrapper["helpers"]}}
    stale = [name for name, digest in current["helpers"].items() if wrapper["helpers"][name] != digest]
    if current["ast_sha256"] != wrapper["ast_sha256"]:
        stale.append(wrapper["method"])
    changed = []
    for row in proof["classes"]:  # every constructor function must still be in its file, unchanged
        for function in row["source_functions"]:
            tree = ast.parse((ROOT / function["path"]).read_text())
            digests = {ast_sha(n) for n in ast.walk(tree)
                       if isinstance(n, ast.FunctionDef) and n.name == function["name"]}
            if function["ast_sha256"] not in digests:
                changed.append(row["declared_class"] + ":" + function["path"] + ":" + function["name"])
    device = {"field": "device", "value": {"parameter": "device"}, "position": "last", "default": None,
              "source": "python/mojolearn/_native_full_reset_params.py _install (every class whose own "
                        "constructor does not name device)",
              "check": "providers/python/device_route_boundary_door.mojo device_admit_class"}
    if "--check" in argv:
        print("stale helpers:", stale or "none")
        print("changed constructors:", changed or "none")
        return 1 if stale or changed or proof.get("installer_device_parameter") != device else 0
    if changed:
        print("REFUSED: constructor source changed; re-derive these rows by reading them:", *changed, sep="\n  ")
        return 1
    note = argv[argv.index("--note") + 1] if "--note" in argv else None
    if stale and not note:
        print("REFUSED: helpers changed (" + ", ".join(stale) + "); pass --note saying why the events are unchanged")
        return 1
    wrapper["ast_sha256"] = current["ast_sha256"]
    wrapper["helpers"] = current["helpers"]
    if stale:
        wrapper["rebound"] = wrapper.get("rebound", "") + " | " + note
    proof["installer_device_parameter"] = device
    PROOF.write_text(json.dumps(proof, indent=2, ensure_ascii=False) + "\n")
    print("rebound:", stale or "nothing stale", "; installer_device_parameter recorded")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
