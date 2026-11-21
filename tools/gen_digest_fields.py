#!/usr/bin/env python3
"""Write docs/api/DIGEST_FIELDS.md: what `model.digest_` hashes, per public class.

Read from SOURCE with `ast` (no built package, nothing imported, nothing run),
from the same declarations the compiled walk reads
(bindings/fitted_digest_door.mojo):

- `python/mojolearn/_digest_fields.py`: REFUSED_BY_NAME (no digest yet, with
  the reason), EXTRA_BY_NAME (private fitted fields added), NOT_STATE (public
  names no class hashes);
- the save schemas the classes register
  (`_native_archive_bridge.register(Class, formats, schema)`): the members
  `save` writes;
- the rule for every other class: its public fitted attributes.

A class's row is found the way the walk finds it: the class or its nearest
ancestor (ancestry resolved from the `class` statements and their imports).

Parallel plan: one source file is one independent unit; the files are parsed
across the machine's cores (`os.cpu_count()` workers, a few megabytes each).
The table build over the parsed classes is a few hundred dictionary reads.

    python tools/gen_digest_fields.py            # writes docs/api/DIGEST_FIELDS.md
    python tools/gen_digest_fields.py --check    # exit 1 when the file is stale
"""
import argparse
import ast
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "python" / "mojolearn"

#: step kind -> the positions of the member names it writes
MEMBER_AT = {
    "format": (1,), "estimator": (1,), "mode": (1,), "str": (1,), "lower": (1,), "init_name": (1,),
    "array": (1,), "flat": (1,), "head": (1,), "reshape": (1,), "i8": (1,), "i4": (1,), "f8": (1,), "f4": (1,),
    "classes": (1,), "utf8": (1,), "class_table": (1,), "stack": (1,), "rows": (1,), "device_array": (1,),
    "concat": (1,), "kernel": (4, 5, 6, 7),
}
#: step kinds that only check or derive, and write no member
NO_MEMBER = {"need", "need_set", "need_true", "need_str", "refuse_set", "refuse_in", "need_eq", "need_in",
             "choice", "check_array", "need_either"}
#: step kinds whose last item is a nested step list (written under a condition)
NESTED = {"when", "when_set", "unless", "when_in", "when_none"}
#: the estimator base every class with the shared `digest_` property descends from
BASE = "CheckMixin"


class Unknown:
    """A schema item that is not a literal in the source (a format constant, a table)."""

    def __repr__(self):
        return "?"


def literal(node, constants):
    """`node` as a Python value where the source spells one; `Unknown()` elsewhere.
    Tuples and lists keep their shape; `*_native_archive_bridge.head(fmt)` and
    `_native_archive_bridge.fitted(field)` become the steps they return."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.Tuple, ast.List)):
        items = []
        for element in node.elts:
            if isinstance(element, ast.Starred):
                inner = literal(element.value, constants)
                items.extend(inner if isinstance(inner, tuple) else (Unknown(),))
            else:
                items.append(literal(element, constants))
        return tuple(items)
    if isinstance(node, ast.Name) and node.id in constants:
        value = constants[node.id]
        if isinstance(value, tuple):
            # A name imported from a sibling module: (module, name there).
            return literal(ast.Name(id=value[1]), module_constants(value[0]))
        return literal(value, constants)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        owner = node.func.value
        if isinstance(owner, ast.Name) and owner.id == "_native_archive_bridge":
            if node.func.attr == "head":
                return (("format", "format"), ("estimator", "estimator"), ("mode", "numeric_mode"))
            if node.func.attr == "fitted" and node.args:
                return ("need", literal(node.args[0], constants))
    return Unknown()


def module_constants(module):
    """The module-level assignments (name -> value node) and sibling imports
    (name -> (module, name there)) of one source file of the package."""
    path = PACKAGE / (module + ".py")
    constants = {}
    if not path.is_file():
        return constants
    for node in ast.parse(path.read_text(), filename=str(path)).body:
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            for name in node.names:
                constants[name.asname or name.name] = (node.module, name.name)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            constants[node.targets[0].id] = node.value
    return constants


def parse(path_text):
    """One source file: its classes (name, bases, has a `digest_` of its own, has `fit`)
    and the save schemas it registers. Returns plain data (it crosses a process)."""
    path = Path(path_text)
    tree = ast.parse(path.read_text(), filename=path_text)
    alias = {}
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            for name in node.names:
                if name.asname:
                    alias[name.asname] = name.name
                if node.level == 1 and node.module:
                    constants[name.asname or name.name] = (node.module, name.name)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            constants[node.targets[0].id] = node.value
    classes = []
    schemas = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            bases = []
            for base in node.bases:
                name = base.id if isinstance(base, ast.Name) else (base.attr if isinstance(base, ast.Attribute) else "")
                bases.append(alias.get(name, name))
            methods = {item.name for item in node.body if isinstance(item, ast.FunctionDef)}
            classes.append((node.name, tuple(bases), "digest_" in methods, "fit" in methods))
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "register"
              and isinstance(node.func.value, ast.Name) and node.func.value.id == "_native_archive_bridge"
              and len(node.args) >= 3 and isinstance(node.args[0], ast.Name)):
            steps = literal(node.args[2], constants)
            if isinstance(steps, tuple):
                schemas.append((node.args[0].id, schema_members(steps), False))
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "schema"
              and isinstance(node.func.value, ast.Name) and node.func.value.id == "_digest_fields"
              and len(node.args) >= 2 and isinstance(node.args[0], ast.Name)):
            # a digest schema: run in place of the class's save schema
            steps = literal(node.args[1], constants)
            if isinstance(steps, tuple):
                schemas.append((node.args[0].id, schema_members(steps), True))
    return path.name, classes, schemas


def schema_members(steps, conditional=False):
    """(member name, conditional) for every member a schema writes, in schema order."""
    found = []
    for step in steps:
        if not isinstance(step, tuple) or not step or not isinstance(step[0], str):
            found.append(("(a step built at import)", conditional))
            continue
        kind = step[0]
        if kind in NESTED:
            inner = step[-1] if isinstance(step[-1], tuple) else ()
            found.extend(schema_members(inner, True))
        elif kind in MEMBER_AT:
            found.extend((str(step[i]), conditional) for i in MEMBER_AT[kind]
                         if i < len(step) and isinstance(step[i], str))
        elif kind not in NO_MEMBER:
            found.append((f"(step {kind})", conditional))
    return found


def tables():
    """EXTRA_BY_NAME, REFUSED_BY_NAME and NOT_STATE of `_digest_fields.py`, as written there."""
    tree = ast.parse((PACKAGE / "_digest_fields.py").read_text())
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("EXTRA_BY_NAME", "REFUSED_BY_NAME", "NOT_STATE"):
                found[node.targets[0].id] = ast.literal_eval(node.value)
    return found["EXTRA_BY_NAME"], found["REFUSED_BY_NAME"], found["NOT_STATE"]


def public_names():
    """The public names of the package: `__all__` and the lazy door names of `__init__.py`."""
    tree = ast.parse((PACKAGE / "__init__.py").read_text())
    names = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target = node.targets[0].id
            if target == "__all__" and isinstance(node.value, (ast.List, ast.Tuple)):
                names.extend(e.value for e in node.value.elts if isinstance(e, ast.Constant))
            elif target == "_LAZY_NAMES" and isinstance(node.value, ast.Dict):
                names.extend(k.value for k in node.value.keys if isinstance(k, ast.Constant))
    return sorted({name for name in names if isinstance(name, str) and name[:1].isupper()})


def sources():
    """The package's source files, without its tests."""
    files = [p for p in sorted(PACKAGE.glob("*.py"))] + [p for p in sorted((PACKAGE / "models").glob("*.py"))]
    return [str(p) for p in files]


def rows(jobs):
    extra, refused, _ = tables()
    files = sources()
    if jobs == 1:
        parsed = [parse(path) for path in files]
    else:
        with ProcessPoolExecutor(max_workers=min(len(files), jobs)) as pool:
            parsed = list(pool.map(parse, files, chunksize=8))
    classes = {}
    schemas = {}
    digest_schemas = {}
    for _, found, registered in parsed:
        for name, bases, own_digest, has_fit in found:
            classes.setdefault(name, (bases, own_digest, has_fit))
        for name, listed, for_digest in registered:
            if for_digest:
                digest_schemas[name] = listed
            else:
                schemas[name] = listed

    def ancestry(name, seen):
        """The class and its ancestors, nearest first (the order the walk looks tables up in)."""
        if name in seen or name not in classes:
            return
        seen.append(name)
        for base in classes[name][0]:
            ancestry(base, seen)

    out = []
    for name in public_names():
        if name not in classes:
            continue
        line = []
        ancestry(name, line)
        if not any(classes[c][2] for c in line):
            continue                                    # not an estimator (no `fit` anywhere in its ancestry)
        if BASE not in line and not any(classes[c][1] for c in line):
            out.append((name, "no `digest_`", "the class is not on the shared estimator base"))
            continue
        reason = next((refused[c] for c in line if c in refused), None)
        if reason is not None:
            out.append((name, "not yet", reason))
            continue
        source = "digest schema"
        listed = next((digest_schemas[c] for c in line if c in digest_schemas), None)
        if listed is None:
            source = "save schema"
            listed = next((schemas[c] for c in line if c in schemas), None)
        if listed is not None:
            text = ", ".join(f"`{m}`" + (" (when present)" if cond else "") for m, cond in listed)
            out.append((name, source, text))
            continue
        added = next((extra[c] for c in line if c in extra), ())
        text = "every public fitted attribute"
        if added:
            text += "; plus " + ", ".join(f"`{n}`" for n in added)
        out.append((name, "public fitted attributes", text))
    return out


def render(table):
    _, _, not_state = tables()
    lines = [
        "# What `digest_` hashes",
        "",
        "Generated by `tools/gen_digest_fields.py` from the source declarations the compiled digest walk reads",
        "(`python/mojolearn/_digest_fields.py` and the registered save schemas). Do not edit.",
        "",
        "Every digest starts with the format word `mojolearn.digest.v1` and the class name. Then each field below, in",
        "name order, as its name, dtype, shape and bytes in C order. A list, tuple or dictionary field is entered item by",
        "item; a member estimator contributes its own digest; a layer object contributes its class name and its learned",
        "arrays. Nothing is kept between reads: the digest is hashed from the model as it is at each read.",
        "",
        "- **save schema**: the members `save` writes for the class.",
        "- **digest schema**: the members `save` writes, built without the refusals of `save` (a model the file format",
        "  does not carry still has a digest).",
        "- **public fitted attributes**: every stored attribute whose name ends in `_` and does not start with `_`,",
        "  value not None, plus the private fitted fields listed (a listed name the model does not hold is passed over).",
        "- Public names no class hashes (they record how a call ran, not what was fitted): "
        + ", ".join(f"`{n}`" for n in not_state) + ".",
        "",
        "| Class | Source | Fields |",
        "|---|---|---|",
    ]
    lines.extend(f"| `{name}` | {source} | {text} |" for name, source, text in table)
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "docs/api/DIGEST_FIELDS.md")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1,
                        help="parse workers (default: every core; 1 on a shared laptop)")
    args = parser.parse_args()
    text = render(rows(max(1, args.jobs)))
    if args.check:
        current = args.out.read_text() if args.out.is_file() else ""
        if current != text:
            print(f"STALE: {args.out}; run tools/gen_digest_fields.py", file=sys.stderr)
            return 1
        return 0
    args.out.write_text(text)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
