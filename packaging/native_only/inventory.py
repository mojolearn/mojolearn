# SPDX-License-Identifier: Apache-2.0
"""Testing/packaging-only public export census; never imports the product."""
from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path


def literal(node):
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return None


def explicit_exports(path):
    """Literal __all__ fragments, excluding dynamic registry extension."""
    names = set()
    for node in ast.parse(path.read_text()).body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            continue
        targets = getattr(node, "targets", [getattr(node, "target", None)])
        if any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
            value = literal(node.value)
            if isinstance(value, (tuple, list)):
                names.update(name for name in value if isinstance(name, str))
    return sorted(names)


def origins(root):
    """Resolve only statically proven aliases; never equate names by spelling."""
    base = root / "python/mojolearn"

    def module_file(module):
        relative = module.removeprefix("mojolearn").lstrip(".")
        path = base.joinpath(*relative.split(".")) if relative else base
        if path.with_suffix(".py").is_file():
            return path.with_suffix(".py")
        if (path / "__init__.py").is_file():
            return path / "__init__.py"
        return None

    def module_of(path):
        parts = list(path.relative_to(base).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        return ".".join(["mojolearn", *parts])

    @lru_cache(None)
    def resolve(module, name, depth=0):
        if depth > 30:
            return module + "." + name
        path = module_file(module)
        if path is None:
            return module + "." + name
        package = module if path.name == "__init__.py" else module.rpartition(".")[0]
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
                return module + "." + name
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if (alias.asname or alias.name) != name:
                        continue
                    source = node.module or ""
                    if node.level:
                        head = package.split(".")
                        head = head[:len(head) - node.level + 1]
                        source = ".".join(head + ([source] if source else []))
                    if source.startswith("mojolearn"):
                        return resolve(source, alias.name, depth + 1)
            if isinstance(node, ast.Assign):
                targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
                if name in targets and isinstance(node.value, ast.Name):
                    return resolve(module, node.value.id, depth + 1)
                if "_LAZY_EXPORTS" in targets:
                    mapping = literal(node.value)
                    if isinstance(mapping, dict) and name in mapping:
                        target_module, attribute = mapping[name]
                        return resolve("mojolearn." + target_module, attribute, depth + 1)
        return module + "." + name

    return resolve, module_of


def canonicalize(root, manifest):
    rows = manifest["inventory"]
    resolve, module_of = origins(root)
    layout_rows = {}
    layout_path = root / "python/mojolearn/_sklearn_modules.py"
    for node in ast.parse(layout_path.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "LAYOUT" for t in node.targets):
            for module, names in literal(node.value).items():
                layout_rows.update({f"mojolearn.{module}.{name}": f"mojolearn.{name}" for name in names})
    by_name = {row["algorithm"]: row for row in rows}
    groups = {}
    for row in rows:
        name = row["algorithm"]
        if name in layout_rows:
            # LAYOUT's lazy_exports explicitly resolves this root attribute.
            target = layout_rows[name]
            origin_row = by_name.get(target)
            if origin_row:
                module = module_of(root / origin_row["source"])
                origin = resolve(module, target.rsplit(".", 1)[-1])
            else:
                origin = target
        else:
            source = root / row["source"]
            origin = (resolve(module_of(source), name.rsplit(".", 1)[-1])
                      if source.is_file() else name)
        groups.setdefault(origin, []).append(row)
    case_by_id = {case["id"]: case for case in manifest["cases"]}
    for origin, group in groups.items():
        canonical = min((row["algorithm"] for row in group), key=lambda name: (name.count("."), len(name), name))
        case_ids = sorted({case for row in group for case in row["cases"]})
        canonical_row = by_name[canonical]
        for row in group:
            row["canonical_algorithm"] = canonical
            row["symbol_origin"] = origin
            row.pop("alias_of", None)
            if row["algorithm"] != canonical:
                row["alias_of"] = canonical
            if case_ids:
                row["cases"] = case_ids
                row["status"] = "ready"
                row["reason"] = "Coverage is limited to each case scope; source presence does not certify compilation or execution."
                for case_id in case_ids:
                    aliases = case_by_id[case_id]["algorithms"]
                    if row["algorithm"] not in aliases:
                        aliases.append(row["algorithm"])
            if canonical_row.get("boundary_blocker"):
                row["boundary_blocker"] = canonical_row["boundary_blocker"]
    # Cases list actual public aliases for exact witness matching. This field
    # keeps the distinct canonical count available without inflating it.
    for case in manifest["cases"]:
        case["algorithms"] = sorted(set(case["algorithms"]))
        case["canonical_algorithms"] = sorted({by_name[a]["canonical_algorithm"] for a in case["algorithms"]})
    canonical_rows = [row for row in rows if "alias_of" not in row]
    algorithms = [row for row in canonical_rows if row["status"] != "not_algorithm"]
    manifest["census"] = {
        "public_export_rows": len(rows),
        "alias_rows": len(rows) - len(canonical_rows),
        "canonical_non_algorithm_exports": sum(row["status"] == "not_algorithm" for row in canonical_rows),
        "canonical_algorithm_entries": len(algorithms),
        "canonical_status_counts": {status: sum(row["status"] == status for row in algorithms)
                                    for status in sorted({row["status"] for row in algorithms})},
        "note": "Algorithm entries include public numerical/helper operations as well as estimator classes; no per-estimator completeness is implied by this denominator.",
    }
    return manifest
