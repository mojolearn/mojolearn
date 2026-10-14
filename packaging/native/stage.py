#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build-time distribution tooling only; nothing here is installed or executed by Mojo callers."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from bincache import EXTERNAL_TOP, parse_imports, resolve  # noqa: E402


def sources(root, entries):
    """Resolve every local import, including parent initializers and conditional imports."""
    pending = [root / entry for entry in entries]
    seen = set()
    while pending:
        path = pending.pop().resolve()
        path.relative_to(root)  # refuse a dependency outside the source tree
        if path in seen:
            continue
        seen.add(path)
        for module, names in parse_imports(path.read_text()):
            if module == "python" or module.startswith(("python.", "std.python")):
                raise ValueError(f"Python runtime dependency in {path.relative_to(root)}: {module}")
            if module == "std" and "python" in names:
                raise ValueError(f"Python runtime dependency in {path.relative_to(root)}: std.python")
            # bincache's conservative resolver can return a parent initializer
            # even when the final module is absent. A distributor must refuse
            # that case rather than emit a package with a broken import.
            if module.startswith("."):
                dots = len(module) - len(module.lstrip("."))
                base = path.parent
                for _ in range(dots - 1):
                    base = base.parent
                candidates = [base.joinpath(*module.lstrip(".").split("."))]
            else:
                candidates = [base.joinpath(*module.split(".")) for base in (root, path.parent)]
            if module.split(".")[0] not in EXTERNAL_TOP and not any(
                candidate.is_dir() or candidate.with_suffix(".mojo").is_file() for candidate in candidates
            ):
                raise ValueError(f"unresolved import in {path.relative_to(root)}: {module}")
            found = resolve(module, names, path, [root])
            if found is None:
                raise ValueError(f"unresolved import in {path.relative_to(root)}: {module}")
            pending.extend(found - seen)
        # An entry point can have parent packages even if nothing imports it.
        for parent in path.relative_to(root).parents:
            if parent == Path("."):
                continue
            init = root / parent / "__init__.mojo"
            if init.is_file() and init not in seen:
                pending.append(init)
    return sorted(seen)


def namespace(text, path, local_roots):
    """Relocate repository imports under mojolearn, leaving code and docstrings intact."""
    prefix = "." * len(path.parts)
    result = []
    in_doc = False
    for line in text.splitlines(keepends=True):
        quotes = line.count('"""')
        if in_doc:
            if quotes % 2:
                in_doc = False
            result.append(line)
            continue
        if quotes % 2:
            in_doc = True
            result.append(line)
            continue
        # Some shared numerical modules also contain a standalone check.
        # Mojo allows importing them from source but precompiled packages may
        # not contain a program entry point. Preserve its body under a private
        # name; no algorithm or caller-facing function is changed.
        if line.startswith("def main("):
            line = line.replace("def main(", "def _standalone_main(", 1)
        match = re.match(r"(\s*from\s+)([\w.]+)(\s+import\s+)", line)
        if match and match[2].split(".")[0] in local_roots:
            line = line[:match.start(2)] + prefix + line[match.start(2):]
        elif re.match(r"\s*import\s+", line):
            # Keep the binding of `import root` and `import root.child as alias`.
            match = re.fullmatch(r"(\s*)import\s+([\w.]+)(?:\s+as\s+(\w+))?(\s*(?:#.*)?\n?)", line)
            if match and match[2].split(".")[0] in local_roots:
                module, _, leaf = match[2].rpartition(".")
                if module and not match[3]:
                    raise ValueError(f"dotted local import needs an alias: {path}: {line.strip()}")
                alias = " as " + match[3] if match[3] else ""
                line = f"{match[1]}from {prefix}{module} import {leaf}{alias}{match[4]}"
            elif any(module.split(".")[0] in local_roots for module, _ in parse_imports(line)):
                raise ValueError(f"unsupported local import: {path}: {line.strip()}")
        result.append(line)
    return "".join(result)


def stage(root, destination):
    # Refuse an existing directory: stale sources must never enter a new artifact.
    destination.mkdir(parents=True, exist_ok=False)
    entries = [line.strip() for line in (root / "packaging/native/entrypoints.txt").read_text().splitlines()
               if line.strip() and not line.lstrip().startswith("#")]
    paths = sources(root, entries)
    local_roots = {path.relative_to(root).parts[0] for path in paths}
    records = []
    package = destination / "mojolearn"
    package.mkdir()
    (package / "__init__.mojo").write_text(
        '# SPDX-License-Identifier: Apache-2.0\n'
        '"""Native machine learning for Mojo.\n\n'
        'In `identical` mode, mojolearn returns the same bits on Apple, NVIDIA and AMD GPUs.\n'
        '"""\n'
        'from .checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, NUMERIC_FAST\n'
    )
    for path in paths:
        rel = path.relative_to(root)
        original = path.read_text()
        relocated = namespace(original, rel, local_roots)
        if rel == Path("core/package_profile.mojo"):
            marker = "comptime NATIVE_PACKAGE = False"
            if relocated.count(marker) != 1:
                raise ValueError("native package profile marker missing or ambiguous")
            relocated = relocated.replace(marker, "comptime NATIVE_PACKAGE = True")
        target = package / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(relocated)
        records.append(dict(path=rel.as_posix(), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            packaged_sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
        for parent in target.parents:
            if parent == package:
                break
            init = parent / "__init__.mojo"
            if not init.exists():
                init.write_text("# SPDX-License-Identifier: Apache-2.0\n")
    manifest = dict(schema=1, default_mode="identical", fast="Apple only", entrypoints=entries,
                    mojo="1.0.0", max_core="26.5.0", sources=records)
    (destination / "native-sources.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Staged {len(entries)} entry points and {len(paths)} source modules")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    stage(ROOT, args.destination.resolve())
