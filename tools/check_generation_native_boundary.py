#!/usr/bin/env python3
"""Diagnostic local dependency inventory.

The target architecture confines Python support to the actual UNPACK/PACK
boundaries. The default inventory follows providers/bindings rather than
exempting their algorithms and flags unresolved local dependencies. Resolving
those imports does not establish that the architecture is satisfied. This tool
cannot prove dynamic targets, admitted types, interpreter execution, hooks or
transitive runtime safety; executed-call checks and source review remain separate.

--require-native-only is an optional architectural assertion for a deliberately
pure root, not product execution policy and not a mandatory source/push gate.
Neither mode inspects compiled upstream standard libraries or runs product code.
"""
import argparse
from pathlib import Path

import re

ROOT = Path(__file__).resolve().parents[1]
ROOTS = ("transformer.session_native_dispatch", "core.transformer_block_session", "core.native_descriptor_abi",
         "core.native_descriptor_call", "training.native_primitive_dispatch",
         "mamba.native_state_dispatch", "transformer.native_lm_resident_dispatch",
         "transformer.causal_lm_program")
FORBIDDEN_IMPORTS = ("std.python", "providers", "bindings")
CALLBACK_TOKENS = re.compile(r"\b(?:PythonObject|PyObjectPtr|Python|py_call|py_call_typed|GILReleased)\b")
# Resolved from Modular's installed native packages, not this repository.
# nn.topk is supplied by lib/mojo/nn.mojoc; package hash/provenance is retained
# in experiments/neural_provider_layout_20261027/external_native.json.
# layout.tensor_engine (DefaultEngine for top_k, Mojo 1.2.0, 7911e021d6) and comm.sync (enable_p2p, 2b4af0abe6) are
# Modular's layout.mojoc and comm.mojoc.
EXTERNAL_MODULES = {"layout", "layout.tile_layout", "layout.tensor_engine", "linalg.gemv", "linalg.matmul", "nn.topk",
                    "comm.sync"}


def unbuilt_generated(problems, root=ROOT):
    """The problems that are only a REGISTERED generated module (tools/generated_sources.GENERATED) a build has not
    written yet in this checkout, with its generator and pin tracked. A build writes them before it compiles
    (generated_sources.py prepare, which downloads pinned Unicode data); audit() never exempts them itself, and a
    present generated file is audited like any source."""
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        from generated_sources import GENERATED
    finally:
        sys.path.pop(0)
    registered = {rel[:-len(".mojo")].replace("/", "."): rel for rel in GENERATED}
    return [(chain, reason) for chain, reason in problems
            if reason == "unresolved local dependency" and chain[-1] in registered
            and not (root / registered[chain[-1]]).exists()
            and all((root / dep).is_file() for dep in GENERATED[registered[chain[-1]]])]


def code_only(source):
    source = re.sub(r'""".*?"""|\'\'\'.*?\'\'\'', "", source, flags=re.S)
    return re.sub(r"#.*", "", source)


def local_file(root, module):
    path = root.joinpath(*module.split("."))
    for candidate in (path.with_suffix(".mojo"), path / "__init__.mojo"):
        if candidate.is_file():
            return candidate
    return None


def audit(root=ROOT, roots=ROOTS, *, require_native_only=False):
    queue = [(module, (module,)) for module in roots]
    seen, violations = set(), []
    while queue:
        module, chain = queue.pop()
        if module in seen:
            continue
        seen.add(module)
        path = local_file(root, module)
        if path is None:
            if not module.startswith(("std.", "max.", "python.")) and module != "python" and module not in EXTERNAL_MODULES:
                violations.append((chain, "unresolved local dependency"))
            continue
        source = path.read_text()
        code = code_only(source)
        without_strings = re.sub(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', "", code)
        if require_native_only and (CALLBACK_TOKENS.search(without_strings) or re.search(r'["\']Py(?:Object|CFunction|Capsule|Import|Err|_)[A-Za-z_]*["\']', code)):
            violations.append((chain, "Python/CPython helper token in native code"))
        imports = re.findall(r"^\s*(?:from|import)\s+([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)", code, re.M)
        for imported in imports:
            if require_native_only and any(imported == prefix or imported.startswith(prefix + ".") for prefix in FORBIDDEN_IMPORTS):
                violations.append((chain + (imported,), "provider/Python dependency"))
            else:
                queue.append((imported, chain + (imported,)))
    return seen, violations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-native-only", action="store_true",
                        help="opt-in import-purity assertion; not runtime policy")
    options = parser.parse_args()
    seen, violations = audit(require_native_only=options.require_native_only)
    for chain, reason in violations:
        print(f"FAIL {reason}: {' -> '.join(chain)}")
    scope = "opt-in pure-root assertion" if options.require_native_only else "dependency/source inventory (not callback-safety proof)"
    print(f"generation {scope}: {len(seen)} modules, {len(violations)} findings")
    return int(bool(violations))


if __name__ == "__main__":
    raise SystemExit(main())
