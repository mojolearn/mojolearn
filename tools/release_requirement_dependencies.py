#!/usr/bin/env python3
"""Frozen, independently reusable build/lane/workload requirement inputs.

This is source analysis, never compilation or runtime evidence. Compilation
uses release_reuse's existing binding identities, including filtered builder
views. Identity uses lane_select's actual dependency closures. Its named-edit
selector can also prove that a changed shared source file leaves a lane alone;
that proof preserves the original execution receipt and tested revision.

Configuration under ``dependency_context`` accepts ``apple_toolchain`` (the
recorded build host toolchain), ``compiler_sha256`` keyed by execution target,
``cpu_target``, ``optional_native_files`` (per-target Boolean presence contracts
for source-proven optional loaders), and ``workloads`` keyed by vendor.
Each workload specifies id, lanes, source_inputs (exact recipe/harness paths),
dataset, settings, boundary, phases and environment. Missing provenance blocks
the corresponding unit; it never expands to a guessed whole-library rerun.
"""
from __future__ import annotations

import copy
from functools import lru_cache
import hashlib
import json
import os
import re
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from release_requirement_routes import loader_routes

SCHEMA = "mojolearn.release-dependencies.v1"
PROOF_SCHEMA = "mojolearn.release-dependency-equivalence.v1"
VENDORS = {"apple": "metal", "nvidia": "cuda", "amd": "hip", "cpu": "host"}
RECIPE_INDEX = "experiments/performance_ideas/README.md"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _revision(root, revision):
    return _git(root, "rev-parse", "--verify", revision + "^{commit}")


def _file_hash(root, path):
    if not isinstance(path, str) or Path(path).is_absolute() or ".." in Path(path).parts or any(c in path for c in "*?["):
        raise ValueError("dependency source must be an exact repository-relative path: " + str(path))
    p = Path(root) / path
    if not p.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("dependency source escapes frozen tree: " + path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def frozen_bindings(rr, target, modes=("identical",)):
    """The frozen revision's own binding registry for MODES, plus host bindings.

    Current registries select tiers with ``bindings(target, modes)``. Releases
    up to 0.8.55 take only ``target`` and list every tier (fast and identical)
    plus host. Filter that frozen list to the same rows; never substitute
    today's module list for an older release.
    """
    import inspect
    params = list(inspect.signature(rr.bindings).parameters)
    if "modes" in params:
        return list(rr.bindings(target, list(modes)))
    if params != ["target"]:
        raise ValueError("unrecognized frozen binding registry signature: bindings(" + ", ".join(params) + ")")
    wanted = set(modes) | {"host"}
    return [binding for binding in rr.bindings(target) if binding.tier in wanted]


def _worker_snapshot(root, request):
    # Import the requested revision's tools, never another checkout's modules.
    root = Path(root).resolve()
    own_tools = str(Path(__file__).resolve().parent)
    sys.path[:] = [str(root / "tools")] + [p for p in sys.path if p not in (own_tools, "")]
    for name in ("release_reuse.py", "lane_select.py", "bincache.py"):
        if not (root / "tools" / name).is_file():
            raise ValueError("frozen dependency tool missing: " + name)
    import release_reuse as rr
    import lane_select as ls
    import bincache
    if Path(rr.ROOT).resolve() != root or Path(ls.ROOT).resolve() != root:
        raise ValueError("dependency tools do not belong to the frozen revision")
    config = request.get("configuration", {})
    context = config.get("dependency_context", {})
    result = dict(schema=SCHEMA, revision=_git(root, "rev-parse", "HEAD"))
    facts = rr.tree_facts(root) if request.get("compile") or request.get("lanes") else None
    result["bindings"] = []
    if request.get("identity_scope") or request.get("lanes"):
        import lane_applicability as applicability
        import verify_lanes
        columns = {vendor: verify_lanes.PASS_COLUMN[backend] for vendor, backend in
                   (("apple", "metal"), ("nvidia", "cuda"), ("amd", "hip"), ("cpu", "cpu"))}
        scoped = {}
        for lane, scope in applicability.scopes().items():
            applicable = {vendor: scope.applicable(column) for vendor, column in columns.items()}
            scoped[lane] = dict(kind=scope.kind, claim_devices=scope.claim_devices,
                oracle=scope.oracle, anchor=scope.anchor,
                required_columns=[v for v in ("apple", "nvidia", "amd", "cpu") if applicable[v][0]],
                excluded={v: reason for v, (ok, reason) in applicable.items() if not ok},
                separate_requirements=(["multi-gpu-nvidia", "multi-gpu-amd"] if scope.claim_devices >= 2 else []))
        paths = ["tools/lane_applicability.py", "tools/lane_select.py", "tools/verify_lanes.py",
                 "tools/identity_break.py", "python/mojolearn/host_surface.py"]
        paths += [str(p.relative_to(root)) for p in (root / "tools/identity_lanes").glob("*.py")]
        proof = dict(schema="mojolearn.release-identity-applicability.v1", source_revision=result["revision"],
                     columns=columns, lanes=scoped, sources={p: _file_hash(root, p) for p in sorted(paths)})
        proof["digest"] = digest(proof)
        result["identity_scope"] = proof
    if facts:
        # Only identical is authorized. Host bindings are included by bindings().
        for target in (rr.LINUX, rr.MACOS):
            for binding in frozen_bindings(rr, target):
                ident = rr.identity(binding, facts, context.get("apple_toolchain"))
                closure = facts["closures"].get(binding.script) or {}
                missing = [rr.unreadable(ident)] if rr.unreadable(ident) else []
                if closure.get("scope") != "closure":
                    missing.append("native import closure unresolved; whole-tree fallback is not admitted")
                provenance = list(closure.get("sources", [])) + list((ident or {}).get("builders", {}))
                result["bindings"].append(dict(key=binding.key, target=target,
                    vendor=binding.vendor, arch=binding.arch, tier=binding.tier,
                    name=binding.name, identity=ident, sources=sorted(set(provenance)),
                    unresolved=missing))
        result["runtime"] = dict(identity=rr.runtime_identity(facts),
            unresolved=rr.unreadable_runtime(rr.runtime_identity(facts)))
        result["toolchains"] = facts["toolchain"]
    if request.get("compile"):
        # This standalone build is outside release_reuse's Python binding table.
        # Follow the same supported Mojo import resolver, plus local C++ headers.
        seeds = ["adapters/native/c_api.mojo"] + [str(p.relative_to(root))
            for p in (root / "adapters/max_ops").rglob("*.mojo")]
        todo, paths, missing = list(seeds), set(), []
        while todo:
            rel = todo.pop()
            if rel in paths:
                continue
            paths.add(rel)
            path = root / rel
            if not path.is_file():
                missing.append("missing adapter source: " + rel)
                continue
            for module, names in bincache.parse_imports(path.read_text()):
                resolved = bincache.resolve(module, names, path, [root, root / "bindings"])
                if resolved is None:
                    missing.append("unresolved adapter import: " + rel + ": " + module)
                else:
                    todo += [str(p.relative_to(root)) for p in resolved]
        import re
        # Directory-level source additions are build inputs to the adapter
        # bundle. This small compile unit is separate from every Python binding.
        todo = [str(p.relative_to(root)) for p in (root / "adapters/torch").rglob("*")
                if p.suffix in (".cpp", ".h", ".hpp")]
        while todo:
            rel = todo.pop()
            if rel in paths:
                continue
            paths.add(rel)
            path = root / rel
            if not path.is_file():
                missing.append("missing adapter C++ source: " + rel)
                continue
            for include in re.findall(r'^\s*#\s*include\s*"([^"\n]+)"', path.read_text(), re.M):
                options = [path.parent / include, root / "adapters/native" / include]
                found = next((p for p in options if p.is_file()), None)
                if found is None:
                    missing.append("unresolved adapter C++ include: " + include)
                else:
                    todo.append(str(found.resolve().relative_to(root)))
        paths.update(["packaging/adapters/build.sh", "packaging/adapters/fetch_torch.py",
                      "packaging/adapters/manifest.py", "packaging/adapters/pins.json"])
        hashes = {p: _file_hash(root, p) for p in sorted(paths)}
        missing += ["missing adapter build input: " + p for p, value in hashes.items() if value is None]
        builder_views = {}
        for target, builders in ((rr.LINUX, rr.LINUX_BUILDERS), (rr.MACOS, rr.MACOS_BUILDERS)):
            builder_views[target] = {rel: rr.builder_view(rel, (root / rel).read_bytes(),
                {"packaging/adapters/build.sh", "adapters"}) if (root / rel).is_file() else None for rel in builders}
        result["adapters"] = dict(sources=hashes, unresolved=missing, builder_views=builder_views)
    if request.get("lanes"):
        sources, reasons = ls.lane_sources()
        result["loader_routes"] = loader_routes(root, ls.host_surface())
        result["lanes"] = {lane: dict(sources={p: _file_hash(root, p) for p in sorted(paths)},
            bindings=reasons[lane].get("bindings", []),
            precision_limitations=reasons[lane].get("precision_limitations", []))
            for lane, paths in sources.items()}
    if request.get("base_revision"):
        base = _revision(root, request["base_revision"])
        paths = _git(root, "diff", "--name-only", "--no-renames", base, "HEAD", "--").splitlines()
        result["selections"] = {}
        # The registry/graph is derived from the NEW frozen tree, so a new file
        # or import edge cannot be hidden by the old snapshot's path listing.
        for vendor in request.get("vendors", ["cpu", "apple", "nvidia", "amd"]):
            # For CPU a missing platform choice must not inherit this analysis
            # host's platform and dismiss changes for the actual runtime host.
            backend = None if vendor == "cpu" else VENDORS[vendor]
            selected = ls.select(paths, ref=base, backend=backend)
            result["selections"][vendor] = dict(selected, base_revision=base,
                source_revision=result["revision"], changed_paths=paths,
                selector_sources={p: _file_hash(root, p) for p in
                    ("tools/lane_select.py", "tools/bincache.py", "tools/identity_break.py")})
    # Exact recipe/harness bytes are collected in the frozen tree as well.
    extra = request.get("extra_paths", [])
    result["extra_sources"] = {p: _file_hash(root, p) for p in extra}
    result["extra_closures"] = {p: {rel: _file_hash(root, rel) for rel in sorted(ls._python_closure([p]))}
                               for p in extra if p.endswith(".py")}
    if request.get("recipes"):
        import re
        index = root / RECIPE_INDEX
        candidates = {RECIPE_INDEX}
        if index.is_file():
            for link in re.findall(r"\]\(([^)]+)\)", index.read_text()):
                path = (index.parent / link.split("#", 1)[0]).resolve()
                if path.is_relative_to(root) and path.suffix == ".py":
                    candidates.add(str(path.relative_to(root)))
        result["recipe_discovery"] = {p: _file_hash(root, p) for p in sorted(candidates)}
    return result


def _tracked_state(root, revision):
    """Verify actual tracked bytes, including assume-unchanged/skip-worktree files."""
    if _revision(root, "HEAD") != revision:
        raise ValueError("dependency source HEAD changed from the requested revision")
    if subprocess.run(["git", "-C", str(root), "diff", "--cached", "--quiet", revision, "--"],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        raise ValueError("dependency source has staged changes")
    algorithm = _git(root, "rev-parse", "--show-object-format")
    entries = subprocess.check_output(["git", "-C", str(root), "ls-tree", "-rz", "--full-tree", revision])
    state = {}
    for entry in entries.split(b"\0"):
        if not entry:
            continue
        metadata, raw_path = entry.split(b"\t", 1)
        mode, kind, expected = metadata.decode().split()
        path = os.fsdecode(raw_path)
        if kind != "blob" or mode not in ("100644", "100755", "120000"):
            raise ValueError("unsupported tracked dependency source kind: " + path)
        source = root / path
        try:
            before = source.lstat()
            if mode == "120000":
                if not stat.S_ISLNK(before.st_mode):
                    raise ValueError("tracked dependency symlink changed: " + path)
                data = os.fsencode(os.readlink(source))
                actual = hashlib.new(algorithm, b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            else:
                if not stat.S_ISREG(before.st_mode) or bool(before.st_mode & stat.S_IXUSR) != (mode == "100755"):
                    raise ValueError("tracked dependency file mode changed: " + path)
                value = hashlib.new(algorithm, b"blob " + str(before.st_size).encode() + b"\0")
                with source.open("rb") as stream:
                    for block in iter(lambda: stream.read(1048576), b""):
                        value.update(block)
                actual = value.hexdigest()
            after = source.lstat()
        except OSError as error:
            raise ValueError("tracked dependency source unavailable: " + path) from error
        # ctime/inode catch an edit followed by restoration of the original
        # bytes or mtime. Access time is intentionally not part of the guard.
        def stamp(info):
            return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
                    info.st_mtime_ns, info.st_ctime_ns)
        if stamp(before) != stamp(after):
            raise ValueError("dependency source changed while being checked: " + path)
        if actual != expected:
            raise ValueError("tracked dependency source differs from requested revision: " + path)
        state[path] = (mode, actual, stamp(after))
    return state


def _clone_tracked_tree(root, tree, state, log):
    """Make a tracked-only CoW view; never hardlink mutable source files."""
    parents = {}
    for name, (mode, _, _) in state.items():
        destination = tree / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if mode == "120000":
            destination.symlink_to(os.readlink(root / name))
        else:
            parents.setdefault(destination.parent, []).append(root / name)
    for parent, paths in parents.items():
        # macOS cp -c uses supported clonefile(2), preserving separate inodes:
        # an analyzer write/chmod cannot alter the original source. It can fall
        # back to copying on a filesystem without cloning; failure still stops
        # analysis. Batching avoids one subprocess per source file.
        for first in range(0, len(paths), 128):
            subprocess.run(["/bin/cp", "-c", "-p", *map(str, paths[first:first + 128]), str(parent)],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    # --no-checkout creates no index. Read the exact requested tree into it
    # without touching the cloned files, so frozen analyzers see normal Git.
    subprocess.run(["git", "-C", str(tree), "read-tree", "HEAD"],
                   stdout=log, stderr=subprocess.STDOUT, check=True)


def _run_snapshot_worker(tree, temporary, request, log):
    inp, out = temporary / "request.json", temporary / "result.json"
    inp.write_text(json.dumps(request))
    env = dict(os.environ)
    for key in ("PYTHONPATH", "MOJOLEARN_RELEASE_BUILD_ONLY"):
        env.pop(key, None)
    # Existing pyc files are not frozen source, and new ones must not enter
    # directory-level source hashes (e.g. portable_math and tokenizer/tools).
    env.update(PYTHONPYCACHEPREFIX=str(temporary / "pycache"), PYTHONDONTWRITEBYTECODE="1")
    process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker",
        str(tree), str(inp), str(out)], cwd=tree, env=env, stdout=log, stderr=subprocess.STDOUT)
    if process.returncode:
        raise ValueError("frozen dependency analysis exited " + str(process.returncode))
    return json.loads(out.read_text())


def frozen_snapshot(root, revision, request, *, log_dir=None):
    """Analyze an exact detached source view, retaining the complete log."""
    root = Path(root).resolve()
    revision = _revision(root, revision)
    logs = Path(log_dir or Path.home() / "mojolearn-evidence/release-dependencies")
    logs.mkdir(parents=True, exist_ok=True)
    token = digest(dict(revision=revision, request=request))[:16]
    report = logs / (token + ".json")
    log_path = logs / (token + ".log")
    # Never reuse a cached result under a different analyzer implementation.
    analyzer = digest({name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                       for name in ("release_requirement_dependencies.py", "release_requirement_routes.py")})
    if report.is_file():
        old = json.loads(report.read_text())
        if old.get("analyzer") == analyzer and old.get("request") == request and old.get("revision") == revision:
            return old["snapshot"]
    with tempfile.TemporaryDirectory(prefix="mojolearn-dependencies-") as temporary:
        temporary = Path(temporary)
        tree = temporary / "tree"
        with log_path.open("w") as log:
            # A fresh physical checkout can exhaust disk on the build Mac.
            # Only reuse exact current bytes, and exclude ALL untracked files:
            # clean Git status alone would admit generated Mojo and pyc files
            # that do not exist in the requested revision's source tree.
            clone_current = sys.platform == "darwin" and _revision(root, "HEAD") == revision
            before = _tracked_state(root, revision) if clone_current else None
            source_view = dict(method="tracked-cow" if clone_current else "git-checkout",
                               source_checkout=str(root), tracked_files_checked=len(before or {}))
            log.write("source view: " + json.dumps(source_view, sort_keys=True) + "\n")
            log.flush()
            command = ["git", "-C", str(root), "worktree", "add", "--detach"]
            if clone_current:
                command.append("--no-checkout")
            added = False
            try:
                subprocess.run(command + [str(tree), revision], stdout=log, stderr=subprocess.STDOUT, check=True)
                added = True
                if clone_current:
                    _clone_tracked_tree(root, tree, before, log)
                view_before = _tracked_state(tree, revision)
                # The detached view is the analysis input. Later edits or new
                # commits in the integration checkout cannot invalidate it.
                # Links must stay inside that view, otherwise an outside edit
                # could change the bytes consumed despite an unchanged link.
                for name, (mode, _, _) in view_before.items():
                    if mode == "120000" and not (tree / name).resolve().is_relative_to(tree.resolve()):
                        raise ValueError("dependency symlink escapes frozen view: " + name)
                snapshot = _run_snapshot_worker(tree, temporary, request, log)
                if snapshot.get("revision") != revision:
                    raise ValueError("frozen dependency snapshot revision mismatch")
                if _tracked_state(tree, revision) != view_before:
                    raise ValueError("frozen dependency view changed during analysis")
            finally:
                # Failed `worktree add` can leave a registration too. Remove
                # only this unique temporary tree; never prune other sessions.
                if added or (tree / ".git").exists():
                    subprocess.run(["git", "-C", str(root), "worktree", "remove", "--force", str(tree)],
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
    report.write_text(json.dumps(dict(analyzer=analyzer, revision=revision, request=request,
                                     source_view=source_view, snapshot=snapshot), sort_keys=True) + "\n")
    return snapshot


def _vendor(spec):
    vendor = spec.get("vendor", spec["id"].rsplit("-", 1)[-1])
    if vendor not in VENDORS:
        raise ValueError(spec["id"] + ": vendor must be explicit")
    return vendor


def _unit(spec, suffix, resolved, sources, missing, coverage):
    result = copy.deepcopy(spec)
    result.update(id=spec["id"] + "/" + suffix, parent_requirement=spec["id"],
        inherit_source_inputs=False, source_inputs={}, configuration=[],
        resolved_inputs=resolved, provenance_sources=sorted(set(sources)),
        unresolved_inputs=sorted(set(missing)), coverage=list(dict.fromkeys(spec["coverage"] + coverage)))
    result.pop("drop_source_inputs", None)
    return result


def _binding_context(snapshot, vendor, lane, context):
    target = "osx-arm64" if vendor == "apple" else "linux-64"
    missing = []
    if vendor == "cpu":
        target = context.get("cpu_target")
        if target not in ("osx-arm64", "linux-64"):
            missing.append("dependency_context.cpu_target is required")
    declared = set(lane["bindings"])
    routes = snapshot.get("loader_routes", {})
    exclusions = routes.get("mode_exclusions", {}).get("identical", {})
    omitted = {name: exclusions[name] for name in declared if name in exclusions}
    names = declared - omitted.keys()
    optional_contracts = {}
    presence = context.get("optional_native_files", {}).get(target, {})
    selected_layout = sorted(b["key"] for b in snapshot["bindings"] if b["target"] == target)
    for name in names & routes.get("optional_files", {}).keys():
        if name not in presence or type(presence[name]) is not bool:
            continue
        witness = routes["optional_files"][name]
        optional_contracts[name] = dict(present=presence[name], predicate=witness,
                                       selected_layout_sha256=digest(selected_layout))
        selected = any(b["name"] == name and b["target"] == target for b in snapshot["bindings"])
        if presence[name] is False:
            if selected:
                missing.append("optional absence contradicts selected native layout: " + name)
            else:
                # This is an installation contract, not a mode-guard inference.
                # Runtime admission must also prove absence in its installed manifest.
                omitted[name] = optional_contracts[name]
    names -= omitted.keys()
    mapped = {}
    if vendor == "cpu":
        aliases = routes.get("cpu_aliases", {})
        for name in list(names):
            if name in aliases:
                mapped[name] = aliases[name]
                names.remove(name)
                names.update(aliases[name]["bindings"])
    matched = [b for b in snapshot["bindings"] if b["target"] == target
        and b["name"] in names and (b["vendor"] == VENDORS[vendor] or b["tier"] == "host")]
    # Every declared binding needs a target identity. A partially matched lane
    # cannot prove an unmapped helper's ABI/flags; host aliases need explicit
    # mapping evidence rather than inheriting whichever binding matched first.
    missing += ["no native binding context mapped for binding: " + name
                for name in sorted(names - {b["name"] for b in matched})]
    toolchain = snapshot.get("toolchains", {}).get(target)
    if not toolchain:
        missing.append("missing execution target toolchain")
    build_context = {}
    for binding in matched:
        ident = binding["identity"]
        if not ident:
            missing.append("missing binding identity: " + binding["key"])
            continue
        build_context[binding["key"]] = {k: v for k, v in ident.items()
            if k not in ("closure", "generators")}
        # A conservative source closure is adequate for compile, but an
        # unreadable target/compiler is never an execution provenance claim.
        missing += [m for m in binding["unresolved"] if "whole-tree" not in m]
    if not matched:
        missing.append("no native binding context mapped for lane")
    environment = dict(target=target, toolchain=toolchain, bindings=build_context)
    if omitted or mapped or optional_contracts:
        environment["loader_routes"] = dict(excluded=omitted, aliases=mapped, optional_files=optional_contracts, sources=routes.get("sources", {}))
    return environment, missing


def _selected_binding_flags(snapshot, configuration):
    """Pin each selected image's actual build settings in every consuming unit.

    Explicit semantic flags are complete, not merged with global flags. Worker
    counts are declared separately but remain fingerprinted: changing compiler
    parallelism has not been established to preserve binary bytes. Raw receipts
    retain the complete invocation, and admission still checks exact values.
    """
    from release_requirements import build_flags
    context = configuration.get("dependency_context", {})
    selected = context.get("selected_binding_build_flags", {})
    jobs = context.get("selected_binding_compile_jobs", {})
    known = {b["key"] for b in snapshot.get("bindings", [])}
    for name, mapping in (("selected_binding_build_flags", selected),
                          ("selected_binding_compile_jobs", jobs)):
        if not isinstance(mapping, dict) or any(not isinstance(key, str) for key in mapping):
            raise ValueError(name + " must be a binding-key mapping")
        if set(mapping) - known:
            raise ValueError(name + " names unknown bindings: " + ", ".join(sorted(set(mapping) - known)))
    routing = {"MOJOLEARN_NUMERIC_MODE", "MOJOLEARN_TARGET_COLUMN", "MOJOLEARN_GPU_ARCHS",
               "MOJOLEARN_LINUX_CPU", "MOJOLEARN_PACKAGE_BYTE_LM", "MOJOLEARN_COMPILE_JOBS"}
    for key, flags in selected.items():
        if (not isinstance(flags, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                for k, v in flags.items()) or build_flags(flags) != flags or routing.intersection(flags)):
            raise ValueError("selected build flags must contain only complete semantic build environment: " + key)
    for key, count in jobs.items():
        if not isinstance(count, str) or re.fullmatch(r"[1-9][0-9]*", count) is None:
            raise ValueError("selected compiler workers must be a positive decimal string: " + key)
    result = copy.deepcopy(snapshot)
    for binding in result.get("bindings", []):
        identity = binding.get("identity")
        if identity is None:
            continue
        if binding["key"] in jobs:
            identity.setdefault("flags", {})["compile_jobs"] = jobs[binding["key"]]
        if binding["key"] in selected:
            identity["selected_build_flags"] = copy.deepcopy(selected[binding["key"]])
    return result


def _selected_macos_cpu_toolchains(snapshot, configuration):
    """Use exact original CPU link witnesses only for named macOS host images.

    CPU build_host_family does not invoke Metal. Keep the full Apple toolchain
    requirement for every GPU image and retain every other unresolved input.
    The selected witness is validated again against original raw bytes by the
    retained producer adapter before it can qualify compilation.
    """
    from macos_cpu_compile_contract import validate_contract
    selected = configuration.get("dependency_context", {}).get("selected_macos_cpu_toolchains", {})
    if not isinstance(selected, dict) or any(not isinstance(k, str) for k in selected):
        raise ValueError("selected_macos_cpu_toolchains must be a binding-key mapping")
    known = {b["key"]: b for b in snapshot.get("bindings", [])}
    if set(selected) - known.keys():
        raise ValueError("selected Mac CPU toolchain names unknown bindings")
    result = copy.deepcopy(snapshot)
    for binding in result.get("bindings", []):
        if binding["key"] not in selected:
            continue
        identity = binding.get("identity")
        if (not identity or identity.get("target") != "osx-arm64" or identity.get("tier") != "host"
                or identity.get("vendor") != "host" or identity.get("flags", {}).get("target_column") != "cpu"):
            raise ValueError("CPU link witness cannot replace a GPU/foreign toolchain")
        witness = validate_contract(selected[binding["key"]])
        if witness["pixi_lock_sha256"] != identity.get("toolchain", {}).get("pixi_lock_sha256"):
            raise ValueError("CPU link witness package lock differs from frozen source")
        identity["host_toolchain"] = copy.deepcopy(witness)
        binding["unresolved"] = [message for message in binding.get("unresolved", [])
                                  if message != "no record of the Apple toolchain"]
    return result


def _snapshot_configuration(configuration):
    """Separate selected-image witnesses from frozen source analysis inputs.

    These three maps are validated/applied only by the two selection helpers
    below, after frozen_snapshot. Keeping them in the worker request repeats
    the same source scan for every separately compiled image. Retain all other
    configuration, including compiler, target, environment and manifest inputs;
    the complete original configuration still determines every final unit.
    """
    result = copy.deepcopy(configuration)
    context = result.get("dependency_context", {})
    if isinstance(context, dict):
        for name in ("selected_binding_build_flags", "selected_binding_compile_jobs",
                     "selected_macos_cpu_toolchains"):
            context.pop(name, None)
    return result


def expand_catalog(root, revision, catalog, configuration=None, *, log_dir=None):
    """Replace compile/identity/performance rows with independently reusable units.

    Unexpanded requirements remain untouched. Unit IDs are stable across commits;
    source revisions appear only in the returned expansion provenance.
    """
    configuration = configuration or {}
    context = configuration.get("dependency_context", {})
    modes = configuration.get("release_modes", ["identical"])
    if modes != ["identical"] and modes != "identical":
        raise ValueError("dependency expansion supports the authorized identical release only")
    specs = catalog["requirements"]
    kinds = {s.get("kind") for s in specs}
    workloads = context.get("workloads", {})
    extra = sorted({p for rows in workloads.values() for row in rows for p in row.get("source_inputs", [])})
    request = dict(compile="compile" in kinds, lanes=bool(kinds & {"identity", "performance"}),
                   recipes="performance" in kinds,
                   configuration=_snapshot_configuration(configuration), extra_paths=extra)
    source_snapshot = frozen_snapshot(root, revision, request, log_dir=log_dir)
    snapshot = source_snapshot
    if kinds & {"compile", "identity", "performance"}:
        snapshot = _selected_binding_flags(snapshot, configuration)
        snapshot = _selected_macos_cpu_toolchains(snapshot, configuration)
    expanded = []
    for spec in specs:
        kind = spec.get("kind")
        if kind not in ("compile", "identity", "performance"):
            expanded.append(copy.deepcopy(spec))
            continue
        vendor = _vendor(spec)
        common = dict(schema=SCHEMA, kind=kind, vendor=vendor,
            release_modes=["identical"], build_flags=configuration.get("build_flags"))
        common_missing = [] if "build_flags" in configuration else ["build_flags"]
        if kind == "compile":
            # Expanded units must retain every declared switch configuration;
            # clearing the parent's broad configuration must not erase a matrix.
            common["compile_variants"] = copy.deepcopy(configuration.get("compile_variants", []))
            target = "osx-arm64" if vendor == "apple" else "linux-64"
            if vendor == "cpu":
                target = context.get("cpu_target")
                if target not in ("osx-arm64", "linux-64"):
                    expanded.append(_unit(spec, "unresolved", dict(unit=common), [],
                        common_missing + ["dependency_context.cpu_target is required"], spec["coverage"]))
                    continue
            if target == "linux-64":
                compiler_hash = context.get("compiler_sha256", {}).get(target)
                common["compiler_sha256"] = compiler_hash
                if not isinstance(compiler_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", compiler_hash):
                    common_missing.append("dependency_context.compiler_sha256.linux-64 is required")
            rows = [b for b in snapshot["bindings"] if b["target"] == target
                and (b["vendor"] == VENDORS[vendor] or b["tier"] == "host")]
            for binding in rows:
                selected_common = dict(common)
                if "selected_build_flags" in (binding["identity"] or {}):
                    selected_common["build_flags"] = binding["identity"]["selected_build_flags"]
                expanded.append(_unit(spec, binding["key"],
                    dict(unit=selected_common, binding=binding["identity"]), binding["sources"],
                    common_missing + binding["unresolved"], [binding["key"]]))
            if target == "linux-64":
                rt = snapshot["runtime"]
                expanded.append(_unit(spec, "runtime", dict(unit=common, runtime=rt["identity"]),
                    [], common_missing + ([rt["unresolved"]] if rt["unresolved"] else []), ["native-runtime-libraries"]))
            arches = sorted({b["arch"] for b in rows if b["tier"] != "host" and b["arch"] != "sm_80"})
            for arch in arches:
                adapter = snapshot["adapters"]
                exemplar = next(b for b in rows if b["arch"] == arch and b["tier"] != "host")
                # Selecting a binding image does not configure the separately
                # built adapter. Keep its own original builder contract.
                original = next(b for b in source_snapshot["bindings"] if b["key"] == exemplar["key"])
                environment = {k: v for k, v in (original["identity"] or {}).items()
                    if k in ("target", "vendor", "arch", "toolchain", "flags", "image", "host_toolchain")}
                environment["builders"] = adapter.get("builder_views", {}).get(target)
                adapter_missing = list(adapter["unresolved"])
                if not environment["builders"] or any(v is None for v in environment["builders"].values()):
                    adapter_missing.append("adapter invocation builder provenance missing")
                expanded.append(_unit(spec, "adapters/" + (arch or "native"),
                    dict(unit=common, adapter_sources=adapter["sources"], environment=environment),
                    list(adapter["sources"]) + list(environment["builders"] or {}),
                    common_missing + adapter_missing + exemplar["unresolved"],
                    ["native-adapter-library", "torch-registration", "max-operation-package"]))
            if not rows:
                expanded.append(_unit(spec, "unresolved", dict(unit=common), [],
                                      ["no frozen binding registry entries"], spec["coverage"]))
            continue
        lanes = snapshot.get("lanes", {})
        recipes = ([dict(id=lane, lanes=[lane]) for lane in sorted(lanes)] if kind == "identity"
                   else workloads.get(vendor, []))
        if not recipes:
            expanded.append(_unit(spec, "unresolved", dict(unit=common), [],
                ["no frozen lane registry" if kind == "identity" else "full-workload recipes have not been mapped"], spec["coverage"]))
            if kind == "performance":
                discovery = snapshot.get("recipe_discovery", {RECIPE_INDEX: None})
                expanded[-1]["provenance_sources"] = sorted(discovery)
                expanded[-1]["recipe_discovery"] = discovery
        for recipe in recipes:
            lane_ids = recipe.get("lanes", [])
            missing, sources, environments, limitations = list(common_missing), {}, {}, {}
            if not lane_ids:
                missing.append("workload has no identity lane mapping")
            for lane in lane_ids:
                if lane not in lanes:
                    missing.append("unknown identity lane: " + lane)
                    continue
                sources.update(lanes[lane]["sources"])
                environment, errors = _binding_context(snapshot, vendor, lanes[lane], context)
                environments[lane] = environment
                missing += errors
                if lanes[lane].get("precision_limitations"):
                    # The existing analyzer retains all reachable dispatcher
                    # branches. That conservative closure can identify a fresh
                    # run; an edit touching an unresolved branch cannot prove
                    # an affected/retained decision and is blocked below.
                    limitations[lane] = lanes[lane]["precision_limitations"]
            missing += ["missing dependency source: " + p for p, h in sources.items() if h is None]
            resolved = dict(unit=dict(common, lanes=sorted(lane_ids)), native_sources=sources,
                            execution_environment=environments)
            coverage = ["lane:" + lane for lane in sorted(lane_ids)]
            if kind == "performance":
                required = ("id", "lanes", "source_inputs", "dataset", "settings", "boundary", "phases", "environment")
                missing += ["workload recipe missing " + k for k in required if k not in recipe or recipe[k] is None]
                if not recipe.get("source_inputs"):
                    missing.append("workload recipe has no retained source")
                recipe_sources = {p: snapshot["extra_sources"].get(p) for p in recipe.get("source_inputs", [])}
                for path in recipe.get("source_inputs", []):
                    recipe_sources.update(snapshot.get("extra_closures", {}).get(path, {}))
                missing += ["missing recipe source: " + p for p, h in recipe_sources.items() if h is None]
                resolved.update(recipe=recipe, recipe_sources=recipe_sources)
                coverage += ["phase:" + p for p in recipe.get("phases", [])]
            expanded.append(_unit(spec, recipe.get("id", "unmapped"), resolved,
                list(sources) + recipe.get("source_inputs", []), missing, coverage))
            expanded[-1]["dependency_limitations"] = limitations
            if kind == "identity" and "identity_scope" in snapshot:
                scope = identity_unit_scope(snapshot["identity_scope"], lane_ids[0], vendor)
                expanded[-1]["dependency_provenance"] = dict(identity_applicability=scope)
                if not scope["applicable"]:
                    expanded[-1]["required"] = False
    result = copy.deepcopy(catalog)
    result["requirements"] = expanded
    result["dependency_provenance"] = dict(schema=SCHEMA, source_revision=snapshot["revision"])
    if "performance" in kinds:
        result["dependency_provenance"]["recipe_discovery"] = snapshot.get("recipe_discovery", {})
    ids = [s["id"] for s in expanded]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate dependency unit IDs")
    return result


def identity_unit_scope(snapshot, lane, vendor):
    """Scope annotation is provenance, never a rewritten compilation/identity input."""
    row = snapshot["lanes"][lane]
    return dict(schema=snapshot["schema"], source_revision=snapshot["source_revision"], lane=lane,
                vendor=vendor, kind=row["kind"], claim_devices=row["claim_devices"],
                oracle=row["oracle"], anchor=row["anchor"], required_columns=row["required_columns"],
                applicable=vendor in row["required_columns"], excluded=row["excluded"],
                separate_requirements=row["separate_requirements"], snapshot_digest=snapshot["digest"])


@lru_cache(maxsize=32)
def identity_scope_snapshot(root, revision, log_dir=None):
    return frozen_snapshot(root, revision, dict(identity_scope=True), log_dir=log_dir)["identity_scope"]


def scope_identity_manifest(root, manifest, *, log_dir=None):
    """Reconcile old catalogs without altering any original run or input fingerprint.

    Multi-device lanes remain linked to the existing separate multi-GPU gates;
    their single-device identity units cannot qualify those separate claims.
    """
    result = copy.deepcopy(manifest)
    scope = identity_scope_snapshot(str(Path(root).resolve()), manifest["source_revision"], str(log_dir) if log_dir else None)
    for wanted in result["requirements"].values():
        if wanted.get("kind") != "identity":
            continue
        unit = wanted["inputs"]["unit"]
        if len(unit.get("lanes", [])) != 1 or unit["lanes"][0] not in scope["lanes"]:
            raise ValueError("identity scope has no exact lane declaration")
        before = wanted["fingerprint"]
        annotation = identity_unit_scope(scope, unit["lanes"][0], unit["vendor"])
        wanted["dependency_provenance"] = dict(wanted.get("dependency_provenance", {}), identity_applicability=annotation)
        if not annotation["applicable"]:
            wanted["required"] = False
        if wanted["fingerprint"] != before or digest(wanted["inputs"]) != before:
            raise ValueError("applicability reconciliation changed identity inputs")
    # These links describe a separate two-device proposition. Preserve the
    # catalog's existing gate policy; source scoping does not reopen deferred
    # multi-GPU/sabotage campaigns or manufacture a publication prerequisite.
    return result


def evidence_restoration(evidence):
    """Missing retained bytes require restoration, not a new execution."""
    restore = []
    for item in evidence:
        from release_evidence import sha256
        try:
            available = sha256(item["path"]) == item.get("sha256")
        except (OSError, ValueError):
            available = False
        if not available:
            restore.append(dict(item))
    return dict(status="restore-required" if restore else "available", restore=restore)


# Reviewed release-label boundary: Sequential puts this value only in the
# unconsumed .verify.json install metadata. The export lane hashes model.ckpt
# and predictions, not that sidecar. Keep exact reviewed consumers/fixture bytes:
# new use of version in computation or a consumed checkpoint requires review.
VERSION_PATH = "python/mojolearn/_version.py"
VERSION_LABEL_ANCHORS = {'python/mojolearn/_x_sequence_model.py': 'bf42be02698ce0f67c5f0dfd94630e0c607841950c24ff3d1a313192ce65f5b6', 'bindings/sequential_checkpoint.mojo': 'ee37e236bbc6846032eeed31a22900723dcd18b561198c8e403a44306a6dcee7', 'tools/identity_lanes/coverage.py': 'db8d3ae94a2a635cb169d97dfd1267267f4a830afe5cae5e53047ff67a4e191d'}
VERSION_LABEL_LANES = frozenset("""public-theta-variants sequence-adamax sequence-croston
sequence-ets sequence-export-roundtrip sequence-garch sequence-gru sequence-lion
sequence-lr-schedulers sequence-lstm sequence-mlp sequence-nadam sequence-prophet
sequence-rnn sequence-stl sequence-tape-sequential sequence-theta sequence-var""".split())


def _version_literal_view(data):
    import ast
    text = data.decode("utf-8")
    body = ast.parse(text).body
    if (len(body) != 1 or not isinstance(body[0], ast.Assign)
            or len(body[0].targets) != 1 or not isinstance(body[0].targets[0], ast.Name)
            or body[0].targets[0].id != "__version__"
            or not isinstance(body[0].value, ast.Constant)
            or not isinstance(body[0].value.value, str)
            or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", body[0].value.value)):
        return None
    value = body[0].value
    if value.lineno != value.end_lineno:
        return None
    lines = text.splitlines(keepends=True)
    line = lines[value.lineno - 1].encode()
    lines[value.lineno - 1] = (line[:value.col_offset] + b'"<release-label>"'
                              + line[value.end_col_offset:]).decode()
    return "".join(lines)


@lru_cache(maxsize=512)
def _git_bytes(root, revision, path):
    return subprocess.check_output(["git", "-C", str(root), "show", revision + ":" + path])


def version_label_proof(root, old, new):
    """Retain original fingerprints; prove only an unconsumed release label changed."""
    if old.get("kind") != "identity" or new.get("kind") != "identity":
        return None
    if old.get("missing_inputs") or new.get("missing_inputs"):
        return None
    if any(w.get("fingerprint") != digest(w.get("inputs")) for w in (old, new)):
        return None
    a, b = copy.deepcopy(old["inputs"]), copy.deepcopy(new["inputs"])
    x, y = a.pop("native_sources", {}), b.pop("native_sources", {})
    lanes = b.get("unit", {}).get("lanes", [])
    changed = {p for p in set(x) | set(y) if x.get(p) != y.get(p)}
    if a != b or changed != {VERSION_PATH} or not lanes or not set(lanes) <= VERSION_LABEL_LANES:
        return None
    old_rev, new_rev = old["source_revision"], new["source_revision"]
    try:
        before, after = (_git_bytes(root, rev, VERSION_PATH) for rev in (old_rev, new_rev))
        views = [_version_literal_view(data) for data in (before, after)]
        if views[0] is None or views[0] != views[1]:
            return None
        if [hashlib.sha256(data).hexdigest() for data in (before, after)] != [x[VERSION_PATH], y[VERSION_PATH]]:
            return None
        for path, sha in VERSION_LABEL_ANCHORS.items():
            if any(hashlib.sha256(_git_bytes(root, rev, path)).hexdigest() != sha for rev in (old_rev, new_rev)):
                return None
    except (OSError, UnicodeError, SyntaxError, subprocess.SubprocessError):
        return None
    proof = dict(schema=PROOF_SCHEMA, kind="unconsumed-release-version-literal",
                 unit_id=b["contract"]["id"], old_revision=old_rev, new_revision=new_rev,
                 old_fingerprint=old["fingerprint"], new_fingerprint=new["fingerprint"],
                 stable_inputs_digest=digest(a), lanes=sorted(lanes),
                 version_source_sha256=dict(old=x[VERSION_PATH], new=y[VERSION_PATH]),
                 reviewed_consumers=VERSION_LABEL_ANCHORS)
    proof["digest"] = digest(proof)
    return proof


def validate_version_label_proof(root, proof, old):
    """Reconstruct the sole permitted delta and independently re-read both commits."""
    if not isinstance(proof, dict) or proof.get("kind") != "unconsumed-release-version-literal":
        return False
    new = copy.deepcopy(old)
    new["source_revision"] = proof.get("new_revision")
    new["inputs"] = dict(new["inputs"], native_sources=dict(new["inputs"]["native_sources"]))
    new["inputs"]["native_sources"][VERSION_PATH] = proof.get("version_source_sha256", {}).get("new")
    new["fingerprint"] = digest(new["inputs"])
    return version_label_proof(root, old, new) == proof


def equivalence_proofs(root, old_manifest, new_manifest, *, log_dir=None, retained_evidence=None):
    """Source-only plan: exact reuse proofs, affected units, or blocked provenance.

    A retained decision never records a new PASS. The ledger must retain the old
    tested revision/receipt and attach this proof when accepting it for the new
    source revision. Missing evidence bytes are a separate restore operation.
    """
    old_rev, new_rev = old_manifest["source_revision"], new_manifest["source_revision"]
    vendors = sorted({w.get("inputs", {}).get("unit", {}).get("vendor")
        for w in new_manifest["requirements"].values() if w.get("kind") in ("identity", "performance")})
    vendors = [v for v in vendors if v in VENDORS]
    snapshot = None
    plan = dict(schema=PROOF_SCHEMA, old_revision=_revision(root, old_rev),
                new_revision=_revision(root, new_rev), units={})
    for rid, new in new_manifest["requirements"].items():
        old = old_manifest["requirements"].get(rid)
        row = dict(unit_id=rid, decision="affected", reason="new requirement")
        plan["units"][rid] = row
        if not old:
            continue
        row.update(old_fingerprint=old["fingerprint"], new_fingerprint=new["fingerprint"])
        if old.get("missing_inputs") or new.get("missing_inputs"):
            row.update(decision="blocked", reason="unresolved requirement inputs")
            continue
        if old["fingerprint"] == new["fingerprint"]:
            row.update(decision="retained", reason="identical dependency fingerprint")
            continue
        a, b = copy.deepcopy(old["inputs"]), copy.deepcopy(new["inputs"])
        a.pop("native_sources", None)
        b.pop("native_sources", None)
        unit = b.get("unit", {})
        if new.get("kind") not in ("identity", "performance") or a != b:
            row["reason"] = "execution, coverage, recipe, flags or build inputs changed"
            continue
        version_proof = version_label_proof(root, old, new)
        if version_proof:
            row.update(decision="retained", reason="only unconsumed release version literal changed", reuse_proof=version_proof)
            continue
        if snapshot is None:
            snapshot = frozen_snapshot(root, new_rev, dict(base_revision=old_rev, vendors=vendors), log_dir=log_dir)
        selection = snapshot["selections"][unit["vendor"]]
        lanes = set(unit.get("lanes", []))
        limited = [row for row in selection.get("precision_limitations", []) if row.get("lane") in lanes]
        if selection.get("unattributed") or limited:
            row.update(decision="blocked", reason="selector provenance unresolved", selection=selection)
            continue
        if not lanes or lanes & set(selection["lanes"]):
            row["reason"] = "source changes reach this unit"
            continue
        proof = dict(schema=PROOF_SCHEMA, unit_id=rid, old_revision=plan["old_revision"],
            new_revision=plan["new_revision"], old_fingerprint=old["fingerprint"],
            new_fingerprint=new["fingerprint"], stable_inputs_digest=digest(a),
            lanes=sorted(lanes), selection=selection)
        proof["digest"] = digest(proof)
        row.update(decision="retained", reason="named-edit selector proves lane inputs unaffected", reuse_proof=proof)
    for rid, row in plan["units"].items():
        if row["decision"] == "retained":
            row["evidence"] = (evidence_restoration(retained_evidence[rid])
                if retained_evidence is not None and rid in retained_evidence
                else dict(status="not-supplied", restore=[]))
    return plan


def validate_equivalence_proof(root, proof, old, new, *, log_dir=None):
    """Recompute frozen selector proof; a caller-authored zero selection is invalid."""
    errors = []
    if proof.get("schema") != PROOF_SCHEMA:
        return ["unknown equivalence proof schema"]
    saved = dict(proof)
    claimed = saved.pop("digest", None)
    if claimed != digest(saved):
        errors.append("equivalence proof digest mismatch")
    rid = proof.get("unit_id")
    if (not isinstance(rid, str)
            or old.get("inputs", {}).get("contract", {}).get("id") != rid
            or new.get("inputs", {}).get("contract", {}).get("id") != rid
            or old.get("source_revision") != proof.get("old_revision")
            or new.get("source_revision") != proof.get("new_revision")):
        return errors + ["equivalence proof source/unit mismatch"]
    if any(wanted.get("fingerprint") != digest(wanted.get("inputs")) for wanted in (old, new)):
        return errors + ["requirement fingerprint does not match its inputs"]
    plan = equivalence_proofs(root,
        dict(source_revision=old["source_revision"], requirements={rid: old}),
        dict(source_revision=new["source_revision"], requirements={rid: new}), log_dir=log_dir)
    actual = plan["units"][rid].get("reuse_proof")
    if actual != proof:
        errors.append("equivalence proof does not reproduce from the frozen source and requirement inputs")
    return errors


if __name__ == "__main__":
    if len(sys.argv) != 5 or sys.argv[1] != "--worker":
        raise SystemExit("library API: expand_catalog / equivalence_proofs / validate_equivalence_proof")
    Path(sys.argv[4]).write_text(json.dumps(_worker_snapshot(sys.argv[2], json.loads(Path(sys.argv[3]).read_text())), sort_keys=True))
