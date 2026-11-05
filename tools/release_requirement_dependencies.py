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
from contextvars import ContextVar
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
import threading
from release_requirement_routes import loader_routes

SCHEMA = "mojolearn.release-dependencies.v1"
PROOF_SCHEMA = "mojolearn.release-dependency-equivalence.v1"
VENDORS = {"apple": "metal", "nvidia": "cuda", "amd": "hip", "cpu": "host"}
RECIPE_INDEX = "experiments/performance_ideas/README.md"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


_REVISIONS = {}


def _revision(root, revision):
    """rev-parse once per (root, revision) in this process: a commit never changes; 0.8.64 --status spawned
    6,010 of these for 2,002 equivalence proofs (127 s)."""
    key = (str(root), revision)
    if key not in _REVISIONS:
        _REVISIONS[key] = _git(root, "rev-parse", "--verify", revision + "^{commit}")
    return _REVISIONS[key]


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
                # No multi-GPU release row since multi-GPU verification was retired
                # (owner, 2026-10-31); no registered lane claims two devices.
                separate_requirements=[])
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
    if request.get("reference_closures"):
        result["reference_closures"] = _reference_closures(root, ls, request["reference_closures"])
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


REFERENCE_SCHEMA = "mojolearn.release-identity-reference-closures.v1"


def reference_key(base, backend):
    """The key of one (base revision, column backend) selection in a snapshot's ``reference_closures``."""
    return base + "|" + (backend or "any")


def _reference_closures(root, ls, spec):
    """THE RECORDER'S CLOSURE COMPARISON, AT THE FREEZE (lane ledger-trusts-reference, 2026-11-04).

    For each requested (base revision, backend, lanes): the lanes among ``lanes`` whose source closure
    changed between the base and this frozen tree, decided by the frozen ``tools/lane_select.py`` exactly
    as ``tools/identity_columns.py changed-lanes`` / ``reuse-check`` decide REUSED vs RECORD
    (``changed_paths(base)``, ``select(..., backend, stop_when=lanes)``). A lane absent from
    ``selected`` with no ``unattributed`` path has an unchanged closure. Precision limitations are not
    a separate verdict: they only ever describe a lane the selection already holds.

    Also returns the frozen harness facts the admitted table is judged against (its sha256, fixtures,
    lane revisions and registry), read from the same frozen tree."""
    h = ls.identity_break()
    harness = dict(sha256=_file_hash(root, "tools/identity_break.py"), fixtures=sorted(h.FIXTURES),
                   lane_revisions=dict(getattr(h, "LANE_REVISIONS", {}) or {}), lanes=sorted(h.LANES))
    selections = {}
    sources = None
    for base, backend, lanes in spec.get("bases", []):
        if sources is None:
            sources, _ = ls.lane_sources()
        base = _revision(root, base)
        wanted = set(lanes)
        paths = list(ls.changed_paths(base))
        sel = ls.select(paths, ref=base, sources=sources, backend=backend, stop_when=wanted)
        selected = sorted(wanted & set(sel["lanes"]))
        by_path = sel.get("by_path", {})
        selections[reference_key(base, backend)] = dict(
            base_revision=base, backend=backend, changed_paths=len(paths), selected=selected,
            lane_paths={lane: sorted(p for p, hit in by_path.items() if lane in hit) for lane in selected},
            unattributed=sorted(sel.get("unattributed", [])), stopped_early=bool(sel.get("stopped_early")))
    return dict(schema=REFERENCE_SCHEMA, harness=harness, selections=selections,
                selector_sources={p: _file_hash(root, p) for p in
                                  ("tools/lane_select.py", "tools/bincache.py", "tools/identity_break.py",
                                   "tools/identity_columns.py")})


#: TOOLING AFTER A FREEZE (lane ledger-trusts-reference, 2026-11-04). Tracked paths whose bytes cannot
#: change a computed bit, a built binary or a retained record: release tooling, prose and CI. A source
#: checkout at the frozen commit may carry a later tooling fix in these paths (0.8.65: the assemble patch
#: in tools/release.py); the frozen view then takes their bytes from the frozen commit's Git objects, so
#: the analysis is still exactly the frozen tree, and the source checkout is not refused for them.
#: Everything else (bindings/, core/ and every kernel tree, python/mojolearn, packaging/, bench/ evidence,
#: pixi and build files) is still byte-checked against the frozen commit. The identity harness inside
#: tools/ decides fixtures and so evidence: it stays checked.
TOOLING_PREFIXES = ("tools/", "docs/", "docs-site/", ".github/", "experiments/")
TOOLING_FILES = frozenset(("CHANGELOG.md", "AGENTS.md", "CLAUDE.md", "README.md"))
EVIDENCE_TOOLING = ("tools/identity_break.py", "tools/identity_lanes/")


def tooling_only(path):
    """True for a tracked path whose bytes cannot determine bits or evidence (TOOLING_PREFIXES)."""
    if path.startswith(EVIDENCE_TOOLING):
        return False
    return path in TOOLING_FILES or path.startswith(TOOLING_PREFIXES)


def _tracked_state(root, revision, *, tooling_from_revision=False):
    """Verify actual tracked bytes, including assume-unchanged/skip-worktree files.

    ``{path: (mode, blob, stamp)}``. With ``tooling_from_revision`` (the source checkout a frozen view is
    cloned from), a tooling-only path (``tooling_only``) whose bytes, mode or presence differ from the
    frozen commit is not refused: it is recorded with stamp None and ``_clone_tracked_tree`` writes the
    frozen commit's own blob into the view. HEAD must still be the frozen commit, and every other tracked
    path (and every staged change outside tooling) must be exactly the frozen commit's."""
    if _revision(root, "HEAD") != revision:
        raise ValueError("dependency source HEAD changed from the requested revision")
    staged = subprocess.run(["git", "-C", str(root), "diff", "--cached", "--name-only", "-z", revision, "--"],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if staged.returncode:
        raise ValueError("dependency source staged state is unreadable")
    if any(not (tooling_from_revision and tooling_only(os.fsdecode(name)))
           for name in staged.stdout.split(b"\0") if name):
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
        try:
            state[path] = (mode, expected, _tracked_stamp(root, path, mode, expected, algorithm))
        except ValueError:
            if not (tooling_from_revision and tooling_only(path)):
                raise
            state[path] = (mode, expected, None)
    return state


def _tracked_stamp(root, path, mode, expected, algorithm):
    """The stat stamp of one tracked file whose bytes and mode are the frozen commit's, else ValueError."""
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
    return stamp(after)


def tooling_overlay(state):
    """Tooling paths a frozen view takes from the frozen commit's Git objects (stamp None)."""
    return sorted(path for path, (_, _, stamp) in state.items() if stamp is None)


def _tooling_commit():
    """The commit of the checkout running this analyzer (release tooling), recorded beside the source."""
    tools = Path(__file__).resolve().parent
    try:
        commit = _git(tools, "rev-parse", "HEAD")
        dirty = bool(_git(tools, "status", "--porcelain", "--untracked-files=no", "--", "."))
    except (OSError, subprocess.CalledProcessError):
        return dict(commit=None, tools_dirty=None)
    return dict(commit=commit, tools_dirty=dirty)


def _clone_tracked_tree(root, tree, state, log):
    """Make a tracked-only CoW view; never hardlink mutable source files."""
    parents = {}
    for name, (mode, blob, stamp) in state.items():
        destination = tree / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if stamp is None:
            # A tooling path that differs in the source checkout: the frozen commit's own bytes.
            data = subprocess.check_output(["git", "-C", str(root), "cat-file", "blob", blob])
            if mode == "120000":
                destination.symlink_to(os.fsdecode(data))
            else:
                destination.write_bytes(data)
                destination.chmod(0o755 if mode == "100755" else 0o644)
        elif mode == "120000":
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


# ---------------------------------------------------------------- parallel and remote analysis workers
#
# 0.8.65: the publish admission bundle ran its frozen dependency workers one at a time, 2h42m of the
# macOS package's 2h46m (10 workers, 6-37 min each; ~/mojolearn-evidence/ledger-storage-notes/MEASUREMENTS.md).
# ``prefetch`` runs the caller's computation in discovery mode: every frozen_snapshot it would start is
# recorded instead (SnapshotPending), then all of them run at once, on all cores or on the box named by
# MOJOLEARN_ADMISSION_WORKER, into the same content-keyed cache (token = digest(revision, request), checked
# against the analyzer digest). A discovery round repeats until nothing new is requested; the caller then
# runs its real computation, which finds every snapshot cached and is exactly the sequential result.

_DISCOVERY = ContextVar("release_dependency_discovery", default=None)
_FAILED = {}
_WORKTREES = threading.Lock()

# TEMPORARY DEPENDENCY WORKTREES ARE REGISTERED AND REMOVED ON EXIT (2026-11-05). 0.8.65 left seven
# <tmp>/mojolearn-dependencies-*/tree worktrees registered in the release checkout: the `finally` in
# frozen_snapshot never runs when the process is killed (a step timeout, a closed terminal). Every tree is recorded in
# WORKTREE_REGISTRY (one JSON per tree: pid, start, root, tree, created) the moment `worktree add` succeeds,
# and its temporary directory carries WORKTREE_OWNER (pid, start, created) from before the add; an atexit hook
# and a SIGTERM/SIGHUP/SIGINT handler remove this process's live trees; `--gc-worktrees` removes registered or
# listed dependency trees whose owner is gone. It never touches a worktree outside a `mojolearn-dependencies-*`
# temporary directory (never ~/mojolearn-wt/*).
#
# OWNER-AWARE GC, RUN BEFORE EVERY ANALYSIS (2026-11-05, 94 GB leaked on the build Mac). The signal handlers
# were installed only from the main thread, and prefetch runs every frozen_snapshot on pool threads, so a
# SIGTERM during a parallel analysis left every tree; and the 24 h gc age kept a whole day of leaks. Now the
# handlers are installed from prefetch (main thread) too; gc removes a tree whose recorded owner process is
# dead (or whose pid now belongs to a later process, by its start time) at any age, keeps a tree whose owner
# is alive, and applies the age rule (WORKTREE_GC_AGE_S, 2 h) only to trees with no recorded owner; and every
# process runs that gc once per checkout before its first analysis.
WORKTREE_PREFIX = "mojolearn-dependencies-"
WORKTREE_OWNER = "owner.json"
WORKTREE_GC_AGE_S = 2 * 3600
_LIVE_WORKTREES = {}
_OWN_START = []


def worktree_registry():
    return Path(os.environ.get("MOJOLEARN_DEPENDENCY_WORKTREES")
                or Path.home() / "mojolearn-evidence/release-dependencies/worktrees")


def _registry_file(tree):
    return worktree_registry() / (hashlib.sha256(str(tree).encode()).hexdigest()[:16] + ".json")


def _process_start(pid):
    """The start time of process PID as `ps` prints it, or None when unknown. A pid reused by a later
    process has a different start, so a recorded (pid, start) pair names one process, not one pid."""
    try:
        out = subprocess.run(["ps", "-o", "lstart=", "-p", str(int(pid))], capture_output=True, text=True,
                             timeout=10)
    except (OSError, ValueError, TypeError, subprocess.SubprocessError):
        return None
    return " ".join((out.stdout or "").split()) or None


def _owner_row():
    import time
    if not _OWN_START:
        _OWN_START.append(_process_start(os.getpid()))
    return dict(pid=os.getpid(), start=_OWN_START[0], created=time.time())


def _write_owner(temporary):
    """Record this process as the owner of a dependency temporary directory, before any worktree is added."""
    try:
        (Path(temporary) / WORKTREE_OWNER).write_text(json.dumps(_owner_row(), sort_keys=True) + "\n")
    except OSError:
        pass  # without it the tree is an unknown-owner tree: the age rule still removes it


def _register_worktree(root, tree):
    _install_worktree_cleanup()
    _LIVE_WORKTREES[str(tree)] = str(root)
    if Path(tree).parent.name.startswith(WORKTREE_PREFIX) and not (Path(tree).parent / WORKTREE_OWNER).exists():
        _write_owner(Path(tree).parent)
    try:
        worktree_registry().mkdir(parents=True, exist_ok=True)
        _registry_file(tree).write_text(json.dumps(dict(_owner_row(), root=str(root), tree=str(tree)),
                                                   sort_keys=True) + "\n")
    except OSError:
        pass  # the in-process hook still removes it; --gc-worktrees finds it in `git worktree list`


def _unregister_worktree(tree):
    _LIVE_WORKTREES.pop(str(tree), None)
    try:
        _registry_file(tree).unlink()
    except OSError:
        pass


def _remove_live_worktrees():
    """This process's registered trees: `git worktree remove --force`, then the temporary directory."""
    import shutil
    for tree, root in list(_LIVE_WORKTREES.items()):
        subprocess.run(["git", "-C", root, "worktree", "remove", "--force", tree],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if Path(tree).parent.name.startswith(WORKTREE_PREFIX):
            shutil.rmtree(Path(tree).parent, ignore_errors=True)
        _unregister_worktree(tree)


_ATEXIT_INSTALLED = []
_SIGNALS_INSTALLED = []


def _install_worktree_cleanup():
    """atexit once; the signal handlers once, from the main thread (Python only installs them there, so a
    first call from a pool thread installs atexit now and the handlers at the next main-thread call)."""
    import atexit
    import signal
    if not _ATEXIT_INSTALLED:
        _ATEXIT_INSTALLED.append(True)
        atexit.register(_remove_live_worktrees)
    if _SIGNALS_INSTALLED or threading.current_thread() is not threading.main_thread():
        return
    _SIGNALS_INSTALLED.append(True)
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        try:
            previous = signal.getsignal(sig)
        except (ValueError, OSError):
            continue
        if sig == signal.SIGINT:
            if previous is not signal.default_int_handler:
                continue  # someone else owns Ctrl-C; atexit still runs after the KeyboardInterrupt

            def handler(_signum, _frame):
                # Ctrl-C keeps Python's behavior (KeyboardInterrupt, `finally` blocks, atexit); the trees go first.
                _remove_live_worktrees()
                raise KeyboardInterrupt
        else:
            if previous not in (signal.SIG_DFL, None):
                continue  # someone else owns it; atexit still runs on a normal exit

            def handler(signum, _frame, _sig=sig):
                _remove_live_worktrees()
                signal.signal(_sig, signal.SIG_DFL)
                os.kill(os.getpid(), signum)
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            continue


def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
    except PermissionError:
        return True  # it exists, owned by another user
    except (OSError, ValueError, TypeError):
        return False
    return True


def _owner_state(row):
    """'self', 'alive', 'dead' or 'unknown' for an owner record {pid, start}."""
    pid = row.get("pid")
    if not pid:
        return "unknown"
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return "unknown"
    if pid == os.getpid():
        return "self"  # this process: its own trees go through _remove_live_worktrees, never through gc
    if not _pid_alive(pid):
        return "dead"
    recorded = row.get("start")
    if recorded:
        current = _process_start(pid)
        if current and current != recorded:
            return "dead"  # the pid now belongs to a later process
    return "alive"


def _listed_worktrees(root, runner=subprocess.run):
    out = runner(["git", "-C", str(root), "worktree", "list", "--porcelain"], capture_output=True, text=True)
    return [line[len("worktree "):] for line in (out.stdout or "").splitlines() if line.startswith("worktree ")]


def _is_dependency_tree(path, protect):
    p = Path(path)
    if any(str(p).startswith(str(x).rstrip("/") + "/") for x in protect):
        return False
    return p.name == "tree" and p.parent.name.startswith(WORKTREE_PREFIX)


def gc_worktrees(root, max_age_s=WORKTREE_GC_AGE_S, now=None, dry=False, runner=subprocess.run,
                 protect=(Path.home() / "mojolearn-wt",)):
    """Remove stale temporary dependency worktrees of ROOT: listed by `git worktree list` or registered,
    under a `mojolearn-dependencies-*` directory. The owner comes from the registry, else from the
    directory's WORKTREE_OWNER file: a dead owner's tree is removed at any age, a live owner's tree (or this
    process's) is kept, and a tree with no recorded owner is removed only when older than MAX_AGE_S.
    A removed or missing tree also loses its registration (`.git/worktrees/<id>`, matched by its gitdir
    file: the targeted equivalent of `git worktree prune`, which would also prune other sessions' missing
    trees). Returns dict(removed, kept, pruned)."""
    import shutil
    import time
    now = time.time() if now is None else now
    registered = {}
    for f in sorted(worktree_registry().glob("*.json")) if worktree_registry().is_dir() else []:
        try:
            row = json.loads(f.read_text())
            registered[row["tree"]] = dict(row, file=str(f))
        except (OSError, ValueError, KeyError):
            continue
    candidates = {t for t in _listed_worktrees(root, runner) if _is_dependency_tree(t, protect)}
    candidates |= {t for t, row in registered.items() if row.get("root") == str(root) and _is_dependency_tree(t, protect)}
    removed, kept, pruned = [], [], []
    for tree in sorted(candidates):
        p = Path(tree)
        row = dict(registered.get(tree) or {})
        if not row.get("pid"):
            try:
                row.update({k: v for k, v in json.loads((p.parent / WORKTREE_OWNER).read_text()).items()
                            if k in ("pid", "start", "created")})
            except (OSError, ValueError, AttributeError):
                pass
        state = _owner_state(row)
        if state == "self":
            kept.append((tree, "owned by this process"))
            continue
        if state == "alive":
            kept.append((tree, "its process %s is alive" % row["pid"]))
            continue
        if state == "unknown":
            try:
                born = row.get("created") or p.parent.stat().st_mtime
            except OSError:
                born = row.get("created") or 0
            if now - born < max_age_s:
                kept.append((tree, "no recorded owner and younger than %g hours" % (max_age_s / 3600)))
                continue
        if dry:
            removed.append(tree)
            continue
        if p.exists():
            runner(["git", "-C", str(root), "worktree", "remove", "--force", tree], capture_output=True, text=True)
            shutil.rmtree(p.parent, ignore_errors=True)
            removed.append(tree)
        admin = Path(root) / ".git" / "worktrees"
        for gitdir in admin.glob("*/gitdir") if admin.is_dir() else []:
            try:
                if Path(gitdir.read_text().strip()).parent.resolve() in (p, p.resolve()):
                    shutil.rmtree(gitdir.parent, ignore_errors=True)
                    pruned.append(tree)
            except OSError:
                continue
        if row.get("file"):
            Path(row["file"]).unlink(missing_ok=True)
    return dict(removed=removed, kept=kept, pruned=pruned)


_GC_DONE = set()
_GC_LOCK = threading.Lock()


def gc_before_analysis(root, say=None):
    """Once per process and checkout, before the first frozen analysis: gc_worktrees with the default rules.
    A gc failure never blocks the analysis (it only leaves disk)."""
    say = say or (lambda line: print(line, file=sys.stderr))
    with _GC_LOCK:
        if str(root) in _GC_DONE:
            return None
        _GC_DONE.add(str(root))
        try:
            with _WORKTREES:
                result = gc_worktrees(Path(root))
        except Exception as error:  # noqa: BLE001 - cleanup is best effort, the analysis is not
            say("dependency worktree gc skipped: " + (str(error) or type(error).__name__))
            return None
    if result["removed"] or result["pruned"]:
        say("dependency worktree gc: removed %d, pruned %d registration(s), kept %d"
            % (len(result["removed"]), len(result["pruned"]), len(result["kept"])))
    return result

class SnapshotPending(ValueError):
    """Discovery only: a frozen snapshot this computation needs is not cached yet."""


def admission_jobs():
    value = os.environ.get("MOJOLEARN_ADMISSION_JOBS", "")
    if value:
        if not value.isdigit() or int(value) < 1:
            raise ValueError("MOJOLEARN_ADMISSION_JOBS must be a positive integer")
        return int(value)
    return os.cpu_count() or 1


def prefetch(compute, *, jobs=None, worker=None, rounds=8, say=None, ran=None):
    """``compute()``, after every frozen snapshot it needs ran in parallel; returns its value.

    Each round runs ``compute`` in discovery mode; a round that needed nothing uncached was an ordinary
    run (discovery only acts on a cache miss), so its value or error is the result: no extra pass. The
    snapshots run in a round are appended to ``ran`` ([{token, seconds, ok}])."""
    jobs = jobs or admission_jobs()
    worker = worker or os.environ.get("MOJOLEARN_ADMISSION_WORKER") or "local"
    say = say or (lambda line: print(line, file=sys.stderr))
    ran = [] if ran is None else ran
    # The snapshots run on pool threads, where no signal handler can be installed: install them here.
    _install_worktree_cleanup()
    for round_ in range(rounds):
        pending, value, error = {}, None, None
        context = _DISCOVERY.set(pending)
        try:
            value = compute()
        except Exception as problem:  # noqa: BLE001 - an error with nothing pending is the real result
            error = problem
        finally:
            _DISCOVERY.reset(context)
        if not pending:
            if error is not None:
                raise error
            return value
        say(f"dependency analysis round {round_ + 1}: {len(pending)} frozen snapshot(s) on {worker}, {jobs} at once")
        ran += run_snapshots(pending, jobs=jobs, worker=worker, say=say)
    return compute()


def run_snapshots(pending, *, jobs, worker="local", say=print):
    """{token: (root, revision, request, log_dir)} -> run each into its cache; failures kept in _FAILED."""
    import time
    from concurrent.futures import ThreadPoolExecutor
    if worker != "local":
        return remote_snapshots(pending, worker, jobs=jobs, say=say)

    def one(item):
        token, (root, revision, request, logs) = item
        started = time.time()
        try:
            frozen_snapshot(root, revision, request, log_dir=logs)
            ok = True
        except Exception as error:  # noqa: BLE001 - recorded; the real computation raises it
            _FAILED[token] = "frozen dependency analysis failed in prefetch: " + (str(error) or type(error).__name__)
            ok = False
        return dict(token=token, seconds=round(time.time() - started, 1), ok=ok)

    with ThreadPoolExecutor(max_workers=max(1, min(jobs, len(pending)))) as pool:
        results = list(pool.map(one, sorted(pending.items())))
    for row in results:
        say(f"  snapshot {row['token']}: {'ok' if row['ok'] else 'FAILED'} {row['seconds']} s")
    return results


ANALYZER_FILES = ("release_requirement_dependencies.py", "release_requirement_routes.py")
_EMPTY_TAR = bytes(10240)


def _analyzer():
    return digest({name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                   for name in ANALYZER_FILES})


def remote_snapshots(pending, target, *, jobs, say=print):
    """Run the snapshots on ``target`` (ssh destination); the Mac only ships, starts and collects.

    Ships the frozen source by ``git push`` of the exact revisions (and every base revision a request
    compares with) to a bare repository on the box, plus this analyzer's own files; runs the workers there
    in parallel (``--remote-batch``); brings each report back by its content-derived token and admits it
    only when its analyzer digest, revision and request equal the local ones. The ledger never leaves
    the Mac: workers read only (revision, request). Commands: MOJOLEARN_ADMISSION_SSH (default
    ``ssh -o BatchMode=yes``), MOJOLEARN_ADMISSION_REMOTE_DIR, MOJOLEARN_ADMISSION_REMOTE_PYTHON."""
    import io
    import shlex
    import tarfile
    import time
    ssh = shlex.split(os.environ.get("MOJOLEARN_ADMISSION_SSH") or "ssh -o BatchMode=yes")
    base = os.environ.get("MOJOLEARN_ADMISSION_REMOTE_DIR") or "mojolearn-admission"
    python = os.environ.get("MOJOLEARN_ADMISSION_REMOTE_PYTHON") or "python3"
    if not re.fullmatch(r"[A-Za-z0-9._/~-]+", base) or not re.fullmatch(r"[A-Za-z0-9._/-]+", python):
        raise ValueError("unsafe MOJOLEARN_ADMISSION_REMOTE_DIR or _REMOTE_PYTHON")
    analyzer = _analyzer()

    def remote(command, data=None):
        return subprocess.run([*ssh, target, command], input=data, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, check=True).stdout

    started = time.time()
    results = []
    roots = {}
    for token, (root, revision, request, logs) in pending.items():
        roots.setdefault(root, []).append((token, revision, request, logs))
    for root, items in roots.items():
        remote(f"mkdir -p {base}/logs {base}/tooling/{analyzer} && "
               f"(test -d {base}/repo.git || git init -q --bare {base}/repo.git)")
        revisions = sorted({revision for _, revision, _, _ in items}
                           | {_revision(root, request["base_revision"]) for _, _, request, _ in items
                              if request.get("base_revision")})
        env = dict(os.environ, GIT_SSH_COMMAND=" ".join(shlex.quote(part) for part in ssh))
        subprocess.run(["git", "-C", root, "push", "--quiet", "--force", f"{target}:{base}/repo.git",
                        *[f"{revision}:refs/admission/{revision}" for revision in revisions]],
                       env=env, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        bundle = io.BytesIO()
        with tarfile.open(fileobj=bundle, mode="w") as archive:
            for name in ANALYZER_FILES:
                archive.add(str(Path(__file__).parent / name), arcname=name, recursive=False)
        remote(f"tar -x -C {base}/tooling/{analyzer}", bundle.getvalue())
        batch = [dict(token=token, revision=revision, request=request) for token, revision, request, _ in items]
        say(f"  remote {target}: {len(batch)} snapshot(s), {len(revisions)} revision(s) pushed")
        remote(f"cd {base} && {python} tooling/{analyzer}/release_requirement_dependencies.py "
               f"--remote-batch repo.git logs {int(jobs)}", json.dumps(batch).encode())
        names = " ".join(f"{token}.json {token}.log" for token, _, _, _ in items)
        archive_bytes = remote(f"cd {base}/logs && set -- && for f in {names}; do [ -f \"$f\" ] && set -- \"$@\" \"$f\"; done; "
                               'if [ $# -gt 0 ]; then tar -cf - "$@"; fi')
        found = {}
        with tarfile.open(fileobj=io.BytesIO(archive_bytes or _EMPTY_TAR), mode="r:") as archive:
            for member in archive:
                if not member.isfile() or not re.fullmatch(r"[0-9a-f]{16}\.(json|log)", member.name):
                    raise ValueError("unexpected member in remote dependency results: " + member.name)
                found[member.name] = archive.extractfile(member).read()
        for token, revision, request, logs in items:
            logs = Path(logs)
            logs.mkdir(parents=True, exist_ok=True)
            data = found.get(token + ".json")
            ok = False
            if data is None:
                _FAILED[token] = "remote dependency analysis produced no report on " + target
            else:
                body = json.loads(data)
                if (body.get("analyzer") != analyzer or body.get("revision") != revision
                        or body.get("request") != request or "snapshot" not in body):
                    _FAILED[token] = "remote dependency report differs from the requested analysis"
                else:
                    body["source_view"] = dict(body.get("source_view") or {}, worker=target,
                                               report_sha256=hashlib.sha256(data).hexdigest())
                    _write_report(logs / (token + ".json"), body)
                    ok = True
            if token + ".log" in found:
                (logs / (token + ".log")).write_bytes(found[token + ".log"])
            results.append(dict(token=token, ok=ok, seconds=None, worker=target))
    say(f"  remote {target}: {sum(r['ok'] for r in results)}/{len(results)} ok in {round(time.time() - started, 1)} s")
    return results


def _remote_batch(repo, logs, jobs):
    """Box side of remote_snapshots: run a batch (stdin JSON) of frozen snapshots in parallel."""
    from concurrent.futures import ThreadPoolExecutor
    batch = json.loads(sys.stdin.read())
    repo, logs = Path(repo).resolve(), Path(logs).resolve()

    def one(item):
        try:
            frozen_snapshot(repo, item["revision"], item["request"], log_dir=logs)
            return item["token"], None
        except Exception as error:  # noqa: BLE001 - reported; the Mac refuses a missing report
            return item["token"], str(error) or type(error).__name__

    with ThreadPoolExecutor(max_workers=max(1, min(int(jobs), len(batch) or 1))) as pool:
        for token, error in pool.map(one, batch):
            print(json.dumps(dict(token=token, ok=error is None, error=error)))
    return 0


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
    analyzer = _analyzer()
    if report.is_file():
        old = json.loads(report.read_text())
        if old.get("analyzer") == analyzer and old.get("request") == request and old.get("revision") == revision:
            return old["snapshot"]
    if token in _FAILED:
        # A prefetch already ran this exact analysis and it failed: report that, never run it twice.
        raise ValueError(_FAILED[token])
    pending = _DISCOVERY.get()
    if pending is not None:
        pending.setdefault(token, (str(root), revision, copy.deepcopy(request), str(logs)))
        raise SnapshotPending(token)
    _install_worktree_cleanup()
    gc_before_analysis(root)
    with tempfile.TemporaryDirectory(prefix=WORKTREE_PREFIX) as temporary:
        temporary = Path(temporary)
        _write_owner(temporary)
        tree = temporary / "tree"
        with log_path.open("w") as log:
            # A fresh physical checkout can exhaust disk on the build Mac.
            # Only reuse exact current bytes, and exclude ALL untracked files:
            # clean Git status alone would admit generated Mojo and pyc files
            # that do not exist in the requested revision's source tree.
            # A bare repository (MOJOLEARN_ADMISSION_WORKER: the source pushed to a box) has no checkout
            # to clone: it is always a fresh checkout of the exact revision.
            clone_current = (sys.platform == "darwin" and not _bare(root)
                             and _revision(root, "HEAD") == revision)
            before = _tracked_state(root, revision, tooling_from_revision=True) if clone_current else None
            # The source commit and the release tooling are two commits (tools/release.py): name both.
            source_view = dict(method="tracked-cow" if clone_current else "git-checkout",
                               source_checkout=str(root), tracked_files_checked=len(before or {}),
                               tooling_from_revision=tooling_overlay(before or {}),
                               tooling=_tooling_commit())
            log.write("source view: " + json.dumps(source_view, sort_keys=True) + "\n")
            log.flush()
            command = ["git", "-C", str(root), "worktree", "add", "--detach"]
            if clone_current:
                command.append("--no-checkout")
            added = False
            try:
                with _WORKTREES:
                    subprocess.run(command + [str(tree), revision], stdout=log, stderr=subprocess.STDOUT, check=True)
                added = True
                _register_worktree(root, tree)
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
                    with _WORKTREES:
                        removal = subprocess.run(["git", "-C", str(root), "worktree", "remove", "--force", str(tree)],
                                                 stdout=log, stderr=subprocess.STDOUT)
                    # A SIGINT handler may already have removed it (_remove_live_worktrees): only a tree that is
                    # still there is a failed removal.
                    if removal.returncode and (tree / ".git").exists():
                        raise subprocess.CalledProcessError(removal.returncode, removal.args)
                _unregister_worktree(tree)
    _write_report(report, dict(analyzer=analyzer, revision=revision, request=request,
                               source_view=source_view, snapshot=snapshot))
    return snapshot


def _write_report(report, body):
    """Atomic: parallel workers and a reader of the cache never see a partial report."""
    fd, name = tempfile.mkstemp(prefix=report.name + ".", dir=report.parent)
    try:
        with os.fdopen(fd, "w") as out:
            out.write(json.dumps(body, sort_keys=True) + "\n")
        os.replace(name, report)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _bare(root):
    try:
        return _git(root, "rev-parse", "--is-bare-repository") == "true"
    except (OSError, subprocess.CalledProcessError):
        return False


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
# Reviewed 2026-11-01 (orchestrator): __version__ reaches the sequential binding context only as
# context["version"], written into the model-verify sidecar's install record next to the platform
# (bindings/sequential_checkpoint.mojo:59); no fitted or predicted value reads it.
VERSION_LABEL_ANCHORS = {'python/mojolearn/_x_sequence_model.py': 'b09cd035b7fa4a326487698e5f1bb39d7dca4a1a12f4a47e2550c82603415b07', 'bindings/sequential_checkpoint.mojo': 'c6ee2b88d4351713378afea4395cdc6dfe9edf7e385e6ad74617ffde68a0e0bf', 'tools/identity_lanes/coverage.py': 'db8d3ae94a2a635cb169d97dfd1267267f4a830afe5cae5e53047ff67a4e191d'}
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


def version_literal_unconsumed(read_old, read_new):
    """THE RELEASE-LABEL RULE, shared by the ledger (`version_label_proof`) and
    tools/lane_select.py: True when `python/mojolearn/_version.py` differs
    only in its `__version__` literal (`_version_literal_view`) and every
    reviewed consumer of that literal (`VERSION_LABEL_ANCHORS`) has its
    reviewed bytes on both sides. `read_old(path)` / `read_new(path)` return
    a file's bytes (or raise OSError)."""
    try:
        views = [_version_literal_view(read(VERSION_PATH)) for read in (read_old, read_new)]
        if views[0] is None or views[0] != views[1]:
            return False
        for path, sha in VERSION_LABEL_ANCHORS.items():
            if any(hashlib.sha256(read(path)).hexdigest() != sha for read in (read_old, read_new)):
                return False
    except (OSError, UnicodeError, SyntaxError, subprocess.SubprocessError, TypeError, AttributeError):
        return False
    return True


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
        if [hashlib.sha256(data).hexdigest() for data in (before, after)] != [x[VERSION_PATH], y[VERSION_PATH]]:
            return None
    except (OSError, UnicodeError, SyntaxError, subprocess.SubprocessError):
        return None
    if not version_literal_unconsumed(lambda p: _git_bytes(root, old_rev, p), lambda p: _git_bytes(root, new_rev, p)):
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
        # FILE-KEYED INPUTS ARE THE SELECTOR'S (lane identity-attribution,
        # 2026-10-31). An identity unit's execution environment carries the
        # hashes of the builders its bindings ran (`builders`) and the loader
        # route sources (`loader_routes.sources`). A difference that is ONLY in
        # those hashes is a set of changed files, and whether a changed file can
        # move a lane is tools/lane_select.py's derived, tested answer (the
        # rootdir-only builder edit, host_surface's named rows, ...), the same
        # one native sources rest on below. Any other difference stays affected.
        moved_files = _file_keyed_difference(a, b) if new.get("kind") == "identity" else None
        if new.get("kind") not in ("identity", "performance") or (a != b and moved_files is None):
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
        if moved_files and not set(moved_files) <= set(selection.get("changed_paths", ())):
            row["reason"] = "an execution input changed that the selector was not asked about"
            continue
        proof = dict(schema=PROOF_SCHEMA, unit_id=rid, old_revision=plan["old_revision"],
            new_revision=plan["new_revision"], old_fingerprint=old["fingerprint"],
            new_fingerprint=new["fingerprint"], stable_inputs_digest=digest(a),
            lanes=sorted(lanes), selection=proof_selection(selection))
        if moved_files:
            proof["selector_proven_files"] = sorted(moved_files)
            proof["stable_inputs_digest"] = digest(_file_keyed_view(a))
        proof["digest"] = digest(proof)
        row.update(decision="retained", reason="named-edit selector proves lane inputs unaffected", reuse_proof=proof)
    for rid, row in plan["units"].items():
        if row["decision"] == "retained":
            row["evidence"] = (evidence_restoration(retained_evidence[rid])
                if retained_evidence is not None and rid in retained_evidence
                else dict(status="not-supplied", restore=[]))
    return plan


#: Explanation the selector attaches to a selection, never part of its decision: `chains` holds ONE example
#: dependency chain per lane and path, whichever the traversal met first, and that choice differs between
#: Python processes (hash-randomized set order). 0.8.64: every stored identity proof failed to reproduce on
#: the next step because of it, and the ledger kept the failed proof (blocked 439 rows).
SELECTION_EXPLANATION = ("chains",)


def proof_selection(selection):
    """The decision-bearing part of a selector selection, as the equivalence proof records it."""
    return {key: value for key, value in selection.items() if key not in SELECTION_EXPLANATION}


def _file_keyed_maps(inputs):
    """((lane, owner), {path: sha256}) for each file-keyed map of one identity unit's execution environment:
    each binding's `builders` (owner: the binding) and the `loader_routes` `sources` (owner: "loader_routes").
    Maps are NAMED, never positional: a stored attempt (JSON, sorted keys) and a freshly materialized
    manifest (insertion order) list the bindings in different orders, and pairing them by position compared
    one binding's builders with another's (0.8.64: 439 identity-apple rows read as "an execution input
    changed that the selector was not asked about")."""
    for lane, env in (inputs.get("execution_environment") or {}).items():
        if not isinstance(env, dict):
            continue
        for name, ident in (env.get("bindings") or {}).items():
            if isinstance(ident, dict) and isinstance(ident.get("builders"), dict):
                yield (lane, name), ident["builders"]
        routes = env.get("loader_routes")
        if isinstance(routes, dict) and isinstance(routes.get("sources"), dict):
            yield (lane, "loader_routes"), routes["sources"]


def _file_keyed_view(inputs):
    """`inputs` with every file-keyed hash replaced by its path."""
    view = copy.deepcopy(inputs)
    for _, table in _file_keyed_maps(view):
        for path in table:
            table[path] = path
    return view


def _file_keyed_difference(a, b):
    """The files whose hashes are ALL that differ between two identity unit
    inputs (an empty set when they are equal), or None when anything else
    differs (a flag, a toolchain, a binding, a file added or removed)."""
    if a == b:
        return set()
    if _file_keyed_view(a) != _file_keyed_view(b):
        return None
    maps_a, maps_b = dict(_file_keyed_maps(a)), dict(_file_keyed_maps(b))
    if set(maps_a) != set(maps_b):
        return None
    moved = set()
    for label, ta in maps_a.items():
        tb = maps_b[label]
        moved |= {p for p in ta if ta[p] != tb.get(p)}
    return moved or None


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
    if actual != decision_proof(proof):
        errors.append("equivalence proof does not reproduce from the frozen source and requirement inputs")
    return errors


def decision_proof(proof):
    """A proof recorded before SELECTION_EXPLANATION existed carries the selector's explanation inside
    its selection; its decision content is compared, with the digest recomputed over that content. A
    proof whose decision fields were altered still fails: the frozen selector does not reproduce them."""
    if not isinstance(proof, dict) or not isinstance(proof.get("selection"), dict):
        return proof
    if not any(key in proof["selection"] for key in SELECTION_EXPLANATION):
        return proof
    saved = dict(proof, selection=proof_selection(proof["selection"]))
    saved.pop("digest", None)
    saved["digest"] = digest(saved)
    return saved


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--gc-worktrees":
        import argparse
        ap = argparse.ArgumentParser(prog="release_requirement_dependencies.py --gc-worktrees")
        ap.add_argument("--gc-worktrees", action="store_true")
        ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
        ap.add_argument("--max-age-hours", type=float, default=WORKTREE_GC_AGE_S / 3600)
        ap.add_argument("--dry-run", action="store_true")
        a = ap.parse_args()
        result = gc_worktrees(Path(a.root), max_age_s=a.max_age_hours * 3600, dry=a.dry_run)
        for tree in result["removed"]:
            print(("would remove " if a.dry_run else "removed ") + tree)
        for tree in result["pruned"]:
            print("pruned the registration of " + tree)
        for tree, why in result["kept"]:
            print("kept %s: %s" % (tree, why))
        raise SystemExit(0)
    if len(sys.argv) == 5 and sys.argv[1] == "--remote-batch":
        raise SystemExit(_remote_batch(*sys.argv[2:5]))
    if len(sys.argv) != 5 or sys.argv[1] != "--worker":
        raise SystemExit("library API: expand_catalog / equivalence_proofs / validate_equivalence_proof")
    Path(sys.argv[4]).write_text(json.dumps(_worker_snapshot(sys.argv[2], json.loads(Path(sys.argv[3]).read_text())), sort_keys=True))
