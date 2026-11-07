#!/usr/bin/env python3
"""Incremental development bindings; release packaging keeps its merged layout.

    python3 tools/development_build.py plan --sets identical,host --json plan.json
    python3 tools/development_build.py build --sets identical,host
    MOJOLEARN_GPU_ARCHS=gfx942 python3 tools/development_build.py build --sets identical

On the Mac the orchestrator wraps build with the existing semaphore:
    bash ~/mojolearn-evidence/compile_slot.sh python3 tools/development_build.py build

This is build tooling, never product runtime. Only identical and host are
supported.

Two layouts (--layout). On macOS the default is `release`: the development
build of a commit IS the macOS wheel's compile. It builds the merged groups of
bindings/merged_groups.json exactly as packaging/macos/build_release_wheel.sh
does (tools/bincache.py build ./bindings/merged_build.sh <kind> --group <stem>,
the wheel's environment, MOJOLEARN_COMPILE_JOBS=1, MOJOLEARN_PACKAGE_BYTE_LM=1)
into ONE Mac cache (--mac-cache, else MOJOLEARN_BINCACHE_DIR, else
~/mojolearn-evidence/bincache-mac); tools/release.py's macOS build seeds from
that cache, so a release of the same commit in the same checkout places every
group instead of compiling it. The section "--layout release" below has the
key-field comparison. On Linux the default is `development`: one group per
binding (bindings/development_groups.json), each with a content-addressed
artifact and receipt in --cache. Unchanged groups reuse their exact bytes. An
unresolved local import widens the key to every project Mojo source and
records that fallback.

Development builds use the standard optimized compiler defaults. Do not pass
-O0 or --optimization-level 0: an unoptimized image returned zero where the
same source built normally returned the correct result, even with explicit
buffer lifetime protection. The compiler internal cause is unconfirmed. Smaller
groups reduce compilation scope; they do not require reduced optimization.

Builds write complete logs and receipts under --evidence. All groups required
by a selected set must be current before its loader index is replaced. A
--group selection can populate the cache without activating an incomplete set
(exit 3). Activation uses immutable files and one atomic index replacement per
set; use a fresh Python process afterwards. Existing per-binding overrides are
refused, since they would silently shadow the selected development binaries.

Release commands do not call this tool. A development-layout index
deliberately fails the release layout check; final release qualification still
builds/checks the compact release layout (from the one Mac cache when this tool
already compiled it with --layout release). No compile-time or numerical equivalence claim follows
from a successful cache/planning test.

Groups compile in parallel by default (--parallel auto). Each group's need is
its RECORDED peak RSS from the last compile (cache/peaks/, with headroom); a
group without a record takes --memory-per-job-gib when given, else the largest
recorded need of its set (identical or host), else the platform group floor.
The scheduler is a sliding window: it starts the longest groups first (their
recorded compiler seconds, else their source closure size) and starts the next
group whenever one finishes and its memory need fits 80% of the memory
available at start and a core is free, at most one group per core, re-reading
available memory before every launch. Groups without a record of their own
pack by the largest need measured so far in this run for their set (their rss
limit stays at least the floor). A group bigger than that budget
runs alone when it fits 80% of total memory, and is refused up front otherwise.
MOJOLEARN_BUILD_GROUP_WORKERS (or --parallel N) caps concurrent groups;
--parallel 1 is the serial run (one compiler, 80% of available memory or the
group floor, as before). Unless the caller sets MOJOLEARN_COMPILE_JOBS, each
group's compiler gets the free cores at its launch shared among the groups
starting then (at least one), so the running groups' threads sum to the core
count; that value is keyed like any compile job count, and the cache remembers which
value built each group so the next plan reuses it. AMD identical groups keep
their fixed worker count (MOJOLEARN_AMD_COMPILE_JOBS, default 8). Each group's
RSS limit is its need; the run stops a compiler when available memory falls
below the 20% kept for the system. --jobs N is the older spelling of
--parallel N. --group-seconds, --stall-seconds and --lock-seconds bound active
work and lock waits; defaults live in tools/release_limits.sh. These controller
settings never reach the compiler.
"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
import contextlib
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import uuid

import bincache
import build_process
import merged_bindings as merged
import release_limits

REPO = Path(__file__).resolve().parent.parent
SCHEMA = "mojolearn.development-build.v1"
KEY_SCHEMA = "mojolearn.development-group.v1"
MANIFEST = "bindings/development_groups.json"
# A group's `builders` key field is its effective compile recipe, not the
# controller's source. The two shell scripts compose every compiler flag,
# define and target from the keyed environment, so their executable lines are
# hashed (comments and blank lines dropped). The root text (merged_bindings.py
# root) is keyed by root_sha256, the members by `members`, and the compiler by
# `toolchain`; tools/development_build.py, build_process.py, build_sizing.py,
# bincache.py and merged_bindings.py never decide the bytes of a group, so
# editing them leaves every key unchanged (2026-10-24: a tool-only commit
# rebuilt 79 of 79 groups on the L4, the M4 and the AMD box).
RECIPE_SCHEMA = "mojolearn.development-recipe.v1"
# bindings/compile_jobs.sh is sourced by merged_build.sh and decides its `-j` when MOJOLEARN_COMPILE_JOBS is unset.
BUILD_INPUTS = ("bindings/merged_build.sh", "bindings/build_defines.sh", "bindings/compile_jobs.sh")
# Non-group runtime artifacts. python/mojolearn/_portable_math.py loads the
# repository-owned host math library from a fixed path in every set; release
# packaging builds it with packaging/portable_math/stage.py. It is a C build
# (cc/clang), keyed on its sources, stage.py's flags and the C compiler.
SUPPORT_SCHEMA = "mojolearn.development-support.v1"
PORTABLE_MATH_INPUTS = ("packaging/portable_math/portable_math.c",
                        "packaging/portable_math/powers_of_ten.h",
                        "packaging/portable_math/stage.py")
# Private transport between controller functions, deliberately outside
# bincache.KEY_ENV_PREFIXES and removed before starting any child process.
GUARD_ENV = "_MOJOLEARN_DEVELOPMENT_GUARD"
MEMORY_FRACTION = 0.8
# Without a measured --memory-per-job-gib, a group compiler may still use at
# least this much: a compiler's need is a property of the group, not of what
# else is running. The largest host groups (tokenizer_host,
# runtime_classical_host) peak near 10 GB on Linux; on a loaded 16 GiB M4
# (2026-10-24) "80% of available" was 2.96 GB and both were killed after
# ~250 s. 12 GiB is that peak plus 20%. A recorded peak raises it.
GROUP_FLOOR_BYTES = {"linux": 12 * 2 ** 30, "darwin": 12 * 2 ** 30}
# Recorded peaks of a run that passed get 20% headroom; a run stopped by a
# limit only shows a lower bound, so it gets 50%.
PEAK_HEADROOM = {"passed": 1.2, "limited": 1.5}
# Cap on concurrently compiling groups (scheduling only; removed from the
# compiler environment and from every cache key in build_environment).
GROUP_WORKERS_ENV = "MOJOLEARN_BUILD_GROUP_WORKERS"
# How long the parallel scheduler waits for a group to finish before it
# re-reads available memory to see whether another group now fits.
SCHEDULER_POLL_SECONDS = 15.0
# Version banners are read from stdout only. On the NVIDIA L4 (2026-10-24,
# source 50de0f93a) `mojo --version` wrote a tcmalloc mbind warning to stderr
# ahead of its banner, carrying the PID and a random address; with stderr
# merged that line became toolchain.version and every key recomputation
# differed, so a correct 183 s compile was refused as "inputs changed".
VERSION_LINE = re.compile(r"^\s*(mojo|pixi)\s+\S", re.I)
# Files that tools, editors or the OS write next to inputs and that never
# reach the compiler: bytecode, editor swap/backup files, Finder metadata.
NON_INPUT_SUFFIXES = (".pyc", ".pyo", ".swp", ".swo", ".tmp", ".orig", ".rej", "~")


class BuildError(RuntimeError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def positive_seconds(value, label):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise BuildError(f"{label} must be finite and positive")
    return value


@contextlib.contextmanager
def lock(path, seconds=None):
    seconds = positive_seconds(seconds if seconds is not None else
                               release_limits.get("DEVELOPMENT_LOCK_SECONDS"), "lock wait")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        deadline = time.monotonic() + seconds
        while True:
            if build_process.cancelled():
                raise KeyboardInterrupt("development build cancelled while waiting for lock")
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise BuildError(f"lock wait exceeded {seconds:g}s: {path}")
                time.sleep(min(0.1, remaining))
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class SourceGraph:
    """Conservative file-level closure using the existing cache import parser."""

    def __init__(self, repo):
        self.repo = Path(repo).resolve()
        self.nodes = {}

    def node(self, path):
        path = path.resolve()
        if path not in self.nodes:
            dependencies, unresolved = set(), []
            for module, names in bincache.parse_imports(path.read_text(errors="replace")):
                got = bincache.resolve(module, names, path, [self.repo, self.repo / "bindings"])
                if got is None:
                    unresolved.append({"file": str(path.relative_to(self.repo)), "import": module})
                else:
                    dependencies.update(p.resolve() for p in got)
            self.nodes[path] = dependencies, unresolved
        return self.nodes[path]

    def closure(self, members):
        files, unresolved = set(), []
        todo = [self.repo / "bindings" / (member + ".mojo") for member in members]
        while todo:
            path = todo.pop().resolve()
            if path in files:
                continue
            if not path.is_file():
                raise BuildError(f"missing binding source: {path}")
            if not path.is_relative_to(self.repo):
                raise BuildError(f"project import leaves repository: {path}")
            files.add(path)
            dependencies, unknown = self.node(path)
            unresolved.extend(unknown)
            todo.extend(dependencies - files)
        scope = "closure"
        if unresolved:
            scope = "tree-fallback"
            files = {p.resolve() for p in bincache.tree_sources(self.repo)}
        # The generated table itself joins the closure once present. Generator
        # sources and the declared Unicode pin are inputs before it exists too.
        if any(str(p.relative_to(self.repo)).startswith("tokenizer/") for p in files):
            files.update(p.resolve() for p in (self.repo / "tokenizer/tools").rglob("*")
                         if p.is_file() and generator_input(p))
            files.add(self.repo / "tokenizer/impl/unicode_class.mojo")
        hashes = {str(p.relative_to(self.repo)): bincache.sha256_file(p) for p in sorted(files)}
        return {"scope": scope, "files": hashes, "unresolved": unresolved}


def generator_input(path):
    """True for a generator source; False for files tools write beside it."""
    if "__pycache__" in path.parts or path.name.startswith("."):
        return False
    return not path.name.endswith(NON_INPUT_SUFFIXES)


def version_banner(argv):
    """The tool's version line from stdout; stderr noise never enters a key."""
    try:
        run = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             stdin=subprocess.DEVNULL, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    lines = run.stdout.decode(errors="replace").splitlines()
    return next((line.strip() for line in lines if VERSION_LINE.match(line)), None)


def first_stdout_line(argv):
    try:
        run = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             stdin=subprocess.DEVNULL, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    return next((line.strip() for line in run.stdout.decode(errors="replace").splitlines()
                 if line.strip()), None)


def c_toolchain_facts(env, plat=None):
    """The C compiler stage.build would pick (CC, else clang/cc), resolved."""
    plat = plat or merged.default_platform()
    name = env.get("CC") or ("clang" if plat == "darwin" else "cc")
    path = shutil.which(name, path=env.get("PATH"))
    if path is None:
        return {"available": False, "compiler": name}
    resolved = str(Path(path).resolve())
    return {"available": True, "compiler": path, "resolved": resolved,
            "sha256": bincache.sha256_file(resolved),
            "version": first_stdout_line([path, "--version"]),
            "machine": platform.machine().lower()}


def support_rows(repo, cache, plat, c_facts):
    missing = [name for name in PORTABLE_MATH_INPUTS if not (repo / name).is_file()]
    if missing:
        raise BuildError("portable math sources missing: " + ", ".join(missing))
    darwin = plat == "darwin"
    fields = {"schema": SUPPORT_SCHEMA, "name": "portable_math", "platform": plat,
              "repo": str(repo), "cache": str(cache),
              "inputs": {name: bincache.sha256_file(repo / name) for name in PORTABLE_MATH_INPUTS},
              "c_toolchain": c_facts}
    row = {"name": "portable_math", "stem": "libMojolearnMath",
           "file": "libMojolearnMath.dylib" if darwin else "libMojolearnMath.so",
           "install": (".dylibs/libMojolearnMath.dylib" if darwin else ".libs/libMojolearnMath.so"),
           "key": digest(fields), "fields": fields}
    row["action"] = "reuse" if cached_artifact(cache, row) else "build"
    return [row]


def changed_fields(before, after, prefix=""):
    """Dotted key-field paths that differ; the refusal names what moved."""
    if isinstance(before, dict) and isinstance(after, dict):
        out = []
        for name in sorted(set(before) | set(after)):
            out.extend(changed_fields(before.get(name), after.get(name),
                                      f"{prefix}.{name}" if prefix else str(name)))
        return out
    return [] if before == after else [prefix or "<root>"]


def input_change_reason(label, before, after):
    moved = changed_fields(before, after)
    shown = ", ".join(moved[:8]) + (f" (+{len(moved) - 8} more)" if len(moved) > 8 else "")
    return f"{label}: {shown or 'group missing from recomputed plan'}"


def recipe_text(text):
    """A build script's executable lines: comments and blank lines dropped."""
    lines = (line.rstrip() for line in text.splitlines())
    return "\n".join(line for line in lines if line.strip() and not line.lstrip().startswith("#")) + "\n"


def group_command(mode, stem, outdir, rootdir):
    """The one compiler invocation of a group; keyed through compile_recipe."""
    return ["sh", "bindings/merged_build.sh", "host" if mode == "host" else "gpu",
            "--layout", "development", "--group", stem, "--outdir", str(outdir),
            "--rootdir", str(rootdir)]


def compile_recipe(scripts, mode, stem, flags_class):
    """Everything that decides a group's compiler command line, deterministic.

    Paths of the work directory are placeholders: they follow from the key."""
    return {"schema": RECIPE_SCHEMA, "scripts": scripts, "mode": mode, "flags_class": flags_class,
            "command": group_command(mode, stem, "<outdir>", "<rootdir>"),
            "artifact": stem + ".so"}


def auto_compile_jobs(mode, env, plat=None):
    """True when the scheduler picks this group's compiler threads.

    The caller's MOJOLEARN_COMPILE_JOBS always wins. AMD identical groups keep
    their fixed worker count (build_environment), since one AMD compiler
    worker crashes."""
    plat = plat or merged.default_platform()
    if "MOJOLEARN_COMPILE_JOBS" in env:
        return False
    return not (plat != "darwin" and mode == "identical"
                and env.get("MOJOLEARN_GPU_ARCHS", "").startswith("gfx"))


def with_compile_jobs(row, env, jobs, plat=None):
    """The same group row keyed for a compiler with `jobs` threads.

    MOJOLEARN_COMPILE_JOBS reaches the key only through fields["environment"]
    (it is keyed: a codegen split has not been shown byte-neutral), so only
    that field and the key change; the source closure is reused."""
    effective = build_environment(row["mode"], dict(env, MOJOLEARN_COMPILE_JOBS=str(int(jobs))),
                                  plat or row["fields"]["platform"])
    fields = dict(row["fields"], environment=bincache.build_env(effective))
    return dict(row, fields=fields, key=digest(fields), compile_jobs=int(jobs))


def build_environment(mode, env, plat=None):
    env = dict(env)
    env.pop(GROUP_WORKERS_ENV, None)  # scheduling only; never a compiler input or key
    plat = plat or merged.default_platform()
    # Reject before key lookup/activation: old O0 artifacts remain evidence,
    # but must not be selected as usable development images. Keep the normal
    # recipe and its cache keys unchanged; do not silently rewrite user flags.
    for variable in ("MOJOLEARN_MOJO_BUILD_FLAGS", "MOJOLEARN_BUILD_EXTRA_DEFINES"):
        try:
            flags = shlex.split(env.get(variable, ""))
        except ValueError as exc:
            raise BuildError(f"invalid {variable}: {exc}") from exc
        for index, flag in enumerate(flags):
            if (flag in ("-O0", "-O=0", "--optimization-level=0") or
                    flag in ("-O", "--optimization-level") and
                    flags[index + 1:index + 2] == ["0"]):
                raise BuildError(
                    f"{variable}: unoptimized (-O0) development images are unsupported; "
                    "remove the optimization override to use the standard optimized build. "
                    "An O0 image returned an incorrect zero; the compiler cause is unconfirmed.")
    if env.get("MOJOLEARN_NUMERIC_MODE", "identical") != "identical":
        raise BuildError("development builds support identical only; unset the other numeric mode")
    env["MOJOLEARN_NUMERIC_MODE"] = "identical"
    env["MOJOLEARN_PACKAGE_BYTE_LM"] = env.get("MOJOLEARN_PACKAGE_BYTE_LM", "1")
    if env["MOJOLEARN_PACKAGE_BYTE_LM"] not in ("0", "1"):
        raise BuildError("MOJOLEARN_PACKAGE_BYTE_LM must be 0 or 1")
    # Match the supported release compiler policy. AMD's one-worker compiler
    # crash is reproduced in recovery; group concurrency is a separate budget.
    default_workers = "1" if plat == "darwin" else "2"
    if plat != "darwin" and mode == "identical" and env.get("MOJOLEARN_GPU_ARCHS", "").startswith("gfx"):
        default_workers = env.get("MOJOLEARN_AMD_COMPILE_JOBS") or "8"
    env["MOJOLEARN_COMPILE_JOBS"] = env.get("MOJOLEARN_COMPILE_JOBS", default_workers)
    if not env["MOJOLEARN_COMPILE_JOBS"].isdigit() or int(env["MOJOLEARN_COMPILE_JOBS"]) < 1:
        raise BuildError("MOJOLEARN_COMPILE_JOBS must be a positive integer")
    env["MOJOLEARN_MERGED_PYTHON"] = sys.executable
    env.pop("MACOSX_DEPLOYMENT_TARGET", None)  # same rule as merged_build.sh
    if mode == "host":
        env.pop("MOJOLEARN_GPU_ARCHS", None)
        env["MOJOLEARN_TARGET_COLUMN"] = "cpu"
    for key in ("MOJOLEARN_EXTRA_DEFINES", "MOJOLEARN_MAMBA_DEFINES"):
        if env.get(key):
            raise BuildError(f"{key} is not supported by merged development groups")
    if mode == "identical" and env.get("MOJOLEARN_BUILD_EXTRA_DEFINES"):
        raise BuildError("MOJOLEARN_BUILD_EXTRA_DEFINES is host-only")
    if env.get("MOJOLEARN_ATTENTION_MEMORY_PROFILE", "estash") != "estash" or env.get("MOJOLEARN_ALLOW_NO_GPU") == "1":
        raise BuildError("unsupported per-binding option in development group environment")
    if bincache.SABOTAGE_RE.search(canonical(bincache.build_env(env)).decode()):
        raise BuildError("fault-injection builds use their dedicated controller, not the development cache")
    return env


def toolchain_facts(repo):
    compiler = repo / ".pixi/envs/default/bin/mojo"
    available = compiler.is_file()
    package_dir = repo / ".pixi/envs/default/lib/mojo"
    packages = {p.name: bincache.sha256_file(p) for p in sorted(package_dir.iterdir())
                if p.is_file() and p.suffix in (".mojoc", ".mojopkg")} if package_dir.is_dir() else {}
    config = repo / "pixi.toml"
    return {"available": available, "compiler": str(compiler.resolve()),
            "compiler_sha256": bincache.sha256_file(compiler) if available else None,
            "version": version_banner([str(compiler), "--version"]) if available else None,
            "locked": bincache.toolchain(repo), "installed": bincache.installed_toolchain(repo),
            "pixi_configuration": hashlib.sha256(bincache.pixi_toml_build_part(config.read_text()).encode()).hexdigest()
                if config.is_file() else None,
            "precompiled_packages": packages, "os": bincache.os_fields(),
            "apple": bincache.darwin_toolchain(), "pixi": version_banner(["pixi", "--version"])}


def cached_artifact(cache, row):
    directory = Path(cache) / "artifacts" / row["key"]
    artifact = directory / row.get("file", row["stem"] + ".so")
    receipt = directory / "receipt.json"
    try:
        saved = json.loads(receipt.read_text())
        if (saved["key"] != row["key"] or saved["fields"] != row["fields"]
                or saved["exit_code"] != 0 or artifact.stat().st_size == 0
                or saved["sha256"] != bincache.sha256_file(artifact)):
            return None
        return artifact
    except (OSError, KeyError, ValueError, TypeError):
        return None


def selected_group(mode, stem, selection):
    if not selection:
        return True
    logical = stem.removeprefix("_mojolearn_dev_").removesuffix("_m1").removesuffix("_host")
    return bool({stem, logical, f"{mode}:{stem}", f"{mode}:{logical}"} & set(selection))


def build_input_scripts(repo):
    """Executable-line hashes of the group build scripts (key field builders.scripts)."""
    return {p: hashlib.sha256(recipe_text((Path(repo) / p).read_text(errors="replace")).encode()).hexdigest()
            for p in BUILD_INPUTS}


def group_fields(repo, cache, mode, plat, stem, flags_class, members, graph, scripts, facts, effective):
    """The key fields of one group; make_plan and the in-flight input check share them."""
    return {"schema": KEY_SCHEMA, "repo": str(repo), "cache": str(cache),
            "mode": mode, "platform": plat, "stem": stem, "flags_class": flags_class,
            "members": members, "source": graph.closure(members),
            "builders": compile_recipe(scripts, mode, stem, flags_class),
            "root_sha256": hashlib.sha256(merged.root_text(stem, members).encode()).hexdigest(),
            "toolchain": facts, "environment": bincache.build_env(effective),
            "link_environment": {k: effective[k] for k in
                ("CC", "CXX", "LD", "SDKROOT", "DEVELOPER_DIR", "LIBRARY_PATH",
                 "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "PIXI_ENVIRONMENT_NAME") if k in effective}}


def current_group_fields(row, repo, cache, env, facts):
    """One group's key fields as its inputs stand now, or None if the group is gone.

    The in-flight input check after a compile. It used to rebuild the whole
    plan of the group's set: every group's closure, every cached artifact's
    sha256 and the support rows, in a pool thread. On the 32-vCPU RunPod box
    (2026-11-07, leg_out/dev.json) the 64 built groups compiled for 45-169 s
    but held their slot and memory reservation for 290-670 s, the compilers
    totalling 2942 s against 30616 s of group time: those whole-set re-plans,
    15 at once in one Python process, kept the box at 1.4 busy cores of 32.
    This rebuilds the same fields for this one group only (make_plan uses the
    same group_fields), so the key comparison is unchanged."""
    repo, cache = Path(repo).resolve(), Path(cache).resolve()
    plat = row["fields"]["platform"]
    effective = build_environment(row["mode"], env, plat)
    byte_lm = effective["MOJOLEARN_PACKAGE_BYTE_LM"] == "1"
    for stem, flags_class, members in merged.plan(row["mode"], plat, byte_lm, "development"):
        if stem == row["stem"]:
            return group_fields(repo, cache, row["mode"], plat, stem, flags_class, members,
                                SourceGraph(repo), build_input_scripts(repo), facts, effective)
    return None


class ToolchainProbe:
    """Toolchain facts read after a given moment, shared by groups that finish together.

    Each finished group must compare its key against toolchain facts read
    after its compiler exited. Reading them hashes the compiler and its
    packages and runs `mojo --version` and `pixi --version`; one read in
    flight serves every group whose compiler exited before that read began,
    so groups finishing together wait for one read, not one each."""

    def __init__(self, repo):
        self.repo = repo
        self.condition = threading.Condition()
        self.busy = False
        self.last = None  # (read start, facts, error)

    def facts_after(self, moment):
        with self.condition:
            while True:
                if self.last is not None and self.last[0] >= moment:
                    _, facts, error = self.last
                    if error is not None:
                        raise error
                    return facts
                if not self.busy:
                    self.busy, start = True, time.monotonic()
                    break
                if build_process.cancelled():
                    raise KeyboardInterrupt("development build cancelled while reading toolchain facts")
                self.condition.wait(1.0)
        facts, error = None, None
        try:
            facts = toolchain_facts(self.repo)
        except BaseException as caught:  # handed to every waiter of this read, then raised
            error = caught
        with self.condition:
            self.busy, self.last = False, (start, facts, error)
            self.condition.notify_all()
        if error is not None:
            raise error
        return facts


def make_plan(repo, cache, modes, env, selection=(), facts=None, plat=None, c_facts=None):
    repo, cache = Path(repo).resolve(), Path(cache).resolve()
    if cache.is_relative_to(repo):
        raise BuildError("keep the development cache outside the source tree")
    if merged.manifest().get("index_file") != "_mojolearn_merged.json":
        raise BuildError("development index contract must match the release loader")
    plat = plat or merged.default_platform()
    facts = facts if facts is not None else toolchain_facts(repo)
    if plat not in ("linux", "darwin"):
        raise BuildError("development builds require Linux or macOS")
    graph, rows = SourceGraph(repo), []
    scripts = build_input_scripts(repo)
    for mode in modes:
        if mode not in ("identical", "host"):
            raise BuildError("development builds support identical and host only")
        effective = build_environment(mode, env, plat)
        arch = effective.get("MOJOLEARN_GPU_ARCHS", "")
        if mode == "identical" and plat == "linux" and not arch:
            raise BuildError("Linux identical development builds need MOJOLEARN_GPU_ARCHS")
        if arch and (plat == "darwin" or not arch.replace("_", "").isalnum()):
            raise BuildError("supply one Linux accelerator target, not a list or an Apple override")
        byte_lm = effective["MOJOLEARN_PACKAGE_BYTE_LM"] == "1"
        for stem, flags_class, members in merged.plan(mode, plat, byte_lm, "development"):
            fields = group_fields(repo, cache, mode, plat, stem, flags_class, members,
                                  graph, scripts, facts, effective)
            row = {"mode": mode, "stem": stem, "members": members, "flags_class": flags_class,
                   "key": digest(fields), "fields": fields,
                   "selected": selected_group(mode, stem, selection)}
            row["auto_compile_jobs"] = auto_compile_jobs(mode, env, plat)
            row["compile_jobs"] = int(effective["MOJOLEARN_COMPILE_JOBS"])
            row["action"] = "reuse" if facts.get("available") and cached_artifact(cache, row) else "build"
            if row["action"] == "build" and row["auto_compile_jobs"] and facts.get("available"):
                # The scheduler chose this group's compiler threads last time;
                # its exact cached build under that keyed value is current.
                built = built_compile_jobs(cache, mode, stem)
                if built and built != row["compile_jobs"]:
                    alternative = with_compile_jobs(row, env, built, plat)
                    if cached_artifact(cache, alternative):
                        row = dict(alternative, action="reuse")
            # Sizing only; never a key field. It survives key changes.
            row["recorded_need_bytes"] = recorded_need(cache, mode, stem)
            row["recorded_seconds"] = recorded_seconds(cache, mode, stem)
            rows.append(row)
    for group in selection:
        if not any(selected_group(r["mode"], r["stem"], [group]) for r in rows):
            raise BuildError(f"group selection matches nothing: {group}")
    support = support_rows(repo, cache, plat, c_facts if c_facts is not None
                           else c_toolchain_facts(env, plat))
    return {"schema": SCHEMA, "layout": "development", "platform": plat,
            "repo": str(repo), "cache": str(cache), "toolchain_available": facts.get("available", False),
            "groups": rows, "support": support, "counts": {"groups": len(rows),
                "reuse": sum(r["action"] == "reuse" for r in rows),
                "build_selected": sum(r["action"] == "build" and r["selected"] for r in rows),
                "build_unselected": sum(r["action"] == "build" and not r["selected"] for r in rows)}}


def build_submission_order(rows):
    """Start larger estimated compile closures before smaller groups.

    File count is a deterministic cost proxy, not a measured duration. Reuse
    the closure already computed for the cache key; scheduling does not scan
    caches/files, rewrite plan rows, or reserve additional compilers. This avoids
    leaving a large owner at the end merely because of manifest order.
    No key fields are added; changes to this tool still invalidate keys once
    because its source is an existing BUILD_INPUT.
    """
    return sorted(rows, key=lambda row: (-len(row["fields"]["source"]["files"]),
                                         row["mode"], row["stem"]))


def available_resources():
    if sys.platform == "linux":
        from build_sizing import box_resources
        return box_resources()
    return os.cpu_count() or 1, build_process.available_memory()


def physical_memory():
    """Total memory this process may use: RAM, lowered to a cgroup limit."""
    if sys.platform == "darwin":
        return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True, timeout=10))
    total = None
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            total = int(line.split()[1]) * 1024
    if total is None:
        raise BuildError("cannot read MemTotal from /proc/meminfo")
    for name in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            text = Path(name).read_text().strip()
        except OSError:
            continue
        if text.isdigit() and int(text) < 2 ** 60:
            total = min(total, int(text))
        break
    return total


def peak_path(cache, mode, stem):
    return Path(cache) / "peaks" / f"{mode}-{stem}.json"


def recorded_need(cache, mode, stem):
    """Bytes the last recorded compile of this group needs, or 0."""
    try:
        saved = json.loads(peak_path(cache, mode, stem).read_text())
        peak = int(saved["peak_rss_bytes"])
        return int(peak * PEAK_HEADROOM["passed" if saved["status"] == "passed" else "limited"])
    except (OSError, KeyError, ValueError, TypeError):
        return 0


def recorded_seconds(cache, mode, stem):
    """Compiler seconds of this group's last passed recorded compile, or 0.0."""
    try:
        saved = json.loads(peak_path(cache, mode, stem).read_text())
        seconds = float(saved["elapsed_seconds"]) if saved["status"] == "passed" else 0.0
        return seconds if math.isfinite(seconds) and seconds > 0 else 0.0
    except (OSError, KeyError, ValueError, TypeError):
        return 0.0


def compile_jobs_path(cache, mode, stem):
    return Path(cache) / "compile-jobs" / f"{mode}-{stem}.json"


def built_compile_jobs(cache, mode, stem):
    """Compiler threads of this group's last cached build, or 0."""
    try:
        jobs = int(json.loads(compile_jobs_path(cache, mode, stem).read_text())["compile_jobs"])
        return jobs if jobs >= 1 else 0
    except (OSError, KeyError, ValueError, TypeError):
        return 0


def record_compile_jobs(cache, row, jobs):
    atomic_json(compile_jobs_path(cache, row["mode"], row["stem"]),
                {"compile_jobs": int(jobs), "key": row["key"],
                 "recorded_utc": datetime.now(timezone.utc).isoformat()})


def record_peak(cache, row, execution, compile_jobs=None):
    """Keep the group's last measured peak where the next plan can size from it.

    A passed run replaces the record; a run stopped by a limit only raises it,
    since its peak is a lower bound of the real need. compile_jobs (the
    compiler threads of that run) is recorded alongside, for reading only."""
    peak = int(execution.get("peak_rss_bytes") or 0)
    status = execution.get("status")
    if peak <= 0 or status not in ("passed", "rss-limit", "memory-pressure", "timeout", "stalled"):
        return
    path = peak_path(cache, row["mode"], row["stem"])
    if status != "passed":
        try:
            prior = json.loads(path.read_text())
            if int(prior["peak_rss_bytes"]) >= peak:
                return
        except (OSError, KeyError, ValueError, TypeError):
            pass
    record = {"peak_rss_bytes": peak, "status": "passed" if status == "passed" else "limited",
              "key": row["key"], "recorded_utc": datetime.now(timezone.utc).isoformat()}
    if compile_jobs is not None:
        record["compile_jobs"] = int(compile_jobs)
    if status == "passed" and execution.get("elapsed_seconds"):
        # Ordering only (packing_order): the longest groups start first.
        record["elapsed_seconds"] = float(execution["elapsed_seconds"])
    atomic_json(path, record)


def resource_budget(requested, workers, per_job_gib, resources=None, floor_bytes=0,
                    recorded_bytes=0, total_bytes=None):
    """One compiler reservation and how many compilers fit.

    floor_bytes is what an unmeasured group compiler may need (the platform
    floor or a recorded peak); recorded_bytes is the largest recorded need of
    the groups about to compile. The serial default is 80% of available
    memory, raised to floor_bytes when that fits 80% of total memory, and
    refused up front when it does not. A measured reservation below a
    recorded need is refused up front too. total_bytes defaults to the
    machine's (or, with explicit resources, to their available memory)."""
    if requested < 1 or workers < 1:
        raise BuildError("--jobs must be positive")
    if per_job_gib is None and requested > 1:
        raise BuildError("parallel builds require a measured --memory-per-job-gib reservation")
    if per_job_gib is not None and (not math.isfinite(per_job_gib) or per_job_gib <= 0):
        raise BuildError("--memory-per-job-gib must be finite and positive")
    explicit = resources is not None
    cores, memory = resources or available_resources()
    usable = int(MEMORY_FRACTION * memory)
    gib = 2 ** 30
    if cores < 1 or usable < 1:
        raise BuildError("available CPU/memory allocation cannot fit one requested compiler reservation")
    source = "measured" if per_job_gib is not None else "available-allocation"
    if per_job_gib is not None:
        reservation = int(per_job_gib * gib)
        if recorded_bytes > reservation:
            raise BuildError(
                f"--memory-per-job-gib {per_job_gib:g} is below a recorded compiler need of "
                f"{recorded_bytes / gib:.1f} GiB for a group in this build; pass "
                f"--memory-per-job-gib {math.ceil(recorded_bytes / gib)} or more")
    else:
        need = max(int(floor_bytes or 0), int(recorded_bytes or 0))
        reservation = max(usable, need)
        if reservation > usable:
            source = "group-floor"
    ceiling = usable
    if reservation > usable:
        total = total_bytes if total_bytes is not None else (memory if explicit else physical_memory())
        ceiling = int(MEMORY_FRACTION * total)
        if reservation > ceiling:
            raise BuildError(
                f"a group compiler reservation of {reservation / gib:.1f} GiB cannot fit: this machine allows "
                f"{ceiling / gib:.1f} GiB (80% of {total / gib:.1f} GiB) and {usable / gib:.1f} GiB "
                f"is available now; free memory or use a larger machine. If you measured a "
                f"smaller peak for these groups, pass --memory-per-job-gib <GiB>.")
    if reservation < 1:
        raise BuildError("available CPU/memory allocation cannot fit one requested compiler reservation")
    # Pack by what is available now; one compiler that fits the machine's
    # total may still start, since its need does not depend on other load.
    by_memory = max(usable // reservation, 1 if reservation <= ceiling else 0)
    actual = min(requested, cores // workers, by_memory)
    if actual < 1:
        raise BuildError("available CPU/memory allocation cannot fit one requested compiler reservation")
    floor = memory - usable if actual * reservation <= usable else max(0, memory - actual * reservation)
    return dict(jobs=actual, cores=cores, available_bytes=memory, compiler_workers=workers,
                rss_bytes=reservation, min_available_bytes=floor, reservation_source=source)


def pool_size(requested, workers, per_job_gib, resources=None):
    return resource_budget(requested, workers, per_job_gib, resources)["jobs"]


def planned_compiler_workers(plan):
    """Reserve for the largest selected compiler, including mixed GPU/host sets."""
    return max((int(row["fields"]["environment"]["MOJOLEARN_COMPILE_JOBS"])
                for row in plan["groups"] if row["selected"] and row["action"] == "build"), default=1)


def parse_parallel(text):
    """--parallel value: 'auto' (None, no cap) or a positive group count."""
    if text == "auto":
        return None
    if not str(text).isdigit() or int(text) < 1:
        raise BuildError("--parallel must be 'auto' or a positive integer")
    return int(text)


def group_cap(parallel, jobs, env_value):
    """The tightest of --parallel N, --jobs N and MOJOLEARN_BUILD_GROUP_WORKERS, or None."""
    caps = []
    if parallel is not None:
        caps.append(parallel)
    if jobs is not None:
        if jobs < 1:
            raise BuildError("--jobs must be positive")
        caps.append(jobs)
    if env_value not in (None, ""):
        if not str(env_value).isdigit() or int(env_value) < 1:
            raise BuildError(f"{GROUP_WORKERS_ENV} must be a positive integer")
        caps.append(int(env_value))
    return min(caps) if caps else None


def group_needs(all_rows, wanted, per_job_gib=None, floor_bytes=0):
    """Packing need (bytes) and its source for each wanted row, in order.

    A recorded need (last measured peak plus headroom) is used as is. A group
    with no record takes --memory-per-job-gib when given, else the largest
    recorded need among all groups of its set (identical and host compilers
    differ), else the platform floor."""
    set_max = {}
    for row in all_rows:
        need = int(row.get("recorded_need_bytes") or 0)
        set_max[row["mode"]] = max(set_max.get(row["mode"], 0), need)
    out = []
    for row in wanted:
        need = int(row.get("recorded_need_bytes") or 0)
        if need > 0:
            out.append((need, "recorded"))
        elif per_job_gib is not None:
            out.append((int(per_job_gib * 2 ** 30), "memory-per-job-gib"))
        elif set_max.get(row["mode"], 0) > 0:
            out.append((set_max[row["mode"]], "set-max"))
        else:
            out.append((int(floor_bytes), "group-floor"))
    if any(need < 1 for need, _ in out):
        raise BuildError("a group has no memory need; pass --memory-per-job-gib <GiB>")
    return out


def source_bytes(row):
    """Bytes of a group's source closure (missing files count zero)."""
    repo = row["fields"].get("repo")
    total = 0
    for name in row["fields"]["source"]["files"] if repo else ():  # driver-loop: over closure files, stat only
        try:
            total += (Path(repo) / name).stat().st_size
        except OSError:
            pass
    return total


def estimated_seconds(rows):
    """Compile-seconds estimate per row, for ordering only.

    A group's recorded compiler seconds when it has them; otherwise its source
    closure bytes, scaled by the recorded seconds per byte of the groups that
    have both. The first run (no records) orders by source bytes alone."""
    sizes = [source_bytes(row) for row in rows]
    timed = [(float(row.get("recorded_seconds") or 0), size) for row, size in zip(rows, sizes)
             if float(row.get("recorded_seconds") or 0) > 0 and size > 0]
    rate = sum(t for t, _ in timed) / sum(b for _, b in timed) if timed else None
    out = []
    for row, size in zip(rows, sizes):  # driver-loop: over development groups
        seconds = float(row.get("recorded_seconds") or 0)
        out.append(seconds if seconds > 0 else size * rate if rate else float(size))
    return out


def packing_order(rows, needs):
    """Longest estimated compile first, then biggest need, then larger closures.

    Longest first keeps the end of the run from waiting on one long group
    started late, with the cores the earlier groups freed sitting idle."""
    seconds = estimated_seconds(rows)
    order = sorted(range(len(rows)), key=lambda i: (-seconds[i], -needs[i],
                                                    -len(rows[i]["fields"]["source"]["files"]),
                                                    rows[i]["mode"], rows[i]["stem"]))
    return [rows[i] for i in order], [needs[i] for i in order]


def next_launches(queue_needs, running_needs, budget_bytes, slots, ceiling_bytes):
    """Indexes into queue_needs (biggest first) to start now.

    Groups are packed while the reserved total stays within budget_bytes and
    at most `slots` run. When nothing runs and the head of the queue is larger
    than the budget, it starts alone if it fits ceiling_bytes (80% of total
    memory); smaller groups never jump ahead of it, so the biggest groups
    still start first."""
    if not queue_needs:
        return []
    if not running_needs and queue_needs[0] > budget_bytes:
        if queue_needs[0] > ceiling_bytes:
            raise BuildError(f"a group compiler reservation of {queue_needs[0] / 2 ** 30:.1f} GiB cannot fit "
                             f"{ceiling_bytes / 2 ** 30:.1f} GiB; free memory or use a larger machine, or pass "
                             "--memory-per-job-gib <GiB> if you measured a smaller peak")
        return [0]
    chosen, reserved, count = [], sum(running_needs), len(running_needs)
    for index, need in enumerate(queue_needs):
        if count >= slots:
            break
        if reserved + need <= budget_bytes:
            chosen.append(index)
            reserved += need
            count += 1
    return chosen


def current_budget(baseline_bytes, available_now, running_needs):
    """Memory the scheduler may reserve now.

    available_now already excludes what running groups use, which is at most
    what they reserved; adding their reservation back and capping at the
    start-of-run budget shrinks the budget only when other load appeared."""
    return min(int(baseline_bytes), int(MEMORY_FRACTION * available_now) + int(sum(running_needs)))


def launch_threads(cores, running_threads, picks_jobs):
    """Compiler threads of the groups started now, in start order.

    picks_jobs holds, per candidate in start order, its fixed compiler thread
    count (the caller's MOJOLEARN_COMPILE_JOBS, AMD's) or None when the
    scheduler chooses. Each candidate takes its fixed count or one thread
    while free cores remain; the first candidate that does not fit and every
    one after it wait for a later launch (a lone group starts even when it
    alone exceeds the cores). The free cores left over go to the
    scheduler-chosen groups, one more each from the first (longest) on, so the
    threads of running groups sum to the core count after every launch. The
    result may be shorter than picks_jobs."""
    free = int(cores) - sum(int(t) for t in running_threads)
    threads = []
    for jobs in picks_jobs:  # driver-loop: over groups started this tick
        want = int(jobs) if jobs is not None else 1
        if want > free and (threads or running_threads):
            break
        threads.append(want)
        free -= want
    chosen = [i for i, jobs in enumerate(picks_jobs[:len(threads)]) if jobs is None]
    if chosen and free > 0:
        share, extra = divmod(free, len(chosen))
        for rank, i in enumerate(chosen):  # driver-loop: over groups started this tick
            threads[i] += share + (1 if rank < extra else 0)
    return threads


def live_need(source, need, measured, floor, ceiling):
    """A queued group's packing need after groups of its set passed in this run.

    Groups with no record of their own (platform floor or set maximum) take
    the largest need measured so far in this run for their set, the same
    set-maximum rule group_needs applies to older records: a floor estimate
    drops to it, a set maximum only rises. Recorded and --memory-per-job-gib
    needs are kept. Its rss limit stays at least the floor (guard_rss)."""
    if not measured:
        return need, source
    if source in ("group-floor", "set-max-live"):
        return min(int(measured), int(ceiling)), "set-max-live"
    if source == "set-max" and measured > need:
        return min(int(measured), int(ceiling)), "set-max"
    return need, source


RSS_GUARD_MARGIN = 1.5
RSS_GUARD_MIN_BYTES = 4 * 2 ** 30


def guard_rss(need, source, floor):
    """The rss kill limit of a group: 1.5x its packing need (a recorded peak is a
    past run's figure; a group that grows a little past it must not be killed:
    mac-n7 2026-11-07 killed four passing groups at limit == peak), never below
    4 GiB, and never below the floor for a live estimate. Packing still uses
    the need itself; this limit is only the runaway guard."""
    limit = max(int(int(need) * RSS_GUARD_MARGIN), RSS_GUARD_MIN_BYTES)
    return max(limit, int(floor)) if source == "set-max-live" else limit


def pressure_floor(available_at_start, baseline_bytes, available_now, need):
    """Available memory below which a running compiler is stopped.

    The 20% of start-of-run available memory kept for the system; lower for a
    group larger than the budget that runs alone (as the serial budget did)."""
    reserve = max(0, int(available_at_start) - int(baseline_bytes))
    return max(0, min(reserve, int(available_now) - int(need)))


def run_logged(argv, repo, env, path):
    child_env = dict(env)
    encoded = child_env.pop(GUARD_ENV, None)
    limits = json.loads(encoded) if encoded else dict(
        resource_budget(1, int(child_env.get("MOJOLEARN_COMPILE_JOBS", "1")), None),
        seconds=release_limits.get("DEVELOPMENT_BUILD_SECONDS"),
        stall_seconds=release_limits.get("DEVELOPMENT_STALL_SECONDS"))
    result = build_process.run(argv, cwd=repo, env=child_env, log=path,
                               seconds=limits["seconds"], stall_seconds=limits["stall_seconds"],
                               rss_bytes=limits["rss_bytes"],
                               min_available_bytes=limits["min_available_bytes"])
    if not result.get("complete") or (result["exit_code"] == 0 and result["status"] != "passed"):
        raise BuildError(f"build guard did not certify completion; see {path}.execution.json")
    return result["exit_code"]


def compile_group(row, repo, cache, env, evidence, toolchain=None, launch=None):
    """Build one group into the cache (or find it there).

    toolchain: a ToolchainProbe shared by the scheduler's workers (default: a
    private read). launch: the scheduler's view at submission, recorded in the
    group receipt (groups_running_at_launch, threads_running_at_launch,
    cores, submitted = time.monotonic() at submission)."""
    started = time.monotonic()
    result = {"mode": row["mode"], "stem": row["stem"], "key": row["key"],
              "state": "failed", "exit_code": 1}
    if launch:
        result.update({k: v for k, v in launch.items() if k != "submitted"})
        if launch.get("submitted") is not None:
            result["queue_wait_seconds"] = max(0.0, started - launch["submitted"])
    limits = json.loads(env.get(GUARD_ENV, "{}"))
    try:
        with lock(cache / "locks" / (row["key"] + ".lock"), limits.get("lock_seconds")):
            hit = cached_artifact(cache, row)
            if hit:
                result.update(state="reused", exit_code=0)
                if row.get("auto_compile_jobs"):
                    record_compile_jobs(cache, row, row.get("compile_jobs") or 1)
            else:
                target = cache / "artifacts" / row["key"]
                work = cache / "work" / row["key"]
                output = work / "output"
                output.mkdir(parents=True, exist_ok=True)
                log = evidence / (row["mode"] + "-" + row["stem"] + ".log")
                execution = log.with_suffix(log.suffix + ".execution.json")
                artifact = output / (row["stem"] + ".so")
                # Neither a failed earlier attempt's output nor its verdict may
                # certify a new compiler that returns zero without writing.
                artifact.unlink(missing_ok=True)
                execution.unlink(missing_ok=True)
                effective = build_environment(row["mode"], env, row["fields"]["platform"])
                result["workers_at_launch"] = int(effective["MOJOLEARN_COMPILE_JOBS"])
                command = group_command(row["mode"], row["stem"], output, work / "roots")
                result.update(log=str(log), command=command, execution_receipt=str(execution))
                compile_started = time.monotonic()
                code = run_logged(command, repo, effective, log)
                compiled = time.monotonic()
                result.update(exit_code=code, compile_seconds=compiled - compile_started)
                if execution.is_file():
                    result["execution"] = json.loads(execution.read_text())
                    record_peak(cache, row, result["execution"], result["workers_at_launch"])
                    result["peak_rss_bytes"] = result["execution"].get("peak_rss_bytes")
                    if code and result["execution"].get("reason"):
                        result["reason"] = result["execution"]["reason"]
                if build_process.cancelled():
                    raise KeyboardInterrupt("development build cancelled before cache publication")
                if code == 0:
                    # This group's own fields only; see current_group_fields.
                    facts = toolchain.facts_after(compiled) if toolchain else toolchain_facts(repo)
                    current = current_group_fields(row, repo, cache, env, facts)
                    if current is None or digest(current) != row["key"]:
                        result.update(exit_code=1, reason=input_change_reason(
                            "build inputs changed during compilation", row["fields"], current or {}))
                    elif not artifact.is_file() or artifact.stat().st_size == 0:
                        result.update(exit_code=1, reason="compiler returned success without nonempty declared output")
                    else:
                        target.mkdir(parents=True, exist_ok=True)
                        sha = bincache.sha256_file(artifact)
                        shutil.copy2(artifact, target / artifact.name)
                        if build_process.cancelled():
                            raise KeyboardInterrupt("development build cancelled before cache receipt")
                        result.update(state="built", sha256=sha)
                        atomic_json(target / "receipt.json", dict(result, fields=row["fields"]))
                        if build_process.cancelled():
                            (target / "receipt.json").unlink(missing_ok=True)
                            raise KeyboardInterrupt("development build cancelled during cache publication")
                        # The plan after this run finds the build under the
                        # keyed thread count the scheduler chose.
                        record_compile_jobs(cache, row, result["workers_at_launch"])
    except KeyboardInterrupt as error:
        result.update(state="interrupted", exit_code=130, reason=str(error))
    except (BuildError, ValueError, OSError, subprocess.SubprocessError) as error:
        result.update(state="failed", exit_code=1, reason=str(error))
    result["seconds"] = time.monotonic() - started
    atomic_json(evidence / (row["mode"] + "-" + row["stem"] + ".json"), result)
    return result


def build_support(row, repo, cache, env, evidence):
    """Compile one non-group artifact into the cache, receipted like a group."""
    started = time.monotonic()
    result = {"support": row["name"], "stem": row["stem"], "key": row["key"],
              "state": "failed", "exit_code": 1}
    limits = json.loads(env.get(GUARD_ENV, "{}"))
    try:
        with lock(cache / "locks" / (row["key"] + ".lock"), limits.get("lock_seconds")):
            if cached_artifact(cache, row):
                result.update(state="reused", exit_code=0)
            else:
                c_facts = row["fields"]["c_toolchain"]
                if not c_facts.get("available"):
                    raise BuildError(f"C compiler {c_facts.get('compiler')!r} not found; "
                                     "portable math needs cc (Linux) or clang (macOS)")
                output = cache / "work" / row["key"] / "output" / row["file"]
                output.parent.mkdir(parents=True, exist_ok=True)
                output.unlink(missing_ok=True)
                log = evidence / ("support-" + row["name"] + ".log")
                command = [sys.executable, "-I", "-c",
                           "import json, pathlib, sys; sys.path.insert(0, sys.argv[1]); import stage; "
                           "print(json.dumps(stage.build(pathlib.Path(sys.argv[2]), sys.argv[3])))",
                           str(repo / "packaging/portable_math"), str(output), c_facts["compiler"]]
                result.update(log=str(log), command=command)
                code = run_logged(command, repo, env, log)
                result["exit_code"] = code
                if build_process.cancelled():
                    raise KeyboardInterrupt("development build cancelled before cache publication")
                if code == 0:
                    plat = row["fields"]["platform"]
                    current = support_rows(repo, cache, plat, c_toolchain_facts(env, plat))[0]
                    if current["key"] != row["key"]:
                        result.update(exit_code=1, reason=input_change_reason(
                            "build inputs changed during compilation", row["fields"], current["fields"]))
                    elif not output.is_file() or output.stat().st_size == 0:
                        result.update(exit_code=1, reason="compiler returned success without nonempty declared output")
                    else:
                        target = cache / "artifacts" / row["key"]
                        target.mkdir(parents=True, exist_ok=True)
                        sha = bincache.sha256_file(output)
                        shutil.copy2(output, target / row["file"])
                        result.update(state="built", sha256=sha)
                        atomic_json(target / "receipt.json", dict(result, fields=row["fields"]))
                        if build_process.cancelled():
                            (target / "receipt.json").unlink(missing_ok=True)
                            raise KeyboardInterrupt("development build cancelled during cache publication")
    except KeyboardInterrupt as error:
        result.update(state="interrupted", exit_code=130, reason=str(error))
    except (BuildError, ValueError, OSError, subprocess.SubprocessError) as error:
        result.update(state="failed", exit_code=1, reason=str(error))
    result["seconds"] = time.monotonic() - started
    atomic_json(evidence / ("support-" + row["name"] + ".json"), result)
    return result


def install_support(row, source, repo):
    """Place one support artifact at the fixed path the runtime loads."""
    destination = repo / "python/mojolearn" / row["install"]
    if destination.is_file() and bincache.sha256_file(destination) == bincache.sha256_file(source):
        return str(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp-" + uuid.uuid4().hex)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return str(destination)


def activate(plan, repo, cache, lock_seconds=None):
    """Publish only complete current sets; every indexed binary is immutable.

    The support artifacts the runtime loads outside the index (portable
    math) are part of every set: activation refuses without them."""
    rows = plan["groups"]
    support = plan.get("support")
    if support is None:
        raise BuildError("plan has no support artifacts; recompute it with this tool")
    sources = {row["key"]: cached_artifact(cache, row) for row in rows + support}
    missing = [row["mode"] + ":" + row["stem"] for row in rows if sources[row["key"]] is None]
    missing += ["support:" + row["name"] for row in support if sources[row["key"]] is None]
    if missing:
        return {"state": "incomplete", "missing": missing, "activated": []}
    modes = sorted({row["mode"] for row in rows})
    for row in rows:
        out = repo / "python/mojolearn" / row["mode"]
        shadows = [str(out / (n + ".so")) for n in row["members"] if (out / (n + ".so")).exists()]
        if shadows:
            raise BuildError("per-binding files would shadow the development index: " + ", ".join(shadows))
    indexes = {}
    for mode in modes:
        out = repo / "python/mojolearn" / mode
        files = {}
        for row in [r for r in rows if r["mode"] == mode]:
            source = sources[row["key"]]
            relative = ".development/" + row["key"] + "/" + source.name
            destination = out / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if bincache.sha256_file(destination) != bincache.sha256_file(source):
                    raise BuildError(f"immutable installed artifact changed: {destination}")
            else:
                temporary = destination.with_name(destination.name + ".tmp-" + uuid.uuid4().hex)
                shutil.copy2(source, temporary)
                os.replace(temporary, destination)
            files[relative] = row["members"]
        indexes[out] = {"format": merged.INDEX_FORMAT, "layout": "development", "files": files,
                        "provenance": {r["stem"]: r["key"] for r in rows if r["mode"] == mode}}
    installed = {row["name"]: install_support(row, sources[row["key"]], repo) for row in support}
    # No index changes until every artifact in every requested set is staged.
    with contextlib.ExitStack() as stack:
        for out in sorted(indexes):
            # Same lock path/protocol as merged.index_lock, with a finite wait.
            key = hashlib.sha256(os.path.realpath(out).encode()).hexdigest()[:16]
            path = Path(tempfile.gettempdir()) / f"mojolearn-merged-index-{key}.lock"
            stack.enter_context(lock(path, lock_seconds))
        if build_process.cancelled():
            raise KeyboardInterrupt("development build cancelled before index activation")
        for out, index in indexes.items():
            if build_process.cancelled():
                raise KeyboardInterrupt("development build cancelled during index activation")
            atomic_json(out / "_mojolearn_merged.json", index)
    return {"state": "activated", "missing": [], "activated": modes, "support": installed}


# ---------------------------------------------------------------------------
# --layout release: the macOS wheel's compile, through the one Mac cache
# ---------------------------------------------------------------------------
#
# WHY (2026-11-07). The Mac compiled the frozen 0.8.66 commit twice: this tool
# built its 79 development groups (development layout, its own cache and key)
# and then tools/release.py built the 15 merged release groups of the same
# commit again inside the macOS wheel (8582 s wall, 25,700+ compiler seconds,
# every group a miss). No key could match, whatever the cache directory:
#
#   key field            development layout             macOS wheel (release)
#   key system           KEY_SCHEMA (this file), cache  tools/bincache.py closure-v2,
#                        ~/.cache/mojolearn-development release-scoped macos-bincache/
#   groups               bindings/development_groups    bindings/merged_groups.json:
#                        .json: 79 `_mojolearn_dev_*`   15 `_mojolearn_native*` files
#                        files, one member each         of 1..10 members
#   merged root text     one member's PyInit            every member's PyInit
#   merged_build.sh args gpu|host --layout development  gpu|host --group <stem>
#                        --group S --outdir --rootdir
#   declared output      <cache>/work/<key>/output      python/mojolearn/<set>/<stem>.so
#   MOJOLEARN_COMPILE_   chosen by the scheduler (free  1 (tools/mac_slot.py run pins
#   JOBS (keyed)         cores at launch)               it; step_macos_build passes it)
#   repository path      where the build ran            the release's source checkout
#   (rpath, keyed)       (e.g. ~/mojolearn-wt/...)      (the main checkout at the commit)
#   build_env            v1 sweep: also keys            closure-v2: only what a keyed
#                        MOJOLEARN_MERGED_PYTHON and    input reads (COMPILE_JOBS,
#                        MOJOLEARN_RUNPOD_KEY_FILE      NUMERIC_MODE, PACKAGE_BYTE_LM)
#   toolchain            compiler binary sha256,        pixi.lock packages, installed
#                        version, packages, Xcode       toolchain check, Xcode
#
# Equal in both: MOJOLEARN_PACKAGE_BYTE_LM=1, MOJOLEARN_NUMERIC_MODE=identical,
# the host column (cpu, no accelerator), the Metal flags per flags class.
# The groups, root text, args, output and job count change the compile itself
# (or are keyed because they might), so the fix is not a looser key: on macOS
# this tool now runs the release compile itself. Every release group of the
# selected sets runs `tools/bincache.py build ./bindings/merged_build.sh
# <kind> --group <stem>` with the environment packaging/macos/
# build_release_wheel.sh gives it (bincache.macos_merged_group) and the
# release's keyed compile settings (bincache.MACOS_RELEASE_COMPILE_ENV),
# writing into the one Mac cache (bincache.mac_cache_dir). The release's
# macOS build seeds from that cache, so when it runs in the same checkout its
# keys are these keys and every group is placed, not compiled.
# The memory-packed scheduler above runs the compiles (one compiler thread
# each, groups in parallel up to the cores and 80% of available memory);
# groups already in the cache are placed in parallel first.
RELEASE_SCHEMA = "mojolearn.development-build.release-layout.v1"
RELEASE_KEY_SCHEMA = "mojolearn.development-group.release-layout.v1"
LAYOUTS = ("development", "release")
MERGED_SCRIPT = "./bindings/merged_build.sh"


def default_layout(plat=None):
    """macOS builds the release layout (the wheel's compile); Linux keeps the
    development groups, since the Linux release legs compile on their own
    boxes with their own accelerator targets."""
    return "release" if (plat or merged.default_platform()) == "darwin" else "development"


def release_environment(mode, env, plat="darwin"):
    """The environment one release-layout group of `mode` starts from: the
    caller's, with the macOS release build's keyed compile settings, through
    build_environment's refusals. A caller value that differs from the
    release's would key another compile, so it is refused, not overridden."""
    env = dict(env)
    for name, value in bincache.MACOS_RELEASE_COMPILE_ENV.items():
        have = env.get(name)
        if have not in (None, value):
            raise BuildError(f"--layout release compiles exactly as the macOS wheel does ({name}={value}); "
                             f"{name}={have} would key another compile (use --layout development)")
        env[name] = value
    return build_environment(mode, env, plat)


def release_group_environment(row, env):
    """The exact environment `bincache.py build` sees for this release row."""
    _kind, _output, genv = bincache.macos_merged_group(
        release_environment(row["mode"], env, row["fields"]["platform"]), row["mode"], row["stem"])
    genv["MOJOLEARN_BINCACHE_DIR"] = row["mac_cache"]
    return genv


def release_archive(mac_cache, key, fields, env):
    """The verified archive of KEY in the one Mac cache or a seed directory, or None."""
    for directory in [Path(mac_cache)] + bincache.seed_dirs(env):
        candidate = directory / (key + ".tar.gz")
        if candidate.is_file():
            try:
                bincache.verify_archive(str(candidate), key, fields)
                return candidate
            except bincache.Reject:
                continue
    return None


def release_plan(repo, cache, mac_cache, modes, env, selection=(), facts=None, plat=None, c_facts=None):
    """The release-layout plan: one row per merged release group of each set,
    keyed by tools/bincache.py exactly as the macOS wheel build keys it."""
    repo, cache, mac_cache = Path(repo).resolve(), Path(cache).resolve(), Path(mac_cache).expanduser().resolve()
    plat = plat or merged.default_platform()
    if plat != "darwin":
        raise BuildError("--layout release is the macOS wheel's compile; Linux release groups compile on the "
                         "release legs (use --layout development here)")
    if mac_cache.is_relative_to(repo) or cache.is_relative_to(repo):
        raise BuildError("keep the caches outside the source tree")
    facts = facts if facts is not None else toolchain_facts(repo)
    script = repo / MERGED_SCRIPT
    rows = []
    for mode in modes:
        if mode not in ("identical", "host"):
            raise BuildError("development builds support identical and host only")
        effective = release_environment(mode, env, plat)
        byte_lm = effective["MOJOLEARN_PACKAGE_BYTE_LM"] == "1"
        for stem, flags_class, members in merged.plan(mode, plat, byte_lm, "release"):
            kind, output, genv = bincache.macos_merged_group(effective, mode, stem)
            genv["MOJOLEARN_BINCACHE_DIR"] = str(mac_cache)
            args = [kind, "--group", stem]
            try:
                bfields, closure = bincache.build_fields(repo, script, args, genv, "local", [output], "sh",
                                                         local=True, plat=plat)
            except bincache.UnkeyableBuild as error:
                raise BuildError(f"{mode}:{stem} cannot be keyed: {error}") from error
            key = bincache.key_of(bfields)
            fields = {"schema": RELEASE_KEY_SCHEMA, "repo": str(repo), "platform": plat, "layout": "release",
                      "mode": mode, "stem": stem, "flags_class": flags_class, "members": members,
                      "source": {"scope": "closure", "files": sorted(closure or {})},
                      "environment": dict(bfields["build_env"],
                                          MOJOLEARN_COMPILE_JOBS=genv["MOJOLEARN_COMPILE_JOBS"]),
                      "bincache": bfields}
            row = {"mode": mode, "stem": stem, "members": members, "flags_class": flags_class,
                   "kind": kind, "output": output, "command": [sys.executable, "tools/bincache.py", "build",
                                                               MERGED_SCRIPT, *args],
                   "mac_cache": str(mac_cache), "key": key, "fields": fields,
                   "selected": selected_group(mode, stem, selection),
                   "auto_compile_jobs": False, "compile_jobs": int(genv["MOJOLEARN_COMPILE_JOBS"])}
            row["action"] = ("reuse" if facts.get("available") and release_archive(mac_cache, key, bfields, genv)
                             else "build")
            row["recorded_need_bytes"] = recorded_need(cache, mode, stem)
            row["recorded_seconds"] = recorded_seconds(cache, mode, stem)
            rows.append(row)
    for group in selection:
        if not any(selected_group(r["mode"], r["stem"], [group]) for r in rows):
            raise BuildError(f"group selection matches nothing: {group}")
    support = support_rows(repo, cache, plat, c_facts if c_facts is not None else c_toolchain_facts(env, plat))
    return {"schema": RELEASE_SCHEMA, "layout": "release", "platform": plat, "repo": str(repo),
            "cache": str(cache), "mac_cache": str(mac_cache),
            "toolchain_available": facts.get("available", False), "groups": rows, "support": support,
            "counts": {"groups": len(rows), "reuse": sum(r["action"] == "reuse" for r in rows),
                       "build_selected": sum(r["action"] == "build" and r["selected"] for r in rows),
                       "build_unselected": sum(r["action"] == "build" and not r["selected"] for r in rows)}}


def provenance_row(directory):
    """(outcome, key) of the last tools/bincache.py provenance row in DIRECTORY, or ("", "")."""
    try:
        lines = (Path(directory) / "provenance.tsv").read_text().splitlines()
    except OSError:
        return "", ""
    parts = lines[-1].split("\t") if lines else []
    return (parts[2], parts[3]) if len(parts) >= 4 else ("", "")


def compile_release_group(row, repo, cache, env, evidence, toolchain=None, launch=None):
    """Place or compile one release group through the one Mac cache.

    The same call signature as compile_group (the schedulers take either).
    tools/bincache.py keys the group in the child; that key must equal the
    plan's, or the group fails (its environment or inputs moved). A placement
    is not a compile attempt (no `command` in the result) and records no
    peak; a compile records its peak for the next plan's packing."""
    started = time.monotonic()
    result = {"mode": row["mode"], "stem": row["stem"], "key": row["key"], "layout": "release",
              "output": row["output"], "state": "failed", "exit_code": 1}
    if launch:
        result.update({k: v for k, v in launch.items() if k != "submitted"})
        if launch.get("submitted") is not None:
            result["queue_wait_seconds"] = max(0.0, started - launch["submitted"])
    limits = json.loads(env.get(GUARD_ENV, "{}"))
    try:
        with lock(Path(cache) / "locks" / (row["key"] + ".lock"), limits.get("lock_seconds")):
            name = row["mode"] + "-" + row["stem"]
            log = evidence / (name + ".log")
            execution = log.with_suffix(log.suffix + ".execution.json")
            provenance = evidence / "bincache" / name
            execution.unlink(missing_ok=True)
            shutil.rmtree(provenance, ignore_errors=True)
            output = Path(repo) / row["output"]
            # The cache never replaces an existing file; build_release_wheel.sh
            # removes the declared output first for the same reason.
            output.unlink(missing_ok=True)
            genv = release_group_environment(row, env)
            genv["MOJOLEARN_BINCACHE_OUT"] = str(provenance)
            result.update(log=str(log), execution_receipt=str(execution), workers_at_launch=row["compile_jobs"])
            code = run_logged(row["command"], repo, genv, log)
            outcome, key = provenance_row(provenance)
            compiled = outcome.startswith(("miss", "rejected", "bypass"))
            result.update(exit_code=code, bincache_outcome=outcome, bincache_key=key)
            if compiled:
                result["command"] = row["command"]
            if execution.is_file():
                result["execution"] = json.loads(execution.read_text())
                if compiled:
                    record_peak(cache, row, result["execution"], row["compile_jobs"])
                    result["peak_rss_bytes"] = result["execution"].get("peak_rss_bytes")
                if code and result["execution"].get("reason"):
                    result["reason"] = result["execution"]["reason"]
            if build_process.cancelled():
                raise KeyboardInterrupt("development build cancelled during a release group")
            if code == 0:
                if key != row["key"]:
                    result.update(exit_code=1, reason=(
                        f"tools/bincache.py keyed this group {key[:16] or '-'}, the plan {row['key'][:16]}: "
                        "its environment or inputs changed since the plan"))
                elif not output.is_file() or output.stat().st_size == 0:
                    result.update(exit_code=1, reason="build returned success without the declared output")
                else:
                    result.update(state="reused" if outcome == "hit" else "built",
                                  sha256=bincache.sha256_file(output),
                                  cached=outcome == "hit" or outcome.endswith("+built-cached"))
    except KeyboardInterrupt as error:
        result.update(state="interrupted", exit_code=130, reason=str(error))
    except (BuildError, ValueError, OSError, subprocess.SubprocessError) as error:
        result.update(state="failed", exit_code=1, reason=str(error))
    result["seconds"] = time.monotonic() - started
    atomic_json(evidence / (row["mode"] + "-" + row["stem"] + ".json"), result)
    return result


def place_release_groups(rows, cache, env, evidence, limits, report, futures):
    """Place the groups the one Mac cache already holds, all at once (no
    compiler runs; each is an archive verify and a file write)."""
    if not rows:
        return
    cores = os.cpu_count() or 1
    place_env = dict(env, **{GUARD_ENV: json.dumps(dict(resource_budget(1, 1, None), **limits))})
    with ThreadPoolExecutor(max_workers=max(1, min(len(rows), cores))) as pool:
        submitted = [pool.submit(compile_release_group, row, REPO, cache, place_env, evidence) for row in rows]
        futures.extend(submitted)
        for future in as_completed(submitted):  # driver-loop: over placed release groups
            collect(report, future.result(), evidence)


def current_release_output(row, repo):
    """True when the package file of ROW holds the bytes of its key's archive."""
    output = Path(repo) / row["output"]
    archive = Path(row["mac_cache"]) / (row["key"] + ".tar.gz")
    try:
        _manifest, blobs = bincache.verify_archive(str(archive), row["key"], row["fields"]["bincache"])
        data = blobs.get(row["output"])
        return data is not None and output.is_file() and bincache.sha256_file(output) == \
            hashlib.sha256(data).hexdigest()
    except (bincache.Reject, OSError):
        return False


def activate_release(plan, repo, cache, lock_seconds=None):
    """Index every complete set exactly as the macOS wheel build indexes it.

    The release layout cannot stage outside the package tree (the declared
    output is a key field), so the group files are already in place; this
    checks each holds its key's archive bytes (a file the cache did not keep
    is refused), installs the support artifacts and rewrites each set's
    `_mojolearn_merged.json` from the plan, one atomic replacement per set."""
    rows = plan["groups"]
    support = plan.get("support")
    if support is None:
        raise BuildError("plan has no support artifacts; recompute it with this tool")
    sources = {row["key"]: cached_artifact(cache, row) for row in support}
    missing = [row["mode"] + ":" + row["stem"] for row in rows if not current_release_output(row, repo)]
    missing += ["support:" + row["name"] for row in support if sources[row["key"]] is None]
    if missing:
        return {"state": "incomplete", "missing": missing, "activated": []}
    modes = sorted({row["mode"] for row in rows})
    for row in rows:
        out = Path(repo) / "python/mojolearn" / row["mode"]
        shadows = [str(out / (n + ".so")) for n in row["members"] if (out / (n + ".so")).exists()]
        if shadows:
            raise BuildError("per-binding files would shadow the merged index: " + ", ".join(shadows))
    installed = {row["name"]: install_support(row, sources[row["key"]], repo) for row in support}
    with contextlib.ExitStack() as stack:
        outs = {mode: Path(repo) / "python/mojolearn" / mode for mode in modes}
        for mode in modes:
            key = hashlib.sha256(os.path.realpath(outs[mode]).encode()).hexdigest()[:16]
            stack.enter_context(lock(Path(tempfile.gettempdir()) / f"mojolearn-merged-index-{key}.lock",
                                     lock_seconds))
        if build_process.cancelled():
            raise KeyboardInterrupt("development build cancelled before index activation")
        for mode in modes:
            files = {row["stem"] + ".so": row["members"] for row in rows if row["mode"] == mode}
            atomic_json(outs[mode] / "_mojolearn_merged.json", {"format": merged.INDEX_FORMAT, "files": files})
    return {"state": "activated", "missing": [], "activated": modes, "support": installed}


def release_build(args, modes, env, cache, mac_cache, evidence, limits, report, cap, futures):
    """`build --layout release`: place what the one Mac cache holds, compile
    the rest with the memory-packed scheduler, then index the sets."""
    report.update(schema=RELEASE_SCHEMA, layout="release", mac_cache=str(mac_cache))
    plan = release_plan(REPO, cache, mac_cache, modes, env, args.group)
    atomic_json(evidence / "plan.json", plan)
    summary(plan)
    if not plan["toolchain_available"]:
        raise BuildError("install the pinned default pixi environment before building")
    selected = [r for r in plan["groups"] if r["selected"]]
    wanted = build_submission_order(r for r in selected if r["action"] == "build")
    report.update(plan=plan, jobs=0, state="building", compile_attempts=0)
    place_release_groups([r for r in selected if r["action"] == "reuse"], cache, env, evidence, limits,
                         report, futures)
    if wanted and cap == 1:
        serial_build(plan, wanted, args, env, cache, evidence, limits, report, compile_release_group)
    elif wanted:
        parallel_build(plan, wanted, args, env, cache, evidence, limits, report, cap, futures,
                       compile_release_group)
    if any(r["exit_code"] for r in report["results"]):
        raise BuildError("one or more release groups failed; the merged indexes were not rewritten")
    uncached = [r["mode"] + ":" + r["stem"] for r in report["results"] if r.get("cached") is False]
    if uncached:
        print("release layout: NOT CACHED (the release will compile these): " + ", ".join(uncached))
    build_support_rows(plan, env, cache, evidence, limits, report)
    final_plan = release_plan(REPO, cache, mac_cache, modes, env, args.group)
    if [r["key"] for r in final_plan["groups"] + final_plan["support"]] != \
            [r["key"] for r in plan["groups"] + plan["support"]]:
        before = {r.get("mode", "support") + ":" + r["stem"]: r["fields"] for r in plan["groups"] + plan["support"]}
        after = {r.get("mode", "support") + ":" + r["stem"]: r["fields"]
                 for r in final_plan["groups"] + final_plan["support"]}
        raise BuildError(input_change_reason(
            "build inputs changed; refusing to index mixed-source outputs", before, after))
    if build_process.cancelled():
        raise KeyboardInterrupt("development build cancelled before activation")
    report.update(activate_release(final_plan, REPO, cache, limits["lock_seconds"]))
    atomic_json(evidence / "result.json", report)
    if args.json:
        atomic_json(args.json, report)
    print(f"{report['state']}: evidence {evidence}")
    return 0 if report["state"] == "activated" else 3


def summary(plan):
    for row in plan["groups"]:
        selection = "selected" if row["selected"] else "unselected"
        scope = row["fields"]["source"]["scope"]
        print(f"{row['mode']} {row['stem']} {row['action']} {selection} {scope} {row['key'][:12]}")
    for row in plan.get("support", []):
        print(f"support {row['stem']} {row['action']} {row['key'][:12]}")
    print("counts " + json.dumps(plan["counts"], sort_keys=True))


def collect(report, result, evidence):
    report["results"].append(result)
    report["compile_attempts"] += int("command" in result)
    atomic_json(evidence / "result.json", report)
    print(f"{result['mode']} {result['stem']} {result['state']} rc={result['exit_code']}")


def host_facts(report, resources=None):
    """Cores and memory of this host, recorded in the summary for the next packing."""
    cores, available = resources or available_resources()
    report.update(host_cores=cores, host_memory_bytes=physical_memory(), host_available_bytes_at_start=available)
    return cores, available


def serial_build(plan, wanted, args, env, cache, evidence, limits, report, compile=None):
    """--parallel 1: one compiler at a time under the serial budget.

    compile: the group compiler (default compile_group; the release layout
    passes compile_release_group)."""
    compile = compile or compile_group
    workers = planned_compiler_workers(plan)
    recorded = max(r["recorded_need_bytes"] for r in wanted)
    floor = GROUP_FLOOR_BYTES.get(plan["platform"], 0)
    budget = resource_budget(1, workers, args.memory_per_job_gib, floor_bytes=floor, recorded_bytes=recorded)
    host_facts(report, (budget["cores"], budget["available_bytes"]))
    env[GUARD_ENV] = json.dumps(dict(budget, **limits))
    report.update(resource_budget=budget, jobs=budget["jobs"], groups_concurrent_max=1,
                  scheduler={"parallel": 1, "slots": 1})
    atomic_json(evidence / "result.json", report)
    toolchain = ToolchainProbe(REPO)
    for row in wanted:  # driver-loop: over development groups, one compiler at a time
        launch = {"groups_running_at_launch": 1, "threads_running_at_launch": row["compile_jobs"],
                  "cores": budget["cores"], "submitted": time.monotonic()}
        result = compile(row, REPO, cache, env, evidence, toolchain, launch)
        collect(report, result, evidence)


def parallel_build(plan, wanted, args, env, cache, evidence, limits, report, cap, futures, compile=None):
    """A sliding window over the groups: start one whenever memory and a core are free.

    Longest estimated compile first. Each start re-plans compiler threads so
    the running groups' threads sum to the core count (launch_threads); a
    group that finishes returns its memory and cores to the next start at
    once. Every launched future is appended to `futures` so the caller keeps
    the verdicts of started workers when the run stops. compile: the group
    compiler (default compile_group; the release layout passes
    compile_release_group)."""
    compile = compile or compile_group
    cores, available_at_start = host_facts(report)
    total = report["host_memory_bytes"]
    baseline = int(MEMORY_FRACTION * available_at_start)
    ceiling = max(baseline, int(MEMORY_FRACTION * total))
    floor = GROUP_FLOOR_BYTES.get(plan["platform"], 0)
    needs = group_needs(plan["groups"], wanted, args.memory_per_job_gib, floor)
    gib = 2 ** 30
    for row, (need, source) in zip(wanted, needs):
        if need > ceiling:
            raise BuildError(
                f"group {row['mode']}:{row['stem']} needs {need / gib:.1f} GiB ({source}) and cannot fit: this "
                f"machine allows {ceiling / gib:.1f} GiB (80% of {total / gib:.1f} GiB) and "
                f"{baseline / gib:.1f} GiB is available now; free memory or use a larger machine. If you "
                f"measured a smaller peak for these groups, pass --memory-per-job-gib <GiB>.")
    # Fixed compiler thread counts (caller's MOJOLEARN_COMPILE_JOBS, AMD's)
    # take their cores through launch_threads; at most one group per core.
    fixed = max((r["compile_jobs"] for r in wanted if not r.get("auto_compile_jobs")), default=1)
    slots = cores if cap is None else max(1, min(cores, cap))
    queue, queue_needs = packing_order(wanted, [need for need, _ in needs])
    sources = {(r["mode"], r["stem"]): source for r, (_, source) in zip(wanted, needs)}
    measured = {}  # mode -> largest need measured in this run (passed peak plus headroom)
    scheduler = {"parallel": "auto" if cap is None else cap, "slots": slots, "budget_bytes": baseline,
                 "ceiling_bytes": ceiling, "fixed_compiler_workers": fixed,
                 "memory_fraction": MEMORY_FRACTION, "order": "longest-first",
                 "threads": "sum-to-cores"}
    report.update(scheduler=scheduler, resource_budget=scheduler, jobs=slots, groups_concurrent_max=0,
                  threads_running_max=0)
    atomic_json(evidence / "result.json", report)
    toolchain = ToolchainProbe(REPO)
    running = {}
    with ThreadPoolExecutor(max_workers=slots) as pool:
        try:
            while queue or running:  # driver-loop: over scheduler ticks of development groups
                if queue and not build_process.cancelled():
                    available_now = available_resources()[1]
                    reserved = [need for _, need, _ in running.values()]
                    threads_now = [r["compile_jobs"] for r, _, _ in running.values()]
                    budget = current_budget(baseline, available_now, reserved)
                    free = cores - sum(threads_now)
                    window = min(slots, len(running) + max(0, free)) if running else slots
                    picks = next_launches(queue_needs, reserved, budget, window, ceiling)
                    threads = launch_threads(cores, threads_now, [
                        None if queue[i].get("auto_compile_jobs") else queue[i]["compile_jobs"] for i in picks])
                    starts = list(zip(picks, threads))
                    taken = [(queue[i], queue_needs[i], jobs) for i, jobs in starts]
                    for index, _ in sorted(starts, reverse=True):  # driver-loop: over groups started this tick
                        queue.pop(index)
                        queue_needs.pop(index)
                    for row, need, jobs in taken:  # driver-loop: over groups started this tick
                        if row.get("auto_compile_jobs"):
                            # Keyed: the plan row takes the key of the compiler it runs.
                            row.update(with_compile_jobs(row, env, jobs, plan["platform"]))
                        source = sources[(row["mode"], row["stem"])]
                        rss = guard_rss(need, source, floor)
                        group_limits = dict(limits, jobs=slots, cores=cores, available_bytes=available_now,
                                            compiler_workers=row["compile_jobs"], rss_bytes=rss,
                                            min_available_bytes=pressure_floor(
                                                available_at_start, baseline, available_now, rss),
                                            reservation_source=source)
                        group_env = dict(env, **{GUARD_ENV: json.dumps(group_limits)})
                        if row.get("auto_compile_jobs"):
                            group_env["MOJOLEARN_COMPILE_JOBS"] = str(row["compile_jobs"])
                        threads_now.append(row["compile_jobs"])
                        launch = {"groups_running_at_launch": len(running) + 1,
                                  "threads_running_at_launch": sum(threads_now), "cores": cores,
                                  "submitted": time.monotonic()}
                        future = pool.submit(compile, row, REPO, cache, group_env, evidence,
                                             toolchain, launch)
                        futures.append(future)
                        running[future] = (row, need, launch)
                    report["groups_concurrent_max"] = max(report["groups_concurrent_max"], len(running))
                    report["threads_running_max"] = max(report["threads_running_max"], sum(threads_now))
                if not running:
                    if queue:
                        raise KeyboardInterrupt("development build cancelled before every group started")
                    break
                done, _ = wait(list(running), timeout=SCHEDULER_POLL_SECONDS, return_when=FIRST_COMPLETED)
                for future in done:  # driver-loop: over groups finished this tick
                    row, need, launch = running.pop(future)
                    result = future.result()
                    result.update(groups_running_at_launch=launch["groups_running_at_launch"],
                                  threads_running_at_launch=launch["threads_running_at_launch"],
                                  need_bytes=need, need_source=sources[(row["mode"], row["stem"])])
                    execution = result.get("execution") or {}
                    if execution.get("status") == "passed" and int(execution.get("peak_rss_bytes") or 0) > 0:
                        measured[row["mode"]] = max(measured.get(row["mode"], 0), int(
                            int(execution["peak_rss_bytes"]) * PEAK_HEADROOM["passed"]))
                    collect(report, result, evidence)
                # Groups without a record of their own pack by what this run measured.
                for index, queued in enumerate(queue):  # driver-loop: over queued development groups
                    key = (queued["mode"], queued["stem"])
                    queue_needs[index], sources[key] = live_need(
                        sources[key], queue_needs[index], measured.get(queued["mode"], 0), floor, ceiling)
        except BaseException:
            # Cancel before ThreadPoolExecutor.__exit__ waits. Its
            # workers' guarded process groups must stop together.
            build_process.cancel_active()
            for future in futures:
                future.cancel()
            raise


def build_support_rows(plan, env, cache, evidence, limits, report):
    """Support artifacts (portable math): small C builds, reused from the
    cache when their inputs are unchanged. Serial by nature: one C file."""
    support_env = dict(env, **{GUARD_ENV: json.dumps(dict(resource_budget(1, 1, None), **limits))}) \
        if any(r["action"] == "build" for r in plan["support"]) else env
    for row in plan["support"]:  # driver-loop: over support artifacts
        if row["action"] != "build":
            continue
        result = build_support(row, REPO, cache, support_env, evidence)
        report["results"].append(result)
        report["compile_attempts"] += int("command" in result)
        atomic_json(evidence / "result.json", report)
        print(f"support {result['stem']} {result['state']} rc={result['exit_code']}")
        if result["exit_code"]:
            raise BuildError(f"support build {row['name']} failed; existing loader indexes were preserved")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("plan", "build"))
    parser.add_argument("--sets", default="identical,host")
    parser.add_argument("--group", action="append", default=[], help="logical name, physical stem or set:name")
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache/mojolearn-development")
    parser.add_argument("--layout", choices=LAYOUTS,
                        help="release (the macOS default): the macOS wheel's merged groups through the one "
                             "Mac cache; development (the Linux default): one group per binding")
    parser.add_argument("--mac-cache", type=Path,
                        help="the one Mac cache of --layout release (default: MOJOLEARN_BINCACHE_DIR, "
                             "else ~/mojolearn-evidence/bincache-mac)")
    parser.add_argument("--json", type=Path, help="write complete plan/provenance")
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--parallel", default="auto",
                        help="'auto' (default): pack groups by recorded peak memory; N: at most N groups; 1: serial")
    parser.add_argument("--jobs", type=int, help="older spelling of --parallel N")
    parser.add_argument("--memory-per-job-gib", type=float,
                        help="need of a group with no recorded peak (GiB)")
    parser.add_argument("--group-seconds", type=float,
                        default=release_limits.get("DEVELOPMENT_BUILD_SECONDS"))
    parser.add_argument("--stall-seconds", type=float,
                        default=release_limits.get("DEVELOPMENT_STALL_SECONDS"))
    parser.add_argument("--lock-seconds", type=float,
                        default=release_limits.get("DEVELOPMENT_LOCK_SECONDS"))
    args = parser.parse_args(argv)
    modes = list(dict.fromkeys(args.sets.split(",")))
    env, cache = dict(os.environ), args.cache.resolve()
    env.pop(GUARD_ENV, None)  # Private settings are made here, never inherited.
    layout = args.layout or default_layout()
    mac_cache = (args.mac_cache.expanduser() if args.mac_cache else bincache.mac_cache_dir(env)).resolve()
    group_workers_env = env.pop(GROUP_WORKERS_ENV, None)  # scheduling only
    evidence = args.evidence or (Path.home() / "mojolearn-evidence/development-builds" /
        (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + str(os.getpid())))
    report = {"schema": SCHEMA, "state": "planning", "results": [], "release_qualified": False,
              "source_commit": bincache.first_line(["git", "-C", str(REPO), "rev-parse", "HEAD"])}
    futures = []
    try:
        if args.command == "plan":
            plan = (release_plan(REPO, cache, mac_cache, modes, env, args.group) if layout == "release"
                    else make_plan(REPO, cache, modes, env, args.group))
            if args.json:
                atomic_json(args.json, plan)
            summary(plan)
            return 0
        evidence.mkdir(parents=True, exist_ok=True)
        limits = dict(seconds=positive_seconds(args.group_seconds, "--group-seconds"),
                      stall_seconds=positive_seconds(args.stall_seconds, "--stall-seconds"),
                      lock_seconds=positive_seconds(args.lock_seconds, "--lock-seconds"))
        cap = group_cap(parse_parallel(args.parallel), args.jobs, group_workers_env)
        if args.memory_per_job_gib is not None:
            positive_seconds(args.memory_per_job_gib, "--memory-per-job-gib")
        report["limits"] = limits
        with build_process.cancellation_scope(), lock(
                cache / "locks" / (digest(str(REPO)) + ".lock"), limits["lock_seconds"]):
            # Generate before computing keys, never after deciding a cache hit.
            # This also preserves the generator's pin/refusal semantics.
            if "host" in modes:
                generator_limits = dict(resource_budget(1, 1, None), **limits)
                env[GUARD_ENV] = json.dumps(generator_limits)
                report["generator_limits"] = generator_limits
                code = run_logged(["sh", "tokenizer/tools/gen_unicode_table.sh"], REPO, env,
                                  evidence / "generate-unicode.log")
                if code:
                    raise BuildError(f"Unicode generator exited {code}; see {evidence / 'generate-unicode.log'}")
            if layout == "release":
                return release_build(args, modes, env, cache, mac_cache, evidence, limits, report, cap, futures)
            plan = make_plan(REPO, cache, modes, env, args.group)
            atomic_json(evidence / "plan.json", plan)
            summary(plan)
            if not plan["toolchain_available"]:
                raise BuildError("install the pinned default pixi environment before building")
            wanted = build_submission_order(
                r for r in plan["groups"] if r["selected"] and r["action"] == "build")
            # A warm cache has no compiler to reserve or start. The generator
            # above still runs under a guard before keys are computed.
            report.update(plan=plan, jobs=0, state="building", compile_attempts=0)
            if wanted and cap == 1:
                serial_build(plan, wanted, args, env, cache, evidence, limits, report)
            elif wanted:
                parallel_build(plan, wanted, args, env, cache, evidence, limits, report, cap, futures)
            atomic_json(evidence / "plan.json", plan)  # keys as built (scheduler thread counts)
            if any(r["exit_code"] for r in report["results"]):
                raise BuildError("one or more group builds failed; existing loader indexes were preserved")
            build_support_rows(plan, env, cache, evidence, limits, report)
            final_plan = make_plan(REPO, cache, modes, env, args.group)
            if ([r["key"] for r in final_plan["groups"] + final_plan["support"]]
                    != [r["key"] for r in plan["groups"] + plan["support"]]):
                before = {r.get("mode", "support") + ":" + r["stem"]: r["fields"]
                          for r in plan["groups"] + plan["support"]}
                after = {r.get("mode", "support") + ":" + r["stem"]: r["fields"]
                         for r in final_plan["groups"] + final_plan["support"]}
                raise BuildError(input_change_reason(
                    "build inputs changed; refusing to activate mixed-source outputs", before, after))
            if build_process.cancelled():
                raise KeyboardInterrupt("development build cancelled before activation")
            report.update(activate(final_plan, REPO, cache, limits["lock_seconds"]))
            atomic_json(evidence / "result.json", report)
            if args.json:
                atomic_json(args.json, report)
            print(f"{report['state']}: evidence {evidence}")
            return 0 if report["state"] == "activated" else 3
    except (BuildError, ValueError, OSError, subprocess.SubprocessError, KeyboardInterrupt) as error:
        interrupted = isinstance(error, KeyboardInterrupt)
        if interrupted:
            build_process.cancel_active()
        # Executor shutdown has now joined started workers. Retain their final
        # guard verdicts even when the main thread stopped consuming futures.
        recorded = {row["key"] for row in report["results"]}
        for future in futures:
            if not future.done() or future.cancelled():
                continue
            try:
                result = future.result()
            except (Exception, KeyboardInterrupt, SystemExit):
                continue
            if result["key"] not in recorded:
                report["results"].append(result)
                recorded.add(result["key"])
        if "compile_attempts" in report:
            report["compile_attempts"] = sum("command" in row for row in report["results"])
        report.update(state="interrupted" if interrupted else "failed", error=str(error),
                      exit_code=130 if interrupted else 1)
        if args.command == "build":
            atomic_json(evidence / "result.json", report)
        if args.json:
            atomic_json(args.json, report)
        print(f"development_build: {error}", file=sys.stderr)
        return report["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
