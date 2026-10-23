#!/usr/bin/env python3
"""Incremental development bindings; release packaging keeps its merged layout.

    python3 tools/development_build.py plan --sets identical,host --json plan.json
    python3 tools/development_build.py build --sets identical,host
    MOJOLEARN_GPU_ARCHS=gfx942 python3 tools/development_build.py build --sets identical

On the Mac the orchestrator wraps build with the existing semaphore:
    bash ~/mojolearn-evidence/compile_slot.sh python3 tools/development_build.py build

This is build tooling, never product runtime. Only identical and host are
supported. Each physical development group has a content-addressed artifact
and receipt. Unchanged groups reuse their exact bytes. An unresolved local
import widens the key to every project Mojo source and records that fallback.

Builds write complete logs and receipts under --evidence. All groups required
by a selected set must be current before its loader index is replaced. A
--group selection can populate the cache without activating an incomplete set
(exit 3). Activation uses immutable files and one atomic index replacement per
set; use a fresh Python process afterwards. Existing per-binding overrides are
refused, since they would silently shadow the selected development binaries.

Release commands do not call this tool. A development index deliberately fails
the release layout check; final release qualification still builds/checks the
compact release layout. No compile-time or numerical equivalence claim follows
from a successful cache/planning test.

Multiple compilers require --jobs plus --memory-per-job-gib, a measured peak
reservation for this source/target, including compiler subprocesses. Available
memory and CPU allocation bound the pool. The default is one compiler until a
group memory measurement is supplied; MOJOLEARN_COMPILE_JOBS controls its workers.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import contextlib
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import uuid

import bincache
import merged_bindings as merged

REPO = Path(__file__).resolve().parent.parent
SCHEMA = "mojolearn.development-build.v1"
KEY_SCHEMA = "mojolearn.development-group.v1"
MANIFEST = "bindings/development_groups.json"
BUILD_INPUTS = ("bindings/merged_build.sh", "bindings/build_defines.sh",
                "tools/merged_bindings.py", "tools/bincache.py", "tools/development_build.py")


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


@contextlib.contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


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
                         if p.is_file() and "__pycache__" not in p.parts)
            files.add(self.repo / "tokenizer/impl/unicode_class.mojo")
        hashes = {str(p.relative_to(self.repo)): bincache.sha256_file(p) for p in sorted(files)}
        return {"scope": scope, "files": hashes, "unresolved": unresolved}


def build_environment(mode, env):
    env = dict(env)
    if env.get("MOJOLEARN_NUMERIC_MODE", "identical") != "identical":
        raise BuildError("development builds support identical only; unset the other numeric mode")
    env["MOJOLEARN_NUMERIC_MODE"] = "identical"
    env["MOJOLEARN_PACKAGE_BYTE_LM"] = env.get("MOJOLEARN_PACKAGE_BYTE_LM", "1")
    if env["MOJOLEARN_PACKAGE_BYTE_LM"] not in ("0", "1"):
        raise BuildError("MOJOLEARN_PACKAGE_BYTE_LM must be 0 or 1")
    env["MOJOLEARN_COMPILE_JOBS"] = env.get("MOJOLEARN_COMPILE_JOBS", "1")
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
    packages = {p.name: bincache.sha256_file(p) for p in sorted(package_dir.glob("*.mojoc"))}
    return {"available": available, "compiler": str(compiler.resolve()),
            "compiler_sha256": bincache.sha256_file(compiler) if available else None,
            "version": bincache.first_line([str(compiler), "--version"]) if available else None,
            "locked": bincache.toolchain(repo), "installed": bincache.installed_toolchain(repo),
            "precompiled_packages": packages, "os": bincache.os_fields(),
            "apple": bincache.darwin_toolchain(), "pixi": bincache.first_line(["pixi", "--version"])}


def cached_artifact(cache, row):
    directory = Path(cache) / "artifacts" / row["key"]
    artifact = directory / (row["stem"] + ".so")
    receipt = directory / "receipt.json"
    try:
        saved = json.loads(receipt.read_text())
        if (saved["key"] != row["key"] or saved["fields"] != row["fields"]
                or saved["exit_code"] != 0 or saved["sha256"] != bincache.sha256_file(artifact)):
            return None
        return artifact
    except (OSError, KeyError, ValueError, TypeError):
        return None


def selected_group(mode, stem, selection):
    if not selection:
        return True
    logical = stem.removeprefix("_mojolearn_dev_").removesuffix("_m1").removesuffix("_host")
    return bool({stem, logical, f"{mode}:{stem}", f"{mode}:{logical}"} & set(selection))


def make_plan(repo, cache, modes, env, selection=(), facts=None, plat=None):
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
    builders = {p: bincache.sha256_file(repo / p) for p in BUILD_INPUTS}
    for mode in modes:
        if mode not in ("identical", "host"):
            raise BuildError("development builds support identical and host only")
        effective = build_environment(mode, env)
        arch = effective.get("MOJOLEARN_GPU_ARCHS", "")
        if mode == "identical" and plat == "linux" and not arch:
            raise BuildError("Linux identical development builds need MOJOLEARN_GPU_ARCHS")
        if arch and (plat == "darwin" or not arch.replace("_", "").isalnum()):
            raise BuildError("supply one Linux accelerator target, not a list or an Apple override")
        byte_lm = effective["MOJOLEARN_PACKAGE_BYTE_LM"] == "1"
        for stem, flags_class, members in merged.plan(mode, plat, byte_lm, "development"):
            fields = {"schema": KEY_SCHEMA, "repo": str(repo), "cache": str(cache),
                      "mode": mode, "platform": plat, "stem": stem, "flags_class": flags_class,
                      "members": members, "source": graph.closure(members), "builders": builders,
                      "root_sha256": hashlib.sha256(merged.root_text(stem, members).encode()).hexdigest(),
                      "toolchain": facts, "environment": bincache.build_env(effective),
                      "link_environment": {k: effective[k] for k in
                          ("CC", "CXX", "LD", "SDKROOT", "DEVELOPER_DIR", "LIBRARY_PATH",
                           "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "PIXI_ENVIRONMENT_NAME") if k in effective}}
            row = {"mode": mode, "stem": stem, "members": members, "flags_class": flags_class,
                   "key": digest(fields), "fields": fields,
                   "selected": selected_group(mode, stem, selection)}
            row["action"] = "reuse" if facts.get("available") and cached_artifact(cache, row) else "build"
            rows.append(row)
    for group in selection:
        if not any(selected_group(r["mode"], r["stem"], [group]) for r in rows):
            raise BuildError(f"group selection matches nothing: {group}")
    return {"schema": SCHEMA, "layout": "development", "platform": plat,
            "repo": str(repo), "cache": str(cache), "toolchain_available": facts.get("available", False),
            "groups": rows, "counts": {"groups": len(rows),
                "reuse": sum(r["action"] == "reuse" for r in rows),
                "build_selected": sum(r["action"] == "build" and r["selected"] for r in rows),
                "build_unselected": sum(r["action"] == "build" and not r["selected"] for r in rows)}}


def available_resources():
    if sys.platform == "linux":
        from build_sizing import box_resources
        return box_resources()
    cores = os.cpu_count() or 1
    # vm_stat's free and inactive pages are a conservative reclaimable budget;
    # do not use the machine's total RAM as if it were currently available.
    raw = subprocess.check_output(["vm_stat"], text=True)
    import re
    page_size = int(re.search(r"page size of (\d+) bytes", raw).group(1))
    values = dict(re.findall(r"^([^:]+):\s+(\d+)\.", raw, re.M))
    memory = page_size * (int(values.get("Pages free", 0)) + int(values.get("Pages inactive", 0)))
    return cores, memory


def pool_size(requested, workers, per_job_gib, resources=None):
    if requested < 1 or workers < 1:
        raise BuildError("--jobs must be positive")
    if requested == 1:
        return 1
    import math
    if per_job_gib is None or not math.isfinite(per_job_gib) or per_job_gib <= 0:
        raise BuildError("parallel builds require a measured --memory-per-job-gib reservation")
    cores, memory = resources or available_resources()
    actual = min(requested, cores // workers, int(0.8 * memory / (per_job_gib * 2 ** 30)))
    if actual < 1:
        raise BuildError("available CPU/memory allocation cannot fit one requested compiler reservation")
    return actual


def run_logged(argv, repo, env, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        result = subprocess.run(argv, cwd=repo, env=env, stdout=stream, stderr=subprocess.STDOUT)
    return result.returncode


def compile_group(row, repo, cache, env, evidence):
    with lock(cache / "locks" / (row["key"] + ".lock")):
        hit = cached_artifact(cache, row)
        if hit:
            return {"mode": row["mode"], "stem": row["stem"], "key": row["key"],
                    "state": "reused", "exit_code": 0}
        target = cache / "artifacts" / row["key"]
        work = cache / "work" / row["key"]
        work.mkdir(parents=True, exist_ok=True)
        output = work / "output"
        output.mkdir(exist_ok=True)
        log = evidence / (row["mode"] + "-" + row["stem"] + ".log")
        effective = build_environment(row["mode"], env)
        command = ["sh", "bindings/merged_build.sh", "host" if row["mode"] == "host" else "gpu",
                   "--layout", "development", "--group", row["stem"], "--outdir", str(output),
                   "--rootdir", str(work / "roots")]
        started = time.monotonic()
        code = run_logged(command, repo, effective, log)
        result = {"mode": row["mode"], "stem": row["stem"], "key": row["key"],
                  "exit_code": code, "seconds": time.monotonic() - started,
                  "log": str(log), "command": command, "state": "failed"}
        artifact = output / (row["stem"] + ".so")
        if code == 0:
            after = make_plan(repo, cache, [row["mode"]], env,
                              facts=toolchain_facts(repo), plat=row["fields"]["platform"])
            current = next(r for r in after["groups"] if r["stem"] == row["stem"])
            if current["key"] != row["key"]:
                result.update(exit_code=1, reason="build inputs changed during compilation")
            elif not artifact.is_file():
                result.update(exit_code=1, reason="compiler returned success without the declared output")
            else:
                target.mkdir(parents=True, exist_ok=True)
                sha = bincache.sha256_file(artifact)
                shutil.copy2(artifact, target / artifact.name)
                result.update(state="built", sha256=sha)
                atomic_json(target / "receipt.json", dict(result, fields=row["fields"]))
        atomic_json(evidence / (row["mode"] + "-" + row["stem"] + ".json"), result)
        return result


def activate(plan, repo, cache):
    """Publish only complete current sets; every indexed binary is immutable."""
    rows = plan["groups"]
    sources = {row["key"]: cached_artifact(cache, row) for row in rows}
    missing = [row["mode"] + ":" + row["stem"] for row in rows if sources[row["key"]] is None]
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
    # No index changes until every artifact in every requested set is staged.
    for out, index in indexes.items():
        with merged.index_lock(out):
            atomic_json(out / "_mojolearn_merged.json", index)
    return {"state": "activated", "missing": [], "activated": modes}


def summary(plan):
    for row in plan["groups"]:
        selection = "selected" if row["selected"] else "unselected"
        scope = row["fields"]["source"]["scope"]
        print(f"{row['mode']} {row['stem']} {row['action']} {selection} {scope} {row['key'][:12]}")
    print("counts " + json.dumps(plan["counts"], sort_keys=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("plan", "build"))
    parser.add_argument("--sets", default="identical,host")
    parser.add_argument("--group", action="append", default=[], help="logical name, physical stem or set:name")
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache/mojolearn-development")
    parser.add_argument("--json", type=Path, help="write complete plan/provenance")
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--memory-per-job-gib", type=float)
    args = parser.parse_args(argv)
    modes = list(dict.fromkeys(args.sets.split(",")))
    env, cache = dict(os.environ), args.cache.resolve()
    evidence = args.evidence or (Path.home() / "mojolearn-evidence/development-builds" /
        (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + str(os.getpid())))
    report = {"schema": SCHEMA, "state": "planning", "results": [], "release_qualified": False,
              "source_commit": bincache.first_line(["git", "-C", str(REPO), "rev-parse", "HEAD"])}
    try:
        if args.command == "plan":
            plan = make_plan(REPO, cache, modes, env, args.group)
            if args.json:
                atomic_json(args.json, plan)
            summary(plan)
            return 0
        evidence.mkdir(parents=True, exist_ok=True)
        with lock(cache / "locks" / (digest(str(REPO)) + ".lock")):
            # Generate before computing keys, never after deciding a cache hit.
            # This also preserves the generator's pin/refusal semantics.
            if "host" in modes:
                code = run_logged(["sh", "tokenizer/tools/gen_unicode_table.sh"], REPO, env,
                                  evidence / "generate-unicode.log")
                if code:
                    raise BuildError(f"Unicode generator exited {code}; see {evidence / 'generate-unicode.log'}")
            plan = make_plan(REPO, cache, modes, env, args.group)
            atomic_json(evidence / "plan.json", plan)
            summary(plan)
            if not plan["toolchain_available"]:
                raise BuildError("install the pinned default pixi environment before building")
            workers = int(build_environment("host", env)["MOJOLEARN_COMPILE_JOBS"])
            jobs = pool_size(args.jobs, workers, args.memory_per_job_gib)
            report.update(plan=plan, jobs=jobs, state="building")
            wanted = [r for r in plan["groups"] if r["selected"] and r["action"] == "build"]
            with ThreadPoolExecutor(max_workers=jobs) as pool:
                futures = [pool.submit(compile_group, row, REPO, cache, env, evidence) for row in wanted]
                for future in as_completed(futures):
                    result = future.result()
                    report["results"].append(result)
                    atomic_json(evidence / "result.json", report)
                    print(f"{result['mode']} {result['stem']} {result['state']} rc={result['exit_code']}")
            if any(r["exit_code"] for r in report["results"]):
                raise BuildError("one or more group builds failed; existing loader indexes were preserved")
            final_plan = make_plan(REPO, cache, modes, env, args.group)
            if [r["key"] for r in final_plan["groups"]] != [r["key"] for r in plan["groups"]]:
                raise BuildError("build inputs changed; refusing to activate mixed-source outputs")
            report.update(activate(final_plan, REPO, cache))
            atomic_json(evidence / "result.json", report)
            if args.json:
                atomic_json(args.json, report)
            print(f"{report['state']}: evidence {evidence}")
            return 0 if report["state"] == "activated" else 3
    except (BuildError, ValueError, OSError, subprocess.SubprocessError) as error:
        report.update(state="failed", error=str(error))
        if args.command == "build":
            atomic_json(evidence / "result.json", report)
        if args.json:
            atomic_json(args.json, report)
        print(f"development_build: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
