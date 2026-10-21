#!/usr/bin/env python3
"""Build native executables and check them without a Python runtime.

This is a test controller. Python never executes inside a tested executable.
Local runs check source/link dependencies; Linux --isolated runs use a minimal
chroot containing only the executables and their inspected shared libraries.
Compilation and execution have separate limits. Missing coverage fails the
full gate; --only is explicitly a partial diagnostic, never release evidence.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import time

import bincache

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = "mojolearn-native-only-v1"
PYTHON_LIB = re.compile(r"(?:libpython|Python\.framework|python[23](?:\.\d+)?(?:$|/))", re.I)
PYTHON_SYMBOL = re.compile(r"\b_?(?:Py[A-Z_][A-Za-z0-9_]*|_Py[A-Za-z0-9_]+)\b")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def closure(root, source):
    """Conservative complete repository import closure, with no Python escape.

    A file containing unused Python bindings is still rejected. That denotes
    an inseparable source dependency, not proof that every function calls it.
    """
    root = root.resolve()
    todo = [root / source]
    seen, errors = {}, []
    while todo:
        path = todo.pop().resolve()
        if path in seen:
            continue
        if not path.is_relative_to(root) or not path.is_file():
            errors.append(f"missing/outside source: {path}")
            continue
        relative = path.relative_to(root).as_posix()
        seen[path] = digest(path)
        text = path.read_text()
        for module, names in bincache.parse_imports(text):
            if module == "python" or module.startswith(("python.", "std.python")):
                errors.append(f"{relative}: Python dependency {module}")
            dependencies = bincache.resolve(module, names, path, [root])
            if dependencies is None:
                errors.append(f"{relative}: unresolved import {module}")
            else:
                todo.extend(dependencies)
        # Catch direct C-API calls/late imports too; comments and docstrings
        # can mention Python legitimately, so remove them for this check.
        code = re.sub(r'(?s)""".*?"""|\'\'\'.*?\'\'\'', "", text)
        code = "\n".join(line.split("#", 1)[0] for line in code.splitlines())
        if re.search(r"\bPythonObject\b|\bPython\s*\.|\b_?Py(?:_[A-Za-z]\w*|[A-Z][A-Za-z0-9]*_[A-Za-z0-9_]+)\b", code):
            errors.append(f"{relative}: Python interop in native-only closure")
    return {p.relative_to(root).as_posix(): h for p, h in sorted(seen.items())}, sorted(set(errors))


def run_command(command, log, timeout, cwd, env=None):
    start = time.monotonic()
    with log.open("w") as stream:
        proc = subprocess.Popen(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)
        timed_out = False
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGKILL)
            code = proc.wait()
    return {"command": command, "exit_code": code, "timeout": timed_out,
            "seconds": round(time.monotonic() - start, 4), "log": str(log)}


def inspect_binary(binary, directory):
    """Inspect undefined CPython symbols and recursively resolved libraries."""
    errors, libraries, inspected = [], set(), set()
    todo = [binary]
    with (directory / "dependencies.log").open("w") as stream:
        while todo:
            path = todo.pop()
            if path in inspected:
                continue
            inspected.add(path)
            cmd = ["otool", "-L", str(path)] if sys.platform == "darwin" else ["ldd", str(path)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            stream.write(f"{cmd!r}\n{result.stdout}{result.stderr}\n")
            if result.returncode:
                errors.append(f"dependency inspection failed: {path}")
                continue
            if PYTHON_LIB.search(result.stdout):
                errors.append(f"Python library dependency: {path}")
            for line in result.stdout.splitlines()[1 if sys.platform == "darwin" else 0:]:
                if "not found" in line:
                    errors.append(f"unresolved library: {line.strip()}")
                match = re.search(r"(?:=>\s*)?(/[^\s(]+)", line)
                if match:
                    lib = Path(match.group(1))
                    # Apple's shared-cache system libraries have no disk file.
                    if lib.is_file():
                        libraries.add(lib)
                        todo.append(lib)
                    elif sys.platform != "darwin":
                        errors.append(f"missing library: {lib}")
                elif sys.platform == "darwin" and "@" in line:
                    # Do not pretend an unresolved rpath has been inspected.
                    errors.append(f"unresolved Mach-O library: {line.strip()}")
    result = subprocess.run(["nm", "-u", str(binary)], capture_output=True, text=True)
    (directory / "symbols.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        errors.append("undefined-symbol inspection failed")
    if PYTHON_SYMBOL.search(result.stdout):
        errors.append("CPython symbols in executable")
    return sorted(libraries), sorted(set(errors))


def load_manifest(path):
    manifest = json.loads(path.read_text())
    cases = manifest["cases"]
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r"[a-z0-9_-]+", i) for i in ids):
        raise ValueError("case ids must be unique safe identifiers")
    for case in cases:
        if not case.get("source") or not case.get("algorithms") or not case.get("marker"):
            raise ValueError(f"incomplete native smoke case: {case['id']}")
        if Path(case["source"]).is_absolute() or ".." in Path(case["source"]).parts:
            raise ValueError("source must stay inside repository")
    return manifest


def coverage_debt(manifest, backend):
    cases = {c["id"]: c for c in manifest["cases"]
             if c.get("status", "ready") == "ready" and backend in c.get("backends", ["cpu"])}
    debt = list(manifest.get("blocked", []))
    inventory = manifest.get("inventory", [])
    if not inventory:
        debt.append({"algorithm": "*", "reason": "algorithm inventory missing"})
    for row in inventory:
        if row.get("status") == "not_algorithm":
            if not row.get("reason"):
                debt.append({"algorithm": row["algorithm"], "reason": "unexplained inventory exclusion"})
            continue
        if backend not in row.get("backends", ["cpu", "metal", "cuda", "hip"]):
            continue
        witnesses = [c for c in row.get("cases", []) if c in cases
                     and row["algorithm"] in cases[c]["algorithms"]]
        if row.get("status") != "ready" or not witnesses:
            debt.append({"algorithm": row["algorithm"], "reason": row.get("reason", "no applicable native smoke")})
        elif row.get("boundary_blocker"):
            debt.append({"algorithm": row["algorithm"], "reason": row["boundary_blocker"]})
    return debt


def build_case(case, args, compiler):
    directory = args.out / case["id"]
    directory.mkdir(parents=True, exist_ok=True)
    if case["id"] in args.preparation_failures:
        record = {"id": case["id"], "status": "preparation_failed", "errors": args.preparation_failures[case["id"]]}
        write_json(directory / "result.json", record)
        return record
    sources, errors = closure(args.source_root, case["source"])
    record = {"id": case["id"], "algorithms": case["algorithms"], "scope": case.get("scope"),
              "source": case["source"], "source_hashes": sources, "status": "source_blocked", "errors": errors}
    if errors:
        write_json(directory / "result.json", record)
        return record
    flags = ["-D", "MOJOLEARN_NUMERIC_IDENTICAL=1", "-I", str(args.source_root)]
    if args.accelerator:
        flags += ["--target-accelerator", args.accelerator]
    inputs = {"sources": sources, "compiler": compiler, "compiler_sha256": digest(args.mojo),
              "toolchain": {p: digest(args.source_root / p) for p in ("pixi.toml", "pixi.lock")},
              "flags": flags, "platform": platform.platform(),
              "machine": platform.machine(), "build_recipe": 1,
              "environment": {k: v for k, v in sorted(os.environ.items()) if k.startswith(("MOJO", "MODULAR", "MAX_"))}}
    key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    cache = args.cache / key
    binary = directory / "smoke"
    receipt = cache / "build.json"
    reused = False
    if receipt.exists() and (cache / "smoke").exists():
        old = json.loads(receipt.read_text())
        if old.get("binary_sha256") == digest(cache / "smoke"):
            shutil.copy2(cache / "smoke", binary)
            record["build"] = {"reused": True, "cache_key": key, "origin": str(receipt)}
            reused = True
    if not reused:
        command = [str(args.mojo), "build", *flags, str(args.source_root / case["source"]), "-o", str(binary)]
        result = run_command(command, directory / "build.log", args.compile_timeout, args.source_root)
        record["build"] = result | {"reused": False, "cache_key": key}
        if result["exit_code"] or result["timeout"]:
            record["status"] = "compile_timeout" if result["timeout"] else "compile_failed"
            write_json(directory / "result.json", record)
            return record
        current_sources, current_errors = closure(args.source_root, case["source"])
        if current_sources != sources or current_errors:
            record["status"] = "source_changed_during_build"
            write_json(directory / "result.json", record)
            return record
        cache.mkdir(parents=True, exist_ok=True)
        shutil.copy2(binary, cache / "smoke")
        shutil.copy2(directory / "build.log", cache / "build.log")
        write_json(receipt, {"inputs": inputs, "binary_sha256": digest(binary), "build": result})
    record["binary_sha256"] = digest(binary)
    libraries, errors = inspect_binary(binary, directory)
    record["libraries"] = {str(lib): digest(lib) for lib in libraries}
    record["errors"] = errors
    record["status"] = "dependency_failed" if errors else "built"
    write_json(directory / "result.json", record)
    return record


def isolated_root(args, records):
    if sys.platform != "linux" or os.geteuid() != 0:
        raise RuntimeError("--isolated needs Linux root/chroot; no simulated interpreter absence")
    root = args.out / "rootfs"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()
    (root / "tmp").mkdir(mode=0o1777)
    (root / "cases").mkdir()
    for record in records:
        if record["status"] != "built":
            continue
        shutil.copy2(args.out / record["id"] / "smoke", root / "cases" / record["id"])
        for name in record["libraries"]:
            source = Path(name)
            if PYTHON_LIB.search(name):
                raise RuntimeError("Python library refused in isolated root")
            target = root / source.relative_to("/")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    # No host filesystem binds, shell, Python executable, site-packages,
    # interpreter library, network tooling, or copied source tree.
    inventory = {p.relative_to(root).as_posix(): digest(p) for p in root.rglob("*") if p.is_file()}
    write_json(args.out / "rootfs-inventory.json", inventory)
    with tarfile.open(args.out / "native-only-runtime.tar.gz", "w:gz") as archive:
        archive.add(root, arcname=".")
    return root


def execute_case(case, record, args, rootfs):
    if record["status"] != "built":
        return record
    directory = args.out / case["id"]
    command = [str(directory / "smoke")]
    if rootfs:
        command = [shutil.which("chroot"), str(rootfs), "/cases/" + case["id"]]
    env = {"PATH": "/usr/bin:/bin", "LANG": "C", "HOME": "/tmp", "TMPDIR": "/tmp"}
    if not rootfs:
        # The local route is explicitly dependency screening, not proof of
        # an interpreter-free machine. Runtime search paths remain native.
        env.update({k: v for k, v in os.environ.items() if k.startswith(("DYLD_", "LD_"))})
    result = run_command(command, directory / "run.log", args.run_timeout, args.out, env)
    record["run"] = result
    marker = case["marker"] in (directory / "run.log").read_text(errors="replace")
    record["marker_found"] = marker
    record["status"] = ("run_timeout" if result["timeout"] else "run_failed" if result["exit_code"]
                        else "missing_result" if not marker else "passed")
    record["interpreter_absent"] = bool(rootfs)
    write_json(directory / "result.json", record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--mojo", type=Path, default=Path(shutil.which("mojo") or "mojo"))
    parser.add_argument("--backend", choices=("cpu", "metal", "cuda", "hip"), default="cpu")
    parser.add_argument("--accelerator", default="")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--compile-timeout", type=float, default=600)
    parser.add_argument("--run-timeout", type=float, default=10)
    parser.add_argument("--isolated", action="store_true")
    parser.add_argument("--only", default="")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args(argv)
    if args.jobs < 1 or args.run_timeout <= 0 or args.compile_timeout <= 0:
        parser.error("job count and timeouts must be positive")
    if args.isolated and args.backend != "cpu":
        parser.error("isolated root currently qualifies CPU; GPU driver isolation is separate")
    args.source_root = args.source_root.resolve()
    args.out = args.out.resolve()
    args.cache = (args.cache or args.out.parent / "native-only-cache").resolve()
    args.manifest = (args.manifest or args.source_root / "packaging/native_only/manifest.json").resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.manifest)
    cases = [c for c in manifest["cases"] if args.backend in c.get("backends", ["cpu"])
             and c.get("status", "ready") == "ready"]
    all_ids = {c["id"] for c in cases}
    selected = set(args.only.split(",")) if args.only else all_ids
    if selected - all_ids:
        parser.error("unknown/inapplicable cases: " + ",".join(sorted(selected - all_ids)))
    cases = [c for c in cases if c["id"] in selected]
    plan = {"schema": SCHEMA, "backend": args.backend, "case_ids": [c["id"] for c in cases],
            "blocked": coverage_debt(manifest, args.backend), "partial_selection": bool(args.only),
            "manifest_sha256": digest(args.manifest), "isolated": args.isolated}
    write_json(args.out / "plan.json", plan)
    if args.plan:
        print(json.dumps({"cases": len(cases), "coverage_debt": len(plan["blocked"]), "plan": str(args.out / "plan.json")}))
        return 0
    args.mojo = Path(shutil.which(str(args.mojo)) or args.mojo).absolute()
    version = subprocess.run([str(args.mojo), "--version"], capture_output=True, text=True, check=True)
    compiler = version.stdout + version.stderr
    (args.out / "compiler-version.txt").write_text(compiler)
    args.preparation_failures = {}
    prepared = {}
    for case in cases:
        for prerequisite in case.get("build_prerequisites", []):
            if prerequisite not in prepared:
                command = shlex.split(prerequisite)
                if command != ["sh", "tokenizer/tools/gen_unicode_table.sh"]:
                    raise ValueError("unreviewed native smoke build prerequisite")
                result = run_command(command, args.out / "unicode-table-build.log", args.compile_timeout, args.source_root)
                prepared[prerequisite] = result
            if prepared[prerequisite]["exit_code"] or prepared[prerequisite]["timeout"]:
                args.preparation_failures[case["id"]] = [prepared[prerequisite]]
    write_json(args.out / "preparation.json", prepared)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        records = list(pool.map(lambda c: build_case(c, args, compiler), cases))
    rootfs = isolated_root(args, records) if args.isolated else None
    # GPU executions are serial. CPU smoke concurrency is bounded and is
    # explicitly not performance evidence or a full-machine benchmark.
    with ThreadPoolExecutor(max_workers=args.jobs if args.backend == "cpu" else 1) as pool:
        records = list(pool.map(lambda pair: execute_case(*pair, args, rootfs), zip(cases, records)))
    failed = [r["id"] for r in records if r["status"] != "passed"]
    complete = bool(records) and not failed and not plan["blocked"] and not args.only
    summary = plan | {"results": records, "passed": len(records) - len(failed), "failed": failed,
                      "coverage_complete": complete, "release_qualified": complete and args.isolated}
    write_json(args.out / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("passed", "failed", "coverage_complete", "release_qualified")}))
    return 1 if failed or (plan["blocked"] and not args.only) or not records else 0


if __name__ == "__main__":
    raise SystemExit(main())
