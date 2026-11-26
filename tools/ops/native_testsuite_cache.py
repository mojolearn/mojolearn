#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reuse of the native-check group programs between builds (tools/ops/native_testsuite.sh build, 2026-11-26).

Why: every native-check build compiled every group program again, on the CPU box and on the L40S, whatever had
changed. Counted over the last 45 commits of main (14 groups a column, the planner's own import parser): 98 percent
of the CPU column's group programs and 99 percent of the GPU column's had the same inputs as in the commit before,
and 7 of 14 (CPU) and 11 of 14 (GPU) had the same inputs 44 commits apart. A program whose inputs did not change is
copied from the cache; one whose inputs changed compiles as before. Nothing is skipped: every check still runs.

    keys --column C --gen DIR --flags "..." [--repo .]   one line per unit of DIR/plan.txt on stdout:
                                                         `group <k> <key>` / `single <name> <key>`
    get <key> <dst>     copy the cached program to <dst>; exit 0 on a hit, 1 on a miss (or no cache directory)
    put <key> <src>     keep a program that just compiled

The cache directory is NC_TS_CACHE; unset or empty means no cache (keys still print). It is a plain directory on the
box: <dir>/<key[:2]>/<key>/{bin,meta.json}. An entry not read for NC_TS_CACHE_DAYS (default 14) is deleted by `put`.

THE KEY is every input that decides the program's bytes, and no path:
  the unit's members, in order; the generated driver and member copies (their bytes); the sha256 of every source
  file the members import, transitively (the import parser of tools/development_build.py, the one the binding cache
  uses; an import it cannot resolve widens the key to every source file of the tree); the compile flags (defines and
  target); the compiler binary's sha256, its version line, the precompiled standard-library packages' sha256 and the
  locked and installed toolchain; the OS, glibc, linker and C compiler; the MOJOLEARN_/MOJO_/MODULAR_ build
  environment; on the GPU column the device architecture the build targets (sm_89, gfx942) and the ptxas version.
  The commit is not in the key: two commits with the same inputs build the same program.
A build whose flags or environment name a fault (sabotage) gets no keys: it never reads or fills the cache.

A cached program was linked in another checkout. Its runpath names that checkout's toolchain directory, so the
runner puts the current checkout's `.pixi/envs/default/lib` on LD_LIBRARY_PATH (same toolchain: it is in the key).
Checked 2026-11-26 with Mojo 1.2.0: a check built in one directory, the directory deleted, ran and passed from
another checkout; without the library path it could not load. The program also carries the source path of the
checkout that built it, which appears only in a failure's file name.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]

SCHEMA = "mojolearn.native-testsuite-program.v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def key_of(fields: dict) -> str:
    return hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def first_line(argv: list[str]) -> str:
    try:
        r = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return ""
    return (r.stdout.decode(errors="replace").splitlines() or [""])[0].strip()


def read_plan(gen: Path) -> list[tuple[str, str, list[str]]]:
    """[(kind, id, members)] from DIR/plan.txt: `group <k> <name>...` and `single <name> <reason>`."""
    units = []
    for line in (gen / "plan.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "group":
            units.append(("group", parts[1], parts[2:]))
        elif len(parts) >= 2 and parts[0] == "single":
            units.append(("single", parts[1], [parts[1]]))
    return units


class Sources:
    """The transitive source closure of check files, hashed once per file (tools/development_build.py SourceGraph)."""

    def __init__(self, repo: Path, roots: list[Path]):
        import development_build
        self.repo = repo
        self.graph = development_build.SourceGraph(repo)
        self.graph.parse_all(roots)
        self.memo: dict[Path, tuple[frozenset, tuple]] = {}

    def closure(self, root: Path) -> tuple[frozenset, tuple]:
        root = root.resolve()
        if root not in self.memo:
            seen, unresolved, todo = set(), set(), [root]
            while todo:
                path = todo.pop()
                if path in seen:
                    continue
                seen.add(path)
                dependencies, missing = self.graph.nodes.get(path, ((), ()))
                unresolved.update(str(m) for m in missing)
                todo.extend(dependencies)
            self.memo[root] = (frozenset(seen), tuple(sorted(unresolved)))
        return self.memo[root]

    def hashes(self, files) -> dict[str, str]:
        files = sorted(files)
        self.graph.hash_files(files)
        return {str(p.relative_to(self.repo)): self.graph.hashes[p] for p in files}

    def tree(self) -> str:
        import bincache
        return key_of(self.hashes(Path(p).resolve() for p in bincache.tree_sources(self.repo)))


def toolchain_fields(repo: Path, flags: list[str]) -> dict | None:
    """What the compiler contributes to the bytes, without its path; None when the checkout has no compiler."""
    import development_build
    facts = development_build.toolchain_facts(repo)
    if not facts.get("available"):
        return None
    out = {k: facts[k] for k in ("compiler_sha256", "version", "locked", "installed", "precompiled_packages", "apple")}
    system = dict(facts.get("os") or {})
    if "--target-cpu" in flags:
        system.pop("cpu", None)  # the target is pinned by the flags, so the build machine's chip is not an input
    out["os"] = system
    return out


def sabotage(flags: list[str], environment: dict) -> bool:
    import bincache
    return bincache.is_sabotage({"build_env": environment, "args": flags})


def unit_keys(repo: Path, column: str, gen: Path, flags: list[str], environ=None) -> list[tuple[str, str, str]]:
    """[(kind, id, key)] for every unit of the plan; [] when this build must not use the cache."""
    import bincache
    import native_checks_report
    environ = os.environ if environ is None else environ
    environment = bincache.build_env(environ)
    if sabotage(flags, environment):
        return []
    toolchain = toolchain_fields(repo, flags)
    if toolchain is None:
        return []
    paths = {name: repo / rel for name, (rel, _) in native_checks_report.testsuite_paths(repo).items()}
    units = read_plan(gen)
    missing = sorted({m for _, _, members in units for m in members} - set(paths))
    if missing:
        raise SystemExit(f"native_testsuite_cache: the plan names files that are not checks: {' '.join(missing[:5])}")
    sources = Sources(repo, [paths[m] for _, _, members in units for m in members])
    device = None
    if column == "gpu":
        device = {"arch": bincache.device_arch(),
                  "ptxas": first_line([environ.get("MODULAR_NVPTX_COMPILER_PATH") or "ptxas", "--version"])}
    common = {"schema": SCHEMA, "column": column, "flags": flags, "toolchain": toolchain, "environment": environment,
              "device": device, "machine": platform.machine().lower()}
    tree = None
    keys = []
    for kind, ident, members in units:
        files, unresolved = set(), set()
        for member in members:
            closure, missing_imports = sources.closure(paths[member])
            files |= closure
            unresolved.update(missing_imports)
        generated = {}
        if kind == "group":
            for path in [gen / f"ts_group_{ident}.mojo"] + [gen / f"ts_{m}.mojo" for m in members]:
                generated[path.name.replace(f"ts_group_{ident}.mojo", "driver")] = sha256_file(path)
        fields = dict(common, kind=kind, members=members, generated=generated, source=sources.hashes(files),
                      unresolved=sorted(unresolved))
        if unresolved:
            tree = tree or sources.tree()
            fields["tree"] = tree
        keys.append((kind, ident, key_of(fields)))
    return keys


def cache_dir(environ=None) -> Path | None:
    value = (os.environ if environ is None else environ).get("NC_TS_CACHE", "")
    return Path(value) if value else None


def entry(root: Path, key: str) -> Path:
    if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
        raise SystemExit(f"native_testsuite_cache: not a key: {key!r}")
    return root / key[:2] / key


def get(root: Path | None, key: str, dst: Path) -> bool:
    if root is None:
        return False
    home = entry(root, key)
    try:
        meta = json.loads((home / "meta.json").read_text())
        if meta.get("key") != key or sha256_file(home / "bin") != meta.get("sha256"):
            return False
    except (OSError, ValueError):
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dst.parent, prefix=dst.name + ".")
    os.close(fd)
    shutil.copyfile(home / "bin", tmp)
    os.chmod(tmp, 0o755)
    os.replace(tmp, dst)
    os.utime(home / "meta.json")
    return True


def prune(root: Path, days: float) -> int:
    removed, limit = 0, time.time() - days * 86400
    for meta in root.glob("??/*/meta.json"):
        try:
            if meta.stat().st_mtime < limit:
                shutil.rmtree(meta.parent)
                removed += 1
        except OSError:
            pass
    return removed


def put(root: Path | None, key: str, src: Path, days: float = 14.0) -> bool:
    if root is None:
        return False
    home = entry(root, key)
    if (home / "meta.json").is_file():
        return True
    home.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=home.parent, prefix=key[:8] + "."))
    try:
        shutil.copyfile(src, tmp / "bin")
        (tmp / "meta.json").write_text(json.dumps({"schema": SCHEMA, "key": key, "sha256": sha256_file(tmp / "bin"),
                                                   "stored": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
        try:
            os.rename(tmp, home)
        except OSError:  # another build stored the same key first
            shutil.rmtree(tmp, ignore_errors=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    prune(root, days)
    return True


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("keys")
    k.add_argument("--column", required=True, choices=("cpu", "gpu"))
    k.add_argument("--gen", required=True)
    k.add_argument("--flags", default="")
    k.add_argument("--repo", default=".")
    g = sub.add_parser("get")
    g.add_argument("key")
    g.add_argument("dst")
    s = sub.add_parser("put")
    s.add_argument("key")
    s.add_argument("src")
    a = p.parse_args(argv)
    if a.cmd == "keys":
        for kind, ident, key in unit_keys(Path(a.repo).resolve(), a.column, Path(a.gen).resolve(), a.flags.split()):
            print(kind, ident, key)
        return 0
    if a.cmd == "get":
        return 0 if get(cache_dir(), a.key, Path(a.dst)) else 1
    put(cache_dir(), a.key, Path(a.src), float(os.environ.get("NC_TS_CACHE_DAYS", "14")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
