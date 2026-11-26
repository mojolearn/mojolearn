#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reuse of the native-check group programs between builds (tools/ops/native_testsuite.sh build, 2026-11-26).

Why: every native-check build compiled every group program again, on the CPU box and on the L40S, whatever had
changed. Counted over the last 45 commits of main (14 groups a column, the planner's own import parser): 98 percent
of the CPU column's group programs and 99 percent of the GPU column's had the same inputs as in the commit before,
and 7 of 14 (CPU) and 11 of 14 (GPU) had the same inputs 44 commits apart. A program whose inputs did not change is
copied from the cache; one whose inputs changed compiles as before. Nothing is skipped: every check still runs.

    keys --column C --gen DIR --flags "..." [--repo .]   one line per unit of DIR/plan.txt on stdout
                                                         (`group <k> <key>` / `single <name> <key>`); the fields of
                                                         each key go to DIR/keys/<key>.json
    get <key> <dst> [--fields DIR]   the program to <dst>: from NC_TS_CACHE, else from R2 through the URL map; exit 0
                                     on a hit, 1 on a miss
    put <key> <src> [--fields DIR]   keep a program that just compiled: in NC_TS_CACHE, and in R2's inbox through the
                                     URL map when there is one

TWO LEVELS. (1) NC_TS_CACHE, a plain directory on the box: <dir>/<key[:2]>/<key>/{bin,meta.json}; unset or empty
means none. An entry not read for NC_TS_CACHE_DAYS (default 14) is deleted by `put`. (2) R2, shared by every box,
through tools/bincache.py's machinery in its "ts" namespace (objects tscache/v1/<partition>/<key>.tar.gz): the Mac
lists the objects and mints short-lived presigned URLs at submit (tools/nvidia_central.sh central_submit, for
tools/ops/cpu_box_checks_job.sh and tools/ops/native_checks_gpu_job.sh); the map reaches the box as a 0600 file named
by NC_TS_URLS. The box can GET objects that exist and PUT only into the job's inbox slots, and records each upload in
NC_TS_UPLOADS (uploads.tsv and keys/<key>.json, inside the job's results). The Mac promotes an inbox object to its
content address after the results come back (tools/box_persist.py fetch runs `bincache.py promote --namespace ts`),
re-deriving the key from the recorded fields and refusing a fault build. The R2 credentials never reach a box.
A fetched archive is placed only when its manifest's key and fields are the ones this box computed and every byte
matches its sha256 (bincache.verify_archive).

THE KEY is sha256 over the canonical JSON of every input that decides the program's bytes, and no path (the same
key_of as the binding cache):
  the unit's members, in order; the generated driver and member copies (their bytes); the sha256 of every source
  file the members import, transitively (the import parser of tools/development_build.py, the one the binding cache
  uses; an import it cannot resolve widens the key to every source file of the tree); the compile flags (`args`:
  defines and target); the compiler binary's sha256, its version line, the precompiled standard-library packages'
  sha256 and the locked and installed toolchain; the OS, glibc, linker and C compiler; the MOJOLEARN_/MOJO_/MODULAR_
  build environment without a job's bookkeeping (`ts_build_env`); on the GPU column the device architecture the build targets (sm_89, gfx942) and
  the ptxas version. The commit is not in the key: two commits with the same inputs build the same program.
A build whose flags or environment name a fault (sabotage) gets no keys: it never reads or fills either level.

A cached program was linked in another checkout. Its runpath names that checkout's toolchain directory, so the
runner puts the current checkout's `.pixi/envs/default/lib` on LD_LIBRARY_PATH (same toolchain: it is in the key).
Checked 2026-11-26 with Mojo 1.2.0: a check built in one directory, the directory deleted, ran and passed from
another checkout; without the library path it could not load. The program also carries the source path of the
checkout that built it, which appears only in a failure's file name.
"""
from __future__ import annotations

import argparse
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

import bincache  # noqa: E402  (tools/ on the path above)

SCHEMA = "mojolearn.native-testsuite-program.v3"  # v3: ts_build_env (rel-fix-1)
NAMESPACE = "ts"
key_of = bincache.key_of
sha256_file = bincache.sha256_file


#: NOT INPUTS OF A CHECK PROGRAM (rel-fix-1, 2026-11-26, review of 3423ae229e..8f0d9a9642). `bincache.build_env`
#: keeps every MOJOLEARN_/MOJO_/MODULAR_ variable it does not know to be neutral, and every box job runs under
#: tools/box_persist_run.sh, which exports MOJOLEARN_RESULTS_DIR, MOJOLEARN_PERSIST_TAG/_JSON/_RUNNER and
#: MOJOLEARN_BOX_PERSIST with values of that job (its results dir, its tag): no key ever repeated on a box. Mojo reads
#: no environment variable while it compiles (a value reaches the program only as a -D define, which is in `args`),
#: and `native_testsuite.sh build` turns none of these into a flag, so they are dropped here: the job's bookkeeping,
#: the box's controller state, the run-time switches of the program and the Python package, and one path whose tool
#: is keyed by its version (MODULAR_NVPTX_COMPILER_PATH: `device.ptxas`). Every other variable with the prefixes stays
#: keyed, as in the binding cache: one not shown neutral is an input.
TS_NON_BUILD_ENV = ("MOJOLEARN_RESULTS_DIR", "MOJOLEARN_BOX_PERSIST", "MOJOLEARN_ALLOW_UNPERSISTED",
                    "MOJOLEARN_ORCHESTRATOR", "MOJOLEARN_REPO_COMMIT", "MOJOLEARN_HOST_PROBE", "MOJOLEARN_TS_FILE",
                    "MOJOLEARN_NUMERIC_MODE", "MODULAR_NVPTX_COMPILER_PATH")
TS_NON_BUILD_PREFIXES = ("MOJOLEARN_PERSIST_", "MOJOLEARN_NVC_", "MOJOLEARN_CPC_", "MOJOLEARN_DEVPOD_")


def ts_build_env(environ) -> dict:
    """The build environment of a check program's key: bincache.build_env without TS_NON_BUILD_ENV."""
    return {k: v for k, v in bincache.build_env(environ).items()
            if k not in TS_NON_BUILD_ENV and not k.startswith(TS_NON_BUILD_PREFIXES)}


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


def unit_fields(repo: Path, column: str, gen: Path, flags: list[str], environ=None) -> list[tuple[str, str, dict]]:
    """[(kind, id, fields)] for every unit of the plan; [] when this build must not use the cache."""
    import native_checks_report
    environ = os.environ if environ is None else environ
    # a fault named anywhere in the environment (dropped variables included) keeps the build out of the cache
    if bincache.is_sabotage({"build_env": bincache.build_env(environ), "args": flags}):
        return []
    environment = ts_build_env(environ)
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
    common = {"schema": SCHEMA, "column": column, "args": flags, "toolchain": toolchain, "build_env": environment,
              "device": device, "machine": platform.machine().lower()}
    tree = None
    out = []
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
        out.append((kind, ident, fields))
    return out


def unit_keys(repo: Path, column: str, gen: Path, flags: list[str], environ=None) -> list[tuple[str, str, str]]:
    """[(kind, id, key)] for every unit of the plan; [] when this build must not use the cache."""
    return [(kind, ident, key_of(fields)) for kind, ident, fields in unit_fields(repo, column, gen, flags, environ)]


# --- level 1: the box's directory -------------------------------------------------------------------------------

def cache_dir(environ=None) -> Path | None:
    value = (os.environ if environ is None else environ).get("NC_TS_CACHE", "")
    return Path(value) if value else None


def entry(root: Path, key: str) -> Path:
    if not bincache.KEY_RE.match(key or ""):
        raise SystemExit(f"native_testsuite_cache: not a key: {key!r}")
    return root / key[:2] / key


def _copy_executable(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dst.parent, prefix=dst.name + ".")
    os.close(fd)
    shutil.copyfile(src, tmp)
    os.chmod(tmp, 0o755)
    os.replace(tmp, dst)


def local_get(root: Path | None, key: str, dst: Path) -> bool:
    if root is None:
        return False
    home = entry(root, key)
    try:
        meta = json.loads((home / "meta.json").read_text())
        if meta.get("key") != key or sha256_file(home / "bin") != meta.get("sha256"):
            return False
    except (OSError, ValueError):
        return False
    _copy_executable(home / "bin", dst)
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


def local_put(root: Path | None, key: str, src: Path, days: float = 14.0) -> Path | None:
    """The entry directory holding `bin`, or None without a cache directory."""
    if root is None:
        return None
    home = entry(root, key)
    if (home / "meta.json").is_file():
        return home
    home.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=home.parent, prefix=key[:8] + "."))
    try:
        shutil.copyfile(src, tmp / "bin")
        (tmp / "meta.json").write_text(json.dumps({"schema": SCHEMA, "key": key, "sha256": sha256_file(tmp / "bin"),
                                                   "stored": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
        try:
            os.rename(tmp, home)
        except OSError:  # another build stored the same key first
            pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    prune(root, days)
    return home


# Kept for the callers and tests of the first version: the directory level alone.
def get(root: Path | None, key: str, dst: Path) -> bool:
    return local_get(root, key, dst)


def put(root: Path | None, key: str, src: Path, days: float = 14.0) -> bool:
    return local_put(root, key, src, days) is not None


# --- level 2: R2 through the URL map ----------------------------------------------------------------------------

def url_map(environ=None) -> dict | None:
    """The staged map (bincache.read_map) when NC_TS_URLS names one of namespace "ts"; else None."""
    path = (os.environ if environ is None else environ).get("NC_TS_URLS", "")
    if not path or not os.path.isfile(path):
        return None
    m = bincache.read_map(path)
    if m["header"].get("namespace") != NAMESPACE or not bincache.SEG_RE.match(m["header"].get("leg", "")):
        return None
    m["path"] = path
    return m


def read_fields(fields_dir: Path | None, key: str) -> dict | None:
    if fields_dir is None:
        return None
    try:
        fields = json.loads((fields_dir / f"{key}.json").read_text())
    except (OSError, ValueError):
        return None
    return fields if key_of(fields) == key else None


def remote_get(m: dict | None, key: str, fields: dict | None, dst: Path, root: Path | None) -> str:
    """'hit' (placed in dst and the directory), 'miss', or 'refused: <why>'."""
    if m is None or fields is None or key not in m["get"]:
        return "miss"
    with tempfile.TemporaryDirectory(prefix="ts-get-") as td:
        arc = Path(td) / "a.tar.gz"
        code = bincache.http_get(m["get"][key], str(arc))
        if code != 200:
            return "miss" if code == 404 else f"refused: GET HTTP {code}"
        try:
            _, blobs = bincache.verify_archive(str(arc), key, fields)
        except bincache.Reject as exc:
            return f"refused: {exc}"
        if set(blobs) != {"bin"}:
            return "refused: the archive holds other files than the program"
        prog = Path(td) / "bin"
        prog.write_bytes(blobs["bin"])
        home = local_put(root, key, prog)
        _copy_executable(home / "bin" if home else prog, dst)
    return "hit"


def remote_put(m: dict | None, key: str, fields: dict | None, home: Path | None, src: Path,
               uploads: Path | None) -> str:
    """'uploaded <slot>', 'present', or 'skipped: <why>'. Records the upload for the Mac's promote."""
    if m is None or fields is None or uploads is None:
        return "skipped: no map, fields or upload record"
    if key in m["get"]:
        return "present"
    slot, url = bincache.claim_slot(m["path"], m["put"])
    if slot is None:
        return "skipped: every inbox slot of this job is used"
    with tempfile.TemporaryDirectory(prefix="ts-put-") as td:
        base = home
        if base is None:
            base = Path(td) / "entry"
            base.mkdir()
            shutil.copyfile(src, base / "bin")
        arc = Path(td) / "a.tar.gz"
        bincache.pack(str(arc), key, fields, str(base), ["bin"], {})
        code = bincache.http_put(url, str(arc))
        if code != 200:
            return f"skipped: PUT HTTP {code}"
    dest = "%s/%s/%s.tar.gz" % (bincache.NAMESPACES[NAMESPACE]["objects"], m["header"]["partition"], key)
    (uploads / "keys").mkdir(parents=True, exist_ok=True)
    (uploads / "keys" / f"{key}.json").write_text(json.dumps(fields, sort_keys=True))
    with open(uploads / "uploads.tsv", "a") as fh:  # one short line per write: O_APPEND keeps concurrent units whole
        fh.write("%s\t%s\t%s\t%s\tsabotage=0\n" % (m["header"]["leg"], slot, dest, key))
    return f"uploaded {slot}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("keys")
    k.add_argument("--column", required=True, choices=("cpu", "gpu"))
    k.add_argument("--gen", required=True)
    k.add_argument("--flags", default="")
    k.add_argument("--repo", default=".")
    for name in ("get", "put"):
        s = sub.add_parser(name)
        s.add_argument("key")
        s.add_argument("path")
        s.add_argument("--fields", help="the keys/ directory `keys` wrote (needed for R2)")
    a = p.parse_args(argv)
    if a.cmd == "keys":
        gen = Path(a.gen).resolve()
        rows = unit_fields(Path(a.repo).resolve(), a.column, gen, a.flags.split())
        if rows:
            (gen / "keys").mkdir(exist_ok=True)
        for kind, ident, fields in rows:
            key = key_of(fields)
            (gen / "keys" / f"{key}.json").write_text(json.dumps(fields, sort_keys=True))
            print(kind, ident, key)
        return 0
    root, fields_dir = cache_dir(), Path(a.fields) if a.fields else None
    if a.cmd == "get":
        if local_get(root, a.key, Path(a.path)):
            return 0
        verdict = remote_get(url_map(), a.key, read_fields(fields_dir, a.key), Path(a.path), root)
        print(f"native_testsuite_cache: R2 {verdict}")
        return 0 if verdict == "hit" else 1
    home = local_put(root, a.key, Path(a.path), float(os.environ.get("NC_TS_CACHE_DAYS", "14")))
    uploads = Path(os.environ["NC_TS_UPLOADS"]) if os.environ.get("NC_TS_UPLOADS") else None
    m = url_map()
    if m is not None:
        print(f"native_testsuite_cache: R2 {remote_put(m, a.key, read_fields(fields_dir, a.key), home, Path(a.path), uploads)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
