#!/usr/bin/env python3
"""Fixture packs: one tracked archive per fixture case in place of dozens of small files.

A fixture case such as a Mamba reference case is about 45 small arrays that one reader opens together. Tracked as
loose files, 46 cases are 2,100 files in the tree. Here each case directory `<root>/<case>/` is tracked as ONE
plain tar, `<root>/<case>.tar`, and the directory is recreated beside it, byte for byte, by `unpack`. The readers
(the Mojo checks, the package tests, the tools) keep opening `<root>/<case>/<member>` exactly as before: no reader
parses the archive, so no member, byte, case or assertion changes.

The pack roots are listed, one repository-relative directory per line, in `checks/native_tests/fixtures/PACKS.txt`.
Every immediate subdirectory of a root that is not itself a root is a case.

    python3 tools/fixture_packs.py unpack          # recreate every case directory that is missing or stale
    python3 tools/fixture_packs.py pack [ROOT...]  # rebuild the archives from the case directories
    python3 tools/fixture_packs.py verify          # every archive member equals the file on disk; nothing extra

`unpack` is what a runner calls before it runs anything that reads a fixture (the CPU box job, the release legs,
the package tests through `ensure()`). It is idempotent: a case whose stamp names its archive's sha256 is skipped.
The archives are deterministic (sorted members, zero times and owners), so packing unchanged files gives the same
bytes. Cases are independent units: pack, unpack and verify run them across all cores.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import io
import os
import shutil
import sys
import tarfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOTS_FILE = "checks/native_tests/fixtures/PACKS.txt"
STAMP = ".pack-sha256"
SKIP_DIRS = {"__pycache__"}


def roots(repo: str = REPO) -> list[str]:
    path = os.path.join(repo, ROOTS_FILE)
    with open(path, encoding="utf-8") as fh:
        return [line.strip().rstrip("/") for line in fh if line.strip() and not line.startswith("#")]


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _case_dirs(repo: str, root: str, all_roots: list[str]) -> list[str]:
    base = os.path.join(repo, root)
    out = []
    for name in sorted(os.listdir(base)):
        full = os.path.join(base, name)
        if os.path.isdir(full) and name not in SKIP_DIRS and f"{root}/{name}" not in all_roots:
            out.append(name)
    return out


def _members(case_dir: str) -> list[str]:
    found = []
    for here, dirs, files in os.walk(case_dir):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for name in sorted(files):
            if name.endswith(".pyc"):
                continue
            found.append(os.path.relpath(os.path.join(here, name), case_dir))
    return sorted(found)


def _stamp(base: str, case: str) -> str:
    """The stamp sits BESIDE the case directory, never in it: readers fingerprint a case by walking its files."""
    return os.path.join(base, "." + case + STAMP)


def _pack_case(base: str, case: str) -> tuple[str, int]:
    case_dir = os.path.join(base, case)
    names = _members(case_dir)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for rel in names:
            with open(os.path.join(case_dir, rel), "rb") as fh:
                data = fh.read()
            info = tarfile.TarInfo(name=f"{case}/{rel}")
            info.size = len(data)
            info.mode = 0o644
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            tar.addfile(info, io.BytesIO(data))
    target = os.path.join(base, case + ".tar")
    payload = buffer.getvalue()
    if not os.path.exists(target) or open(target, "rb").read() != payload:
        with open(target, "wb") as fh:
            fh.write(payload)
    with open(_stamp(base, case), "w", encoding="utf-8") as fh:
        fh.write(hashlib.sha256(payload).hexdigest() + "\n")
    return case, len(names)


def _unpack_case(base: str, archive: str) -> tuple[str, str]:
    case = archive[: -len(".tar")]
    source = os.path.join(base, archive)
    case_dir = os.path.join(base, case)
    want = _sha256_file(source)
    stamp = _stamp(base, case)
    if os.path.isfile(stamp) and open(stamp, encoding="utf-8").read().strip() == want:
        return case, "current"
    if os.path.isdir(case_dir):
        shutil.rmtree(case_dir)
    with tarfile.open(source, mode="r") as tar:
        for info in tar.getmembers():
            if not info.isfile() or not info.name.startswith(case + "/") or ".." in info.name.split("/"):
                raise SystemExit(f"fixture_packs: {source}: unexpected member {info.name!r}")
            target = os.path.join(base, info.name)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with tar.extractfile(info) as member, open(target, "wb") as fh:
                shutil.copyfileobj(member, fh)
    with open(stamp, "w", encoding="utf-8") as fh:
        fh.write(want + "\n")
    return case, "unpacked"


def _verify_case(base: str, archive: str) -> tuple[str, list[str]]:
    case = archive[: -len(".tar")]
    case_dir = os.path.join(base, case)
    problems = []
    inside = {}
    with tarfile.open(os.path.join(base, archive), mode="r") as tar:
        for info in tar.getmembers():
            inside[info.name[len(case) + 1:]] = hashlib.sha256(tar.extractfile(info).read()).hexdigest()
    on_disk = _members(case_dir) if os.path.isdir(case_dir) else []
    for rel in on_disk:
        if rel not in inside:
            problems.append(f"{case}/{rel}: on disk, not in the archive")
        elif _sha256_file(os.path.join(case_dir, rel)) != inside[rel]:
            problems.append(f"{case}/{rel}: bytes differ")
    for rel in inside:
        if rel not in on_disk:
            problems.append(f"{case}/{rel}: in the archive, not on disk")
    return case, problems


def _workers() -> int:
    return max(1, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1))


def _each(function, jobs):
    with concurrent.futures.ThreadPoolExecutor(max_workers=_workers()) as pool:
        return list(pool.map(lambda job: function(*job), jobs))


def pack(repo: str = REPO, only: list[str] | None = None) -> int:
    all_roots = roots(repo)
    jobs = []
    for root in only or all_roots:
        root = root.rstrip("/")
        if root not in all_roots:
            raise SystemExit(f"fixture_packs: {root} is not listed in {ROOTS_FILE}")
        jobs += [(os.path.join(repo, root), case) for case in _case_dirs(repo, root, all_roots)]
    done = _each(_pack_case, jobs)
    print(f"fixture_packs: packed {len(done)} cases, {sum(count for _, count in done)} members")
    return 0


def _archives(repo: str) -> list[tuple[str, str]]:
    jobs = []
    for root in roots(repo):
        base = os.path.join(repo, root)
        if os.path.isdir(base):
            jobs += [(base, name) for name in sorted(os.listdir(base)) if name.endswith(".tar")]
    return jobs


def unpack(repo: str = REPO, quiet: bool = False) -> int:
    done = _each(_unpack_case, _archives(repo))
    if not quiet:
        fresh = sum(1 for _, state in done if state == "unpacked")
        print(f"fixture_packs: {len(done)} cases, {fresh} unpacked, {len(done) - fresh} already current")
    return 0


def ensure(repo: str = REPO) -> None:
    """Called by Python readers before they open a fixture case directory."""
    unpack(repo, quiet=True)


def verify(repo: str = REPO) -> int:
    done = _each(_verify_case, _archives(repo))
    problems = [line for _, lines in done for line in lines]
    for line in problems[:40]:
        print("fixture_packs: " + line)
    print(f"fixture_packs: verified {len(done)} cases, {len(problems)} problems")
    return 1 if problems else 0


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[1] == "pack":
        return pack(only=argv[2:] or None)
    if len(argv) == 2 and argv[1] == "unpack":
        return unpack()
    if len(argv) == 2 and argv[1] == "verify":
        return verify()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
