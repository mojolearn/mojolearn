#!/usr/bin/env python3
"""Create/verify a compact manifest for a clean detached neural trial tree."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


SCHEMA = "mojolearn.clean-detached-source.v1"
SCOPES = ("bindings", "gemm", "training", "transformer", "checks", "core",
          "python/mojolearn", "tools/lm_step_memory_probe.py",
          "tools/lm_fold_specialize_probe.py", "tools/lm_fold_specialize_body.sh",
          "tools/trial_source_manifest.py", "pixi.toml", "pixi.lock")
REQUIRED = {"bindings/_mojolearn_byte_lm.mojo", "gemm/checks/gemm_identical.mojo",
            "python/mojolearn/_byte_lm_impl.py", "tools/lm_step_memory_probe.py",
            "tools/lm_fold_specialize_probe.py", "tools/lm_fold_specialize_body.sh",
            "tools/trial_source_manifest.py"}


def run_git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def digest(path):
    if path.is_symlink():
        data = os.readlink(path).encode()
        return hashlib.sha256(data).hexdigest(), len(data), "symlink"
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
            size += len(block)
    return h.hexdigest(), size, "file"


def cmd_create(args):
    root = args.root.resolve()
    status = run_git(root, "status", "--porcelain", "--untracked-files=no")
    if status:
        raise SystemExit("tracked source tree is not clean:\n" + status)
    commit = run_git(root, "rev-parse", "HEAD")
    if args.commit and commit != args.commit:
        raise SystemExit("HEAD %s != requested %s" % (commit, args.commit))
    tree = run_git(root, "rev-parse", "HEAD^{tree}")
    raw = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "-z", "--", *SCOPES])
    names = sorted(p.decode() for p in raw.split(b"\0") if p)
    if not REQUIRED <= set(names):
        raise SystemExit("manifest scopes miss required paths: %r" % sorted(REQUIRED - set(names)))
    files = {}
    for name in names:
        sha, size, kind = digest(root / name)
        files[name] = {"sha256": sha, "bytes": size, "kind": kind}
    record = {"schema": SCHEMA, "commit": commit, "git_tree": tree,
              "tracked_status": "clean", "scopes": list(SCOPES), "files": files}
    args.out.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    print("SOURCE MANIFEST create commit=%s tree=%s files=%d" % (commit, tree, len(files)))


def cmd_verify(args):
    root = args.root.resolve()
    record = json.loads(args.manifest.read_text())
    if record.get("schema") != SCHEMA or record.get("tracked_status") != "clean":
        raise SystemExit("invalid clean detached source manifest schema/status")
    if record.get("commit") != args.commit:
        raise SystemExit("manifest commit %r != required %r" % (record.get("commit"), args.commit))
    files = record.get("files", {})
    if not REQUIRED <= set(files) or len(files) < len(REQUIRED):
        raise SystemExit("source manifest is incomplete")
    failures = []
    for name, expected in files.items():
        path = root / name
        if not path.exists() and not path.is_symlink():
            failures.append(name + ":missing")
            continue
        sha, size, kind = digest(path)
        if (sha, size, kind) != (expected.get("sha256"), expected.get("bytes"), expected.get("kind")):
            failures.append(name + ":changed")
    if failures:
        raise SystemExit("source manifest verification failed: " + ", ".join(failures[:20]))
    print("SOURCE MANIFEST verify commit=%s tree=%s files=%d" %
          (record["commit"], record.get("git_tree"), len(files)))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    create = sub.add_parser("create")
    create.add_argument("--root", type=Path, required=True)
    create.add_argument("--commit")
    create.add_argument("--out", type=Path, required=True)
    create.set_defaults(func=cmd_create)
    verify = sub.add_parser("verify")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--commit", required=True)
    verify.set_defaults(func=cmd_verify)
    args = parser.parse_args()
    return args.func(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
