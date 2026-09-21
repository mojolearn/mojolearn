#!/usr/bin/env python3
"""Create/verify a compact manifest for a clean detached neural trial tree."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile


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


def cmd_bundle(args):
    """Put the generated manifest inside the exact git-archive shipment."""
    root = args.root.resolve()
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mojolearn-source-bundle-") as td:
        temp = Path(td)
        manifest = temp / "SHIPPED_SOURCE_MANIFEST.json"
        cmd_create(argparse.Namespace(root=root, commit=args.commit, out=manifest))
        commit = json.loads(manifest.read_text())["commit"]
        archive = temp / "source.tar"
        subprocess.run(["git", "-C", str(root), "archive", "--format=tar",
                        "--output", str(archive), commit], check=True)
        manifest_bytes = manifest.read_bytes()
        commit_bytes = (commit + "\n").encode()
        with tarfile.open(archive, "a") as tf:
            for name, data in (("SHIPPED_SOURCE_MANIFEST.json", manifest_bytes),
                               ("SHIPPED_COMMIT.txt", commit_bytes)):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                info.mode = 0o644
                info.mtime = 0
                import io
                tf.addfile(info, io.BytesIO(data))
        # Verify the payload itself, not only the source worktree from which
        # it was made. Every manifested byte must be present in the tar.
        record = json.loads(manifest_bytes)
        with tarfile.open(archive, "r") as tf:
            if tf.extractfile("SHIPPED_COMMIT.txt").read() != commit_bytes:
                raise SystemExit("bundled commit witness differs")
            bundled_manifest = tf.extractfile("SHIPPED_SOURCE_MANIFEST.json").read()
            if bundled_manifest != manifest_bytes:
                raise SystemExit("bundled source manifest differs")
            for name, expected in record["files"].items():
                member = tf.getmember(name)
                if expected["kind"] == "symlink":
                    data = member.linkname.encode()
                else:
                    data = tf.extractfile(member).read()
                if (hashlib.sha256(data).hexdigest() != expected["sha256"]
                        or len(data) != expected["bytes"]):
                    raise SystemExit("bundled source differs: " + name)
        os.replace(archive, out)
    archive_sha = _sha_path(out)
    sidecar = Path(str(out) + ".sha256")
    sidecar.write_text("%s  %s\n" % (archive_sha, out.name))
    print("SOURCE BUNDLE commit=%s files=%d bytes=%d sha256=%s" %
          (commit, len(record["files"]), out.stat().st_size, archive_sha))


def _sha_path(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


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
    bundle = sub.add_parser("bundle")
    bundle.add_argument("--root", type=Path, required=True)
    bundle.add_argument("--commit", required=True)
    bundle.add_argument("--out", type=Path, required=True)
    bundle.set_defaults(func=cmd_bundle)
    args = parser.parse_args()
    return args.func(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
