#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The shipped SOURCE and the release TOOLING are two commits (2026-09-25).

    python3 tools/release_tooling.py overlay <source-commit> [--json]

A release ships exactly its frozen source commit. The machinery that drives it
(tools/release.py, the leg runners, the guards, the wheel smoke, the
publisher) runs from the tooling checkout, normally current main, so a fix to
that machinery lands without moving the source and without discarding a build.

Where a build leg ships tools to a box, the box unpacks the SOURCE archive and
then the ROUTE OVERLAY: the tooling checkout's copy of each file in
BOX_OVERLAY whose bytes differ from the source commit's (the pattern
tools/release_linux_build.sh uses; route.txt there, route-overlay.txt here).
Nothing in the build's source inventory is ever overlaid: a Mojo file,
bindings/, packaging/, python/, tokenizer/, pixi.toml, pixi.lock and
tools/linux_surface_qualification.sh are refused by name, as is anything the
native inventory (check_linux_release_qualification.is_native_source) counts.

An overlaid file that is also a binding-identity builder
(release_reuse.LINUX_BUILDERS, tools/release061_remote_build.sh) enters the
binding identities with the OVERLAID bytes, so the reuse plan and the
identity-keyed leg reuse judge what actually ran on the box.

The tooling digest names the build tooling that ran: the sha256 of every file
in BUILD_TOOLING at the tooling commit. A completed build leg is reused under
a later freeze only when its set's identities AND this digest are unchanged.
"""
import argparse
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from check_linux_release_qualification import is_native_source  # noqa: E402

SCHEMA = "mojolearn.release-route-overlay.v1"
#: Tools a GPU build leg's box runs out of the unpacked source tree.
BOX_OVERLAY = ("tools/release061_remote_build.sh", "tools/nvidia_serial_guard.py", "tools/amd_serial_guard.py",
               "tools/cpu_build_guard.py", "tools/bincache.py")
#: Everything that drives a Linux build leg: the local runners, the box
#: helpers and BOX_OVERLAY. Their bytes at the tooling commit are the tooling
#: digest a reused leg must match.
BUILD_TOOLING = ("tools/gemm_remote_leg.sh", "tools/do_release061_leg.sh", "tools/hotaisle_release_leg.sh",
                 "tools/hotaisle_vm_lib.sh", "tools/release_ubuntu22_build.sh", "tools/bincache_leg.sh",
                 "tools/release_linux_build.sh", "tools/runpod_cpu_leg.sh", "tools/release_linux_cpu_box.sh",
                 "tools/runpod_guard.sh") + BOX_OVERLAY
REFUSED_PREFIXES = ("bindings/", "packaging/", "python/", "tokenizer/")
REFUSED_NAMES = ("pixi.toml", "pixi.lock", "tools/linux_surface_qualification.sh")


class OverlayRefused(Exception):
    pass


def refused(rel):
    """Why REL may never be overlaid onto the frozen source, or None."""
    if rel.endswith((".mojo", ".mojopkg")):
        return "a Mojo source"
    if rel.startswith(REFUSED_PREFIXES) or rel in REFUSED_NAMES:
        return "in the build's source inventory"
    if is_native_source(rel):
        return "native build input (check_linux_release_qualification.is_native_source)"
    if ".." in rel.split("/") or rel.startswith("/"):
        return "not a repository-relative path"
    return None


def git(root, *args, text=True):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=text, check=True).stdout


def blob(root, commit, rel):
    """The bytes of REL at COMMIT, or None when the commit has no such file."""
    p = subprocess.run(["git", "-C", str(root), "cat-file", "blob", f"{commit}:{rel}"], capture_output=True)
    return p.stdout if p.returncode == 0 else None


def sha(data):
    return hashlib.sha256(data).hexdigest() if data is not None else None


def dirty(root, files):
    """The FILES whose working-tree copy differs from HEAD in the tooling checkout."""
    out = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no", "--", *files],
                         capture_output=True, text=True, check=True).stdout
    return sorted(line[3:] for line in out.splitlines() if line.strip())


def tooling_digest(root, commit, files=BUILD_TOOLING):
    """(digest, {file: sha256}) of FILES at the tooling COMMIT."""
    rows = {rel: sha(blob(root, commit, rel)) for rel in files}
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(), rows


def overlay_manifest(root, tooling_commit, source_commit, files=BOX_OVERLAY):
    """What the route overlay would carry: every file of FILES whose bytes at
    the tooling commit differ from the source commit's. Refuses a file in the
    source inventory before looking at any bytes."""
    for rel in files:
        why = refused(rel)
        if why:
            raise OverlayRefused(f"{rel} may not be overlaid onto the frozen source: {why}")
    rows = {}
    for rel in files:
        new = blob(root, tooling_commit, rel)
        if new is None:
            raise OverlayRefused(f"{rel} is not in the tooling commit {tooling_commit[:12]}")
        old = blob(root, source_commit, rel)
        if old != new:
            rows[rel] = dict(source_sha256=sha(old), tooling_sha256=sha(new))
    return dict(schema=SCHEMA, source_commit=source_commit, tooling_commit=tooling_commit, files=rows,
                digest=hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest())


def write_overlay(root, manifest, dest):
    """The overlay tarball (bytes from the tooling COMMIT, not the working tree;
    fixed mtime, owner and order so the same files always make the same
    bytes) and its manifest beside it. Returns (tarball, sha256), or
    (None, None) when nothing differs."""
    dest = Path(dest)
    if not manifest["files"]:
        return None, None
    dest.mkdir(parents=True, exist_ok=True)
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for rel in sorted(manifest["files"]):
            data = blob(root, manifest["tooling_commit"], rel)
            if sha(data) != manifest["files"][rel]["tooling_sha256"]:
                raise OverlayRefused(f"{rel} changed at the tooling commit while the overlay was written")
            info = tarfile.TarInfo(rel)
            info.size, info.mtime, info.mode, info.uid, info.gid = len(data), 0, 0o755, 0, 0
            info.uname = info.gname = "root"
            tar.addfile(info, io.BytesIO(data))
    import gzip
    gz = gzip.compress(raw.getvalue(), mtime=0)
    tgz = dest / "route-overlay.tgz"
    tgz.write_bytes(gz)
    digest = sha(gz)
    doc = dict(manifest, tarball=str(tgz), tarball_sha256=digest)
    (dest / "route-overlay.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return tgz, digest


def effective_builders(manifest):
    """{builder: sha256} for the binding-identity builders the overlay replaces."""
    import release_reuse
    return {rel: row["tooling_sha256"] for rel, row in (manifest or {}).get("files", {}).items()
            if rel in release_reuse.LINUX_BUILDERS}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("overlay",))
    ap.add_argument("source_commit")
    ap.add_argument("--tooling", default="HEAD")
    a = ap.parse_args(argv)
    tooling = git(ROOT, "rev-parse", a.tooling).strip()
    source = git(ROOT, "rev-parse", a.source_commit + "^{commit}").strip()
    m = overlay_manifest(ROOT, tooling, source)
    digest, _ = tooling_digest(ROOT, tooling)
    print(json.dumps(dict(m, tooling_digest=digest), indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
