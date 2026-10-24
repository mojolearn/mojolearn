#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The framework adapters' manifest: written at build, finalized after staging, checked at pack.

A release set's adapters live in one directory, `_adapters/`, beside the set's
tiers (Linux: `mojolearn/<vendor dir>/<arch>/_adapters/` in the vendor wheel;
macOS: `mojolearn/_adapters/`), and hold exactly four files:

    libmojolearn_adapter.so   the native adapter library (device code: vendor wheel)
    mojolearn_torch.so        the PyTorch dispatcher registration
    mojolearn_ops.mojoc       the MAX custom-operation package
    adapters.json             this manifest (schema mojolearn.adapters.v1)

The manifest records the ABI, vendor and architecture, the sha256 and size of
every file AS SHIPPED, the Torch wheel the registration was linked against
(version, minor series, backend, C++ ABI, wheel sha256), the MAX and Mojo
versions, the exact build commands, and the sha256 of every adapter source.
python/mojolearn/adapters/_shipped.py reads it to refuse a Torch or MAX the
files were not built for.

    manifest.py command FILE -- ARGV...     write one build command (JSON)
    manifest.py record OUT --vendor V --arch A --platform P --logs DIR --torch-build JSON
                       --pixi-prefix DIR --mojo-version S --source-commit SHA
                                            the build record (OUT/adapters-build.json)
    manifest.py finalize OUT                adapters.json from the record and the staged bytes
    manifest.py check OUT [--vendor V --arch A]
                                            exactly the four files, each matching the manifest

Build and packaging glue only (stdlib); it never runs in the library.
"""
import argparse
import hashlib
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
SCHEMA = "mojolearn.adapters.v1"
ABI = 1
DIRECTORY = "_adapters"
MANIFEST = "adapters.json"
BUILD_RECORD = "adapters-build.json"
#: role -> shipped file name
FILES = {"native": "libmojolearn_adapter.so", "torch": "mojolearn_torch.so", "max": "mojolearn_ops.mojoc"}
#: The sources whose bytes decide the adapters (their sha256 is recorded).
SOURCE_ROOTS = ("adapters", "packaging/adapters")
#: A qualification-only route; refused in every shipped build command.
TEST_ONLY = "MOJOLEARN_ADAPTER_TEST_HOST_STAGING"
VENDORS = {"cuda": "linux", "hip": "linux", "metal": "macos"}
ARCH_RE = re.compile(r"^(sm_[0-9]+a?|gfx[0-9a-f]+|apple)$")


class ManifestError(ValueError):
    """An adapters directory whose files and manifest disagree."""


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pins(path=HERE / "pins.json"):
    return json.loads(pathlib.Path(path).read_text())


def source_digests(repo=REPO):
    """{repo-relative path: sha256} of every adapter source file."""
    out = {}
    for root in SOURCE_ROOTS:
        base = pathlib.Path(repo) / root
        for path in sorted(base.rglob("*")):
            rel = path.relative_to(repo).as_posix()
            if path.is_file() and "__pycache__" not in path.parts and not rel.endswith((".pyc", ".json")) \
                    or rel == "packaging/adapters/pins.json":
                out[rel] = sha256(path)
    return out


def conda_version(prefix, name):
    """The version of conda package `name` installed in the pixi env at `prefix`, or None."""
    rows = sorted(pathlib.Path(prefix, "conda-meta").glob(f"{name}-[0-9]*.json"))
    for path in rows:
        doc = json.loads(path.read_text())
        if doc.get("name") == name:
            return doc.get("version")
    return None


def file_rows(directory):
    directory = pathlib.Path(directory)
    return {role: dict(name=name, sha256=sha256(directory / name), bytes=(directory / name).stat().st_size)
            for role, name in FILES.items()}


def record(out, vendor, arch, platform, logs, torch_build, pixi_prefix, mojo_version, source_commit):
    out, logs = pathlib.Path(out), pathlib.Path(logs)
    if VENDORS.get(vendor) != platform or not ARCH_RE.match(arch) or (vendor == "metal") != (arch == "apple"):
        raise ManifestError(f"vendor {vendor}, arch {arch}, platform {platform} do not name one release set")
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit or ""):
        raise ManifestError("a full source commit is required")
    p = pins()
    torch = json.loads(pathlib.Path(torch_build).read_text())
    pin = p["torch"]["build"][vendor]
    if torch.get("version") != pin["version"] or torch.get("series") != p["torch"]["series"]:
        raise ManifestError(f"the registration was linked against torch {torch.get('version')}, pins.json {pin['version']}")
    max_version = conda_version(pixi_prefix, "max")
    if max_version != p["max"]["version"]:
        raise ManifestError(f"the build environment has max {max_version}, pins.json {p['max']['version']}")
    commands = {role: json.loads((logs / f"{role}-command.json").read_text()) for role in FILES}
    if any(TEST_ONLY in word for argv in commands.values() for word in argv):
        raise ManifestError(f"{TEST_ONLY} is a qualification-only route; a shipped adapter never carries it")
    doc = dict(schema=SCHEMA, abi=ABI, vendor=vendor, arch=arch, platform=platform,
               source_commit=source_commit, numeric_mode="identical",
               torch={k: torch.get(k) for k in ("version", "series", "backend", "backend_version", "cxx11_abi",
                                                "wheel", "wheel_sha256", "index_url", "python_tag")},
               max=dict(version=max_version), mojo=dict(version=mojo_version, conda=conda_version(pixi_prefix, "mojo")),
               requirements=dict(torch=p["torch"]["requirement"], max=p["max"]["requirement"]),
               commands=commands, sources=source_digests(),
               unstaged=file_rows(out))
    (out / BUILD_RECORD).write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return doc


def finalize(out):
    """adapters.json = the build record plus the shipped bytes' digests; the record is removed."""
    out = pathlib.Path(out)
    doc = json.loads((out / BUILD_RECORD).read_text())
    doc["files"] = file_rows(out)
    (out / MANIFEST).write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    (out / BUILD_RECORD).unlink()
    return check(out)


def validate(doc, digests, vendor=None, arch=None):
    """Check a parsed manifest against {file name: (sha256, bytes)} of the files beside it."""
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA or doc.get("abi") != ABI:
        raise ManifestError(f"not a {SCHEMA} manifest of ABI {ABI}")
    if vendor is not None and doc.get("vendor") != vendor:
        raise ManifestError(f"the manifest names vendor {doc.get('vendor')}, the set is {vendor}")
    if arch is not None and doc.get("arch") != arch:
        raise ManifestError(f"the manifest names architecture {doc.get('arch')}, the set is {arch}")
    if VENDORS.get(doc.get("vendor")) != doc.get("platform") or not ARCH_RE.match(str(doc.get("arch"))):
        raise ManifestError("the manifest names no valid vendor, platform and architecture")
    files = doc.get("files")
    if not isinstance(files, dict) or set(files) != set(FILES):
        raise ManifestError(f"the manifest must name exactly {sorted(FILES)}")
    if set(digests) != set(FILES.values()):
        raise ManifestError(f"the adapters directory must hold exactly {sorted(FILES.values())} and {MANIFEST}, "
                            f"holds {sorted(digests)}")
    for role, name in FILES.items():
        row = files[role]
        if row.get("name") != name or (row.get("sha256"), row.get("bytes")) != tuple(digests[name]):
            raise ManifestError(f"{name}: the shipped bytes differ from the manifest")
    torch = doc.get("torch") or {}
    if not (torch.get("series") and torch.get("backend") in ("cuda", "rocm", "mps")
            and re.fullmatch(r"[0-9a-f]{64}", str(torch.get("wheel_sha256")))):
        raise ManifestError("the manifest does not name the Torch build the registration links")
    if doc["platform"] == "linux" and torch.get("cxx11_abi") not in (0, 1):
        raise ManifestError("a Linux manifest must name the Torch C++ ABI")
    if not (doc.get("max") or {}).get("version"):
        raise ManifestError("the manifest does not name the MAX version")
    if any(TEST_ONLY in word for argv in (doc.get("commands") or {}).values() for word in argv):
        raise ManifestError(f"{TEST_ONLY} is a qualification-only route")
    return doc


def check(out, vendor=None, arch=None):
    out = pathlib.Path(out)
    if not (out / MANIFEST).is_file():
        raise ManifestError(f"{out} has no {MANIFEST}")
    digests = {p.name: (sha256(p), p.stat().st_size) for p in sorted(out.iterdir()) if p.name != MANIFEST}
    return validate(json.loads((out / MANIFEST).read_text()), digests, vendor, arch)


def refresh(manifest_bytes, read, note):
    """The manifest with every file's digest recomputed from `read(name)` (bytes),
    for a packer that rewrites shipped binaries in place (packaging/macos/shrink_wheel.py)."""
    doc = json.loads(manifest_bytes)
    for row in doc["files"].values():
        data = read(row["name"])
        row.update(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
    doc.setdefault("post_build", []).append(note)
    return (json.dumps(doc, indent=2, sort_keys=True) + "\n").encode()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["command"]:
        if len(argv) < 4 or argv[2] != "--":
            raise SystemExit("usage: manifest.py command FILE -- ARGV...")
        pathlib.Path(argv[1]).write_text(json.dumps(argv[3:], indent=1) + "\n")
        return 0
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="action", required=True)
    r = sub.add_parser("record")
    r.add_argument("out")
    for flag in ("--vendor", "--arch", "--platform", "--logs", "--torch-build", "--pixi-prefix",
                 "--mojo-version", "--source-commit"):
        r.add_argument(flag, required=True)
    f = sub.add_parser("finalize")
    f.add_argument("out")
    c = sub.add_parser("check")
    c.add_argument("out")
    c.add_argument("--vendor")
    c.add_argument("--arch")
    a = ap.parse_args(argv)
    try:
        if a.action == "record":
            record(a.out, a.vendor, a.arch, a.platform, a.logs, a.torch_build, a.pixi_prefix,
                   a.mojo_version, a.source_commit)
        elif a.action == "finalize":
            doc = finalize(a.out)
            print(f"adapters: {doc['vendor']}/{doc['arch']} torch {doc['torch']['version']} "
                  f"({doc['torch']['backend']}), max {doc['max']['version']}, "
                  + ", ".join(f"{r['name']} {r['sha256'][:12]}" for r in doc["files"].values()))
        else:
            check(a.out, a.vendor, a.arch)
    except (ManifestError, OSError, KeyError, ValueError) as exc:
        print(f"manifest.py: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
