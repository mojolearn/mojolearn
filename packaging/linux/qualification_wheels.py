#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Assemble local installed-test wheels from recorded binaries without compiling.

This is a qualification controller, not a release/reuse admission route. It uses
pack_wheel's split writer and current ownership metadata, verifies every retained
payload against its installed RECORD, and records each replacement separately.
The linux_x86_64 tag deliberately cannot be published to PyPI. No qualification,
manylinux floor, or common native source identity is inherited from the base.
The current source COMMIT and any retained PTX manifest are never falsified:
the ordinary loader must refuse PTX from a different source commit.

Run with Python 3.11+, outside the base environment. --source-root is an immutable
checkout; --installed-site is the old environment's site-packages. Optional
--native-overlay is a JSON list of {member, path, source_commit, build_evidence}.
Evidence is a list of retained build log/receipt paths, hashed into the manifest;
it records provenance supplied by the caller, not an assertion of build success.
"""
import argparse
import base64
import csv
from email.parser import BytesParser
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tomllib
from types import SimpleNamespace
import zipfile


NOTICE = ("LOCAL QUALIFICATION ONLY: current package source and individually inventoried "
          "native assets; compatibility and numerical qualification are not inherited. "
          "This is not a release build or a release-reuse proof.")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_member(name):
    parts = PurePosixPath(name).parts
    require(name and "\\" not in name and not name.startswith("/")
            and all(p not in ("", ".", "..") for p in parts)
            and "/".join(parts) == name, "unsafe package member: " + name)
    return name


def regular(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "missing or symlink file: " + str(path))
    return path


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def load_writer(source):
    spec = importlib.util.spec_from_file_location("qualification_split_writer",
                                                source / "packaging/linux/pack_wheel.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def retained_payload(site):
    """Read only the installed distributions' owned package files, never import them."""
    site = site.resolve()
    entries, provenance, distributions = {}, {}, []
    for dist in sorted(site.glob("mojolearn*.dist-info")):
        metadata_path = regular(dist / "METADATA")
        metadata = BytesParser().parsebytes(metadata_path.read_bytes())
        name = metadata.get("Name", "").lower().replace("_", "-")
        if name not in {"mojolearn", "mojolearn-core", "mojolearn-amd",
                        "mojolearn-nvidia", "mojolearn-nvidia-ptx"}:
            continue
        record = regular(dist / "RECORD")
        distributions.append(dict(name=name, version=metadata["Version"],
                                  metadata_sha256=digest(metadata_path), record_sha256=digest(record),
                                  dist_info=str(dist)))
        seen = set()
        for row in csv.reader(record.read_text().splitlines()):
            require(len(row) == 3 and row[0] not in seen, "invalid/duplicate RECORD row: " + str(dist))
            member, recorded_hash, recorded_size = row
            seen.add(member)
            # pip also records its generated scripts and caches; they are never
            # input assets. Current committed source supplies every Python file.
            if not member.startswith("mojolearn/") or member.endswith((".py", ".pyc")):
                continue
            safe_member(member)
            require("__pycache__" not in PurePosixPath(member).parts, "cache in payload RECORD")
            path = regular(site / member)
            require(path.resolve().is_relative_to(site), "payload escapes installed site: " + member)
            actual = digest(path)
            encoded = base64.urlsafe_b64encode(bytes.fromhex(actual)).decode().rstrip("=")
            require(recorded_hash == "sha256=" + encoded and recorded_size == str(path.stat().st_size),
                    "installed RECORD mismatch: " + member)
            require(member not in entries, "two distributions own " + member)
            entries[member] = path
            provenance[member] = dict(origin="installed-record", distribution=name,
                                      version=metadata["Version"], path=str(path), sha256=actual)
    require("mojolearn/identity_columns/COMMIT" in entries, "installed identity source witness missing")
    require(any("/.libs/" in p for p in entries), "installed shared native runtime missing")
    return entries, provenance, distributions


def source_payload(source, pw):
    entries = pw.python_package_entries()
    entries["mojolearn_diagnostics.py"] = source / "python/mojolearn_diagnostics.py"
    package = source / "python/mojolearn"
    for rel in ("ALPHA_API.md", "models/catalog.json"):
        entries["mojolearn/" + rel] = package / rel
    for rel in ("identity_trace_diff", "identity_break"):
        entries["mojolearn/_" + rel + ".py"] = source / "tools" / (rel + ".py")
    for path in sorted((source / "tools/identity_lanes").glob("*.py")):
        entries[f"mojolearn/_identity_lane_{path.stem}.py"] = path
    for directory in (package / "reference_cards", package / "verify_reference"):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and not path.name.endswith(".pyc"):
                entries["mojolearn/" + path.relative_to(package).as_posix()] = path
    from verification_ctr_payload import model_entries
    for name, path in model_entries(source).items():
        entries["mojolearn/verify_reference/ctr_models/" + name] = path
    record_dir = pw.host_surface.training_gpu_column_record()
    for column in pw.host_surface.TRAINING_GPU_COLUMNS:
        path = source / column
        entries[f"mojolearn/identity_columns/{record_dir}/{path.name}"] = path
    entries["mojolearn/CITATION.cff"] = source / "CITATION.cff"
    return entries


def assert_committed(source, commit, entries):
    blobs = {}
    for row in git(source, "ls-tree", "-r", "-z", commit).split(b"\0"):
        if row:
            metadata, path = row.split(b"\t", 1)
            blobs[path.decode()] = metadata.split()[2].decode()
    for path in entries.values():
        regular(path)
        require(path.resolve().is_relative_to(source), "source asset outside immutable checkout")
        rel = path.relative_to(source).as_posix()
        data = path.read_bytes()
        actual = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
        require(blobs.get(rel) == actual, "uncommitted source asset: " + rel)
    return blobs


def apply_native_overlay(entries, provenance, overlay):
    replaced = {}
    for row in overlay:
        member = safe_member(row["member"])
        require(member.startswith("mojolearn/") and member.endswith(".so"), "overlay must name a native .so")
        require(member not in replaced, "duplicate native overlay member: " + member)
        merged = str(PurePosixPath(member).parent / "_mojolearn_merged.json")
        require(merged not in entries, "replace a merged directory through its declared build artifact: " + member)
        require(re.fullmatch("[0-9a-f]{40}", row["source_commit"]), "invalid native source commit")
        path = regular(Path(row["path"]))
        with path.open("rb") as stream:
            require(stream.read(4) == b"\x7fELF", "replacement is not an ELF binary: " + str(path))
        path = path.resolve()
        evidence = [dict(path=str(regular(Path(p).resolve())), sha256=digest(p))
                    for p in row.get("build_evidence", [])]
        require(evidence, "native replacement needs retained build evidence: " + member)
        replaced[member] = provenance.get(member)
        entries[member] = path
        provenance[member] = dict(origin="explicit-native-artifact", path=str(path), sha256=digest(path),
                                  source_commit=row["source_commit"], build_evidence=evidence)
    return replaced


def assemble(source, commit, site, out, overlay=()):
    source, site, out = source.resolve(), site.resolve(), out.resolve()
    require(re.fullmatch("[0-9a-f]{40}", commit), "invalid package source commit")
    require(git(source, "rev-parse", "HEAD").decode().strip() == commit, "source checkout is not the requested freeze")
    require(not out.exists(), "output directory already exists")
    require(not out.is_relative_to(site) and not out.is_relative_to(source), "output must be outside input trees")
    pw = load_writer(source)
    entries, provenance, distributions = retained_payload(site)
    retained_commit = entries["mojolearn/identity_columns/COMMIT"].read_text().strip()
    current = source_payload(source, pw)
    # All shipped source/resource bytes and packaging inputs are committed;
    # the old installed environment is never modified or imported.
    packaging_inputs = {p: source / p for p in ("python/pyproject.toml", "README.md", "LICENSE", "NOTICE",
                        "packaging/linux/pack_wheel.py", "python/mojolearn/gpu_plugins.py")}
    blobs = assert_committed(source, commit, dict(current, **packaging_inputs))
    source_hashes = {}
    for member, path in current.items():
        entries[member] = path
        source_hashes[member] = digest(path)
        provenance[member] = dict(origin="package-source", source_commit=commit,
                                  path=str(path), sha256=source_hashes[member])
    replaced = apply_native_overlay(entries, provenance, overlay)
    plugins = pw.gpu_plugins
    # Only canonical installed roots are emitted; preserving depth preserves
    # the retained ELF RUNPATH. Refuse accidental two-source ownership.
    normalized, normalized_provenance = {}, {}
    for member, path in entries.items():
        installed = plugins.installed_member(member)
        require(installed not in normalized, "duplicate canonical payload: " + installed)
        normalized[installed] = path
        normalized_provenance[installed] = provenance[member]
    entries, provenance = normalized, normalized_provenance
    manifest_member = "mojolearn/cuda_ptx/sm_80/" + plugins.BASELINE_MANIFEST
    bundle, ptx = None, None
    if manifest_member in entries:
        path = entries[manifest_member]
        ptx = dict(source_commit=json.loads(path.read_text())["source_commit"], manifest_sha256=digest(path))
        ptx["current_source_admissible"] = ptx["source_commit"] == commit
        bundle = dict(manifest_sha256=ptx["manifest_sha256"])
    sets = sorted({(plugins.by_profile(plugins.member_payload(m)), m.split("/")[2])
                   for m in entries if plugins.member_payload(m) is not None})
    kinds = pw.split_kinds("", {v for v, _ in sets}, sets)
    config = tomllib.loads((source / "python/pyproject.toml").read_text())["project"]
    version = pw.read_version(source)
    require(config["version"] == version, "source package versions differ")
    dist, tag = f"mojolearn-{version}.dist-info", "py3-none-linux_x86_64"
    # Keep all native compile-source hashes separately from the binaries:
    # neither their presence nor the caller's receipt proves those binaries
    # were compiled from this source. Every binary origin remains explicit.
    native_sources = {p: source / p for p in sorted(blobs)
                      if p.endswith((".mojo", ".mojopkg")) or p in ("pixi.toml", "pixi.lock")}
    assert_committed(source, commit, native_sources)
    native_source_hashes = {p: digest(v) for p, v in native_sources.items()}
    current_witness = (commit + "\n").encode()
    provenance["mojolearn/identity_columns/COMMIT"] = dict(origin="generated-package-source-witness",
        source_commit=commit, sha256=hashlib.sha256(current_witness).hexdigest())
    report = dict(schema="mojolearn.local-qualification-wheels.v1", notice=NOTICE,
                  package_source_commit=commit, retained_source_commit=retained_commit,
                  source_root=str(source), installed_site=str(site), version=version,
                  distributions=distributions, assets=provenance, replaced_assets=replaced,
                  package_source_sha256=source_hashes, native_source_sha256=native_source_hashes,
                  packaging_inputs_sha256={p: digest(v) for p, v in packaging_inputs.items()},
                  controller_sha256=digest(Path(__file__)), ptx=ptx,
                  numerical_qualification="pending; not inherited", publishable=False)
    raw = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    generated = {
        f"{dist}/METADATA": pw.metadata_text(config, NOTICE).encode(),
        f"{dist}/WHEEL": pw.wheel_file_text(tag).encode(),
        f"{dist}/entry_points.txt": ("[console_scripts]\n" + "".join(
            f"{k} = {v}\n" for k, v in config.get("scripts", {}).items())).encode(),
        f"{dist}/QUALIFICATION_INPUTS.json": raw,
        "mojolearn/identity_columns/COMMIT": current_witness,
    }
    entries.pop("mojolearn/identity_columns/COMMIT")
    for license_name in config.get("license-files", []):
        generated[f"{dist}/licenses/{license_name}"] = (source / license_name).read_bytes()
    out.mkdir(parents=True)
    (out / "qualification-inputs.json").write_bytes(raw)
    built = pw.write_split(out, kinds, entries, generated, dist, config, version, tag, None,
                           [SimpleNamespace(vendor=v, arch=a) for v, a in sets], NOTICE, bundled_ptx=bundle)
    # Independently read the ZIP payload to detect changed inputs during packing.
    for wheel, _ in built:
        with zipfile.ZipFile(wheel) as archive:
            for member in archive.namelist():
                if member in entries:
                    require(hashlib.sha256(archive.read(member)).hexdigest() == provenance[member]["sha256"],
                            "asset changed during packing: " + member)
    from wheel_api_audit import split_audit
    audit = split_audit([p for p, _ in built])
    (out / "split-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    require(not audit["problems"], "split ownership/metadata audit failed: see split-audit.json")
    report["wheels"] = [dict(path=str(p), sha256=digest(p), profile=k, bytes=p.stat().st_size) for p, k in built]
    (out / "qualification-inputs.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--installed-site", type=Path, required=True)
    parser.add_argument("--native-overlay", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    overlay = json.loads(args.native_overlay.read_text()) if args.native_overlay else []
    require(isinstance(overlay, list), "native overlay must be a JSON list")
    report = assemble(args.source_root, args.source_commit, args.installed_site, args.out, overlay)
    print(json.dumps(dict(wheels=len(report["wheels"]), assets=len(report["assets"]),
                          evidence=str(args.out / "qualification-inputs.json"), ptx=report["ptx"])))


if __name__ == "__main__":
    main()
