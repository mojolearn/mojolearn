#!/usr/bin/env python3
"""Private, content-addressed transport of canonical release ledger evidence."""
import argparse
from contextlib import contextmanager
from email.parser import BytesParser
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import subprocess
import tarfile
import tempfile
import zipfile

import release_evidence as evidence
import release_requirements as requirements

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "mojolearn.release-ledger-admission.v1"
PREFIX = "release-readiness/v1/"
POLICY_FILES = ("tools/release_publish_admission.py", "tools/release_evidence.py", "tools/release_requirements.py",
                "tools/release_requirements.json", "tools/release_requirement_receipts.py",
                "tools/release_requirement_dependencies.py", "tools/candidate_wheel_install.py",
                "tools/gate_smoke_box.py", "tools/adapter_gate.py", "tools/autograd_smoke.py",
                "tools/gate_autograd_artifacts.py", "tools/pypi_wheel_install.py",
                "tools/release_gate_reuse.py", "tools/release_incremental_gate.py", "tools/native_adapter_composition.py",
                "tools/retained_release_composition.py", "tools/release_macos_assembly.py",
                "tools/coverage_sabotage.py", "tools/identity_lanes/coverage_sabotage.tsv",
                "x_neighbors/sabotage/coverage_precise_p2m_transpose_i.patch",
                "x_neighbors/sabotage/coverage_precise_p2m_const_cols.patch")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def strict_json(data):
    def distinct(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate admission JSON key: " + key)
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=distinct)


def policy(root):
    result = {name: sha256(Path(root) / name) for name in POLICY_FILES}
    for module in ("tools/release_requirements_storage.py", "tools/release_requirement_routes.py",
                   "tools/release_compile_evidence.py"):
        result[module] = sha256(Path(root) / module) if (Path(root) / module).is_file() else None
    return result


def source_revision(root, source):
    if not re.fullmatch(r"[0-9a-f]{40}", source):
        raise ValueError("publication requires a full frozen source SHA")
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", source + "^{commit}"], text=True).strip()
    if actual != source:
        raise ValueError("frozen source is unavailable or differs")
    return actual


def wheel_identity(path, version, source, configuration):
    """Bind published bytes to qualified artifacts; audit file-free metadata separately."""
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(name.startswith("/") or ".." in Path(name).parts for name in names):
            raise ValueError("unsafe or duplicate wheel members")
        metas = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metas) != 1:
            raise ValueError("wheel must contain one package metadata record")
        metadata = BytesParser().parsebytes(archive.read(metas[0]))
        name = canonical(metadata.get("Name", ""))
        if not name or metadata.get("Version") != version:
            raise ValueError("wheel version/name differs from the canonical release ledger")
        prefix = metas[0].rsplit("/", 1)[0]
        is_meta = name == "mojolearn" and all(member.startswith(prefix + "/") for member in names)
        if "macosx" in path.name:
            if archive.read("mojolearn/identity_columns/COMMIT").decode().strip() != source:
                raise ValueError("Apple wheel belongs to another source freeze")
        else:
            from candidate_wheel_install import wheel_record
            wheel_record(path, version, source)  # full Linux RECORD/tags/source admission
        if is_meta:
            allowed = {prefix + "/" + item for item in ("METADATA", "WHEEL", "RECORD", "LINUX_PAYLOAD.json", "gpu_plugin.json")}
            if any(member not in allowed and not member.startswith(prefix + "/licenses/") for member in names):
                raise ValueError("file-free metadata wheel contains undeclared entry points or payload")
            expected = {"mojolearn-core==" + version, "mojolearn-nvidia==" + version, "mojolearn-amd==" + version}
            requires = [re.sub(r"\s+", "", value).replace("(", "").replace(")", "")
                        for value in metadata.get_all("Requires-Dist", [])]
            actual = {r for r in requires if "extra==" not in r}
            if actual != expected:
                raise ValueError("file-free mojolearn must pin exactly the qualified core and both vendors")
            # Extras only forward to the core (packaging/linux/pack_wheel.py meta_project).
            extras = set(metadata.get_all("Provides-Extra", []))
            forwarded = {r for r in requires if "extra==" in r}
            if (extras - {"numpy", "verify", "torch", "max"}
                    or forwarded != {f'mojolearn-core[{x}]=={version};extra=="{x}"' for x in extras}):
                raise ValueError("file-free mojolearn extras must forward exactly to mojolearn-core at this version")
    found = sha256(path)
    qualified = [mapping for key, mapping in configuration.items()
                 if key.startswith("artifacts:") and isinstance(mapping, dict)
                 and (mapping.get(name) == found or mapping.get(path.name) == found)]
    if not is_meta and not qualified:
        raise ValueError("published wheel hash is absent from the qualified artifact inputs: " + path.name)
    if is_meta and not all(isinstance(configuration.get("artifacts:" + vendor), dict)
                           for vendor in ("nvidia", "amd")):
        raise ValueError("metadata wheel requires the qualified NVIDIA and AMD artifact sets")
    return dict(file=path.name, distribution=name, version=version, sha256=found, source_revision=source,
                scope="file-free pinned dependencies" if is_meta else "qualified artifact bytes")


def verify_ledger(ledger_path, root, source, *, log_dir=None):
    source_revision(root, source)
    ledger = requirements.RequirementsLedger(ledger_path, source_root=root, evidence_log_dir=log_dir)
    doc = ledger.read()
    if doc.get("source_revision") != source or not isinstance(doc.get("version"), str):
        raise ValueError("canonical ledger belongs to another frozen source/version")
    # Do not infer compiler flags from this transport process's environment.
    # Regenerate every required unit from the frozen source and recorded inputs;
    # a caller-authored shorter requirements list cannot authorize publication.
    manifest = requirements.requirements(root, source, doc["version"], configuration=doc.get("configuration", {}),
                                         evidence_log_dir=log_dir)
    ready, reasons = ledger.gate(manifest, stage="pre")
    waiver_path = os.environ.get("MOJOLEARN_OWNER_WAIVER")
    if not ready and waiver_path:
        # The owner's explicit waiver (tools/release.py --owner-waiver): rows whose requirement id
        # starts with a waived prefix are excused and printed; every other row still blocks.
        waiver = json.loads(Path(waiver_path).read_text())
        prefixes = tuple(waiver["prefixes"])
        waived = [r.split(":", 1)[0] for r in reasons if r.split(":", 1)[0].startswith(prefixes)]
        reasons = [r for r in reasons if not r.split(":", 1)[0].startswith(prefixes)]
        print("publication admission: %d ledger row(s) WAIVED by %s (%s)" % (len(waived), waiver["approved_by"], waiver_path), file=sys.stderr)
        ready = not reasons
    if not ready:
        raise ValueError("canonical release ledger blocks publication: " + "; ".join(reasons[:12]))
    config = manifest["configuration"]
    nvidia, amd = config.get("artifacts:nvidia", {}), config.get("artifacts:amd", {})
    if nvidia and amd and nvidia.get("mojolearn-core") != amd.get("mojolearn-core"):
        raise ValueError("qualified NVIDIA and AMD sets identify different shared core wheel bytes")
    return doc, manifest


def _tar_bytes(archive, name, data):
    info = tarfile.TarInfo(name)
    info.size, info.mode, info.mtime = len(data), 0o600, 0
    archive.addfile(info, io.BytesIO(data))


def create_bundle(ledger_path, root, source, wheels, output):
    ledger_path, root, output = Path(ledger_path), Path(root), Path(output)
    before = ledger_path.read_bytes()
    reads = {}
    with evidence.evidence_context(collect=reads):
        doc, manifest = verify_ledger(ledger_path, root, source, log_dir=output.parent / "dependency-logs")
    if ledger_path.read_bytes() != before:
        raise ValueError("canonical ledger changed during admission; retry metadata admission only")
    artifacts = [wheel_identity(path, doc["version"], source, manifest["configuration"]) for path in wheels]
    if not artifacts or len({row["file"] for row in artifacts}) != len(artifacts):
        raise ValueError("admission needs distinct actual wheel files")
    if not reads:
        raise ValueError("canonical pre gate retained no actual producer evidence")
    admission = dict(schema=SCHEMA, source_revision=source, version=doc["version"], artifacts=artifacts,
                     ledger_sha256=hashlib.sha256(before).hexdigest(), evidence=reads, policy=policy(root),
                     manifest_requirements_digest=digest(manifest["requirements"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        _tar_bytes(archive, "admission.json", json.dumps(admission, sort_keys=True).encode())
        _tar_bytes(archive, "ledger.json", before)
        stored = set()
        for original, expected in sorted(reads.items()):
            if sha256(original) != expected:
                raise ValueError("evidence changed after admission: " + original)
            if expected not in stored:
                archive.add(original, arcname="objects/" + expected, recursive=False)
                stored.add(expected)
    found = sha256(output)
    with open_bundle(output, found):
        pass  # detect evidence changing between the final hash and archive copy
    return dict(bundle_sha256=found, source_revision=source, version=doc["version"],
                key=PREFIX + found + ".tar.gz", artifacts=artifacts)


@contextmanager
def open_bundle(path, expected_sha):
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha) or sha256(path) != expected_sha:
        raise ValueError("admission bundle SHA256 differs")
    with tempfile.TemporaryDirectory(prefix="mojolearn-ledger-evidence-") as temporary:
        root = Path(temporary)
        with tarfile.open(path, "r:gz") as archive:
            seen = set()
            for member in archive:
                name = member.name
                valid = name in ("admission.json", "ledger.json") or re.fullmatch(r"objects/[0-9a-f]{64}", name)
                if not valid or name in seen or not member.isfile() or member.issym() or member.islnk():
                    raise ValueError("unsafe, duplicate or unsupported admission archive member")
                seen.add(name)
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open("wb") as out:
                    import shutil
                    shutil.copyfileobj(source, out)
        admission = strict_json((root / "admission.json").read_text())
        ledger_doc = strict_json((root / "ledger.json").read_text())
        if not isinstance(admission, dict) or not isinstance(ledger_doc, dict):
            raise ValueError("admission and ledger records must be JSON objects")
        if admission.get("schema") != SCHEMA or sha256(root / "ledger.json") != admission.get("ledger_sha256"):
            raise ValueError("admission schema or original ledger hash differs")
        mapping = admission.get("evidence")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("admission contains no producer evidence mapping")
        for original, value in mapping.items():
            if not isinstance(original, str) or not Path(original).is_absolute() or not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError("invalid original evidence path/object identity")
            if not (root / "objects" / value).is_file() or sha256(root / "objects" / value) != value:
                raise ValueError("admission evidence object is missing or corrupt")
        if seen != {"admission.json", "ledger.json", *("objects/" + value for value in mapping.values())}:
            raise ValueError("unreferenced admission archive members")
        yield root, admission


def verify_bundle(path, expected_sha, root, source):
    with open_bundle(path, expected_sha) as (unpacked, admission):
        if admission["source_revision"] != source or admission.get("policy") != policy(root):
            raise ValueError("admission frozen source or validation policy differs")
        with evidence.evidence_context(objects=unpacked / "objects", mapping=admission["evidence"]):
            doc, manifest = verify_ledger(unpacked / "ledger.json", root, source, log_dir=unpacked / "dependency-logs")
        if admission["version"] != doc["version"] or admission["manifest_requirements_digest"] != digest(manifest["requirements"]):
            raise ValueError("canonical frozen requirements differ from admitted inputs")
        return admission


def verify_wheels(path, expected_sha, source, wheels, *, ledger_path=None):
    # CI runs this only after its ledger_admission job independently passed.
    # The immutable bundle digest identifies that exact reviewed artifact map.
    with open_bundle(path, expected_sha) as (unpacked, admission):
        if admission["source_revision"] != source:
            raise ValueError("published artifacts have another frozen source")
        if ledger_path is not None and sha256(ledger_path) != admission["ledger_sha256"]:
            raise ValueError("canonical ledger changed after admission; recheck before public upload")
        doc = requirements.RequirementsLedger(unpacked / "ledger.json").read()
        declared = admission.get("artifacts", [])
        indexed = {row["file"]: row for row in declared}
        if not indexed or len(indexed) != len(declared) or not wheels:
            raise ValueError("admission artifact files are missing or duplicated")
        for wheel in wheels:
            if Path(wheel).name not in indexed or sha256(wheel) != indexed[Path(wheel).name]["sha256"]:
                raise ValueError("publication wheel bytes differ from verified ledger admission: " + Path(wheel).name)
            actual = wheel_identity(wheel, admission["version"], source, doc["configuration"])
            if actual != indexed[Path(wheel).name]:
                raise ValueError("publication wheel source/metadata differs from ledger admission")
        return len(wheels)


def r2_client():
    from release_ledger import R2, read_creds
    names = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")
    creds = {key: os.environ.get(key) for key in names}
    if not all(creds.values()):
        path = Path(os.environ.get("MOJOLEARN_R2_CREDS", "~/.mojolearn_r2")).expanduser()
        creds = read_creds(path) if path.is_file() else None
    if not creds:
        raise ValueError("private R2 credentials are required for ledger transport")
    return R2(creds)


def fetch_bundle(expected_sha, destination):
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        raise ValueError("missing or invalid requirements_bundle_sha256")
    data = r2_client().get(PREFIX + expected_sha + ".tar.gz")
    if data is None or hashlib.sha256(data).hexdigest() != expected_sha:
        raise ValueError("private ledger bundle is missing or has another digest")
    Path(destination).write_bytes(data)


def upload_bundle(path, key):
    """Stream large private evidence through the established multipart client.

    The JSON ledger client's single urllib PUT broke its connection on a
    232-MiB admission archive. AWS CLI multipart uploads retain the exact
    content-addressed archive and retry transport without rerunning tests.
    Credentials remain in the local process environment, never command text.
    """
    client = r2_client()
    environment = dict(os.environ, AWS_ACCESS_KEY_ID=client.creds['R2_ACCESS_KEY_ID'],
                       AWS_SECRET_ACCESS_KEY=client.creds['R2_SECRET_ACCESS_KEY'],
                       AWS_DEFAULT_REGION='auto')
    environment.pop('AWS_SESSION_TOKEN', None)
    subprocess.run(['aws', 's3', 'cp', str(path), 's3://' + client.creds['R2_BUCKET'] + '/' + key,
                    '--endpoint-url', 'https://' + client.creds['R2_ACCOUNT_ID'] + '.r2.cloudflarestorage.com',
                    '--only-show-errors'], env=environment, check=True, stdout=sys.stderr)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("bundle", "verify", "wheels"))
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source", required=True)
    parser.add_argument("--wheel", action="append", type=Path, default=[])
    parser.add_argument("--wheel-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--bundle-sha256")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args(argv)
    wheels = args.wheel + (sorted(args.wheel_dir.glob("*.whl")) if args.wheel_dir else [])
    if args.command == "bundle":
        if args.ledger is None or args.output is None:
            parser.error("bundle requires --ledger and --output")
        result = create_bundle(args.ledger, args.root, args.source, wheels, args.output)
        if args.upload:
            upload_bundle(args.output, result["key"])
        print(json.dumps(result, sort_keys=True))
        return 0
    with tempfile.TemporaryDirectory(prefix="mojolearn-private-admission-") as temporary:
        bundle = args.bundle or Path(temporary) / "admission.tar.gz"
        if args.bundle is None:
            fetch_bundle(args.bundle_sha256 or "", bundle)
        if args.command == "verify":
            result = verify_bundle(bundle, args.bundle_sha256 or "", args.root, args.source)
            print(json.dumps(dict(source_revision=result["source_revision"], version=result["version"],
                                  qualified_artifacts=len(result["artifacts"]))))
        else:
            print(json.dumps(dict(verified_wheels=verify_wheels(bundle, args.bundle_sha256 or "", args.source, wheels,
                                                               ledger_path=args.ledger))))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError, tarfile.TarError, zipfile.BadZipFile) as error:
        raise SystemExit("publication admission refused: " + str(error))
