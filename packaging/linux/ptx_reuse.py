"""Release-only admission of unchanged PTX bytes from a completed build leg.

The v2 manifest's top-level source_commit binds the current core. An admission
retains the original build manifest, build proof and runtime manifest as hashed
raw witnesses; their original source commits and arithmetic evidence never move.
This module stages metadata only. It neither compiles nor changes loaded bytes.
"""
import hashlib
import json
from pathlib import Path
import re

SCHEMA = "mojolearn.ptx-release-admission.v1"
#: HOST_OWNER_ADMISSION (2026-10-31, lane release-auto-reuse-2): the PTX set's
#: GPU bytes, indexes, runtime and adapters are the retained PTX leg's own,
#: proved against its original build proof exactly as before; its host/ files
#: are the host owner's (tools/release.py host_owner_leg, cuda-sm_89), proved
#: against THAT leg's original build proof. Only host rows of the set identity
#: and the runtime identity digest may differ from the PTX leg's original
#: identity: every GPU binding identity must be equal. tools/release.py looks
#: for this marker in the frozen source before taking the PTX set this way.
HOST_OWNER_ADMISSION = "host-owner"
HOST_OWNER_PREFIX = "mojolearn/cuda/sm_89/"
MANIFEST = "PTX_BASELINE.json"
INDEX = "_mojolearn_merged.json"
PREFIX = "mojolearn/cuda/sm_80/"


def digest(doc):
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def valid_hash(value, size=64):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{%d}" % size, value) is not None


def require(condition, message):
    if not condition:
        raise ValueError("PTX reuse: " + message)


def normalized_set_identity(record):
    """Validate an identity, dropping only the obsolete PTX release-commit salt.

    Validate the saved digest *before* normalization. A corrupt saved identity,
    unknown field, missing closure/runtime digest or non-PTX salt is not reusable.
    """
    if not isinstance(record, dict):
        return None
    doc = {k: v for k, v in record.items() if k != "digest"}
    if (set(doc) not in ({"set", "bindings", "runtime"},
                        {"set", "bindings", "runtime", "source_commit"})
            or not isinstance(doc.get("set"), str)
            or not isinstance(doc.get("bindings"), dict) or not doc["bindings"]
            or not all(isinstance(k, str) and valid_hash(v) for k, v in doc["bindings"].items())
            or not valid_hash(doc.get("runtime")) or record.get("digest") != digest(doc)):
        return None
    if "source_commit" in doc:
        if doc["set"] != "cuda/sm_80" or not valid_hash(doc["source_commit"], 40):
            return None
        del doc["source_commit"]
    return dict(doc, digest=digest(doc))


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _actual(root):
    gpu, host, indexes = {}, {}, {}
    for sub in ("", "identical", "host"):
        directory = root / sub
        for p in sorted(directory.glob("*.so")):
            (host if sub == "host" else gpu)[p.relative_to(root).as_posix()] = file_hash(p)
        index = directory / INDEX
        if index.is_file():
            indexes[index.relative_to(root).as_posix()] = file_hash(index)
    runtime = {p.name: file_hash(p) for p in sorted((root / ".libs").iterdir()) if p.is_file()}
    return gpu, host, indexes, runtime


def validate_admission(doc, root, origins, expected_identity=None):
    """Check a release envelope against original proofs and the staged bytes.

    Called during assembly and again by the packer. The packer's existing core
    source check independently binds the envelope to the actual release checkout.
    """
    from staged_native_admission import complete_native_proof, decoded
    from verify_linux_surface_qualification import load_gpu_plugins

    root = Path(root)
    admission = doc.get("release_admission")
    require(isinstance(admission, dict) and admission.get("schema") == SCHEMA, "missing release admission")
    origin = admission.get("origin")
    require(isinstance(origin, dict) and origin.get("leg") == "cuda-sm_80"
            and origins.get("cuda-sm_80") == origin, "admission differs from retained leg origin")
    require(all(valid_hash(origin.get(k)) for k in ("proof_sha256", "set_identity_digest", "tooling_digest"))
            and all(valid_hash(origin.get(k), 40) for k in ("source_commit", "admitted_for")),
            "incomplete leg identity/tooling/proof")
    current = normalized_set_identity(admission.get("set_identity"))
    original = normalized_set_identity(origin.get("original_set_identity"))
    owner = admission.get("host_owner")
    if owner is None:
        require(current is not None and original == current and current["set"] == "cuda/sm_80"
                and current["digest"] == origin["set_identity_digest"], "changed or missing set identity")
    else:
        require(isinstance(owner, dict) and owner.get("kind") == HOST_OWNER_ADMISSION,
                "unsupported host owner admission")

        def gpu_rows(identity):
            # a host row's key is `<target>/host/-/host/<name>` (`host/<name>` in
            # older plans); every other row is a GPU binding of the set
            return {k: v for k, v in identity["bindings"].items()
                    if "/host/" not in k and not k.startswith("host/")}
        require(current is not None and original is not None and current["set"] == original["set"] == "cuda/sm_80"
                and current["digest"] == origin["set_identity_digest"] and gpu_rows(current)
                and gpu_rows(current) == gpu_rows(original), "changed or missing GPU binding identity")
    require(origin["original_set_identity"].get("source_commit", origin["source_commit"])
            == origin["source_commit"], "original identity names a different build source")
    if expected_identity is not None:
        require(normalized_set_identity(expected_identity) == current, "current set identity differs")
    require(doc.get("source_commit") == origin["admitted_for"] and doc.get("source_dirty") is False,
            "release core binding differs or is dirty")
    original_manifest = decoded(admission.get("original_manifest"))
    proof = decoded(admission.get("original_build_proof"))
    stage = decoded(admission.get("original_stage_manifest"))
    require(all(isinstance(d, dict) for d in (original_manifest, proof, stage)), "malformed original witness")
    require("release_admission" not in original_manifest, "original manifest is not a build manifest")
    require(original_manifest.get("source_commit") == origin["source_commit"]
            and original_manifest.get("source_dirty") is False, "original source differs or is dirty")
    require(proof.get("source_commit") == origin["source_commit"] and complete_native_proof(proof)
            and admission["original_build_proof"]["sha256"] == origin["proof_sha256"],
            "missing, incomplete or stale original build proof")
    restored = {k: v for k, v in doc.items() if k != "release_admission"}
    restored["source_commit"] = original_manifest["source_commit"]
    require(restored == original_manifest, "original arithmetic/build metadata was changed")
    gpu, host, indexes, runtime = _actual(root)
    load_gpu_plugins().validate_baseline_manifest(original_manifest, gpu, indexes={
        k: v for k, v in indexes.items() if not k.startswith("host/")})
    require(proof.get("extensions") == {PREFIX + k: v for k, v in gpu.items()} and gpu,
            "GPU bytes differ from original build proof")
    owner_indexes = {}
    if owner is None:
        require(proof.get("host_extension", {}) == {PREFIX + k: v for k, v in host.items()},
                "host bytes differ from original build proof")
    else:
        owner_origin = owner.get("origin")
        require(isinstance(owner_origin, dict) and owner_origin.get("leg") == "cuda-sm_89"
                and origins.get("cuda-sm_89") == owner_origin
                and owner_origin.get("admitted_for") == origin["admitted_for"]
                and valid_hash(owner_origin.get("proof_sha256")), "host owner differs from its retained leg origin")
        owner_proof = decoded(owner.get("build_proof"))
        require(isinstance(owner_proof, dict) and complete_native_proof(owner_proof)
                and owner["build_proof"]["sha256"] == owner_origin["proof_sha256"]
                and owner_proof.get("source_commit") == owner_origin.get("source_commit"),
                "missing, incomplete or stale host owner build proof")
        require(host and owner_proof.get("host_extension", {}) == {HOST_OWNER_PREFIX + k: v for k, v in host.items()},
                "host bytes differ from the host owner's build proof")
        owner_indexes = owner_proof.get("merged_indexes", {})
    proved_indexes = proof.get("merged_indexes", {})
    require(proof.get("binding_layout", "per-binding") == ("merged" if indexes else "per-binding"),
            "original proof binding layout differs")
    # Every index the set carries is proved: its GPU tier indexes by the PTX
    # leg's proof, and in host-owner mode its host/ index by the owner's proof.
    own = {k for k in indexes if owner is None or not k.startswith("host/")}
    proved_own = {k for k in proved_indexes if owner is None or not k.startswith(PREFIX + "host/")} \
        if isinstance(proved_indexes, dict) else None
    require(proved_own == {PREFIX + k for k in own}, "missing or stale original index proof")
    for rel, value in indexes.items():
        if owner is not None and rel.startswith("host/"):
            row = owner_indexes.get(HOST_OWNER_PREFIX + rel)
            index = json.loads((root / rel).read_text())
            require(isinstance(row, dict) and row.get("sha256") == value
                    and isinstance(index, dict) and row.get("files") == index.get("files"),
                    "host index differs from the host owner's build proof: " + rel)
            continue
        row = proved_indexes[PREFIX + rel]
        index = json.loads((root / rel).read_text())
        require(isinstance(row, dict) and row.get("sha256") == value
                and isinstance(index, dict) and row.get("files") == index.get("files"),
                "index differs from original build proof: " + rel)
    records = stage.get("staged_libs")
    require(isinstance(records, list) and all(isinstance(r, dict) for r in records),
            "missing original runtime inventory")
    require(runtime and len(records) == len(runtime)
            and {r.get("name"): r.get("sha256") for r in records} == runtime,
            "runtime bytes differ from original staged inventory")
    if "runtime_libraries" in proof:
        require(proof["runtime_libraries"] == runtime, "runtime bytes differ from original build proof")
    adapters = {p.name: file_hash(p) for p in sorted((root / "_adapters").glob("*")) if p.is_file()}
    adapter_proof = {PREFIX + "_adapters": dict(files=adapters, manifest_sha256=adapters.get("adapters.json"))} if adapters else {}
    require(proof.get("adapters", {}) == adapter_proof, "adapter bytes differ from original build proof")
    return True


def rebind_manifest(root, original_root, origin, current_commit, current_identity, host_owner=None):
    """Write only the new release manifest, preserving original raw witnesses.
    `host_owner` ({"origin": the host owner leg's origin}) admits the set's
    host/ files as that leg's (HOST_OWNER_ADMISSION)."""
    from staged_native_admission import witness, decoded

    require(valid_hash(current_commit, 40) and origin.get("admitted_for") == current_commit,
            "admission is not for the current core")
    require(origin.get("release_build"), "missing original release build directory")
    original_root = Path(original_root)
    proof_path = Path(origin["release_build"]) / "build" / "build-provenance.json"
    manifest = witness(original_root / MANIFEST)
    doc = decoded(manifest)
    doc["source_commit"] = current_commit
    doc["release_admission"] = dict(
        schema=SCHEMA, origin=origin, set_identity=current_identity,
        original_manifest=manifest, original_build_proof=witness(proof_path),
        original_stage_manifest=witness(original_root / "manifest.json"))
    origins = {"cuda-sm_80": origin}
    if host_owner is not None:
        owner_origin = host_owner["origin"]
        require(owner_origin.get("release_build"), "missing host owner release build directory")
        doc["release_admission"]["host_owner"] = dict(
            kind=HOST_OWNER_ADMISSION, origin=owner_origin,
            build_proof=witness(Path(owner_origin["release_build"]) / "build" / "build-provenance.json"))
        origins["cuda-sm_89"] = owner_origin
    validate_admission(doc, root, origins, expected_identity=current_identity)
    (Path(root) / MANIFEST).write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return doc
