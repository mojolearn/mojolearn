"""Read-only equivalence proof for retained autograd executables; never builds.

Use committed bytes, not the working tree. Resolve every transitive import with
the normal cache resolver. Missing/generated/unresolved inputs refuse reuse.
The original receipt, compiler hash, command, targets and binary hashes remain
unchanged. This proves no new compilation is necessary, not runtime success.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import bincache

SCHEMA = "mojolearn.autograd-artifact-reuse/1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def inputs(root, revision, seeds):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args])
    revision = git("rev-parse", "--verify", revision + "^{commit}").decode().strip()
    entries = {}
    for record in git("ls-tree", "-rz", revision).split(b"\0"):
        if not record:
            continue
        meta, path = record.split(b"\t", 1)
        mode, kind, oid = meta.decode().split()
        name = path.decode()
        if name.endswith(".mojo") or name in ("pixi.lock", "pixi.toml"):
            if kind != "blob" or mode not in ("100644", "100755"):
                raise ValueError("unsupported source mode in autograd closure: " + name)
            entries[name] = (mode, oid)
    if any(name not in entries for name in [*seeds, "pixi.lock", "pixi.toml"]):
        raise ValueError("autograd source or toolchain lock absent at " + revision)
    # A small source-only scratch tree supplies exact case-sensitive discovery
    # semantics to bincache.resolve. No checkout, untracked bytes, or shared files.
    with tempfile.TemporaryDirectory(prefix="autograd-inputs-") as temporary:
        tree = Path(temporary)
        identifiers = list(dict.fromkeys(oid for _, oid in entries.values()))
        batch = subprocess.check_output(["git", "-C", str(root), "cat-file", "--batch"],
                                        input="".join(oid + "\n" for oid in identifiers).encode())
        blobs, offset = {}, 0
        for oid in identifiers:
            end = batch.index(b"\n", offset)
            header = batch[offset:end].split()
            if len(header) != 3 or header[1] != b"blob":
                raise ValueError("unavailable committed autograd source")
            size = int(header[2])
            blobs[oid] = batch[end + 1:end + 1 + size]
            offset = end + size + 2
        for name, (_, oid) in entries.items():
            path = tree / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blobs[oid])
        selected, pending = set(), list(seeds)
        while pending:
            name = pending.pop()
            if name in selected:
                continue
            selected.add(name)
            path = tree / name
            for module, symbols in bincache.parse_imports(path.read_text()):
                resolved = bincache.resolve(module, symbols, path, [tree])
                if resolved is None:
                    raise ValueError("unresolved autograd import: " + name + ": " + module)
                pending.extend(str(item.relative_to(tree)) for item in resolved)
        selected.update(("pixi.lock", "pixi.toml"))
        return {name: dict(mode=entries[name][0], sha256=hashlib.sha256(blobs[entries[name][1]]).hexdigest())
                for name in sorted(selected)}


def prove(root, original, requested):
    seeds = [row["source"] for row in original["files"]]
    before = inputs(root, original["source_commit"], seeds)
    after = inputs(root, requested, seeds)
    if before != after:
        changed = sorted(name for name in set(before) | set(after) if before.get(name) != after.get(name))
        raise ValueError("autograd inputs changed; affected artifacts need a new build: " + ", ".join(changed[:12]))
    if any(before[row["source"]]["sha256"] != row["source_sha256"] for row in original["files"]):
        raise ValueError("retained source hash differs from its committed build source")
    return dict(schema=SCHEMA, original_source_commit=original["source_commit"], source_commit=requested,
                original_manifest_sha256=digest(original), original_inputs=before, requested_inputs=after,
                compiler=original["compiler"], target=original["target"], target_cpu=original["target_cpu"],
                reason="unchanged transitive sources, toolchain locks, compiler, targets and compile commands")


def source_matches(document, requested):
    if document.get("source_commit") == requested:
        return True
    reuse = document.get("reuse") or {}
    original = {key: value for key, value in document.items() if key != "reuse"}
    before = reuse.get("original_inputs")
    return (reuse.get("schema") == SCHEMA and reuse.get("source_commit") == requested
            and reuse.get("original_source_commit") == document.get("source_commit")
            and reuse.get("original_manifest_sha256") == digest(original)
            and isinstance(before, dict) and bool(before) and before == reuse.get("requested_inputs")
            and reuse.get("compiler") == document.get("compiler")
            and reuse.get("target") == document.get("target")
            and reuse.get("target_cpu") == document.get("target_cpu"))
