"""Bind wheel verification inputs to one committed MojoLearn source tree.

This checks retained bytes and provenance, not numerical qualification. It
never downloads records, regenerates expected values or imports product code.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import subprocess


TABLE = "python/mojolearn/verify_reference/table.json"
RECORDS = "bench/results/identity_break"


def committed_files(repo, commit):
    raw = subprocess.check_output(["git", "-C", str(repo), "ls-tree", "-r", "-z", commit])
    result = {}
    for row in raw.split(b"\0"):
        if row:
            meta, path = row.split(b"\t", 1)
            mode, kind, oid = meta.decode().split()
            if kind == "blob" and mode in ("100644", "100755"):
                result[path.decode()] = oid
    return result


def validate(root, commit="HEAD", git_root=None):
    """Check checkout or exported staging tree against its source commit.

    Return the wheel-member -> source-file mapping for verification data only.
    Native builds and vendor libraries have their own provenance contracts.
    """
    root = Path(root).resolve()
    blobs = committed_files(git_root or root, commit)

    def checked(rel):
        path = root / rel
        if (not path.is_file() or path.is_symlink()
                or not path.resolve().is_relative_to(root)):
            raise ValueError("verification input missing or outside repository: " + rel)
        data = path.read_bytes()
        oid = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
        if blobs.get(rel) != oid:
            raise ValueError("verification input not identical to source commit: " + rel)
        return path, data

    _, raw = checked(TABLE)
    table = json.loads(raw)
    if not isinstance(table.get("records"), list):
        raise ValueError("reference table has no record inventory")
    for record in table["records"]:
        rel = str(Path(RECORDS) / record["dir"] / record["file"])
        # Reject traversal lexically, even when it would land back inside root.
        if Path(record["dir"]).is_absolute() or ".." in Path(rel).parts or Path(record["file"]).name != record["file"]:
            raise ValueError("reference record must be inside " + RECORDS + ": " + rel)
        _, data = checked(rel)
        if record.get("sha256") != hashlib.sha256(data).hexdigest():
            raise ValueError("reference record checksum missing or mismatched: " + rel)
        column = json.loads(data)
        if column.get("commit") != record["commit"] or column.get("vendor") != record["vendor"]:
            raise ValueError("reference record provenance mismatch: " + rel)

    entries = {}
    package = root / "python/mojolearn"
    for directory in ("verify_reference", "reference_cards"):
        for path in sorted((package / directory).rglob("*")):
            if path.is_file():
                rel = path.relative_to(root).as_posix()
                # CTR files are generated copies from committed fixture sources.
                if directory == "verify_reference" and "ctr_models" in path.relative_to(package).parts:
                    continue
                checked(rel)
                entries["mojolearn/" + path.relative_to(package).as_posix()] = path
    checked("python/mojolearn/host_surface.py")
    manifest = runpy.run_path(str(package / "host_surface.py"))
    record_dir = manifest["training_gpu_column_record"]()
    for rel in manifest["TRAINING_GPU_COLUMNS"]:
        path, _ = checked(rel)
        entries[f"mojolearn/identity_columns/{record_dir}/{path.name}"] = path
    checked("python/mojolearn/_verification_ctr_models.py")
    from verification_ctr_payload import model_entries
    for name, path in model_entries(root).items():
        checked(path.relative_to(root).as_posix())
        entries["mojolearn/verify_reference/ctr_models/" + name] = path
    # Historical inventory shipped as Python is also evidence, not build output.
    checked("python/mojolearn/_verification_evidence_data.py")
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--git-root", type=Path)
    parser.add_argument("--commit", default="HEAD")
    args = parser.parse_args()
    entries = validate(args.root, args.commit, args.git_root)
    print(f"Verified {len(entries)} verification payload files against {args.commit}")


if __name__ == "__main__":
    main()
