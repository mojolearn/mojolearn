"""Shrink the built macOS wheel under PyPI's per-file limit, in place.

PyPI refuses any file over 100 MiB (python/mojolearn/gpu_plugins.py wheel_size_limit; no larger allowance exists).
0.8.55's macOS wheel was 100.6 MiB and the upload got HTTP 400. Two changes that leave the code untouched:

- `strip -x` drops each binding's local symbols (names no loader can look up); exported symbols, the module init and
  the embedded Metal library stay. `strip` voids the ad-hoc signature, so each binding is signed again with `codesign -s -`.
- every member is written with deflate level 9 (setuptools writes level 6).
- if this still exceeds the limit and AdvanceCOMP's `advzip` is available,
  recompress with standard Zopfli DEFLATE, accepting only byte-identical members
  and installation metadata. `*.compression.json` retains the tool and proof.

Measured on 0.8.55: 100.6 MiB -> about 97 MiB. RECORD is rewritten with the new hashes and sizes, and so is
the framework adapters' manifest (mojolearn/_adapters/adapters.json, packaging/adapters/manifest.py): it names
the shipped bytes, which this rewrite changes, and is checked against them once more here.
Exits 1 if the wheel is still over the limit, so an oversized wheel fails the build, not the upload.

    python3 packaging/macos/shrink_wheel.py python/dist/mojolearn-*.whl
"""
import base64
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import zipfile

LIMIT = 100 * 1024**2
ADAPTER_MANIFEST = "mojolearn/_adapters/adapters.json"


def _adapter_manifest():
    import importlib.util
    path = pathlib.Path(__file__).resolve().parents[1] / "adapters" / "manifest.py"
    spec = importlib.util.spec_from_file_location("mojolearn_adapter_manifest", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record_hash(data):
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def zip_witness(path):
    """Bind every uncompressed member (including RECORD) and install metadata."""
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len({i.filename for i in infos}) != len(infos):
            raise ValueError("duplicate wheel members before/after recompression")
        if any(i.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED) for i in infos):
            raise ValueError("wheel compression is not standard store/DEFLATE")
        return dict(comment=archive.comment.hex(), members={i.filename: dict(
            sha256=hashlib.sha256(archive.read(i)).hexdigest(), bytes=i.file_size,
            external_attr=i.external_attr, internal_attr=i.internal_attr,
            create_system=i.create_system, date_time=list(i.date_time),
            extra=i.extra.hex(), comment=i.comment.hex()) for i in infos})


def recompress_deflate(wheel):
    """Optional standard ZIP optimization; never change a shipped member byte.

    Separately compiled retained groups can exceed PyPI's limit after strip and
    zlib9. AdvanceCOMP's documented Zopfli mode searches for a smaller standard
    DEFLATE stream. It does not relink code or require a nonstandard installer.
    The original archive survives any command failure or witness mismatch.
    """
    tool = shutil.which("advzip")
    if tool is None:
        print("shrink_wheel: stronger standard DEFLATE unavailable (install advancecomp for advzip)")
        return
    wheel = pathlib.Path(wheel)
    before = zip_witness(wheel)
    original = hashlib.sha256(wheel.read_bytes()).hexdigest()
    candidate = wheel.with_suffix(".recompressing")
    if candidate.exists():
        raise ValueError("recompression output already exists: " + str(candidate))
    argv = [tool, "--recompress", "--shrink-insane", "--iter=1", "--keep-file-time", str(candidate)]
    receipt = dict(schema="mojolearn.wheel.deflate-recompression.v1", command=argv,
                   tool=str(pathlib.Path(tool).resolve()), tool_sha256=hashlib.sha256(pathlib.Path(tool).read_bytes()).hexdigest(),
                   tool_version=subprocess.check_output([tool, "--version"], text=True, timeout=30).strip(),
                   before_sha256=original, before_bytes=wheel.stat().st_size, witness=before,
                   complete=False, members_and_metadata_identical=None, exit_code=None)
    shutil.copy2(wheel, candidate)
    try:
        result = subprocess.run(argv, check=True, timeout=1200)
        receipt["exit_code"] = result.returncode
        receipt["members_and_metadata_identical"] = zip_witness(candidate) == before
        if not receipt["members_and_metadata_identical"]:
            raise ValueError("recompression changed wheel members or install metadata")
        if hashlib.sha256(wheel.read_bytes()).hexdigest() != original:
            raise ValueError("original wheel changed during recompression")
        receipt.update(after_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
                       after_bytes=candidate.stat().st_size, complete=True)
        receipt["selected"] = receipt["after_bytes"] < receipt["before_bytes"]
        if receipt["selected"]:
            candidate.replace(wheel)
    except Exception as exc:
        receipt["error"] = str(exc)
        if isinstance(exc, subprocess.CalledProcessError):
            receipt["exit_code"] = exc.returncode
        raise
    finally:
        wheel.with_suffix(".compression.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        candidate.unlink(missing_ok=True)


def shrink(wheel):
    wheel = pathlib.Path(wheel)
    before = wheel.stat().st_size
    with zipfile.ZipFile(wheel) as zin:
        infos = zin.infolist()
        members = {i.filename: zin.read(i) for i in infos}
    record_name = next(n for n in members if n.endswith(".dist-info/RECORD"))
    with tempfile.TemporaryDirectory() as tmp:
        for name in members:
            if not name.endswith((".so", ".dylib")):
                continue
            path = pathlib.Path(tmp) / "b.so"
            path.write_bytes(members[name])
            subprocess.run(["strip", "-x", str(path)], check=True, capture_output=True)
            subprocess.run(["codesign", "-f", "-s", "-", str(path)], check=True, capture_output=True)
            members[name] = path.read_bytes()
    if ADAPTER_MANIFEST in members:
        manifest = _adapter_manifest()
        base = ADAPTER_MANIFEST.rsplit("/", 1)[0] + "/"
        members[ADAPTER_MANIFEST] = manifest.refresh(members[ADAPTER_MANIFEST], lambda n: members[base + n],
                                                     "shrink_wheel.py: strip -x and codesign -f -s -")
        shipped = {n[len(base):]: (hashlib.sha256(d).hexdigest(), len(d)) for n, d in members.items()
                   if n.startswith(base) and n != ADAPTER_MANIFEST}
        import json
        manifest.validate(json.loads(members[ADAPTER_MANIFEST]), shipped, vendor="metal", arch="apple")
    rows = []
    for i in infos:
        if i.filename == record_name:
            continue
        rows.append(f"{i.filename},{_record_hash(members[i.filename])},{len(members[i.filename])}")
    rows.append(f"{record_name},,")
    members[record_name] = ("\n".join(rows) + "\n").encode()
    out = wheel.with_suffix(".shrinking")
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zout:
        for i in infos:
            info = zipfile.ZipInfo(i.filename, date_time=i.date_time)
            info.external_attr = i.external_attr
            info.compress_type = zipfile.ZIP_DEFLATED
            zout.writestr(info, members[i.filename], compresslevel=9)
    out.replace(wheel)
    if wheel.stat().st_size > LIMIT:
        recompress_deflate(wheel)
    after = wheel.stat().st_size
    print(f"shrink_wheel: {wheel.name} {before / 2**20:.2f} MiB -> {after / 2**20:.2f} MiB (PyPI limit {LIMIT / 2**20:.0f} MiB)")
    if after > LIMIT:
        print(f"shrink_wheel: ERROR {wheel.name} is over PyPI's per-file limit; PyPI would refuse it", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(max([shrink(w) for w in sys.argv[1:]] or [2]))
