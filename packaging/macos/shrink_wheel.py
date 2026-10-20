"""Shrink the built macOS wheel under PyPI's per-file limit, in place.

PyPI refuses any file over 100 MiB (python/mojolearn/gpu_plugins.py wheel_size_limit; no larger allowance exists).
0.8.55's macOS wheel was 100.6 MiB and the upload got HTTP 400. Two changes that leave the code untouched:

- `strip -x` drops each binding's local symbols (names no loader can look up); exported symbols, the module init and
  the embedded Metal library stay. `strip` voids the ad-hoc signature, so each binding is signed again with `codesign -s -`.
- every member is written with deflate level 9 (setuptools writes level 6).

Measured on 0.8.55: 100.6 MiB -> about 97 MiB. RECORD is rewritten with the new hashes and sizes.
Exits 1 if the wheel is still over the limit, so an oversized wheel fails the build, not the upload.

    python3 packaging/macos/shrink_wheel.py python/dist/mojolearn-*.whl
"""
import base64
import hashlib
import pathlib
import subprocess
import sys
import tempfile
import zipfile

LIMIT = 100 * 1024**2


def _record_hash(data):
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


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
    after = wheel.stat().st_size
    print(f"shrink_wheel: {wheel.name} {before / 2**20:.2f} MiB -> {after / 2**20:.2f} MiB (PyPI limit {LIMIT / 2**20:.0f} MiB)")
    if after > LIMIT:
        print(f"shrink_wheel: ERROR {wheel.name} is over PyPI's per-file limit; PyPI would refuse it", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(max([shrink(w) for w in sys.argv[1:]] or [2]))
