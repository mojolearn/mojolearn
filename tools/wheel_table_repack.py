#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""DATA-ONLY REPACK of a built release wheel with the admitted reference table (Andrew 2026-11-05: one
identity run per release, on the built wheels; the admitted table ships without rebuilding any binary).

    python3 tools/wheel_table_repack.py WHEEL --table TABLE.json --out DIR [--receipt FILE]

Writes DIR/<same wheel file name>: every member byte-for-byte the input's (same order, compression and
timestamps) except `mojolearn/verify_reference/table.json`, which becomes TABLE.json, and the matching line of
`*.dist-info/RECORD` (its sha256 and size). Then it reopens both wheels and checks, by sha256, that every other
member is unchanged, so no binding, payload or Python file moved, and writes a receipt
(schema mojolearn.wheel-table-repack.v1) naming the input and output wheel digests, the old and new table
digests and the digest of every native member (.so, .dylib, .metallib, .ptx, .hsaco, .cubin, .bin).
A wheel without the table (the GPU plugins) is refused: it needs no repack, its bytes ship as built.
Nothing here builds, installs or runs a lane. Tests: tools/test_wheel_table_repack.py.
"""
import argparse
import base64
import hashlib
import json
import sys
import zipfile
from pathlib import Path

SCHEMA = "mojolearn.wheel-table-repack.v1"
TABLE_MEMBER = "mojolearn/verify_reference/table.json"
NATIVE = (".so", ".dylib", ".metallib", ".ptx", ".hsaco", ".cubin", ".bin")


class RepackError(RuntimeError):
    pass


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def record_hash(data):
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def members(path):
    with zipfile.ZipFile(path) as z:
        return {i.filename: sha256(z.read(i.filename)) for i in z.infolist() if not i.filename.endswith("/")}


def new_record(text, table):
    """RECORD with only the table's line rewritten; every other line kept byte-for-byte."""
    out, hit = [], 0
    for line in text.splitlines(keepends=True):
        if line.split(",", 1)[0] == TABLE_MEMBER:
            end = "\r\n" if line.endswith("\r\n") else ("\n" if line.endswith("\n") else "")
            out.append("%s,%s,%d%s" % (TABLE_MEMBER, record_hash(table), len(table), end))
            hit += 1
        else:
            out.append(line)
    if hit != 1:
        raise RepackError("RECORD names %s %d time(s), not once" % (TABLE_MEMBER, hit))
    return "".join(out)


def repack(wheel, table_path, out_dir):
    wheel, out_dir = Path(wheel), Path(out_dir)
    table = Path(table_path).read_bytes()
    json.loads(table)   # an unreadable table never ships
    out = out_dir / wheel.name
    if out.resolve() == wheel.resolve():
        raise RepackError("the output would overwrite the input wheel; use another --out")
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(wheel) as src:
        infos = src.infolist()
        names = [i.filename for i in infos]
        if TABLE_MEMBER not in names:
            raise RepackError("%s carries no %s: ship it as built" % (wheel.name, TABLE_MEMBER))
        records = [n for n in names if n.endswith(".dist-info/RECORD")]
        if len(records) != 1:
            raise RepackError("%s has %d RECORD files" % (wheel.name, len(records)))
        old_table = src.read(TABLE_MEMBER)
        tmp = out.with_suffix(".tmp")
        with zipfile.ZipFile(tmp, "w") as dst:
            for info in infos:
                data = src.read(info.filename)
                if info.filename == TABLE_MEMBER:
                    data = table
                elif info.filename == records[0]:
                    data = new_record(data.decode(), table).encode()
                clone = zipfile.ZipInfo(info.filename, date_time=info.date_time)
                clone.compress_type, clone.external_attr = info.compress_type, info.external_attr
                clone.create_system, clone.comment, clone.extra = info.create_system, info.comment, info.extra
                dst.writestr(clone, data)
        tmp.replace(out)
    before, after = members(wheel), members(out)
    moved = sorted(n for n in set(before) | set(after)
                   if before.get(n) != after.get(n) and n not in (TABLE_MEMBER, records[0]))
    if moved or set(before) != set(after):
        out.unlink()
        raise RepackError("a member other than the table and RECORD changed: %s" % ", ".join(moved[:10]))
    if after[TABLE_MEMBER] != sha256(table):
        out.unlink()
        raise RepackError("the repacked table is not the admitted table")
    return dict(schema=SCHEMA, input=dict(file=wheel.name, sha256=sha256(wheel.read_bytes())),
                output=dict(file=out.name, path=str(out), sha256=sha256(out.read_bytes())),
                table=dict(member=TABLE_MEMBER, old_sha256=sha256(old_table), new_sha256=sha256(table),
                           source=str(Path(table_path).resolve())),
                unchanged_members=len(before) - 2,
                native_members={n: d for n, d in sorted(before.items()) if n.endswith(NATIVE)})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("wheel")
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--receipt", default="")
    a = ap.parse_args(argv)
    try:
        doc = repack(a.wheel, a.table, a.out)
    except (RepackError, OSError, ValueError, zipfile.BadZipFile) as exc:
        print("wheel_table_repack: REFUSED %s" % exc, file=sys.stderr)
        return 1
    receipt = Path(a.receipt or (Path(a.out) / (Path(a.wheel).name + ".repack.json")))
    receipt.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print("REPACKED %s table %s -> %s, %d member(s) unchanged (%d native); receipt %s" % (
        doc["output"]["file"], doc["table"]["old_sha256"][:12], doc["table"]["new_sha256"][:12],
        doc["unchanged_members"], len(doc["native_members"]), receipt))
    return 0


if __name__ == "__main__":
    sys.exit(main())
