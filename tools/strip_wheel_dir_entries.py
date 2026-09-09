#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Remove zero-length DIRECTORY entries from a repaired wheel. DEVIATION 2295.

`pack_wheel.py` writes files only; `auditwheel repair` rewrites the archive and
adds explicit directory entries (15 on the 0.7.0 wheel). RECORD does not list
directories, and never has, so the qualification driver's
`declared == set(paths)` check refused the very wheel it exists to admit.

The right fix is in the checkers, and it is made. But the on-device driver
lives in `tools/linux_surface_qualification.sh`, which is IN the native
inventory, so changing it moves the source fingerprint and invalidates three
per-architecture build proofs that cost real rentals. This tool takes the
other route: it makes the artifact match what the toolchain already expects,
without touching a single file the inventory covers.

WHAT CHANGES: the zip's central directory loses N zero-length entries whose
names end in "/". WHAT DOES NOT: every file's name, bytes, order, compression
and hash, RECORD itself, and therefore everything RECORD attests. A zip
directory entry carries no data; pip has never required one.

This is a post-audit byte change, which the runbook allows only WITH explicit
provenance, so a receipt is written beside the wheel naming both digests and
every entry removed. It is never a bypass: run auditwheel first, always.

  python3 tools/strip_wheel_dir_entries.py IN.whl OUT.whl [--receipt R.json]
"""
import argparse
import hashlib
import json
import pathlib
import shutil
import sys
import zipfile


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('source')
    ap.add_argument('dest')
    ap.add_argument('--receipt', default='')
    args = ap.parse_args()
    src, dst = pathlib.Path(args.source), pathlib.Path(args.dest)
    if dst.exists():
        sys.exit('refusing to overwrite ' + str(dst))
    dst.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(src) as z:
        infos = z.infolist()
        removed = [i.filename for i in infos if i.filename.endswith('/')]
        kept = [i for i in infos if not i.filename.endswith('/')]
        if not removed:
            shutil.copyfile(src, dst)
            print('no directory entries; copied unchanged')
        else:
            # Preserve every entry's own compression and metadata; only the
            # directory entries are dropped.
            with zipfile.ZipFile(dst, 'w') as out:
                for info in kept:
                    out.writestr(info, z.read(info.filename),
                                 compress_type=info.compress_type)

    # The bytes RECORD attests must be identical on both sides. Anything else
    # would make this a repack rather than a removal, so it is checked.
    with zipfile.ZipFile(src) as a, zipfile.ZipFile(dst) as b:
        an = [n for n in a.namelist() if not n.endswith('/')]
        bn = b.namelist()
        assert an == bn, 'file list changed'
        for n in an:
            assert hashlib.sha256(a.read(n)).digest() == hashlib.sha256(b.read(n)).digest(), \
                'content changed for ' + n

    receipt = dict(schema='mojolearn.wheel.dir-entry-strip.v1', deviation=2295,
                   source=src.name, source_sha256=sha256(src),
                   dest=dst.name, dest_sha256=sha256(dst),
                   removed_directory_entries=sorted(removed),
                   files_unchanged=len(bn),
                   note='zip directory entries removed; RECORD, file names, bytes and hashes identical')
    text = json.dumps(receipt, indent=2)
    if args.receipt:
        pathlib.Path(args.receipt).write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
