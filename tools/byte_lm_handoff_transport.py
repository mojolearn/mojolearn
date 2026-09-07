#!/usr/bin/env python3
"""Bounded file-only compact handoff transport. Never imports the model."""
import argparse
import hashlib
import json
from pathlib import Path
import stat
import zipfile

from byte_lm_resume_handoff import bundle_bound, load_handoff

CAP = 32 * 1024 * 1024


def check(path, digest, vendor, kind):
    bundle_bound(path)
    return load_handoff(path, digest, kind=kind, vendor=vendor)


def pack(path, output, digest, vendor, kind):
    check(path, digest, vendor, kind)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_STORED) as archive:
        for item in sorted(path.rglob('*')):
            if item.is_file():
                archive.write(item, item.relative_to(path).as_posix())


def unpack(path, output, digest, vendor, kind):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > CAP + 1024 * 1024:
        raise ValueError('transport archive type/size refused')
    with zipfile.ZipFile(path) as archive:
        items = archive.infolist()
        names = [item.filename for item in items]
        if not 1 <= len(items) <= 128 or len(set(names)) != len(names):
            raise ValueError('archive member count/duplicate refused')
        total = 0
        for item in items:
            mode = item.external_attr >> 16
            parts = item.filename.split('/')
            if (len(item.filename) > 1024 or len(parts) > 8 or
                    any(p in ('', '.', '..') for p in parts) or
                    '\\' in item.filename or not stat.S_ISREG(mode) or
                    item.compress_type != zipfile.ZIP_STORED or item.flag_bits & 1 or
                    item.file_size > 16 * 1024 * 1024):
                raise ValueError('archive path/type/size refused')
            total += item.file_size
        if total > CAP:
            raise ValueError('archive exceeds 32 MiB')
        output.mkdir(mode=0o700, parents=False, exist_ok=False)
        for item in items:
            target = output / item.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(archive.read(item))
    check(output, digest, vendor, kind)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'pack', 'unpack', 'install'))
    parser.add_argument('path', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--vendor', choices=('cuda', 'hip'), required=True)
    parser.add_argument('--kind', choices=('baseline128', 'head64'), required=True)
    args = parser.parse_args()
    if args.action != 'check' and args.output is None:
        parser.error('--output required')
    if args.action == 'pack':
        pack(args.path, args.output, args.sha256, args.vendor, args.kind)
    elif args.action == 'unpack':
        unpack(args.path, args.output, args.sha256, args.vendor, args.kind)
    else:
        bundle = check(args.path, args.sha256, args.vendor, args.kind)
        if args.action == 'install':
            if args.kind != 'baseline128':
                raise ValueError('only retained baseline binding can be installed')
            raw = (bundle['directory'] / bundle['data']['binding_file']).read_bytes()
            if hashlib.sha256(raw).hexdigest() != bundle['data']['binding_sha256']:
                raise ValueError('retained binding changed')
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('xb') as stream:
                stream.write(raw)
    print(json.dumps(dict(action=args.action, handoff_sha256=args.sha256,
                          vendor=args.vendor, kind=args.kind, identity_admitted=False)))


if __name__ == '__main__':
    main()
