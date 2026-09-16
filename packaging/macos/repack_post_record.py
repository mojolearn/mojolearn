#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Repack a RECORDED macOS wheel with the post-record data of this checkout (0.8.6).

    python3 packaging/macos/repack_post_record.py <recorded wheel> --out <dir> [--root <checkout>]

The release identity record is taken from the wheel packaging/macos/build_release_wheel.sh
built; the record then lands in the manifest's record lists and the verify
reference table. Rebuilding would ship bindings nobody recorded, so the final
macOS wheel is the recorded wheel with exactly these members replaced:

  mojolearn/host_surface.py                  only when it parses to the same module as
                                             the recorded copy once the POST_RECORD_NAMES
                                             assignments are removed
                                             (tools/verify_linux_surface_qualification.py)
  mojolearn/verify_reference/table.json      the regenerated reference table
  mojolearn/identity_columns/<record>/*.json the columns TRAINING_GPU_COLUMNS now names,
                                             which follow from the record lists

Every other member, every .so and .dylib included, is copied with its bytes and
zip attributes unchanged; identity_columns/COMMIT keeps the build commit. RECORD
is regenerated. Anything the checkout would change beyond the list is refused.
"""
import argparse
import base64
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
TABLE = 'mojolearn/verify_reference/table.json'
MANIFEST = 'mojolearn/host_surface.py'
COLUMNS = 'mojolearn/identity_columns/'
COMMIT = COLUMNS + 'COMMIT'


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record_row(name, data):
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
    return [name, 'sha256=' + digest, str(len(data))]


def repack(recorded, out_dir, root):
    recorded, out_dir, root = Path(recorded), Path(out_dir), Path(root)
    admission = _load(root / 'tools' / 'verify_linux_surface_qualification.py', 'repack_admission')
    with zipfile.ZipFile(recorded) as zin:
        infos = zin.infolist()
        names = [i.filename for i in infos]
        records = [n for n in names if n.endswith('.dist-info/RECORD')]
        if len(records) != 1:
            raise SystemExit(f'repack: expected one RECORD in {recorded.name}, found {records}')
        for required in (MANIFEST, TABLE, COMMIT):
            if required not in names:
                raise SystemExit(f'repack: the recorded wheel has no {required}')
        old_manifest = zin.read(MANIFEST)
        new_manifest = (root / 'python' / 'mojolearn' / 'host_surface.py').read_bytes()
        if not admission.post_record_equivalent(old_manifest.decode(), new_manifest.decode()):
            raise SystemExit('repack: python/mojolearn/host_surface.py differs from the recorded copy outside '
                             + ', '.join(sorted(admission.POST_RECORD_NAMES)))
        with tempfile.TemporaryDirectory() as tmp:
            staged = Path(tmp) / 'host_surface.py'
            staged.write_bytes(new_manifest)
            surface = _load(staged, 'repack_host_surface')
        record_dir = surface.training_gpu_column_record()
        new_columns = {}
        for col in surface.TRAINING_GPU_COLUMNS:
            src = root / col
            if not src.is_file():
                raise SystemExit(f'repack: the manifest names {col}, which is not in {root}')
            new_columns[f'{COLUMNS}{record_dir}/{src.name}'] = src.read_bytes()
        replaced = {MANIFEST: new_manifest,
                    TABLE: (root / 'python' / 'mojolearn' / 'verify_reference' / 'table.json').read_bytes()}
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / recorded.name
        if target.resolve() == recorded.resolve():
            raise SystemExit('repack: --out must not be the recorded wheel\'s directory')
        rows, changed = [], []
        with zipfile.ZipFile(target, 'w') as zout:
            for info in infos:
                name = info.filename
                if name == records[0] or (name.startswith(COLUMNS) and name != COMMIT):
                    continue
                data = zin.read(name)
                if name in replaced:
                    if replaced[name] != data:
                        changed.append(name)
                    data = replaced[name]
                zout.writestr(info, data)
                rows.append(_record_row(name, data))
            for name, data in sorted(new_columns.items()):
                old = zin.read(name) if name in names else None
                if old != data:
                    changed.append(name)
                info = zipfile.ZipInfo(name, date_time=zin.getinfo(COMMIT).date_time)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = zin.getinfo(COMMIT).external_attr
                zout.writestr(info, data)
                rows.append(_record_row(name, data))
            removed = sorted(n for n in names if n.startswith(COLUMNS) and n != COMMIT and n not in new_columns)
            buffer = io.StringIO()
            writer = csv.writer(buffer, lineterminator='\n')
            writer.writerows(rows)
            writer.writerow([records[0], '', ''])
            zout.writestr(zin.getinfo(records[0]), buffer.getvalue().encode())
    allowed = {MANIFEST, TABLE}
    unexpected = [n for n in changed if n not in allowed and not n.startswith(COLUMNS)]
    if unexpected:
        target.unlink()
        raise SystemExit(f'repack: members outside the post-record list would change: {unexpected}')
    with zipfile.ZipFile(recorded) as a, zipfile.ZipFile(target) as b:
        kept_binaries = [n for n in a.namelist() if n.endswith(('.so', '.dylib'))]
        if any(a.read(n) != b.read(n) for n in kept_binaries):
            target.unlink()
            raise SystemExit('repack: a binary changed; refusing')
    return dict(wheel=str(target), sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                changed=sorted(changed), removed=removed, binaries_unchanged=len(kept_binaries),
                commit_witness=zipfile.ZipFile(target).read(COMMIT).decode().strip())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('recorded')
    parser.add_argument('--out', required=True)
    parser.add_argument('--root', default=str(HERE.parent.parent))
    args = parser.parse_args(argv)
    print(json.dumps(repack(args.recorded, args.out, args.root), indent=1, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
