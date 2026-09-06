#!/usr/bin/env python3
"""Remove empty ZIP directory entries; preserve every wheel payload byte.

Some repair tools add unrecorded directory entries. This normalization keeps
strict RECORD inventory admission without treating directory metadata as files.
It never changes a payload, RECORD, binary, or platform tag.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def normalize(source, target):
    source, target = Path(source), Path(target)
    if source.resolve() == target.resolve() or target.exists():
        raise ValueError('Use a fresh output path; retain the repaired original')
    with zipfile.ZipFile(source) as original:
        infos = original.infolist()
        if len(infos) != len({i.filename for i in infos}):
            raise ValueError('Duplicate ZIP member')
        removed = [i.filename for i in infos if i.is_dir()]
        if any(i.file_size != 0 for i in infos if i.is_dir()):
            raise ValueError('Nonempty directory member')
        expected = {i.filename: hashlib.sha256(original.read(i)).hexdigest()
                    for i in infos if not i.is_dir()}
        target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target, 'x') as output:
            for info in infos:
                if not info.is_dir():
                    output.writestr(info, original.read(info))
        with zipfile.ZipFile(target) as output:
            actual = {i.filename: hashlib.sha256(output.read(i)).hexdigest()
                      for i in output.infolist()}
        if actual != expected:
            raise ValueError('Payload changed during normalization')
    return {'removed_empty_directories': removed, 'unchanged_payload_files': len(expected),
            'original_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'normalized_sha256': hashlib.sha256(target.read_bytes()).hexdigest()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('target', type=Path)
    args = parser.parse_args()
    print(json.dumps(normalize(args.source, args.target), indent=2))
