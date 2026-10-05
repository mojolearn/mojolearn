#!/usr/bin/env python3
"""Metadata-only spec to stdout: quality TAG, or timing TAG DATASET QUALITY_JSON.
Run from the pinned harness tree. No native import, test, build, queue edit.
Timing reads/hashes existing receipt and board files, so run before admission,
serially outside scored work. Manager applies the shared preflight separately.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

COMPILED = '7eacaa2b2c1a6fa84fa3aa6e4b7d6c14b2d0d817'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    action, tag, *rest = sys.argv[1:]
    assert action in ('quality', 'timing') and re.fullmatch('[A-Za-z0-9_.-]+', tag)
    root = Path(__file__).resolve().parents[1]
    harness = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    artifacts = [dict(id='resample', kind='pair', compiled_source=COMPILED,
                      binding='resample', numeric_mode='fast', defines_A='',
                      defines_B='-D MOJOLEARN_RESAMPLE_FAST_GATHER'),
                 dict(id='ibase', kind='single', compiled_source=COMPILED,
                      binding='ibase', numeric_mode='identical', defines='', artifact='_mojolearn.so')]
    script = 'tools/resample_gpu_gather_pair.py'
    args = [COMPILED, tag, COMPILED]
    prerequisites = []
    needs = ['resample', 'ibase']
    files = ['tools/resample_gpu_gather_quality.py', 'tools/resample_gpu_gather_pair.py',
             'resample/gather_fast.mojo', 'resample/estimator.mojo', 'python/mojolearn/resample.py']
    case = 'exact-draw-output-lifetime'
    if action == 'quality':
        assert not rest
    else:
        dataset, quality_path = rest
        assert dataset in ('taxi', 'istella')
        qp = Path(quality_path).expanduser().resolve()
        quality = json.loads(qp.read_text())
        assert quality['status'] == 'PASS' and quality['source_sha'] == COMPILED
        qhash = digest(qp)
        prerequisites.append(dict(id='quality', kind='file', path=str(qp), sha256=qhash,
                                  json_equals=dict(status='PASS', source_sha=COMPILED,
                                                   fixture='resample-gpu-gather-v1')))
        needs.append('quality')
        data = Path.home() / 'board-0834/cache/algos-data/rows-full'
        for suffix in ('npz', 'json'):
            path = data / ('reg-' + dataset + '.' + suffix)
            identifier = 'board_' + suffix
            prerequisites.append(dict(id=identifier, kind='file', path=str(path), sha256=digest(path)))
            needs.append(identifier)
        script = 'tools/resample_gpu_gather_timing.py'
        args.extend([str(qp), qhash, dataset])
        files.extend(['tools/resample_gpu_gather_timing.py', 'tools/bench_board_algos.py',
                      'tools/bench_board_more.py', 'tools/classical_two_datasets.py',
                      'tools/bench_board_probe.py'])
        case = 'actual-board-resample-' + dataset
    print(json.dumps(dict(version=1, tag=tag, harness_source=harness, script=script, args=args,
                          artifacts=artifacts, prerequisites=prerequisites,
                          cases=[dict(name=case, requires=needs, source_files=files)]), indent=2))


if __name__ == '__main__':
    main()
