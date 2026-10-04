#!/usr/bin/env python3
"""SOURCE TAG IBASE_SOURCE QUALITY_JSON QUALITY_SHA256 taxi|istella.
Exactly one existing board worker round per arm, without a warmup/replay.
The board resample runner includes the first full X/y output read (means)
inside runner.fit. No opponent, native build, or alternate/synthetic fixture.
"""
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile
from resample_gpu_gather_pair import DEFINE, FIXTURE, digest, verify_artifacts


def main():
    source, tag, ibase_source, quality_path, quality_hash, dataset = sys.argv[1:]
    assert re.fullmatch('[A-Za-z0-9_.-]+', tag)
    assert re.fullmatch('[0-9a-f]{64}', quality_hash)
    assert dataset in ('taxi', 'istella')
    root, arms, hashes, bm, base_binary, harness = verify_artifacts(source, ibase_source)
    qp = Path(quality_path).expanduser().resolve()
    assert qp.is_relative_to(Path.home() / 'mq/out')
    assert digest(qp) == quality_hash
    quality = json.loads(qp.read_text())
    for key, value in dict(status='PASS', fixture=FIXTURE, source_sha=source,
                           hashes=hashes, ibase_source=ibase_source,
                           ibase_sha256=bm['sha256'], defines_A='', defines_B=DEFINE).items():
        assert quality[key] == value, key
    assert quality['quality_script_sha256'] == digest(root / 'tools/resample_gpu_gather_quality.py')
    subprocess.run(['git', 'merge-base', '--is-ancestor', quality['harness_source'], harness], check=True)
    # Reject any source/board change since the quality receipt except this new
    # timing helper, its spec generator, policy declaration and documentation.
    changes = subprocess.check_output(['git', 'diff', '--name-only', quality['harness_source'], harness], text=True).splitlines()
    allowed = {'tools/resample_gpu_gather_timing.py', 'tools/resample_gpu_gather_spec.py',
               'tools/apple_fast_job_policy.py'}
    assert all(p in allowed or p.startswith('docs/') for p in changes), changes
    board = root / 'tools/bench_board_algos.py'
    data = Path.home() / 'board-0834/cache/algos-data/rows-full'
    data_npz = data / ('reg-' + dataset + '.npz')
    data_json = data / ('reg-' + dataset + '.json')
    assert data_npz.is_file() and data_json.is_file()
    import numpy as np
    shapes = {}
    with zipfile.ZipFile(data_npz) as archive:
        for name in ('X.npy', 'y.npy', 'Xq.npy'):
            with archive.open(name) as stream:
                version = np.lib.format.read_magic(stream)
                shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
                shapes[name] = list(shape)
                if name == 'X.npy':
                    assert shape == (1000000, 11 if dataset == 'taxi' else 220), shape
                    assert not fortran and dtype == np.dtype('float32'), (fortran, dtype)
                if name == 'y.npy':
                    assert shape == (1000000,), shape
    dataset_hashes = dict(npz=digest(data_npz), metadata=digest(data_json))
    out = Path.home() / 'mq/out' / (tag + '-timing')
    out.mkdir(parents=True, exist_ok=False)
    pkg = root / 'python/mojolearn'
    installed = pkg / '_mojolearn_resample.so'
    base_installed = pkg / 'identical/_mojolearn.so'
    assert not installed.exists() and not base_installed.exists()
    base_installed.parent.mkdir(exist_ok=True)
    shutil.copy2(base_binary, base_installed)
    env = os.environ.copy()
    env.update(MOJOLEARN_NUMERIC_MODE='fast', MOJOLEARN_VENDOR='apple',
               MOJOLEARN_BENCH_INSTALLED='0', MOJOLEARN_REPO_COMMIT=harness,
               PYTHONPATH=str(root / 'python'), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    env.pop('MOJOLEARN_ALGOS_SMOKE_ROWS', None)
    records = {}
    try:
        for arm in ('A', 'B'):
            shutil.copy2(arms / (arm + '.so'), installed)
            assert digest(installed) == hashes[arm]
            capture = out / (arm + '.npz')
            command = [sys.executable, str(board), 'worker', '--arm', 'ours-fast',
                       '--lane', 'resample', '--dataset', dataset, '--data', str(data)]
            # The existing board worker performs no call before receiving round.
            # Send only round1, save its existing quality summaries, then quit.
            with (out / (arm + '.log')).open('x') as log:
                result = subprocess.run(command, input='round 1\nsave ' + str(capture) + '\nquit\n',
                                        stdout=subprocess.PIPE, stderr=log, text=True,
                                        env=env, timeout=1800)
            (out / (arm + '.protocol.jsonl')).write_text(result.stdout)
            if result.returncode:
                raise RuntimeError('board worker failed: ' + arm + ', retained log ' + str(out))
            events = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
            assert [e['event'] for e in events] == ['ready', 'round', 'saved', 'bye'], events
            ready, scored = events[0], events[1]
            assert ready['info']['numeric_mode_used'] == 'fast', ready
            assert scored['round'] == 1 and math.isfinite(scored['ms']) and scored['ms'] > 0
            assert events[2]['path'] == str(capture)
            records[arm] = dict(ready=ready, scored=scored, binding_sha256=hashes[arm])
            # Persist each completed score immediately; no automatic retry.
            (out / (arm + '.json')).write_text(json.dumps(records[arm], sort_keys=True))
            print('RESAMPLE_BOARD arm=' + arm + ' dataset=' + dataset + ' ms=' + str(scored['ms']), flush=True)
        with np.load(out / 'A.npz') as a, np.load(out / 'B.npz') as b:
            assert sorted(a.files) == sorted(b.files) == ['xmean', 'ymean']
            for key in a.files:
                assert a[key].shape == b[key].shape and a[key].dtype == b[key].dtype
                assert np.isfinite(a[key]).all() and np.isfinite(b[key]).all()
                assert a[key].tobytes() == b[key].tobytes(), key
        assert records['A']['scored']['digest'] == records['B']['scored']['digest']
        receipt = dict(status='PASS', compiled_source=source, harness_source=harness,
                       quality_path=str(qp), quality_sha256=quality_hash, dataset=dataset,
                       dataset_hashes=dataset_hashes, shapes=shapes, hashes=hashes, records=records,
                       board_script_sha256=digest(board), scored_calls_per_arm=1, warmup_calls=0,
                       boundary='existing board resample fit including complete X/y float64 output means',
                       opponent_runs=0, automatic_retries=0)
        (out / 'PASS.json').write_text(json.dumps(receipt, sort_keys=True))
    finally:
        installed.unlink(missing_ok=True)
        base_installed.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
