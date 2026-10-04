#!/usr/bin/env python3
"""One-shot stage driver for a composed wave revision; never times, never repeats.

Order: verify pinned harness hashes -> repaired supplemental gates (append-only)
-> success-receipt reconciliation -> broad identity ON/OFF once. Each stage
takes an exclusive claim directory, so a restart refuses instead of repeating
work. Status lives in <wave>/stage-status.json. Supplement or reconciliation
failure is recorded but does not block identity, which needs only the frozen
prepare receipt; timing stays separately gated on quality + cross-vendor proof.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wave', required=True, type=Path)
    p.add_argument('--harness', required=True, type=Path)
    p.add_argument('--harness-commit', required=True)
    p.add_argument('--vendor', required=True, choices=('nvidia', 'amd'))
    p.add_argument('--gpu-arch', required=True)
    p.add_argument('--python', default='/root/mojolearn/.pixi/envs/default/bin/python')
    p.add_argument('--repo', default='/root/mojolearn')
    p.add_argument('--data', default='/root/board-0833/cache/algos-data/rows-small')
    p.add_argument('--cell', action='append', default=[])
    p.add_argument('--skip-identity', action='store_true')
    p.add_argument('--attempt', type=int, default=1, help='Fresh claim/log names for a new attempt; earlier attempts stay untouched')
    a = p.parse_args()
    wave, harness = a.wave.resolve(), a.harness.resolve()
    plan = harness / 'identical_wave_plan.json'
    state = {'pid': os.getpid(), 'vendor': a.vendor, 'arch': a.gpu_arch, 'harness_commit': a.harness_commit,
             'status': 'STARTING', 'stages': []}

    def save():
        status = 'stage-status.json' if a.attempt == 1 else 'stage-status-attempt%d.json' % a.attempt
        tmp = wave / (status + '.new')
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        tmp.replace(wave / status)

    def stage(name, argv):
        name = name if a.attempt == 1 else name + '-attempt' + str(a.attempt)
        (wave / (name + '.claim')).mkdir()  # refuses a repeated stage
        state['status'] = 'RUNNING_' + name.upper(); save()
        with (wave / (name + '-controller.log')).open('xb') as log:
            rc = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, cwd=str(harness)).returncode
        (wave / (name + '-controller.rc')).write_text(str(rc) + '\n')
        state['stages'].append({'stage': name, 'rc': rc, 'argv': argv,
                                'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}); save()
        return rc

    try:
        expected = json.loads((harness / 'harness-sha256.json').read_text())
        for name, wanted in expected.items():
            if hashlib.sha256((harness / name).read_bytes()).hexdigest() != wanted:
                raise RuntimeError('harness changed: ' + name)
        common = ['--vendor', a.vendor, '--gpu-arch', a.gpu_arch]
        if a.cell:
            argv = [a.python, str(harness / 'tools/identical_wave_supplement.py'), '--wave', str(wave), '--plan', str(plan),
                    '--harness', str(harness), '--harness-commit', a.harness_commit, '--python', a.python, '--data', a.data] + common
            for cell in a.cell:
                argv += ['--cell', cell]
            stage('supplements', argv)
            stage('reconcile', [a.python, str(harness / 'tools/identical_wave_reconcile.py'), '--wave', str(wave), '--plan', str(plan)])
        if not a.skip_identity:
            if (wave / 'identity.json').exists() or (wave / 'on/identity').exists() or (wave / 'off/identity').exists():
                raise RuntimeError('identity has prior evidence')
            stage('identity', [a.python, str(harness / 'tools/identical_wave_runner.py'), 'identity', '--plan', str(plan), '--sha',
                               json.loads((wave / 'wave.json').read_text())['sha'], '--repo', a.repo, '--out', str(wave),
                               '--python', a.python, '--data', a.data] + common)
        failed = [s['stage'] for s in state['stages'] if s['rc']]
        state['status'] = 'COMPLETE_WITH_FAILURES' if failed else 'COMPLETE_NO_TIMING'; state['failed_stages'] = failed; save()
    except Exception as exc:
        state['status'] = 'STOPPED'; state['error'] = str(exc); save(); traceback.print_exc(); return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
