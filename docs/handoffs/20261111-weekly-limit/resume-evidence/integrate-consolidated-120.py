"""Finite root-owned combined build followed by the authorized development merge.

The existing builder owns compile scheduling and cache validation. This is not
a release controller, runtime qualification, identity run, or background watcher.
"""
from pathlib import Path
import datetime
import json
import os
import subprocess
import sys

E = Path(__file__).resolve().parent
ROOT = Path('/Users/andrewhendel/mojolearn-wt/integ-python-followon-20261111')
MAIN = Path('/Users/andrewhendel/mojolearn-wt/main-integration-20261110')
HOLD = MAIN / 'PREINTEGRATION_MERGE_HOLD_20261111'
ENV = dict(os.environ, MOJOLEARN_ORCHESTRATOR='1', MOJOLEARN_WIDE_REBUILD_OK='1')


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root, text=True).strip()


def main():
    source, base, attempt = sys.argv[1:]
    evidence = E / ('consolidated-120-' + attempt)
    evidence.mkdir(exist_ok=False)
    record = dict(source=source, base=base, state='building', compiled=False,
                  merged=False, pushed=False, evidence=str(evidence),
                  scope='Mac identical and CPU compilation only; no runtime, vendor identity, or performance claim')

    def save():
        record['updated_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        (evidence / 'integration.json').write_text(json.dumps(record, indent=2) + '\n')

    def run(name, command, cwd):
        with (evidence / (name + '.log')).open('x') as log:
            return subprocess.run(command, cwd=cwd, env=ENV, stdout=log,
                                  stderr=subprocess.STDOUT).returncode

    save()
    try:
        assert git(ROOT, 'rev-parse', 'HEAD') == source
        assert not git(ROOT, 'status', '--porcelain'), 'Combined source has uncommitted edits'
        code = run('build', ['bash', 'tools/compile_slot.sh', 'pixi', 'run',
            'python3', 'tools/development_build.py', 'build', '--sets', 'identical,host',
            '--layout', 'release', '--evidence', str(evidence / 'build')], ROOT)
        record['build_exit'] = code
        if code:
            record['state'] = 'compile_failed'
            save()
            return code
        result = json.loads((evidence / 'build/result.json').read_text())
        assert result['source_commit'] == source and result['state'] == 'activated'
        assert len(result['results']) == 25 and all(r['exit_code'] == 0 for r in result['results'])
        record['compiled'] = True
        record['binding_units'] = 25
        save()
        assert git(ROOT, 'rev-parse', 'HEAD') == source and not git(ROOT, 'status', '--porcelain')
        assert git(MAIN, 'rev-parse', 'HEAD') == base, 'Main advanced; integrate its changes before merging'
        assert git(MAIN, 'branch', '--show-current') == 'main'
        status = git(MAIN, 'status', '--porcelain')
        if status == '?? PREINTEGRATION_MERGE_HOLD_20261111':
            old = json.loads((E / 'next-cohort-repair1-integration-result.json').read_text())
            assert old['state'] == 'stopped' and old['compiled'] and not old['merged']
            assert HOLD.read_text().startswith('User requested one combined pre-integration merge on 2026-11-11.')
            HOLD.unlink()
        else:
            assert not status, 'Main contains unrelated edits; preserved'
        assert run('merge', ['git', 'merge', '--ff-only', source], MAIN) == 0
        record['merged'] = True
        save()
        code = run('push', ['git', 'push', 'origin', 'main'], MAIN)
        record.update(push_exit=code, pushed=(code == 0),
                      state='integrated' if code == 0 else 'push_failed')
        save()
        return code
    except Exception as error:
        record.update(state='stopped', error=str(error))
        save()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
