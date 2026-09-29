"""Reject incomplete artifact identity evidence without running GPU work."""
import copy
from pathlib import Path
import subprocess
import unittest

SCRIPT = Path(__file__).with_name('release_identity_installed.sh')
ns = {'__name__': 'identity_gate_test'}
exec(compile(SCRIPT.read_text().split('# IDENTITY_PY_BEGIN\n')[1].split('# IDENTITY_PY_END')[0], str(SCRIPT), 'exec'), ns)


def evidence():
    return dict(format='mojolearn.verify-all-report.v1', exit=0, verdict='VERIFIED',
        fixtures=['base'], repeats=1, lanes=['ridge'], models_checked=0,
        execution=dict(mode='fresh-process-per-cell', timeout_s=120, interrupted=None,
                       completed_cells=1, total_cells=1), bindings=[{'sha256':'a'*64}],
        cells=[dict(lane='ridge',fixture='base',part=p,state='IDENTICAL',value='a'*16,error=None)
               for p in ns['PARTS']])


class IdentityAdmissionTests(unittest.TestCase):
    def test_complete_core_identity(self):
        ns['admit_identity'](evidence(), ['ridge'])

    def test_structural_na_is_explicit(self):
        d=evidence();d['cells'][-1].update(state='N/A',value='n/a:no-decode-state')
        ns['admit_identity'](d, ['ridge'])

    def test_rejects_missing_duplicate_wrong_lane_or_part(self):
        changes=[lambda d:d['cells'].pop(), lambda d:d['cells'].append(d['cells'][0]),
                 lambda d:d['cells'][0].update(lane='ols'),
                 lambda d:d['cells'][0].update(part='other')]
        for change in changes:
            d=evidence();change(d)
            with self.subTest(change=change), self.assertRaises(AssertionError):
                ns['admit_identity'](d,['ridge'])

    def test_refusal_owed_divergence_and_undeclared_never_pass(self):
        for state,value in [('REFUSED',None),('OWED','a'*16),('DIVERGENT','b'*16),
                            ('N/A','n/a:UNDECLARED'),('N/A','n/a:skipped')]:
            d=evidence();d['cells'][0].update(state=state,value=value)
            with self.subTest(state=state,value=value), self.assertRaises(AssertionError):
                ns['admit_identity'](d,['ridge'])

    def test_incomplete_worker_run_or_extra_scope_never_pass(self):
        for change in [lambda d:d['execution'].update(completed_cells=0),
                       lambda d:d['execution'].update(interrupted='timeout'),
                       lambda d:d.update(fixtures=['base','odd']),
                       lambda d:d.update(bindings=[])]:
            d=evidence();change(d)
            with self.subTest(change=change), self.assertRaises(AssertionError):
                ns['admit_identity'](d,['ridge'])

    def test_shell_syntax(self):
        for name in ('release_identity_installed.sh','release_installed_checks.sh','release_wheel_smoke.sh'):
            subprocess.run(['bash','-n',str(SCRIPT.with_name(name))],check=True)

if __name__=='__main__':unittest.main()
