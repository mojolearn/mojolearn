import copy
import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path

import release_repair_gate as gate


class RepairAdmissionTests(unittest.TestCase):
    def test_graph_requires_every_case_and_negative_control(self):
        names = [str(n) for n in range(31)]
        doc = dict(exit_code=0, expected_cases=names,
                   cases={n: dict(status='passed') for n in names},
                   comparator_controls=dict(same_report='passed', changed_digest='rejected',
                       failed_case='rejected', missing_case='rejected', wrong_mode='rejected'))
        gate.graph(doc)
        for mutate in (lambda d: d['cases'].pop('0'),
                       lambda d: d['cases']['0'].update(status='cannot_run'),
                       lambda d: d['comparator_controls'].update(changed_digest='passed')):
            broken = copy.deepcopy(doc); mutate(broken)
            with self.assertRaises(ValueError): gate.graph(broken)

    def test_full_summary_without_executed_cells_does_not_qualify(self):
        doc = dict(verdict='VERIFIED', exit=0, native_provenance_complete=True,
                   counts=dict(IDENTICAL=11843, DIVERGENT=0, REFUSED=0, OWED=0, **{'N/A':22860}),
                   execution=dict(total_cells=3843, completed_cells=3843), bindings=[{'sha256':'x'}],
                   cross_check=dict(passed=True, complete=True, differ=0))
        with self.assertRaisesRegex(ValueError, 'raw verifier cell coverage'):
            gate.full_report(doc)

    def test_comparison_refuses_incomplete_or_different_parts(self):
        doc = dict(verdict='AGREE', exit=0, agree=2547, differ=0, incomparable=0, uncomputed=0)
        gate.comparison(doc, 2547)
        for key in ('differ', 'incomparable', 'uncomputed'):
            with self.assertRaises(ValueError): gate.comparison(dict(doc, **{key:1}), 2547)

    def test_delta_rejects_unqualified_runtime_and_component_mismatch(self):
        with tempfile.TemporaryDirectory() as folder:
            def wheel(name, members):
                p=Path(folder)/name
                with zipfile.ZipFile(p,'w') as z:
                    for n, value in members.items():z.writestr(n,value)
                return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            base=wheel('before.whl', {'mojolearn/_backend.py':b'old','mojolearn/ops.py':b'unchanged'})
            after=wheel('after.whl', {'mojolearn/_backend.py':b'new','mojolearn/ops.py':b'unchanged'})
            qualified={'mojolearn/_backend.py':hashlib.sha256(b'new').hexdigest()}
            self.assertEqual(gate.wheel_delta(base,after,qualified),['mojolearn/_backend.py'])
            with self.assertRaises(ValueError):gate.wheel_delta(base,after,{})
            with self.assertRaises(ValueError):gate.wheel_delta(base,after,{'mojolearn/_backend.py':'0'*64})
            extra=wheel('extra.whl', {'mojolearn/_backend.py':b'new','mojolearn/ops.py':b'changed'})
            with self.assertRaises(ValueError):gate.wheel_delta(base,extra,qualified)
            with self.assertRaises(ValueError):gate.wheel_delta(base,extra,dict(qualified,**{'mojolearn/ops.py':hashlib.sha256(b'changed').hexdigest()}))


if __name__ == '__main__':unittest.main()
