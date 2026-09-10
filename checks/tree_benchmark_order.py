#!/usr/bin/env python3
"""Host-only runner control checks; no GPU timing or learner qualification."""
import ast
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import time

source = Path(__file__).resolve().parents[1] / 'tools/speed_gbdt_arm.py'
node = next(n for n in ast.parse(source.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == 'run')


def check(rotate, fail_second=False, profile=True):
    visits, refusals, ranges, active = [], [], [], []
    namespace = dict(time=time, contextlib=contextlib, device_string=lambda: 'test',
                     per_arm_budget_s=lambda: 60, process_deadline_s=lambda: 60,
                     hash_predictions=lambda _: 'digest',
                     _check_same_library_agreement=lambda *_: None,
                     emit_scale_reminder=lambda *_: None)
    for name in ('emit_header', 'emit_warmup', 'emit_round', 'emit_acc'):
        namespace[name] = lambda *args: None
    namespace['emit_refused'] = lambda *args: refusals.append(args)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    arms = []
    counts = {}
    for name in ('a', 'b', 'c'):
        counts[name] = 0

        def fit(model, data, name=name):
            assert active == ([name] if profile else []), active
            counts[name] += 1
            visits.append(name)
            if fail_second and name == 'b' and counts[name] == 2:
                raise RuntimeError('planted fit failure')

        def score(*_):
            assert not active, 'scoring must remain outside fit ranges'
            return [('quality', 1., [1.])]

        arms.append(SimpleNamespace(name=name, library=name, make=object,
                                    fit=fit, sync=lambda: None,
                                    score=score))
    @contextlib.contextmanager
    def fit_context(name, round_index):
        assert not active
        active.append(name)
        ranges.append((name, round_index))
        try:
            yield
        finally:
            active.pop()

    with contextlib.redirect_stdout(io.StringIO()):
        live = namespace['run']('rf', arms, SimpleNamespace(tag='test'), 3,
                                'smoke', rotate_order=rotate, fit_context=fit_context if profile else None)
    assert not active, 'failed fits must close their profiling range'
    if profile:
        assert [name for name, _ in ranges] == visits
        assert ranges[:3] == [('a', 0), ('b', 0), ('c', 0)]
    else:
        assert not ranges
    if fail_second:
        assert visits == list('abc' + 'abc' + 'ca' + 'ac'), visits
        assert [a.name for a in live] == ['a', 'c']
        assert len(refusals) == 1 and 'planted fit failure' in refusals[0][-1]
    else:
        assert visits == list('abc' + ('abc' + 'bca' + 'cab' if rotate else 'abcabcabc')), visits
        assert not refusals


check(False, profile=False)
check(True, profile=False)
check(False)
check(True)
check(True, fail_second=True)
print('PASS: fixed and rotating order; failed arm removed without skipping surviving fits')
