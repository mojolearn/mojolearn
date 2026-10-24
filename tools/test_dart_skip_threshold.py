"""Scalar argument encoding tests without requiring a compiled binding.

The DART skip_drop threshold moved from `_DARTBase._dart_thr` (Python) into the
native public DART fit, bindings/trees_public_boost.mojo `_dart_thr`
(6ea508cd2: one native call per public method). This test reads that Mojo
function's body, restates it statement by statement as Python, and checks it
against the exact counter comparison."""
import ast
from fractions import Fraction
import math
from pathlib import Path
import re

ROOT = Path(__file__).parents[1]
MOJO = ROOT / 'bindings/trees_public_boost.mojo'
PY = ROOT / 'python/mojolearn/_expansion_trees.py'


def _mojo_threshold():
    text = MOJO.read_text()
    m = re.search(r'^def _dart_thr\(v: Float64\) -> Int:\n(.*?)(?=^\S)', text, re.S | re.M)
    assert m, '_dart_thr(v: Float64) -> Int is missing from %s' % MOJO
    body = [line for line in m.group(1).splitlines() if line.strip()]
    if body and body[0].strip().startswith('"""'):
        body = body[1:]
    allowed = {'if not v > 0.0:', 'return 0', 'if v >= 1.0:', 'return 1 << 53',
               'return Int(ceil(v * 9007199254740992.0))'}
    # every statement is one this restatement understands; anything else means the encoding changed
    assert {line.strip() for line in body} <= allowed, body
    src = 'def _dart_thr(v):\n' + '\n'.join(body).replace('Int(ceil(', 'int(math.ceil(') + '\n'
    namespace = {'math': math}
    exec(compile(src, str(MOJO), 'exec'), namespace)
    return namespace['_dart_thr']


def test_skip_drop_matches_exact_counter_comparison():
    threshold = _mojo_threshold()
    probabilities = [0., 1., .5, .1, math.nextafter(0., 1.),
                     math.nextafter(1., 0.), 2.**-53, math.nextafter(2.**-53, 1.)]
    for probability in probabilities:
        limit = threshold(probability)
        assert isinstance(limit, int) and 0 <= limit <= 1 << 53
        for word in {0, (1 << 53) - 1, max(0, limit - 1), min((1 << 53) - 1, limit)}:
            assert (word < limit) == (Fraction(word, 1 << 53) < Fraction(probability))


def test_the_native_fit_uses_it_and_python_computes_no_threshold():
    text = MOJO.read_text()
    assert 'var skip_thr = _dart_thr(Float64(py=_float(fields["skip_drop"])))' in text
    classes = {node.name: node for node in ast.parse(PY.read_text()).body if isinstance(node, ast.ClassDef)}
    for name in ('DARTRegressor', 'DARTClassifier'):
        assert any(isinstance(b, ast.Name) and b.id == '_DARTBase' for b in classes[name].bases)
    assert not any(isinstance(n, ast.FunctionDef) and n.name == '_dart_thr' for n in ast.walk(classes['_DARTBase']))
