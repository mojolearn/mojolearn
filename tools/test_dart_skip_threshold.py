"""Scalar argument encoding tests without requiring a compiled GPU binding."""
import ast
from fractions import Fraction
import math
from pathlib import Path


def test_skip_drop_matches_exact_counter_comparison():
    source = Path(__file__).parents[1] / 'python/mojolearn/_expansion_trees.py'
    classes = {node.name: node for node in ast.parse(source.read_text()).body
               if isinstance(node, ast.ClassDef)}
    base = classes['_DARTBase']
    method = next(node for node in base.body
                  if isinstance(node, ast.FunctionDef) and node.name == '_dart_thr')
    # Execute the actual scalar method, avoiding package import's GPU gate.
    method.decorator_list = []
    namespace = {'math': math}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
    threshold = namespace['_dart_thr']
    for name in ('DARTRegressor', 'DARTClassifier'):
        assert any(isinstance(b, ast.Name) and b.id == '_DARTBase'
                   for b in classes[name].bases)
    probabilities = [0., 1., .5, .1, math.nextafter(0., 1.),
                     math.nextafter(1., 0.), 2.**-53, math.nextafter(2.**-53, 1.)]
    for probability in probabilities:
        limit = threshold(probability)
        assert isinstance(limit, int) and 0 <= limit <= 1 << 53
        for word in {0, (1 << 53) - 1, max(0, limit - 1), min((1 << 53) - 1, limit)}:
            assert (word < limit) == (Fraction(word, 1 << 53) < Fraction(probability))
