"""Host-independent exceptional arithmetic in the actual metric epilogue."""
import ast
import math
from pathlib import Path
import struct


def assemble():
    path = Path(__file__).resolve().parents[1] / 'python/mojolearn/_expansion_metrics.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_assemble')
    class Array:
        @staticmethod
        def from_list(values, dtype):
            return values
    env = {'_math': math, 'Array': Array}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), env)
    return env['_assemble']


def test_zero_weight_infinite_score_is_explicit_positive_nan():
    score = assemble()
    for weights in ('variance_weighted', [1.0, 0.0]):
        result = score([1.0, 1.0], [2.0, 0.0], weights, False)
        assert struct.pack('>d', result).hex() == '7ff8000000000000'
    assert score([1.0, 1.0], [2.0, 0.0], 'variance_weighted', True) == 0.5


def test_finite_scores_and_nonzero_weight_infinity_keep_semantics():
    score = assemble()
    assert score([1.0, 1.0], [2.0, 4.0], 'variance_weighted', False) == 2/3
    assert score([1.0, 1.0], [2.0, 0.0], 'uniform_average', False) == -math.inf
    assert score([1.0, 1.0], [2.0, 0.0], 'raw_values', False) == [0.5, -math.inf]
    assert math.isnan(score([0.0, 0.0], [0.0, 0.0], 'variance_weighted', False))
