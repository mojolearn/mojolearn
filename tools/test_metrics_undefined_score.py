"""Host-independent exceptional arithmetic in the actual metric epilogue.

The per-output scores `1 - num / den` and their multioutput average are the
x_metrics program's reg_epi stage (ASM, then AVG with the NaN rule), so the
contract is checked on the package's own r2_score: a NaN score, or an
infinite one with weight zero, averages to the positive quiet NaN;
variance_weighted weighs by `den` unless it is all zero.
"""
import math
import struct

import numpy as np
import pytest

from needs_built_package import require_built_package
require_built_package()
from mojolearn import _expansion_metrics as metrics


def score(y_true, y_pred, multioutput, force_finite):
    return metrics.r2_score_options(np.asarray(y_true, dtype=np.float32), np.asarray(y_pred, dtype=np.float32),
                                    None, multioutput, force_finite, None)


# num (residual sums) [1, 1], den (total sums) [2, 0]: scores [0.5, -inf]
Y_INF, P_INF = [[0.0, 3.0], [2.0, 3.0]], [[0.0, 3.0], [1.0, 4.0]]
# num [1, 2], den [2, 8]: scores [0.5, 0.75], variance_weighted (2*0.5 + 8*0.75) / 10
Y_FIN, P_FIN = [[0.0, 0.0], [2.0, 4.0]], [[0.0, 1.0], [1.0, 3.0]]
# num [0, 0], den [0, 0]: scores [nan, nan]
Y_ZERO = P_ZERO = [[1.0, 1.0], [1.0, 1.0]]


def test_zero_weight_infinite_score_is_explicit_positive_nan():
    for weights in ('variance_weighted', [1.0, 0.0]):
        result = score(Y_INF, P_INF, weights, False)
        assert struct.pack('>d', result).hex() == '7ff8000000000000'
    assert score(Y_INF, P_INF, 'variance_weighted', True) == 0.5


def test_finite_scores_and_nonzero_weight_infinity_keep_semantics():
    assert score(Y_FIN, P_FIN, 'variance_weighted', False) == pytest.approx(0.7, rel=1e-15)
    assert score(Y_INF, P_INF, 'uniform_average', False) == -math.inf
    assert list(score(Y_INF, P_INF, 'raw_values', False)) == [0.5, -math.inf]
    assert math.isnan(score(Y_ZERO, P_ZERO, 'variance_weighted', False))
