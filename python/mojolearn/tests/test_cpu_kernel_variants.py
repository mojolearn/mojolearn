# SPDX-License-Identifier: Apache-2.0
"""CPU kernel variants: degree transport through save and reload, row
invariance of the reloaded model, and the degree range refusal.

Cross-vendor identity is established by recorded harness lanes, not by these
single-machine tests. They must run with the newly built kernel host binding.
"""
import numpy as np
import pytest
import mojolearn as ml
from mojolearn import _backend

GATE_BACKENDS = ("cpu",)


def cpu():
    if _backend._CPU_ONLY is None:
        pytest.skip('requires a CPU-only package')


def bits(a):
    a = np.asarray(a)
    return a.shape, a.dtype.str, a.tobytes()


@pytest.mark.parametrize('family', ['KernelRidge', 'Nystroem'])
@pytest.mark.parametrize('kernel,degree', [('linear', 3), ('rbf', 3), ('sigmoid', 3),
    ('laplacian', 3), ('poly', 0), ('poly', 1), ('poly', 2), ('poly', 3), ('poly', 7), ('poly', 32)])
def test_kernel_host_reload_and_batch(family, kernel, degree, tmp_path):
    cpu()
    rng = np.random.default_rng(293)
    X = (rng.standard_normal((24, 5)) * .125).astype('float32')
    Q = (rng.standard_normal((7, 5)) * .125).astype('float32')
    y = rng.standard_normal((24, 2)).astype('float32')
    kwargs = dict(kernel=kernel, degree=degree, gamma=.5, coef0=.25)
    if family == 'KernelRidge':
        model = ml.KernelRidge(alpha=64., **kwargs)
        method = 'predict'
    else:
        model = ml.Nystroem(n_components=7, random_state=3, **kwargs)
        method = 'transform'
    model.fit(X, y)
    expected = getattr(model, method)(Q)
    path = tmp_path / 'model.npz'
    model.save(path)
    restored = ml.host_model(path)
    assert restored.vendor_used() == 'cpu'
    assert restored._kernel_params[1] == degree
    assert bits(expected) == bits(getattr(restored, method)(Q))
    split = np.concatenate([np.asarray(getattr(restored, method)(Q[:2])),
                            np.asarray(getattr(restored, method)(Q[2:]))])
    alone = np.concatenate([np.asarray(getattr(restored, method)(Q[i:i+1])) for i in range(len(Q))])
    assert bits(expected) == bits(split) == bits(alone)


@pytest.mark.parametrize('degree', [-1, 33])
def test_host_rejects_out_of_range_polynomial_degree(degree):
    cpu()
    X = np.eye(3, dtype='float32')
    with pytest.raises(Exception, match='degree'):
        ml.KernelRidge(kernel='poly', degree=degree).fit(X, np.ones(3, dtype='float32'))
