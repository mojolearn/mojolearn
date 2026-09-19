# SPDX-License-Identifier: Apache-2.0
"""CPU kernel variants: degree transport, reload and row invariance.

Cross-vendor identity is established by recorded harness lanes, not by these
single-machine tests. They must run with the newly built kernel host binding.
"""
import numpy as np
import pytest
import mojolearn as ml
from mojolearn import _backend, host_surface
from mojolearn._cpu_reference import reference_training

GATE_BACKENDS = ("cpu",)
VARIANTS = [f'{family}-{kernel}' for family in ('kernel-ridge', 'nystroem')
            for kernel in ('poly', 'sigmoid', 'laplacian')]


def test_variants_are_public_but_saved_model_qualification_is_still_pending():
    """The six kernel variants are declared, covered and now PUBLIC, and their
    saved-model debt is untouched by that.

    They were promoted 2026-09-19 when the RTX 4090 column was admitted to the
    shipped table, so the old `not ... & public_reference_lanes()` reading is
    inverted here rather than deleted: it still fails if one silently drops out
    of the public set.

    The last line is the one that has NOT changed, and the reason is worth
    keeping in front of a reader. An identity-table cell and a `classical_host`
    saved-model recording are different artifacts. The NVIDIA identity column
    paid the first; the only kernel-variant saved-model recordings in the tree
    are Apple's, so it could not pay the second. A lane can be a public
    reference lane and still owe a saved-model recording, and collapsing the
    two would overstate what an install can replay.
    """
    from mojolearn._verify_all import load_harness
    harness = load_harness()
    assert set(VARIANTS) <= set(harness.LANES) & set(harness.BATCH)
    assert set(VARIANTS) <= set(host_surface.covered_lanes())
    assert set(VARIANTS) <= set(host_surface.public_reference_lanes())
    assert set(VARIANTS) <= set(host_surface.PUBLIC_REFERENCE_PROMOTED)
    assert set(VARIANTS) <= set(host_surface.saved_model_inference_owed())


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
    with reference_training():
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


def test_polynomial_degree_affects_host_fit_and_inference():
    cpu()
    X = np.asarray([[.25, -.5], [-.25, .125], [.5, .25], [-.125, -.25]], dtype='float32')
    y = np.asarray([.25, -.5, .75, .125], dtype='float32')
    answers = []
    with reference_training():
        for degree in (0, 1, 2, 3):
            m = ml.KernelRidge(kernel='poly', degree=degree, gamma=.5, coef0=.25, alpha=4.).fit(X, y)
            answers.append(bits(m.predict(X)))
    assert len(set(answers)) == 4, 'degree was ignored in fit or prediction'


@pytest.mark.parametrize('degree', [-1, 33])
def test_host_rejects_out_of_range_polynomial_degree(degree):
    cpu()
    X = np.eye(3, dtype='float32')
    with reference_training(), pytest.raises(Exception, match='degree'):
        ml.KernelRidge(kernel='poly', degree=degree).fit(X, np.ones(3, dtype='float32'))
