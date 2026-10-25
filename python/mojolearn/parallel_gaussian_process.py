# SPDX-License-Identifier: Apache-2.0
"""Class-level GPU scheduling for GaussianProcessClassifier.

One complete binary covariance problem must fit on one GPU. Multiclass models
schedule independent one-vs-rest fits and predictions across devices; this is
not a distributed factorization of a single binary covariance matrix.
"""
__all__ = ['fit_gaussian_process_classifier', 'predict_gaussian_process_classifier']

from . import _gpc_impl
from . import _native_parallel_shard as _shard
from ._native_parallel_shard import _native_parallel_shard

# GaussianProcessClassifier, for the Python-sharded drivers' native owner
# (bindings/parallel_shard_native.mojo): one class column's binary fit or
# latent pass is the plain native one in a worker process.
_shard.register(gpc_class=_gpc_impl.GaussianProcessClassifier)


def fit_gaussian_process_classifier(estimator, X, y, *, devices=(0,)):
    """Fit independent binary class problems, publishing state only on success.

    Class order, target encoding, binary solver and likelihood fold are the same
    as ordinary ``fit``. The whole training matrix is replicated per active
    worker; class-level scheduling helps multiclass throughput, not binary fit
    capacity. Binary classification creates only one task.
    """
    # One native call (bindings/parallel_shard_native.mojo
    # `fit_gaussian_process_classifier`): one binary Laplace fit per
    # one-vs-rest class column on a worker process per device (column i on
    # device i % len(devices)), stored as the plain fit stores them.
    return _native_parallel_shard('fit_gaussian_process_classifier',
                                  (estimator, X, y, devices))


def predict_gaussian_process_classifier(estimator, X, *, devices=(0,), method='predict'):
    """Predict independent fitted classes and fold columns in canonical order.

    ``method`` is ``predict`` or ``predict_proba``. Each worker receives only one
    class covariance state plus the common training/query matrices. The complete
    fitted estimator remains host-resident; no single GPU receives every class.
    """
    # One native call (`predict_gaussian_process_classifier`): one latent
    # pass per fitted class on a worker process, the class columns folded in
    # canonical order; the plain method's public conversion for our caller.
    return _native_parallel_shard('predict_gaussian_process_classifier',
                                  (estimator, X, devices, method))
