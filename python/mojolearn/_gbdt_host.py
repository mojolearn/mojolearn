# SPDX-License-Identifier: Apache-2.0
"""CPU inference for saved gradient-boosting models.

Model loading, parsing, input preparation and prediction run through compiled
Mojo. ``predict`` returns float32 raw approximations; binary classification
``predict_proba`` returns float64 probabilities.
"""

#: `_MODEL_FORMAT` in ensemble.py.
GBDT_FORMAT = 'mojolearn-gbdt-1'
#: `MODEL_TEXT_NAME` and `MODEL_TEXT_VERSION`, `gbdt/models/model_text.mojo:225-229`.
_TEXT_NAME = 'mojolearn-model'
_TEXT_VERSION = 2
#: `NAN_TREATMENT_*`, `gbdt/data/quantization.mojo:73-76`, keyed by the
#: token `nan_treatment_token` writes (`model_text.mojo:232-245`).
_NAN_CODES = {'as_is': 0, 'as_false': 1, 'as_true': 2}
_PROBA_LOSSES = ('Logloss', 'CrossEntropy')
_MULTI_LOSSES = ('MultiClass', 'MultiClassOneVsAll')
_CLASSIFICATION_LOSSES = _PROBA_LOSSES + _MULTI_LOSSES
#: `ctr_type_name`, `gbdt/ctrs/ctr.mojo`: the names a `ctr_table` may carry.
_CTR_TYPE_CODES = {'Borders': 0, 'Buckets': 1, 'BinarizedTargetMeanValue': 2,
                   'FloatTargetMeanValue': 3, 'Counter': 4, 'FeatureFreq': 5}
#: The types `TCtrValueTable.value_for` has apply-time arithmetic for; the
#: others raise there and are refused here by name.
_CTR_APPLIED = ('Borders', 'Counter', 'FeatureFreq')
#: Records no reader of this version understands, refused by name.
_REFUSED_RECORDS = ()


def parse_model_text(text):
    """Parse saved gradient-boosting model text into native CPU inference buffers."""
    return _native_host_archive(None, "host_tree_parse", (text,), _ARCHIVE_CONTEXT)


#: The `estimator` members a `mojolearn-gbdt-1` archive may carry, with the
#: losses each class can save. `OrderedRMSE` and
#: `ExperimentalTwoLevelFeatureFreq` subclass `GradientBoosting` and inherit
#: its `save`, so their archives are the same format holding the same model
#: text records; the name is kept so a report says which class trained the
#: file. Simple and tensor CTR records use their respective native layouts.
GBDT_ESTIMATORS = {
    'GradientBoosting': None,
    'OrderedRMSE': ('RMSE',),
    'ExperimentalTwoLevelFeatureFreq': ('RMSE',),
}


class HostGBDT:
    """CPU inference for a saved gradient-boosting model.

    Loads a file written by ``GradientBoosting.save`` and predicts on the CPU,
    with no GPU needed. ``predict`` returns what ``GradientBoosting.predict``
    returns for the same file. Models with categorical (CTR) columns are
    supported; a model that uses CTR table types with no prediction-time
    formula (Buckets, BinarizedTargetMeanValue, FloatTargetMeanValue) is
    refused at load.

    Build one with :meth:`from_file`. The constructor takes the archive's
    fields directly and refuses a file whose fields disagree with its model
    text as corrupt.

    Parameters
    ----------
    loss : str
        The loss the model was fitted with.
    text : str
        The model text (``GradientBoosting.model_``).
    n_features_in : int
    approx_dim : int
        Number of raw outputs per row.
    n_classes : int or None, default None
    numeric_mode : str or None, default None
        The numeric mode recorded with the model.
    bias : float or None, default None
    estimator : str, default 'GradientBoosting'
        The class that saved the model.

    Attributes
    ----------
    n_features_in_ : int
    approx_dim_ : int
    bias_ : float
    n_classes_ : int or None
    model_ : str

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> y = (X[:, 0] > 0.5).astype(np.float32)
    >>> mojolearn.GradientBoosting(loss="Logloss", n_estimators=50).fit(X, y).save("gb.npz")
    >>> host = mojolearn.HostGBDT.from_file("gb.npz")
    >>> proba = host.predict_proba(X)
    """

    estimator = 'GradientBoosting'

    def __init__(self, *, loss, text, n_features_in, approx_dim, n_classes=None,
                 numeric_mode=None, bias=None, estimator='GradientBoosting'):
        return _native_host_archive(self, "host_tree_init_gbdt",
            (loss, text, n_features_in, approx_dim, n_classes, numeric_mode, bias, estimator), _ARCHIVE_CONTEXT)

    @classmethod
    def from_file(cls, path):
        """Load a model from a file written by ``GradientBoosting.save``.

        Parameters
        ----------
        path : str or path-like

        Returns
        -------
        HostGBDT

        Raises
        ------
        ValueError
            If the file was saved by another class or is corrupt.
        """
        return _native_host_archive(cls, "host_archive_load", (path,), _ARCHIVE_CONTEXT)

    @property
    def n_trees(self):
        """Number of trees in the model."""
        return _native_host_archive(self, "host_tree_count", (), _ARCHIVE_CONTEXT)

    @property
    def inference_engine(self):
        """Name of the prediction engine, ``'sequential'``."""
        return _native_host_archive(self, "host_tree_engine", (), _ARCHIVE_CONTEXT)

    def model_sha256(self):
        """Return the SHA-256 hex digest of the model text (UTF-8).

        It equals the digest of the archive's ``model`` member.
        """
        return _native_host_archive(self, "host_tree_hash", (), _ARCHIVE_CONTEXT)

    def predict(self, X):
        """Return raw float32 scores, as ``GradientBoosting.predict`` does.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        mojolearn.Array
            Shape ``(n_samples,)`` for a single-output loss, otherwise
            ``(n_samples, approx_dim)``.
        """
        # `GradientBoosting._check_fitted` stages X COLUMN-major through
        # `as_f32_colmajor`, whose 2-D flip is the base binding's
        # `transpose_f32`, a stub on a CPU-only install. The float32 cast is
        # the same `cast_f64_to_f32` helper here (the host binding carries
        # it), and the flip is the package's own `_reorder`, which
        # `transpose_f32` documents itself as ("a pure move, no arithmetic:
        # every float32 bit pattern, NaN payloads included, arrives
        # unchanged", `bindings/_mojolearn.mojo:765-768`); the bytes the
        # binding reads are therefore the GPU path's by that contract.
        return _native_host_archive(self, "host_tree_predict", (X,), _ARCHIVE_CONTEXT)

    def predict_proba(self, X):
        """Return float64 class probabilities ``[1 - p, p]`` for ``Logloss`` and ``CrossEntropy`` models.

        Matches ``GradientBoosting.predict_proba`` for these losses. ``MultiClass``
        and ``MultiClassOneVsAll`` models raise ``NotImplementedError``; any other
        loss raises ``ValueError``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        mojolearn.Array
            Shape ``(n_samples, 2)``.
        """
        return _native_host_archive(self, "host_tree_proba", (X,), _ARCHIVE_CONTEXT)


# CPU-only entries the gbdt `_HostBinding` serves beside the host binding's
# own (`_backend._HOST_PY_ENTRIES`): a CPU-trained model with CTR records,
# whose records the host binding's parser refuses by name. GPU-path modules
# reach these only through a CPU-vendor binding, never by import.
def gbdt_ctr_model_dim(text):
    """The approx dim of a CTR model text, read by `parse_model_text`."""
    return _native_host_archive(None, "host_tree_ctr_dim", (text,), _ARCHIVE_CONTEXT)


def gbdt_ctr_predict(X, *, loss, text, n_features_in, approx_dim, n_classes, estimator):
    """`HostGBDT(...).predict(X)` for a CTR model text."""
    return _native_host_archive(None, "host_tree_ctr_predict",
        (X, loss, text, n_features_in, approx_dim, n_classes, estimator), _ARCHIVE_CONTEXT)


# Defer this import until HostGBDT exists: the shared archive registry also
# connects this class when HostForest is the first public name imported.
from ._forest_host import _load, _native_host_archive, _ARCHIVE_CONTEXT
