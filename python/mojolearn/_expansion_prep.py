# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE PREP LANE'S PUBLIC DOOR (docs/lanes/ALGORITHM_EXPANSION_BRIEFS.md).

Owned by the `prep` expansion lane. `mojolearn/__init__.py` imports this
module and makes every name in `__all__` public as `mojolearn.<name>`; a name
that is already public is refused at import. Put the classes here, or import
them here from the lane's own modules. Two optional hooks route a SAVED model
to the CPU (`mojolearn.host_model`); `_classical_host.py` merges them when it is
first imported, after the package, so both may rely on every module existing:

  CLASSICAL_HOST_BASENAMES     {"_mojolearn_x_prep": "_mojolearn_x_prep_host"}
  def classical_host_formats() -> {"mojolearn-<format>-1": {"Estimator": HostEstimator}}
                               import `_classical_host` INSIDE it and subclass
                               its `_HostBound`; never import it at module level
"""
from . import _backend
from ._arrays import _addr, _addr_ro
from ._buffer import empty
from .preprocessing import _ScalerProtocol

__all__ = ["ExpansionDummyScaler"]


class ExpansionDummyScaler(_ScalerProtocol):
    """THE PROOF DUMMY (lane/algos-prep; removed before merge): scale_ is each
    column's mean absolute value, fitted on `_mojolearn_x_prep`."""
    _parameters = ()

    def fit(self, X, y=None):
        from ._cpu_reference import require_training
        require_training(self)
        values = self._input(X)
        n, d = values.shape
        mode = _backend.default_mode()
        out = empty((d,), "<f4")
        _backend.binding("_mojolearn_x_prep", mode).x_prep_l1_mean(_addr_ro(values), _addr(out), [n, d])
        self.scale_, self.numeric_mode_, self.n_features_in_ = out, mode, d
        return self

    def transform(self, X):
        values = self._input(X)
        n, d = values.shape
        out = empty((n, d), "<f4")
        _backend.binding("_mojolearn_x_prep", self.numeric_mode_).x_prep_scale(
            _addr_ro(values), _addr_ro(self.scale_), _addr(out), [n, d])
        return out
