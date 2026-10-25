# SPDX-License-Identifier: Apache-2.0
"""Verification metadata for explicitly authorized Python compatibility work.

This module does not implement or select a product fallback. The verifier uses
it to report an existing callback without calling that operation Python-free.
"""
import sys
from types import MethodType, ModuleType

from ._array import Array

ARRAY_LENGTH_METADATA = {
    'id': 'array-length-metadata',
    'authorization': 'Andrew, 2026-10-25: documented Python fallbacks authorized',
    'reason': 'Retained first-axis length metadata after a Codex Daybreak interruption; native migration remains incomplete.',
    'callee': 'mojolearn._array.Array.__len__',
    'upstream_context': 'https://github.com/modular/modular/issues/7399',
}
_LENGTH_CODE = Array.__len__.__code__
_OWNERS = ('VotingRegressor', 'VotingClassifier', 'StackingRegressor',
           'StackingClassifier', 'MultiOutputRegressor', 'MultiOutputClassifier',
           'OneVsRestClassifier', 'CalibratedClassifierCV')
_METHODS = frozenset(('fit', 'fit_transform', 'predict', 'predict_proba', 'transform', 'score'))


def approved_metadata_callbacks(operation):
    """Return exact callback code identities for the declared public owners."""
    if type(operation) is not MethodType or operation.__func__.__name__ not in _METHODS:
        return {}
    module = sys.modules.get('mojolearn._expansion_trees')
    if type(module) is not ModuleType:
        return {}
    owner = type(operation.__self__)
    if not any(owner is vars(module).get(name) for name in _OWNERS):
        return {}
    # A subclass or monkeypatched length implementation is not authorized by
    # this exception. The callback must be the originally connected body.
    length = vars(Array).get('__len__')
    if getattr(length, '__code__', None) is not _LENGTH_CODE:
        return {}
    return {_LENGTH_CODE: ARRAY_LENGTH_METADATA}
