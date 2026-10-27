"""Explicit availability contracts for over-broad installed-method census rows.

The census assigned shared installer method names to unrelated classes. These
rows are NOT numerical executions. A newly exposed method fails this guard so
it must receive a real fixture rather than silently inheriting an exemption.
"""
__mojolearn_user_boundary__ = True
from ._verify_api import resolve
from ._verify_api_metadata import factory
import inspect

ABSENT = frozenset(('mojolearn.CalibratedClassifierCV.fit_transform', 'mojolearn.CalibratedClassifierCV.transform', 'mojolearn.DecisionTreeRegressor.predict_proba', 'mojolearn.ExperimentalTwoLevelFeatureFreq.apply', 'mojolearn.ExperimentalTwoLevelFeatureFreq.predict_with_leaves', 'mojolearn.ExperimentalTwoLevelFeatureFreq.score', 'mojolearn.ExperimentalTwoLevelFeatureFreq.staged_predict', 'mojolearn.ExtraTreesRegressor.predict_proba', 'mojolearn.GradientBoosting.apply', 'mojolearn.GradientBoosting.predict_with_leaves', 'mojolearn.GradientBoosting.score', 'mojolearn.GradientBoosting.staged_predict', 'mojolearn.GradientBoostingClassifier.apply', 'mojolearn.GradientBoostingClassifier.predict_with_leaves', 'mojolearn.GradientBoostingClassifier.staged_predict', 'mojolearn.GradientBoostingRegressor.apply', 'mojolearn.GradientBoostingRegressor.predict_proba', 'mojolearn.GradientBoostingRegressor.predict_with_leaves', 'mojolearn.GradientBoostingRegressor.staged_predict', 'mojolearn.IsolationForest.apply', 'mojolearn.IsolationForest.predict_proba', 'mojolearn.IsolationForest.predict_with_leaves', 'mojolearn.IsolationForest.score', 'mojolearn.IsolationForest.staged_predict', 'mojolearn.MultiOutputClassifier.fit_transform', 'mojolearn.MultiOutputClassifier.transform', 'mojolearn.MultiOutputRegressor.fit_transform', 'mojolearn.MultiOutputRegressor.predict_proba', 'mojolearn.MultiOutputRegressor.transform', 'mojolearn.OneVsRestClassifier.fit_transform', 'mojolearn.OneVsRestClassifier.transform', 'mojolearn.OrderedRMSE.apply', 'mojolearn.OrderedRMSE.predict_with_leaves', 'mojolearn.OrderedRMSE.score', 'mojolearn.OrderedRMSE.staged_predict', 'mojolearn.RandomForestRegressor.predict_proba', 'mojolearn.StackingRegressor.predict_proba', 'mojolearn.VotingRegressor.predict_proba'))
CONDITIONAL = frozenset('mojolearn.'+name+'.fit_transform' for name in (
    'AlternatingLeastSquares','BernoulliNB','CategoricalNB','ComplementNB',
    'EllipticEnvelope','GaussianNB','MinCovDet','MultinomialNB'))


def check(recorder,public,method):
    endpoint=public+'.'+method
    def operation():
        cls=resolve(public)
        if endpoint in ABSENT:
            try:inspect.getattr_static(cls,method)
            except AttributeError:pass
            else:raise AssertionError('newly exposed endpoint requires a numerical fixture: '+endpoint)
        obj=factory(public)
        if endpoint in CONDITIONAL:
            assert not hasattr(obj,'transform'), 'new transform route requires fit_transform fixture'
        try:getattr(obj,method)
        except AttributeError:return {'availability':'not exposed','numerical_execution':False}
        raise AssertionError('expected absent method was exposed: '+endpoint)
    recorder.check(endpoint,operation,kind='availability contract')
