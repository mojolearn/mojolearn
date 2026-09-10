#!/usr/bin/env python3
"""Run GPU RF/ET, scoring and scaler smoke checks with NumPy imports blocked.

Run with an installed package (no PYTHONPATH) for wheel qualification, or
PYTHONPATH=python for a source-tree correctness check. Requires a GPU and
current native bindings. This is deliberately not a performance benchmark.
"""
import importlib.abc, sys, json
class RejectNumpy(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname=='numpy' or fullname.startswith('numpy.'):
   raise ImportError('NumPy blocked for runtime qualification')
sys.meta_path.insert(0,RejectNumpy())
import mojolearn as ml
from mojolearn import metrics
X=ml.Array.from_list([[float(i),float(i%3)] for i in range(16)],'<f4')
y=[i%2 for i in range(16)]
results=[]
for cls in (ml.RandomForestClassifier,ml.ExtraTreesClassifier):
 m=cls(n_estimators=2,max_depth=2,numeric_mode='identical')
 m.fit(X,y)
 pred=m.predict(X)
 score=metrics.accuracy_score(y,pred,numeric_mode='identical')
 results.append(dict(estimator=cls.__name__,shape=pred.shape,score=score))
for cls in (ml.StandardScaler,ml.MinMaxScaler):
 m=cls(numeric_mode='identical');out=m.fit_transform(X);restored=m.inverse_transform(out)
 assert out.shape==X.shape and restored.shape==X.shape
 results.append(dict(estimator=cls.__name__,shape=out.shape))
folds=ml.cross_val_score(ml.RandomForestClassifier(n_estimators=2,max_depth=2,numeric_mode='identical'),X,y,cv=2)
assert folds.shape==(2,)
results.append(dict(operation='cross_val_score',scores=folds.tolist()))
assert not any(n=='numpy' or n.startswith('numpy.') for n in sys.modules)
print(json.dumps(dict(numpy_loaded=False,scope='GPU correctness smoke, not timing',package_file=ml.__file__,results=results)))
