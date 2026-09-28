#!/usr/bin/env python3
"""Run GPU RF/ET, scoring and scaler smoke checks with NumPy imports blocked.

Run with an installed package (no PYTHONPATH) for wheel qualification, or
PYTHONPATH=python for a source-tree correctness check. Requires a GPU and
current native bindings. This is deliberately not a performance benchmark.
"""
import argparse, importlib.abc, importlib.metadata, pathlib, sys, json, re

def metadata_errors(required, extras, plugins=()):
 """Allow optional NumPy extras and exact split-core pins, never base NumPy.

 This audit runs in a clean environment, so it uses only the standard library.
 The wheel audit independently validates the split marker and its version.
 """
 errors = []
 remaining = list(required)
 for plugin in plugins:
  if remaining.count(plugin) != 1:
   errors.append('missing or duplicated exact GPU plugin requirement: ' + plugin)
  if plugin in remaining:
   remaining.remove(plugin)
 for dependency in remaining:
  name, separator, marker = dependency.partition(';')
  optional_numpy = (re.fullmatch(r"numpy(?:\s*[<>=!~].*)?", name.strip(), re.I)
                    and separator and any(extra in extras
                    and re.fullmatch(r"\s*extra\s*==\s*[\"']" + extra + r"[\"']\s*", marker)
                    for extra in ('numpy', 'verify')))
  if not optional_numpy:
   errors.append('unexpected runtime requirement: ' + dependency)
 return errors


def main():
 parser = argparse.ArgumentParser(description=__doc__)
 parser.add_argument("--no-gpu", action="store_true")
 parser.add_argument("--installed", action="store_true")
 args = parser.parse_args()
 class RejectNumpy(importlib.abc.MetaPathFinder):
  def find_spec(self, fullname, path=None, target=None):
   if fullname=='numpy' or fullname.startswith('numpy.'):
    raise ImportError('NumPy blocked for runtime qualification')
 sys.meta_path.insert(0,RejectNumpy())
 import mojolearn as ml
 if args.installed:
  try:
   importlib.metadata.distribution('numpy')
  except importlib.metadata.PackageNotFoundError:
   pass
  else:
   raise AssertionError('Installed qualification requires a clean environment without NumPy')
  assert pathlib.Path(ml.__file__).resolve().is_relative_to(pathlib.Path(sys.prefix).resolve()), ml.__file__
  distribution = importlib.metadata.distribution('mojolearn')
  from mojolearn import gpu_plugins
  plugins = (gpu_plugins.core_requirements(distribution.version)
             if distribution.read_text(gpu_plugins.CORE_MARKER) is not None else [])
  errors = metadata_errors(distribution.requires or [],
                           distribution.metadata.get_all('Provides-Extra', []), plugins)
  assert not errors, errors
 if args.no_gpu:
  assert ml.Array.from_list([1, 2], '<i4').tolist() == [1, 2]
  print(json.dumps(dict(numpy_loaded=False, package_file=ml.__file__, scope='import and Array only; GPU not tested')))
  sys.exit(0)
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


if __name__ == '__main__':
 main()
