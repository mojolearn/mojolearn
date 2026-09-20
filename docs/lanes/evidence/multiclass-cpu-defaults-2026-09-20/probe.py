import sys,json,hashlib
import numpy as np
import mojolearn as ml
from mojolearn import _backend
rng=np.random.default_rng(17)
x=rng.standard_normal((240,6)).astype(np.float32)
y=np.digitize(x[:,0]+x[:,2]*.5,[-.4,.4]).astype(np.int32)
h=rng.standard_normal((37,6)).astype(np.float32)
def sha(v): return hashlib.sha256(v).hexdigest()
results={}
for loss in ['MultiClass','MultiClassOneVsAll']:
 for boot in ['Bayesian','Bernoulli','Poisson','No']:
  for strength in [0.,1.]:
   kw=dict(n_estimators=4,max_depth=3,loss=loss,bootstrap_type=boot,random_strength=strength,class_weights=[1.,2.,.5])
   if boot in ['Poisson','Bernoulli']:kw['subsample']=.66
   m=ml.GradientBoosting(**kw).fit(x,y)
   parts={'model':sha(m.model_.encode()),'predict':sha(np.asarray(m.predict(h)).tobytes()),'proba':sha(np.asarray(m.predict_proba(h)).tobytes())}
   results[f'{loss}/{boot}/{strength}']=parts
   print(loss,boot,strength,parts,flush=True)
json.dump({'vendor':_backend.vendor(),'parts':results},open(sys.argv[1],'w'),indent=2)
