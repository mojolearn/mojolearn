import hashlib,json
import numpy as np
import mojolearn as ml
from mojolearn._gp_impl import GaussianProcessRegressor,RBF
from mojolearn.randomforest import RandomForestClassifier
assert ml.vendor()=='metal',ml.vendor()
x=np.array([[i/16,(i%3)/4,(i%5)/8] for i in range(16)],dtype=np.float32)
y=np.array([i/8-0.5 for i in range(16)],dtype=np.float32)
gp=GaussianProcessRegressor(kernel=RBF(1.0),alpha=2**-20,optimizer=None).fit(x,y)
pred,std=gp.predict(x,return_std=True)
pred,std=np.asarray(pred),np.asarray(std)
assert np.isfinite(pred).all() and np.isfinite(std).all() and (std>=0).all()
forest=RandomForestClassifier(n_estimators=4,max_depth=3,max_features='log2',random_state=7).fit(x,np.arange(16,dtype=np.int32)%2)
labels=np.asarray(forest.predict(x));assert labels.shape==(16,)
print(json.dumps({'vendor':ml.vendor(),'gp_prediction':hashlib.sha256(pred.tobytes()).hexdigest(),'gp_std':hashlib.sha256(std.tobytes()).hexdigest(),'forest_prediction':hashlib.sha256(labels.tobytes()).hexdigest()}))
