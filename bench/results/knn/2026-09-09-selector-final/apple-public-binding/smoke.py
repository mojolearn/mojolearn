import numpy as np
import mojolearn as ml
assert ml.numeric_mode() == 'identical'
x = ((np.arange(1024*8).reshape(1024,8)*37 % 251)-125).astype(np.float32)/16
q = x[[0,10,37,80,123,511]].copy()
sq = ((q[:,None,:]-x[None,:,:])**2).sum(axis=2)
for k in (1,10,15,33):
    model=ml.NearestNeighbors(n_neighbors=k, numeric_mode='identical').fit(x)
    d,i=model.kneighbors(q)
    expected=np.argsort(sq,axis=1,kind='stable')[:,:k]
    np.testing.assert_array_equal(i,expected)
    np.testing.assert_array_equal(d,np.sqrt(np.take_along_axis(sq,expected,axis=1)))
    d2,i2=model.kneighbors(q)
    np.testing.assert_array_equal(i,i2)
    np.testing.assert_array_equal(d.view(np.uint32),d2.view(np.uint32))
print('PASS: rebuilt IDENTICAL Python kNN, k=1/10/15/33, exact dyadic oracle, tie order and repeat bits')
