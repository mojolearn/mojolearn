"""Tiny public cluster regressions for Python 3.10/3.11 Array buffers.

No NumPy import. The native fitting work is eight rows by two columns. Tests
also refuse direct Array buffer exports on newer Python, so they cannot mask
an accidental dependency on the Python 3.12 __buffer__ protocol.
"""
import builtins
import unittest
from unittest.mock import patch

import mojolearn as ml
from mojolearn import _buffer
from mojolearn import _expansion_cluster as cluster
from mojolearn._array import Array


def data():
    return Array._from_flat([0.,0.,.2,.1,.1,.3,.3,.2,4.,4.,4.2,4.1,4.1,4.3,4.3,4.2], (8,2), '<f4')


def capture_paths():
    """Exercise every Array flattening path, retain exact returned bits."""
    x=data();out={}
    def keep(name,a):out[name]=dict(shape=a.shape,dtype=a.dtype,bytes=_buffer.flat_bytes(a).hex())
    fit=ml.MiniBatchKMeans(2,init=[[0.,0.],[4.,4.]],n_init=1,max_iter=2,batch_size=4,random_state=0).fit(x)
    keep('minibatch-fit',fit.cluster_centers_)
    fit.partial_fit(x);keep('minibatch-after-fit',fit.cluster_centers_)
    part=ml.MiniBatchKMeans(2,init=[[0.,0.],[4.,4.]],n_init=1,batch_size=4,random_state=0)
    part.partial_fit(x);keep('partial-first',part.cluster_centers_)
    part.partial_fit(x);keep('partial-second',part.cluster_centers_);keep('partial-counts',part.counts_)
    bisect=ml.BisectingKMeans(2,max_iter=2,random_state=0).fit(x)
    keep('bisect-predict',bisect.predict(x))
    for kind in ('full','tied'):
        bg=ml.BayesianGaussianMixture(n_components=2,covariance_type=kind,covariance_prior=[[1.,0.],[0.,1.]],max_iter=2,random_state=0).fit(x)
        keep('bgmm-'+kind,bg.score_samples(x))
    gm=ml.GaussianMixture(n_components=2,means_init=[[0.,0.],[4.,4.]],covariance_type='full',max_iter=2,random_state=0).fit(x)
    keep('gmm-means',gm.means_);keep('gmm-score',gm.score_samples(x))
    return out


class ArrayBufferCompatibility(unittest.TestCase):
    def test_public_cluster_paths_without_array_buffer_export(self):
        def no_array_buffer(obj):
            if isinstance(obj,Array):raise TypeError('Array has no native buffer exporter on Python 3.11')
            return builtins.memoryview(obj)
        with patch.object(cluster,'memoryview',no_array_buffer,create=True):
            result=capture_paths()
        self.assertEqual(len(result),10)
        self.assertTrue(all(row['bytes'] for row in result.values()))

    def test_finite_reference_arm_and_typed_bytes(self):
        x=data()
        with patch.object(_buffer,'hotpath_enabled',return_value=False):
            self.assertEqual(_buffer.flat_bytes(cluster._f32(x)).hex(),_buffer.flat_bytes(x).hex())
            with self.assertRaisesRegex(ValueError,'NaN or infinity'):
                cluster._f32([[float('nan'),0.]])
        # Preserve signed-zero/subnormal storage and signed integer tree nodes.
        values=Array._from_flat([-0.,2**-149],(1,2),'<f4')
        self.assertEqual(_buffer.flat_bytes(values).hex(),'0000008001000000')
        nodes=Array._from_flat([-1,0,2],(1,3),'<i4')
        self.assertEqual(list(_buffer.flat_bytes(nodes).cast('i')),[-1,0,2])


if __name__=='__main__':unittest.main()
