"""Stored-array quality only: no fit, device, opponent, or timing."""
import unittest
import numpy as np
import classical_two_datasets as classical
import bench_board_more as more


class IVFRecallQuality(unittest.TestCase):
    def setUp(self):
        self.data={'index':np.arange(128,dtype=np.float32)[:,None],
                   'queries':np.zeros((1,1),dtype=np.float32)}
    def test_ivf_k_is_independent_of_classical_knn_k(self):
        self.assertNotEqual(more.IVF_K,classical.KNN_K)
        q=more.quality('ivf',self.data,{'ours-fast':{'ind':np.arange(more.IVF_K)[None,:]}})
        self.assertEqual(q['ours-fast']['recall_at_k'],1.0)
        self.assertEqual(q['ours-fast']['rows_with_repeated_ids'],0)
    def test_bad_ivf_neighbors_score_zero(self):
        q=more.quality('ivf',self.data,{'ours-fast':{'ind':np.arange(100,100+more.IVF_K)[None,:]}})
        self.assertEqual(q['ours-fast']['recall_at_k'],0.0)
    def test_classical_default_k_unchanged(self):
        q=classical.quality('knn',self.data,{'ours':{'ind':np.arange(classical.KNN_K)[None,:]}},{})
        self.assertEqual(q['ours']['recall_at_k'],1.0)
    def test_declared_k_and_array_width_must_agree(self):
        with self.assertRaisesRegex(ValueError,'shape'):
            classical.quality('knn',self.data,{'ours':{'ind':np.arange(10)[None,:]}},{})
    def test_invalid_k_refused(self):
        for k in (0,129):
            with self.assertRaisesRegex(ValueError,'index row'):
                classical.quality('knn',self.data,{}, {},knn_k=k)

if __name__=='__main__':unittest.main()
