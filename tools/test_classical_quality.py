"""CPU-only checks for full-input clustering validation; no GPU timing."""
from unittest.mock import patch
import numpy as np
from sklearn.metrics import silhouette_score
import classical_two_datasets as C


def test_hdbscan_silhouette_uses_every_consumed_row_and_keeps_noise():
    x = np.array([[-3, 0], [-2, 0], [0, 0], [.1, 0], [3, 0], [3.1, 0], [3.2, 0], [3.3, 0], [99, 99]], dtype=np.float32)
    labels = np.array([-1, -1, 0, 0, 1, 1, 1, 1], dtype=np.int32)
    with patch('sklearn.metrics.silhouette_score', wraps=silhouette_score) as score:
        quality = C.quality('hdbscan', {'X': x}, {'ours': {'labels': labels}, 'library': {'labels': labels}}, {'hdbscan': {'rows': 8}})
    assert score.call_count == 2
    for call in score.call_args_list:
        assert np.array_equal(call.args[0], x[:8])
        assert np.array_equal(call.args[1], labels)
        assert call.kwargs == {'metric': 'euclidean'}
    assert quality['ours']['silhouette'] == quality['library']['silhouette']
    assert quality['ours']['rows'] == 8
    assert quality['library']['ari_vs_ours'] == 1
