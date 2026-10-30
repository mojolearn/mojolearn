"""Real padded-token copies must retain their input storage through the copy."""
import numpy as np
import pytest


@pytest.mark.parametrize('threaded', [False, True])
def test_host_ragged_tokens_match_independent_rows(threaded):
    import mojolearn as ml
    from mojolearn import _identity_break as fixtures

    shape = ml.ByteLanguageModelConfig()
    _, parameters = fixtures._byte_lm_params(shape)
    model = ml.LanguageModelInference(parameters, shape=shape, threaded=threaded, threads=2)
    lengths = (shape.length, max(1, shape.length // 2), 1)
    ids = (np.arange(len(lengths) * shape.length, dtype=np.int32).reshape(len(lengths), shape.length) * 13 + 7) % shape.vocab_size
    expected = [np.asarray(model.logits(np.ascontiguousarray(ids[i:i + 1, :n]))).copy()
                for i, n in enumerate(lengths)]
    padded_ids = ids.copy()
    for row, size in enumerate(lengths):
        padded_ids[row, size:] = -12345
    for _ in range(4):
        actual = np.asarray(model.logits(padded_ids, lengths=lengths)).copy()
        next_ids = np.asarray(model.next_bytes(padded_ids, lengths=lengths))
        for row, size in enumerate(lengths):
            np.testing.assert_array_equal(actual[row:row + 1, :size].view(np.uint32), expected[row].view(np.uint32))
            assert actual[row, size:].tobytes() == bytes(actual[row, size:].nbytes)
            assert next_ids[row] == int(np.argmax(expected[row][0, -1]))
