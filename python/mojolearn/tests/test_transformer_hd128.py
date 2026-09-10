"""IDENTICAL hd128 public forward/backward, cache and mutable-weight gate."""
import os
import numpy as np
from mojolearn.transformer import TransformerBlock
from mojolearn.tests.test_transformer_surface import _weights, _uniform


def same_bits(left, right, name):
    assert left.shape == right.shape, name
    assert np.array_equal(left.view(np.uint32), right.view(np.uint32)), name


def main():
    assert os.environ.get("MOJOLEARN_NUMERIC_MODE") == "identical"
    rng = np.random.default_rng(0x128B10C)
    weights = _weights(rng, 128, 1, 1, 128, 256)
    x = _uniform(rng, (2, 5, 128), -2, 2)
    dy = _uniform(rng, x.shape, -0.5, 0.5)
    previous = os.environ.get("MOJOLEARN_TRANSFORMER_ATTN_PATH")
    try:
        for window in (0, 3):
            block = TransformerBlock(weights, n_heads=1, n_kv_heads=1,
                                     head_dim=128, window=window)
            os.environ["MOJOLEARN_TRANSFORMER_ATTN_PATH"] = "auto"
            cache = block.allocate_state(2, 5)
            y = block.forward(x, cache)
            grads = block.backward(x, dy)
            split = block.allocate_state(2, 5)
            prefix = block.forward(x[:, :4, :], split)
            tail = block.step(x[:, 4:5, :], split)
            same_bits(np.concatenate((prefix, tail), axis=1), y, "split output")
            same_bits(split.k_cache, cache.k_cache, "split key cache")
            same_bits(split.v_cache, cache.v_cache, "split value cache")
            assert split.cached_tokens == cache.cached_tokens == 5
            os.environ["MOJOLEARN_TRANSFORMER_ATTN_PATH"] = "eager"
            same_bits(block.forward(x), y, "eager vs fused forward")
            eager_grads = block.backward(x, dy)
            for name in grads:
                same_bits(grads[name], eager_grads[name], "eager vs fused grad " + name)
            os.environ["MOJOLEARN_TRANSFORMER_ATTN_PATH"] = "auto"
            # Caller-owned weights remain live; no address-only cache may
            # hide a mutation after a successful call. Named order beats
            # a lower offending index in the later weight tensor.
            q = weights["q_proj.weight"].reshape(-1)
            k = weights["k_proj.weight"].reshape(-1)
            saved_q, saved_k = q[3], k[0]
            q[3], k[0] = np.float32(np.nan), np.float32(np.inf)
            error = ""
            try:
                block.forward(x)
            except Exception as exc:
                error = str(exc)
            finally:
                q[3], k[0] = saved_q, saved_k
            assert "NaN in q_proj.weight at flat index 3" in error, error
            same_bits(block.forward(x), y, "restored mutable weights")
    finally:
        if previous is None:
            os.environ.pop("MOJOLEARN_TRANSFORMER_ATTN_PATH", None)
        else:
            os.environ["MOJOLEARN_TRANSFORMER_ATTN_PATH"] = previous
    print("Transformer hd128 public PASS: both windows, all gradients, split caches, ordered mutable-weight refusal")


if __name__ == "__main__":
    main()
