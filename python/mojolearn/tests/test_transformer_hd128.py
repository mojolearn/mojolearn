"""IDENTICAL hd128 public forward/backward, cache and mutable-weight gate."""
import os
import numpy as np
from mojolearn.transformer import TransformerBlock
from mojolearn.tests.test_transformer_surface import _weights, _uniform


def _u32(a):
    """`a`'s bits as uint32, whether `a` is an ndarray or a mojolearn Array.

    `Array` has NO `.view` (it exposes `tobytes()` and the buffer protocol),
    so `a.view(np.uint32)` raised AttributeError on every library output once
    the numpy-free migration landed. `np.ascontiguousarray` materializes
    either kind without touching the bits, which is what makes this a bitwise
    comparison and not a value one.
    """
    return np.ascontiguousarray(a, dtype=np.float32).ravel().view(np.uint32)


def same_bits(left, right, name):
    assert np.shape(left) == np.shape(right), name
    assert np.array_equal(_u32(left), _u32(right)), name


def main():
    assert os.environ.get("MOJOLEARN_NUMERIC_MODE") == "identical"
    rng = np.random.default_rng(0x128B10C)
    weights = _weights(rng, 128, 1, 1, 128, 256)
    x = _uniform(rng, (2, 5, 128), -2, 2)
    dy = _uniform(rng, x.shape, -0.5, 0.5)
    # The attention path is a build define since lane envgate-neural-lm
    # (2026-11-15): -D MOJOLEARN_TRANSFORMER_ATTN_PATH_EAGER builds the eager
    # arm (it was MOJOLEARN_TRANSFORMER_ATTN_PATH=eager, flipped in this
    # process). Eager against fused crosses two builds: run this file on the
    # default wheel and on the eager wheel with MOJOLEARN_TWO_BUILD_RECORD /
    # _REFERENCE (python/mojolearn/tests/_build_arms.py); without them the
    # cross-build words are skipped and every within-build check runs.
    import hashlib
    words = {}
    for window in (0, 3):
        block = TransformerBlock(weights, n_heads=1, n_kv_heads=1,
                                 head_dim=128, window=window)
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
        words["window%d.forward" % window] = hashlib.sha256(_u32(y).tobytes()).hexdigest()
        for name in sorted(grads):
            words["window%d.grad.%s" % (window, name)] = hashlib.sha256(_u32(grads[name]).tobytes()).hexdigest()
        # The constructor converts the weights once into arrays the block
        # owns (bindings/neural_block_admission_door.mojo admit_block_f32),
        # and every forward hands those to the kernels through the one
        # weight door (frozen, epoch-keyed, as Mamba). An in-place edit of
        # the caller's array therefore never reaches the block.
        q = weights["q_proj.weight"].reshape(-1)
        saved_q = q[3]
        q[3] = np.float32(np.nan)
        try:
            same_bits(block.forward(x), y, "caller edit after construction")
        finally:
            q[3] = saved_q
        # New weights go through the public setter; the next forward
        # refuses the non-finite one, in named order (q_proj before
        # k_proj, whatever the flat indices).
        bad = {name: np.array(value, copy=True) for name, value in weights.items()}
        bad["q_proj.weight"].reshape(-1)[3] = np.float32(np.nan)
        bad["k_proj.weight"].reshape(-1)[0] = np.float32(np.inf)
        error = ""
        try:
            block.set_params(weights=bad)
            block.forward(x)
        except Exception as exc:
            error = str(exc)
        assert "NaN in q_proj.weight at flat index 3" in error, error
        block.set_params(weights=weights)
        same_bits(block.forward(x), y, "restored weights")
    print("Transformer hd128 public PASS: both windows, all gradients, split caches, frozen weights, ordered non-finite refusal")
    return words


def test_eager_and_fused_builds_give_the_same_bits():
    """The words `main` hashes, compared across the default (fused) wheel and
    the -D MOJOLEARN_TRANSFORMER_ATTN_PATH_EAGER wheel."""
    from mojolearn.tests._build_arms import two_build_compare
    two_build_compare("transformer_hd128_attention_path", main())


if __name__ == "__main__":
    main()
