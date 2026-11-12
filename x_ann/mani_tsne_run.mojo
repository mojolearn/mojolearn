# SPDX-License-Identifier: Apache-2.0
"""TSNE door RUN: the checks on typed values and the typed fit plan.

`bindings/mani_tsne_native.mojo` is the UNPACK/PACK side: it reads the
estimator's fields, converts the scalars, admits the start array, calls
`pca_fit_owned` and `x_ann_tsne_fit` and publishes the fitted attributes. It
calls these checks at the point the public method made them, so the refusal
order and wording are unchanged. A refusal whose wording quotes the caller's
value takes that text from the door. Nothing here imports Python.
"""

comptime TSNE_EXPLORATION_MAX_ITER = 250
comptime TSNE_INIT_ARRAY = 0
comptime TSNE_INIT_RANDOM = 1
comptime TSNE_INIT_PCA = 2


def tsne_check_samples(n: Int, d: Int) raises:
    # scikit-learn's TSNE validates X with ensure_min_samples=2
    if n < 2:
        raise Error("Found array with " + String(n) + " sample(s) (shape=(" + String(n) + ", " + String(d)
                    + ")) while a minimum of 2 is required by TSNE.")


def tsne_perplexity_positive(perplexity: Float64) -> Bool:
    return perplexity > 0.0


def tsne_perplexity_refusal(perplexity_repr: String) -> Error:
    return Error("The 'perplexity' parameter of TSNE must be a float in the range (0.0, inf). Got "
                 + perplexity_repr + " instead.")


def tsne_perplexity_below_samples(perplexity: Float64, n: Int) -> Bool:
    return perplexity < Float64(n)


def tsne_perplexity_samples_refusal(perplexity_str: String, n: Int) -> Error:
    return Error("perplexity (" + perplexity_str + ") must be less than n_samples (" + String(n) + ")")


def tsne_auto_learning_rate(n: Int, exaggeration: Float64) -> Float64:
    """max(n / exag / 4.0, 50.0), Python's order (the first unless the second
    is larger)."""
    var auto_rate = Float64(n) / exaggeration / 4.0
    return Float64(50.0) if 50.0 > auto_rate else auto_rate


def tsne_post_factor_valid(factor: Float64) -> Bool:
    """`0.0 < factor < inf`: positive, and not infinite or NaN (inf - inf is NaN)."""
    return 0.0 < factor and factor - factor == 0.0


def tsne_post_factor_refusal(factor_repr: String) -> Error:
    return Error("mojolearn TSNE: post_exaggeration_learning_rate_factor must be positive and finite, got "
                 + factor_repr)


def tsne_exploration_iterations(max_iter: Int) -> Int:
    return min(TSNE_EXPLORATION_MAX_ITER, max_iter)


def tsne_init_mode(init: String) -> Int:
    """The binding-built start: 1 random from random_state, 2 the PCA
    embedding scaled; 0 for anything else."""
    if init == "random":
        return TSNE_INIT_RANDOM
    if init == "pca":
        return TSNE_INIT_PCA
    return TSNE_INIT_ARRAY


def tsne_check_pca_components(k: Int, d: Int) raises:
    """`PCA._fit`'s n_components refusal."""
    if k < 1 or k > d:
        raise Error("mojolearn n_components must be in [1, " + String(d) + "], got " + String(k))


def tsne_init_shape_matches(shape: List[Int], n: Int, k: Int) -> Bool:
    return len(shape) == 2 and shape[0] == n and shape[1] == k


def tsne_init_shape_refusal(shape_repr: String, n: Int, k: Int) -> Error:
    return Error("mojolearn TSNE: an init array must have shape (" + String(n) + ", " + String(k)
                 + "), got " + shape_repr)


def tsne_check_input_status(status: Int) raises:
    """X's finite status word from the device upload; nonzero refuses X."""
    if status != 0:
        raise Error("Input X contains NaN or infinity")


def tsne_fit_tests_input(mode: Int) -> Bool:
    """The PCA start already tested X on the device; every other start asks
    `x_ann_tsne_fit` for X's finite status word."""
    return mode != TSNE_INIT_PCA
