# SPDX-License-Identifier: Apache-2.0
"""TSNE door RUN (`x_ann/mani_tsne_run.mojo`) without Python.

Each fixture is one typed check or plan value: accepted and refused values,
the refusal wording, the automatic learning rate and the start modes.
Independent fixtures run across the host's cores.
"""
from core.host_parallel import host_parallelize
from x_ann.mani_tsne_run import (
    TSNE_EXPLORATION_MAX_ITER, TSNE_INIT_ARRAY, TSNE_INIT_RANDOM, TSNE_INIT_PCA, tsne_check_samples,
    tsne_perplexity_positive, tsne_perplexity_refusal, tsne_perplexity_below_samples,
    tsne_perplexity_samples_refusal, tsne_auto_learning_rate, tsne_post_factor_valid, tsne_post_factor_refusal,
    tsne_exploration_iterations, tsne_init_mode, tsne_check_pca_components, tsne_init_shape_matches,
    tsne_init_shape_refusal, tsne_check_input_status, tsne_fit_tests_input,
)

comptime MANI_TSNE_RUN_FIXTURES = 10


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def fixture(unit: Int) raises:
    if unit == 0:
        tsne_check_samples(2, 3)
        try:
            tsne_check_samples(1, 3)
            raise Error("accepted")
        except e:
            require(String(e) == "Found array with 1 sample(s) (shape=(1, 3)) while a minimum of 2 is required by TSNE.",
                    "sample count refusal wording")
    elif unit == 1:
        require(tsne_perplexity_positive(30.0) and not tsne_perplexity_positive(0.0), "perplexity sign")
        var zero = 0.0
        require(not tsne_perplexity_positive(zero / zero), "NaN perplexity is refused")
        require(String(tsne_perplexity_refusal("-1.0"))
                == "The 'perplexity' parameter of TSNE must be a float in the range (0.0, inf). Got -1.0 instead.",
                "perplexity refusal wording")
    elif unit == 2:
        require(tsne_perplexity_below_samples(4.5, 5) and not tsne_perplexity_below_samples(5.0, 5),
                "perplexity below n_samples")
        require(String(tsne_perplexity_samples_refusal("5.0", 5)) == "perplexity (5.0) must be less than n_samples (5)",
                "perplexity samples refusal wording")
    elif unit == 3:
        # max(n / exag / 4.0, 50.0)
        require(tsne_auto_learning_rate(100, 12.0) == 50.0, "the floor of 50")
        require(tsne_auto_learning_rate(4800, 12.0) == 100.0, "n / exag / 4")
    elif unit == 4:
        var zero = 0.0
        var infinite = 1.0 / zero
        require(tsne_post_factor_valid(1.0), "a positive finite factor")
        require(not tsne_post_factor_valid(0.0) and not tsne_post_factor_valid(-1.0), "a non-positive factor")
        require(not tsne_post_factor_valid(infinite) and not tsne_post_factor_valid(zero / zero), "inf and NaN")
        require(String(tsne_post_factor_refusal("0.0"))
                == "mojolearn TSNE: post_exaggeration_learning_rate_factor must be positive and finite, got 0.0",
                "post factor refusal wording")
    elif unit == 5:
        require(tsne_exploration_iterations(1000) == TSNE_EXPLORATION_MAX_ITER, "exploration capped at 250")
        require(tsne_exploration_iterations(100) == 100, "a short run explores throughout")
    elif unit == 6:
        require(tsne_init_mode("random") == TSNE_INIT_RANDOM and tsne_init_mode("pca") == TSNE_INIT_PCA,
                "builtin start modes")
        require(tsne_init_mode("spectral") == TSNE_INIT_ARRAY, "other words fall to the array arm")
        require(tsne_fit_tests_input(TSNE_INIT_RANDOM) and tsne_fit_tests_input(TSNE_INIT_ARRAY)
                and not tsne_fit_tests_input(TSNE_INIT_PCA), "who asks for X's finite status")
    elif unit == 7:
        tsne_check_pca_components(2, 2)
        for k in range(2):
            try:
                tsne_check_pca_components(k * 3, 2)
                raise Error("accepted")
            except e:
                require(String(e) == "mojolearn n_components must be in [1, 2], got " + String(k * 3),
                        "pca components refusal wording")
    elif unit == 8:
        var good: List[Int] = [4, 2]
        var wide: List[Int] = [4, 3]
        var flat: List[Int] = [8]
        require(tsne_init_shape_matches(good, 4, 2), "matching init shape")
        require(not tsne_init_shape_matches(wide, 4, 2) and not tsne_init_shape_matches(flat, 4, 2),
                "mismatched init shapes")
        require(String(tsne_init_shape_refusal("(8,)", 4, 2))
                == "mojolearn TSNE: an init array must have shape (4, 2), got (8,)", "init shape refusal wording")
    else:
        tsne_check_input_status(0)
        try:
            tsne_check_input_status(1)
            raise Error("accepted")
        except e:
            require(String(e) == "Input X contains NaN or infinity", "input status refusal wording")


def run_checks() raises:
    var failures = List[Int](length=MANI_TSNE_RUN_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL mani_tsne_run fixture", unit, String(error))
    host_parallelize(task, MANI_TSNE_RUN_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    _ = failures^
    require(failed == 0, "mani_tsne_run failed " + String(failed) + " fixtures")
    print("PASS mani_tsne_run:", MANI_TSNE_RUN_FIXTURES, "typed check and plan fixtures")
