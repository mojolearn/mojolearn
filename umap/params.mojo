# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632


from std.math import isfinite
from umap.curve import fit_umap_curve


#: The largest n_components the layout optimizers take (one register row of
#: this many floats per vertex); above it, refused by name.
comptime UMAP_MAX_COMPONENTS = 32


struct UMAPParams(Copyable, Movable):
    var n_neighbors: Int
    var n_components: Int
    var local_connectivity: Float32
    var set_op_mix_ratio: Float32
    var min_dist: Float32
    var spread: Float32
    var n_epochs: Int
    var random_seed: UInt64
    var learning_rate: Float32
    var repulsion_strength: Float32
    var negative_sample_rate: Int
    # lane/algos-decomp option parity (2026-09-27). Each default is the
    # behaviour before the option existed, so a default fit keeps its bits.
    #: the k-NN metric, a cuVS DistanceType (-1: euclidean, the legacy
    #: sentinel; 0 sqeuclidean, 2 cosine, 3 manhattan, 7 chebyshev, 9
    #: minkowski with `metric_arg` = p)
    var metric: Int
    var metric_arg: Float32
    #: the curve's a and b given directly (both > 0), else fitted from
    #: min_dist and spread (0, 0)
    var curve_a: Float32
    var curve_b: Float32

    def __init__(
        out self,
        n_neighbors: Int = 15,
        n_components: Int = 2,
        local_connectivity: Float32 = Float32(1.0),
        set_op_mix_ratio: Float32 = Float32(1.0),
        min_dist: Float32 = Float32(0.1),
        spread: Float32 = Float32(1.0),
        n_epochs: Int = 0,
        random_seed: UInt64 = UInt64(0),
        learning_rate: Float32 = Float32(1.0),
        repulsion_strength: Float32 = Float32(1.0),
        negative_sample_rate: Int = 5,
        metric: Int = -1,
        metric_arg: Float32 = Float32(2.0),
        curve_a: Float32 = Float32(0.0),
        curve_b: Float32 = Float32(0.0),
    ):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.local_connectivity = local_connectivity
        self.set_op_mix_ratio = set_op_mix_ratio
        self.min_dist = min_dist
        self.spread = spread
        self.n_epochs = n_epochs
        self.random_seed = random_seed
        self.learning_rate = learning_rate
        self.repulsion_strength = repulsion_strength
        self.negative_sample_rate = negative_sample_rate
        self.metric = metric
        self.metric_arg = metric_arg
        self.curve_a = curve_a
        self.curve_b = curve_b

    def validate(self, n_samples: Int) raises:
        if n_samples < 2:
            raise Error("UMAP requires at least two samples")
        if self.n_neighbors < 2 or self.n_neighbors > n_samples:
            raise Error("UMAP n_neighbors must be in [2, n_samples]")
        if self.n_components < 1:
            raise Error("UMAP n_components must be positive")
        if not isfinite(self.local_connectivity) or self.local_connectivity < Float32(0.0):
            raise Error("UMAP local_connectivity must be finite and >= 0")
        if not isfinite(self.set_op_mix_ratio):
            raise Error("UMAP set_op_mix_ratio must be finite")
        if not isfinite(self.min_dist) or not isfinite(self.spread):
            raise Error("UMAP min_dist and spread must be finite")
        if self.set_op_mix_ratio < Float32(0.0) or (
            self.set_op_mix_ratio > Float32(1.0)
        ):
            raise Error("UMAP set_op_mix_ratio must be in [0, 1]")
        if self.min_dist < Float32(0.0) or not (
            self.spread > Float32(0.0)
        ) or self.min_dist > self.spread:
            raise Error("UMAP requires 0 <= min_dist <= spread")
        if not isfinite(self.learning_rate) or self.learning_rate <= Float32(0):
            raise Error("UMAP learning_rate must be positive and finite")
        if not isfinite(self.repulsion_strength) or self.repulsion_strength < Float32(0):
            raise Error("UMAP repulsion_strength must be nonnegative and finite")
        if self.negative_sample_rate < 0 or self.negative_sample_rate > 2147483647:
            raise Error("UMAP negative_sample_rate must fit a nonnegative Int32")
        if self.n_epochs < 0:
            raise Error("UMAP n_epochs must be non-negative")
        if not (
            self.metric == -1 or self.metric == 0 or self.metric == 1 or self.metric == 2
            or self.metric == 3 or self.metric == 7 or self.metric == 9
        ):
            raise Error("UMAP metric must be euclidean, sqeuclidean, cosine, manhattan, chebyshev or minkowski")
        if self.metric == 9 and (not isfinite(self.metric_arg) or not (self.metric_arg > Float32(0.0))):
            raise Error("UMAP minkowski p must be positive and finite")
        if not isfinite(self.curve_a) or not isfinite(self.curve_b) or (
            (self.curve_a > Float32(0.0)) != (self.curve_b > Float32(0.0))
        ) or self.curve_a < Float32(0.0) or self.curve_b < Float32(0.0):
            raise Error("UMAP a and b must both be given (positive) or both be fitted")

    def curve(self) raises -> Tuple[Float32, Float32]:
        """(a, b): given directly, or umap-learn's find_ab_params fit from
        min_dist and spread."""
        if self.curve_a > Float32(0.0):
            return (self.curve_a, self.curve_b)
        var c = fit_umap_curve(self.min_dist, self.spread)
        return (c.a, c.b)

