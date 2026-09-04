# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632


struct UMAPParams(Copyable, Movable):
    var n_neighbors: Int
    var n_components: Int
    var local_connectivity: Float32
    var set_op_mix_ratio: Float32
    var min_dist: Float32
    var spread: Float32
    var n_epochs: Int
    var random_seed: UInt64

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
    ):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.local_connectivity = local_connectivity
        self.set_op_mix_ratio = set_op_mix_ratio
        self.min_dist = min_dist
        self.spread = spread
        self.n_epochs = n_epochs
        self.random_seed = random_seed

    def validate(self, n_samples: Int) raises:
        if n_samples < 2:
            raise Error("UMAP requires at least two samples")
        if self.n_neighbors < 2 or self.n_neighbors > n_samples:
            raise Error("UMAP n_neighbors must be in [2, n_samples]")
        if self.n_components < 1:
            raise Error("UMAP n_components must be positive")
        if self.local_connectivity != Float32(1.0):
            raise Error("UMAP local_connectivity != 1 is not implemented")
        if self.set_op_mix_ratio < Float32(0.0) or (
            self.set_op_mix_ratio > Float32(1.0)
        ):
            raise Error("UMAP set_op_mix_ratio must be in [0, 1]")
        if self.min_dist < Float32(0.0) or not (
            self.spread > Float32(0.0)
        ) or self.min_dist > self.spread:
            raise Error("UMAP requires 0 <= min_dist <= spread")
        if self.n_epochs < 0:
            raise Error("UMAP n_epochs must be non-negative")

