# SPDX-License-Identifier: Apache-2.0
"""Validated SpectralEmbedding configuration; no estimator/dictionary required."""


@fieldwise_init
struct SpectralEmbeddingConfig(Copyable, Movable):
    var affinity: String
    var components: Int
    var neighbors: Int
    var seed: Int
    var gamma: Float64
    var tolerance: Float64
    var automatic_neighbors: Bool
    var automatic_gamma: Bool

    def neighbors_for(self, rows: Int) -> Int:
        return max(rows // 10, 1) if self.automatic_neighbors else self.neighbors

    def gamma_for(self, features: Int) -> Float64:
        return 1.0 / Float64(features) if self.automatic_gamma else self.gamma
