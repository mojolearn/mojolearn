# SPDX-License-Identifier: Apache-2.0
"""KernelDensity cache identity and take-once resident ownership words."""
from core.kde_native_api import KDENativeDriver


@fieldwise_init
struct KDEFitKey(Copyable, Movable):
    var train: Int
    var rows: Int
    var features: Int
    var weights: Int
    var kernel: String
    var metric: String
    var bandwidth: Float64

    def matches(self, other: Self) -> Bool:
        return (self.train == other.train and self.rows == other.rows and self.features == other.features
            and self.weights == other.weights and self.kernel == other.kernel and self.metric == other.metric
            and self.bandwidth == other.bandwidth)


@fieldwise_init
struct KDEResidentState(Movable):
    var owner: Int
    var provider: Int
    var key: KDEFitKey
    var handle: Int
    var driver: KDENativeDriver

    def matches(self, owner: Int, provider: Int, key: KDEFitKey) -> Bool:
        return self.handle >= 0 and self.owner == owner and self.provider == provider and self.key.matches(key)

    def take_handle(mut self) -> Int:
        var handle = self.handle
        self.handle = -1
        return handle


@fieldwise_init
struct KDEAdmissionWords(Movable):
    var words: List[Int]

    def matches(self, words: List[Int]) -> Bool:
        if len(self.words) != len(words):
            return False
        for i in range(len(words)):  # driver-loop: fixed fitted-array metadata, never data values
            if self.words[i] != words[i]:
                return False
        return True
