# SPDX-License-Identifier: Apache-2.0
# Default-off experimental whole-read policy and committed host publication.
from core.resident_generation_read import ResidentGreedyRead


struct ResidentSamplingRead(Movable):
    var progress: ResidentGreedyRead
    var temperature: Float32
    var top_k: Int
    var top_p: Float32
    var seed: UInt64
    var output_position: Int
    # Boundary retains admitted independent host destinations and their owners
    # across the entire native call. Native recovery never uses ordinary export
    # from a failed (unusable) session.
    var cache_keys: List[Int]
    var cache_values: List[Int]
    var cached_tokens: Int
    var kv_ready: Bool
    var failure_is_forward: Bool

    def __init__(out self, var progress: ResidentGreedyRead,
                 temperature: Float32, top_k: Int, top_p: Float32,
                 seed: UInt64, output_position: Int,
                 var cache_keys: List[Int], var cache_values: List[Int],
                 cached_tokens: Int):
        self.progress = progress^
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.seed = seed
        self.output_position = output_position
        self.cache_keys = cache_keys^
        self.cache_values = cache_values^
        self.cached_tokens = cached_tokens
        self.kv_ready = False
        self.failure_is_forward = False
