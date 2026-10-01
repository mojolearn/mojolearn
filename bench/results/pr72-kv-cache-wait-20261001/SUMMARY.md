# PR #72 no hard wait after each layer's KV cache update (lane/neural-pass67), 2026-10-01

## M3 Ultra (~/pr72-metal; main = mirror main 3befccd28)

Same bits (4a8e781b0739a038, same losses). lm-forward stage 89.8 -> 90.9 ms, race 86.5 -> 92.5; train stage 311.2 -> 308.2, race 291.7 -> 293.6.
attn.rope_and_cache 0.510 -> 0.502 ms/layer: the wait was not the M3's rope_and_cache cost. Neutral on Apple.
NVIDIA (nvc1-0016) and AMD (pass13) pending.
