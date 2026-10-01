# PR #72 no hard wait after each layer's KV cache update (lane/neural-pass67), 2026-10-01

## M3 Ultra (~/pr72-metal; main = mirror main 3befccd28)

Same bits (4a8e781b0739a038, same losses). lm-forward stage 89.8 -> 90.9 ms, race 86.5 -> 92.5; train stage 311.2 -> 308.2, race 291.7 -> 293.6.
attn.rope_and_cache 0.510 -> 0.502 ms/layer: the wait was not the M3's rope_and_cache cost. Neutral on Apple.

## AMD MI325X (pass13, /root/ab-pr72-amd)

Same bits on every arm. lm-forward main 23.324/23.022 vs pass67 23.446/23.041 ms; train main 58.484/58.352 vs pass67 58.518/58.573. Neutral on AMD.
R2 `measurements/2026-10-01/ab-pr72-amd.tar.gz`.

NVIDIA: the nvc1-0016 output was lost when nvc1 self-deleted; rerun queued on nvc2 (/root/ab-pr72-nvidia).
