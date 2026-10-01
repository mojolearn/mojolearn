# Apple byte-LM transport defaults (PR #51, lane/neural-pass43), M3 Ultra, 2026-10-01

Main (537fdc625) vs the branch, both built on the M3 (Metal). Digests equal everywhere: lm-forward 4a8e781b0739a038,
transformer-forward d5a2b289afdb5709, lm-train-step losses identical.

| | main | branch default | MOJOLEARN_DOWNLOAD_STAGE=0 | MOJOLEARN_BYTE_LM_LAYER_SYNC=1 | MOJOLEARN_ATTN_SPECULATIVE=0 |
|---|---:|---:|---:|---:|---:|
| lm-forward (stage tool, median of 5) | 114.7 / 115.9 / 115.0 | 90.8 / 93.0 / 92.9 | 114.0 | 89.2 | 91.5 |
| board lm-forward race (3 rounds) | 108.1 | 81.1 | | | |
| board lm-train-step race (3 rounds) | 194.1 | 188.8 | | | |

lm-forward 1.26-1.33x; the staged download (pinned buffer for the 67 MB logits) is the change that moves it; the other
two defaults are neutral at this shape. The stage tool's lm-train-step median is noisy on this box (main 272-331 ms,
branch 278-339 ms across four runs, with transformer-forward, which no change touches, moving 12-17 ms in the same
runs); the board race shows 194 -> 189 ms. NVIDIA/AMD columns untouched (Apple-only defaults).
Raw: R2 measurements/2026-10-01/pr51-metal.tar.gz.
