# Byte-LM session buffers from device arenas on Apple (PR #53, lane/neural-pass44), M3 Ultra, 2026-10-01

Main (8beab1c70, with #51) vs main + #53, both built on the M3; arena on (default on Apple) vs MOJOLEARN_DEVICE_ARENA=0.
Digests and losses identical in every run (lm-forward 4a8e781b0739a038, transformer-forward d5a2b289afdb5709).

| | main | arena (default) | arena off |
|---|---:|---:|---:|
| launch probe, open board-shape session | | 22.2 us | 65.9 us |
| launch probe, fresh context | | 15.3 us | 18.4 us |
| lm-forward, stage tool (2 runs) | 90.9 / 91.2 | 77.9 / 77.8 | 90.5 / 93.2 |
| lm-train-step, stage tool (2 runs) | 275.4 / 271.5 | 183.0 / 186.8 | 275.5 / 281.4 |
| board lm-forward race | 82.2 | 72.6 | 82.1 |
| board lm-train-step race | 188.8 | 174.7 | 188.6 |

966 views from 12 chunks. Metal's per-launch cost scales with live allocations; carving the session's ~1,300
buffers from a few arenas brings an open-session launch back near a fresh context's. NVIDIA/AMD unaffected (Apple-only
default). Raw: R2 measurements/2026-10-01/pr53-metal.tar.gz.
