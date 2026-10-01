# moe expert products as tiled device kernels (PR #34, lane/neural-pass29), 2026-10-01

Board moe cell (synthetic), ours, 3 rounds; restore = branch with MOJOLEARN_SEQ_MOE_TILED=0:
| | before | after (tiled) | restore | digest |
|---|---|---|---|---|
| AMD MI325X (released 0.8.32 vs branch) | 1,271.4 ms | 168.7 ms (7.5x) | 1,257.2 ms | 393fefe91af8350f |
| NVIDIA L40S (released 0.8.32 vs branch) | 951.1 ms | 157.7 ms (6.0x) | 952.4 ms | 393fefe91af8350f |
| Apple M3 Ultra (main vs branch, Metal) | 979.8 ms | 871.6 ms (1.12x) | 962.4 ms | f831134ea9e61e93 |
AMD and NVIDIA digests are equal. The Mac digest differs because the cell's torch-generated inputs differ on arm64
torch (see embedding-transfers-20261001); the author's fixed fixture gives a74a239c300d7131 on host, Metal tiled and
Metal items. Files: amd/, nvidia/ (race3-* is the run with torch installed), m3-ultra/ (race2-*).
