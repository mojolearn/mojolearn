# PRs #43 (TSQR split), #45 (sequence pool), #39 (IVF transport + staged scan), #46 (QR tiles), 2026-10-01

Digests identical in every run on every vendor (before / after / restore).

**#43 TSQR split (merged)**, on main with #41; restore = MOJOLEARN_QR_SPLIT=0:
| | AMD MI325X | NVIDIA L40S | Apple M3 Ultra |
|---|---|---|---|
| svd istella split / restore | 10,265 / 27,535 ms (2.7x) | 14,968 / 30,256 ms (2.0x) | 23,973 / 43,974 ms (1.8x; branch without #41) |
| PCA(svd_solver="full") 1M x 220 split / restore | 2,601 / 21,036 ms (8.1x) | 3,945 / 20,342 ms (5.2x) | — |
| PCA Z sha | a66bad1a27c69cf9 | a66bad1a27c69cf9 | — |
qr cells unchanged (the reduced route is #41's).

**#45 sequence pool (merged)**: Apple layernorm 71.4 -> 34.0 ms (2.1x); AMD and NVIDIA flat (AMD layernorm
82 -> 83, NVIDIA 28.9 -> 27.7, cross-entropy 342 -> 333 ms). Other Apple cells need the x_cnn binding (not measured).

**#39 IVF (merged, staged scan the NVIDIA/AMD default)**, 400,000 x 220, sha 95089ab768e8be7a everywhere:
search AMD 560 -> 453 ms staged (477 plain); NVIDIA 787 -> 646 staged (698 plain); fit about even.

**#46 QR dot/update tiles (NOT merged)**: on the L40S (the only vendor on the device route) tiles on vs off:
qr taxi 495 vs 278 ms, istella 15,697 vs 8,252; svd taxi 534 vs 300, istella 37,611 vs 30,677: slower.
