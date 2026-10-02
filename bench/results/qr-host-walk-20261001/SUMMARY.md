# geqrf/orgqr as a block-interleaved host walk (PR #41, lane/neural-pass37), 2026-10-01

Measured on head 31266d3a6 (the walk taken on every column for m >= 65,536); merged at 4ccd78813, which keeps the walk
as the default on AMD and Apple and the device route on NVIDIA (MOJOLEARN_XD_QR_HOST=0/1 forces either), per these
numbers. Digests identical in every run on all three vendors: qr e54f8db0f5ca6525 / 76e60d3379a2735a, svd
f07891b9aaf42a98 / 92c189cb2337dc73.

walk vs device route (MOJOLEARN_XD_QR_HOST=0, = main):
| cell | AMD MI325X | Apple M3 Ultra | NVIDIA L40S |
|---|---|---|---|
| qr taxi | 888.4 -> 182.7 ms (4.9x) | 498.0 -> 195.3 ms (2.5x) | 283.7 -> 544.3 ms (slower) |
| qr istella | 16,732.6 -> 5,081.6 ms (3.3x) | 16,954.2 -> 7,729.1 ms (2.2x) | 8,351.7 -> 14,344.6 ms (slower) |
| svd taxi | 914.8 -> 279.0 ms | 533.5 -> 234.4 ms | 305.2 -> 723.0 ms (slower) |
| svd istella | 38,029.7 -> 27,372.1 ms | 43,914.8 -> 35,698.9 ms | 30,533.6 -> 33,944.6 ms (slower) |
Released 0.8.32 (before): AMD qr 4,049 / 76,780 ms, svd 4,080 / 92,194 ms; NVIDIA qr 966 / 21,112, svd 1,062 / 42,174.
