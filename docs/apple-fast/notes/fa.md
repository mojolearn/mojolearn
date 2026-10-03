# FactorAnalysis on the Apple GPU (FAST): profile of main's fit (lane/apple-fast-fa, 2026-10-03)

Board lane `factor-analysis` (tools/bench_board_algos.py, AFC_FAMILY=algos, xlane decomp): `FactorAnalysis(n_components=8,
max_iter=1000, tol=1e-2, svd_method="randomized", iterated_power=3, random_state=SEED)`, fit on the cls block's fit rows
(taxi 1,000,000 x 11; istella 1,000,000 x 220), `transform(Xq)` clocked apart as infer_ms. M3 Ultra (gaps file
m3-gaps-0834.tsv): taxi FAST 144.3 / IDENTICAL 157.3 / sklearn 144,607 ms. No istella row in the gaps file and none in
BOARD_M3_FAST.md (fastica, nmf, cca and incremental-pca have istella rows): the istella race of this lane has not landed on the
board; the lane's `datasets` default is both, so it is raced on istella when the board runs it. The 220-column EM is the
heavier case and is requested too (deciding dataset taxi, the one with a number).

Code: python/mojolearn/_expansion_decomp.py `FactorAnalysis.fit` (line ~1913) on the decomp kit `_Kit` (the GPU binding
_mojolearn_x_decomp, x_decomp/device.mojo `DevExec`, x_decomp/resident.mojo for device-resident matrices). `_RES_MIN = 1`, so
on the GPU binding every `ew`, `mm`, `colsum` is a resident launch (no sync) on pooled device buffers; `qr_r`, `svd`, `eigh`,
`lu` are host-address entries: the operand is downloaded (`_M.s`, one sync), re-uploaded inside `DevExec`, the result
downloaded (sync). Reading `.s` of a device matrix is a download plus a sync.

## Once per fit (n >= d, the board's case)

| step | launches | syncs | host copies | buffers |
|---|---|---|---|---|
| `M = _M.from_input(X)` | 0 | 0 | one copy of X into an array.array (host) | - |
| `k.colmean(M)`: colsum + scale | 2 (`colsum_part_kernel` + `fold_kernel`) + 1 | 1 (upload of X, 44 / 880 MB, waits) | - | X on device (n x d), 1 x d x2 |
| `Xc = k.ew("sub", M, mean)` | 1 | 0 | - | Xc (n x d: a second 880 MB buffer on istella) |
| `var = scale(colsum(sq(Xc)))/n` | 1 + 2 + 1 | 0 | - | sq(Xc): a THIRD n x d buffer; 1 x d x2 |
| `Rx = k.qr_r(Xc)` | `qr_factor` (core/householder_qr, sliced; one launch per slice per column: ~2 x 220 x slices on istella) | 1 download of Xc (`Xc.addr` -> `.s`: 880 MB staged in 64 MB pieces, ~21 ms / 64 MB = ~290 ms on istella, ~15 ms on taxi) + `device_qr_r`'s upload/sync/download | `DevExec.qr_r` copies all m x n floats into a `List[Float32]` ONE `append` AT A TIME (220 million on istella, 11 million on taxi), then `device_qr_r` uploads that list | fresh da (n x d), scratch, R |
| `psi = k.const(1.0, 1, d)` | 0 | 0 | host 1 x d | - |

So before the first EM iteration the data crosses the bus three times on the n >= d route (upload, download for qr_r, upload
again), two extra n x d buffers are written and read (Xc, sq), and the host walks every value once. The decomp-linalg lane's
unmerged `MOJOLEARN_FA_FAST_QRR` removes only the List copy (same download, same re-upload).

## Per EM iteration (main; d = 11 taxi, 220 istella; nc = 8)

| step | launches | syncs | note |
|---|---|---|---|
| `sqrt_psi = adds(sqrt(psi), 1e-12)` | 2 | 1 (psi is host after iteration 1: upload waits) | 1 x d |
| `k.ew("scale", k.ew("div", Rx, sqrt_psi), 1/sqrt(n))` | 2 | 1 (Rx host -> upload waits, iteration 1 only; later Rx is on device) | d x d |
| `k.svd(.)`: `DevExec.svd_cells` | `qr_factor_bounded` of the d x d (several launches) + `pj_transpose` + `pj_identity` + (d - 1 + d % 2) `rs_round_kernel` launches PER SWEEP + `rs_norm` + `pj_transpose` | 1 (download of the operand, `.addr`) + 1 (`ctx.synchronize()` before the QR) + 1 PER SWEEP (the rotation flags read back) + 1 (s, v download) + 2 (buffer frees) | 8 fresh device buffers created and freed per call; Python then sorts s, `take_cols`, `.T` on host copies |
| `s2 = sq(sv)`, `Vt.rows`, `s2.cols(nc, d)` -> `_dsum` | 1 | 1 (upload sv) + 1 (`.s` of the tail) | f64 sum on the host |
| `W = mul(Vt, sqrt(maxs(adds(sk, -1), 0)).T)`; `W = mul(W, sqrt_psi)` | 3 + 1 + 1 | 1 (upload sk) + 1 (`.T` downloads) + 1 (upload Vt) + 1 (upload the column) | nc x d |
| `slog = _dsum(logs(sk).s)`, `plog = _dsum(logs(psi).s)` | 2 | 2 uploads + 2 downloads | f64 sums on the host; the convergence test `(ll - old_ll) < tol` in Python |
| `psi = maxs(sub(var, colsum(sq(W))), 1e-12)` | 1 + 1 + 1 + 1 | 0 | psi stays on device (1 x d) |

Per iteration: ~20 small launches outside the SVD, ~12 syncs outside the SVD, plus the SVD's QR launches, its (d - 1) round
launches per sweep and one sync per sweep. Every launch and wait costs ~0.2 ms on Apple before any arithmetic; at d = 11 the
arithmetic is nothing, so the fit is launch-and-wait bound: 144 ms is consistent with a few dozen iterations of ~3 to 4 ms
of waits. At d = 220 the per-sweep round launches dominate (219 launches a sweep, several sweeps an iteration) and the QR of
1,000,000 x 220 plus the 880 MB download and the 220-million-append host copy dominate the setup.

## transform(Xq)

`M = sub(X, mean)` (n x d pass, writes a second n x d buffer), `Wpsi = div(W, psi)`, `cov_z = inv(I + Wpsi W^T)` (nc x nc LU
on the host-address path: 2 syncs), `mm(M, Wpsi^T)` (n x nc), `mm(., cov_z)` (n x nc), `.out()` (download). Two passes over
the n x d data (sub, gemm) plus the n x nc intermediate.

## Mechanisms of the candidates (docs/apple-fast/ab/fa.md has the per-define paragraphs)

- FA's EM depends on the data only through the sample covariance: G = Xc^T Xc (d x d) computed once in a tiled pass over X
  (centering at load, no Xc buffer, no sq buffer, no download, no host copy) and var = diag(G) / n; each iteration then
  needs only the d x d scaled Gram's eigendecomposition (the route main already takes when n < d).
- The whole EM loop as one binding call with every operand resident: per iteration one scale launch, the eigh, one finish
  launch (ordering, sign, W, psi update, the log terms) and ONE small readback (2 d + 3 floats) for the f64 log-likelihood
  and the convergence test, which keep main's arithmetic exactly.
- The eigh as ONE launch of one threadgroup for d <= 256 (the round-robin cells of x_decomp/rr.mojo unchanged, every round
  behind a device barrier) in place of 2 (d - 1) launches and a sync per sweep.
- transform as one launch over rows with (d x nc) P = Wpsi^T cov_z and the mean in threadgroup memory; one download.
- One arena buffer for the loop's scratch; W and psi read back in one copy.
- The log-likelihood test on the device (double-float float32 sums; Metal has no float64), the host reading a flag every
  4 iterations; the ll values reconstructed at the end.
