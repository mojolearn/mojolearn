# What `--fp-mode contract=off` does to the IDENTICAL columns (2026-09-26)

Measurement only. No default changed, nothing merged.

## Setup

- Flag: `mojo build --fp-mode contract=off` (Mojo 1.0.0 `mojo build --help`: the only
  feature is `contract`, `fast` is the default and fuses `a*b + c` into an FMA across
  statements, `off` disables contraction).
- Hook: `MOJOLEARN_MOJO_BUILD_FLAGS`, empty by default, placed after `--emit shared-lib`
  on all 24 `mojo build` lines in `bindings/build_*.sh` (host shims reach it through
  `build_host_family.sh`). `tools/bincache.py` keys it as a `MOJOLEARN_*` variable.
- Source: the 0.8.19 commit 69a519c1522d plus only the hook (commit 1de164534 on
  `lane/fp-contract-off-exp`), so every difference below is the flag.
- Cells: the 0.8.19 release selection, 209 lanes x base,denormal,odd = 627 cells, one fit,
  batch and rlpair parts left out (as in the release passes).
- References: 0.8.19 Metal column (`release-check/69a519c1522d/metal/column.json`) and the
  0.8.19 NVIDIA RTX 4090 column (`release/0.8.19/69a519c1522d/smoke-linux/column-cuda.json`);
  those two agree 627/627, infer/model 984/984.
- Evidence (outside the repo): `~/mojolearn-evidence/fp-contract-off/` (build logs,
  columns, every diff below, the NVIDIA leg).

## Probe: host code contracts too

`@no_inline def muladd(a, b, c): var t = a*b; return t + c` (f32) compiled for
`--target-cpu apple-m1` emits one `fmadd`; with `contract=off`, none. For
`x86-64-v3` (the Linux wheel's CPU target): one `vfmadd`, and none with `off`. So the host
bindings are subject to the same contraction as the device code.

## Results (contract=off against 0.8.19, contract=fast)

| column (contract=off) | train cells MOVED | infer/model MOVED | total cells touched |
|---|---|---|---|
| Apple M4 Metal (local) | 12 of 627 | 10 of 984 | 16 of 627 |
| NVIDIA RTX 4090 sm_89 (RunPod) | 12 of 627 | 10 of 984 | 16 of 627 |
| CPU host bindings (this Mac) | 12 of 627 | 10 of 984 | 16 of 627 |

The same 16 cells, six lanes, on every backend:

| lane | fixtures | what moved |
|---|---|---|
| gbdt-parametric-losses | base, denormal, odd | train parts Poisson, Tweedie (other 9 losses agree) |
| gbdt-stochastic-arms | base, denormal, odd | train parts sym-Poisson, sym-Tweedie |
| gpc | base, denormal, odd | train part `proba` and infer (L, pi, W_sr, lml, n_iter, predict agree) |
| gpc-multiclass | base, denormal, odd | train part `proba` and infer |
| umap | base, denormal, odd | infer |
| gbdt-ordered-rmse | denormal | model (saved bytes) |

Nothing in gemm, linear models, rf/et, svm, kmeans/knn, the neural lanes (mlp, mamba,
transformer, samba, byte LM) or the low-bit lanes moved: their contracted and uncontracted
builds give the same bits at these shapes.

## Cross-vendor agreement with contraction off

- Metal-off vs NVIDIA-off: IDENTICAL 627/627, infer/model 984/984.
- CPU-off vs Metal-off and vs NVIDIA-off: **6 cells DIVERGENT**, the Poisson loss only:
  `gbdt-parametric-losses` part Poisson and `gbdt-stochastic-arms` part sym-Poisson, on all
  three fixtures. Tweedie, gpc, umap and gbdt-ordered-rmse agree across all three.
  (6 `model` cells of gbdt-categorical-ctr-tables / gbdt-tensor-ctr-tables read ONE-COLUMN
  by design: a CPU column loads the GPU model and writes none.)

So with contraction on, all four columns agree (0.8.19). With contraction off, the two GPUs
still agree with each other, but the CPU host path parts from them on Poisson: something in
that path gave the same bits on CPU and GPU only because both contracted it.

## Speed

Not resolvable at release-check shapes. Metal cell time summed 326 s (0.8.19) vs 373 s (off),
with per-lane ratios scattered 0.73x to 1.24x under shared-Mac load; the NVIDIA off column
ran from a source tree with more parts than the recorded wheel column, so its times are not
comparable. A speed answer needs a 1M+ row A/B on one box.

## Cost

One RunPod RTX 4090, pod sy6wt8z0mkrqi3, about 43 minutes at $0.74/h, about $0.53;
terminated and verified gone (HTTP 404). Local: 55 bindings built at -j 1 in 27 min, the
Metal column 406 s, the CPU column 1482 s (one shard).

## Recommendation

Do not switch to `contract=off`. It moves 16 of 627 cells and, unlike the default, breaks
CPU vs GPU agreement on the Poisson loss (6 cells). The default's agreement is real but
rests on every backend contracting the same expressions; the cells that depend on it are
now named (the Poisson/Tweedie gradients, GPC `proba`, UMAP transform, one ordered-RMSE
model). The durable fix is local, not global: write those expressions with explicit
`fma(...)` (or explicit separate rounding) in the source, so their result stops depending on
a compiler choice, then rerun this study to confirm the move count drops to zero.
