# spectral-embedding, 2026-09-20 (lane/expose-spectral-embedding)

The first columns for the `spectral-embedding` lane: `mojolearn.SpectralEmbedding`
on the k-NN graph arm and on a precomputed affinity, and
`mojolearn.manifold.spectral_embedding` with the unnormalized Laplacian and the
trivial eigenvector kept. All nine fixtures, every cell fitted once
(`--repeats 1`).

| file | device | commit | cells |
|---|---|---|---|
| `cpu-x86.json` | RunPod CPU pod, AMD EPYC 4564P, host bindings `core` and `metrics` built from source | 078a0fb65 | 9 STABLE, 0 refused |
| `nvidia-nvidia-geforce-rtx-4090-sm_89.spectral-embedding-2026-09-20.json` | RunPod RTX 4090, `tools/gap_column_leg.sh` | 849d8e2b8 | 9 STABLE, 0 refused |
| `cpu-x86-sabotage.json` | the CPU pod, `-D MOJOLEARN_HOST_SABOTAGE=1` host build | 078a0fb65 | all nine cells move |

The two commits differ in documentation and evidence files only.

`tools/identity_break.py --diff cpu-x86.json nvidia-...json` reads IDENTICAL x2 on
all nine train cells and exits 0. The infer, model and batch parts are N/A: the
estimator embeds the fitted rows only and has no `transform` or `save`.

`base` and `dupes` share a hash, as do `denormal` and `denormal_ftz`. The
shipped table shows the same two pairs for `spectral-precomputed`, which reads
the same leading rows and four columns; why those fixture pairs agree there was
not measured here.

Admitted into `python/mojolearn/verify_reference/table.json` by a scoped
`verify --all --emit-reference --lanes spectral-embedding --batch-checks`: nine
cells added, none changed, 0 conflicts. No Apple or AMD column has been taken.
