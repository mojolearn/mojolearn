# linear-apple3: Apple (Metal) FAST speed round 3, the linear family

Brief: ~/mojolearn-evidence/apple3_speed_brief.md (2026-09-28 ~20:10Z).
Branch `lane/linear-apple3` off lane/apple3-merged 6856b5f8f. Round 2:
docs/lanes/progress/linear-apple2.md. Jobs go through
`tools/apple_steward.py submit --kind speed --target m3ultra-b`; the before
and after arms are in the SAME job (tools/linear_apple3_ab.sh: an arm is a
set of build defines, so both arms are one commit's source).

## Item 1: the x_linear fits past one block

Finding (docs/lanes/progress/py-bugs.md item 5): `fit_device` launches
`fit_kernel` with grid_dim=1, so every x_linear fit is one program on ONE
block of LINEAR_TPB = 256 threads, and each row pass is one thread's chain
per output over all n rows.

Why a fit cannot simply get more blocks: the fit is ONE launch whose threads
meet at block barriers (`Team.sync`). A block cannot wait on another block
inside a launch (Metal has no grid barrier, and a spin on device memory can
hang when the blocks are not scheduled together). Past one block the fit
has to be several launches.

Change (x_linear/blocks.mojo, WIP, opt-in `-D MOJOLEARN_X_LINEAR_BLOCKS=1`,
FAST on Apple only, n >= 8192 rows): the fit's control and its m x m algebra
run on the host; every pass over the rows is ONE launch of n / 1024 blocks
of 256 threads. Block k maps its 1024 rows (linear predictor, per-row
terms), then one thread per output folds the block's rows; the host sums
each output's block partials in float64. Fits converted: Poisson / Gamma /
Tweedie (Newton-Cholesky), Huber and LogisticRegressionCV (L-BFGS),
Quantile (ADMM). The algorithms, constants and stopping rules are the
one-block fits'; only the grouping of each row sum differs, so FAST words
change and a paired quality check is owed. The partition depends on n
alone, so every Apple GPU gives the same FAST words.

IDENTICAL: untouched by construction (the hook in x_linear/device.mojo is
comptime-gated on FAST + Apple + the define). Can a multi-block schedule be
bitwise identical? Partly. The per-row maps (linear predictor, loss terms,
residuals) are row-independent and could run on any number of blocks with
the same bits, and a fit with more than 256 outputs (d >= 22 for a Hessian)
could give every output its own thread. But every sum over the rows is, by
the contract, ONE thread's ascending chain over all n rows, and that chain
is where the time goes; more blocks cannot shorten it without regrouping
the sum, which moves bits. Not attempted in this round.

## Changes (every one WIP and opt-in until its A/B and quality check are on record)

| define | what | mode | files | shared code |
|---|---|---|---|---|
| `-D MOJOLEARN_X_LINEAR_BLOCKS=1` | Poisson / Gamma / Tweedie, Huber, LogisticRegressionCV, Quantile: row passes on n / 1024 blocks, control on the host; isotonic predict over the whole grid | FAST, Apple | x_linear/blocks.mojo (new), x_linear/device.mojo (hook) | no |
| `-D MOJOLEARN_X_LINEAR_BLOCKS_GRAM=1` | LassoCV / ElasticNetCV, RidgeClassifier / RidgeCV, BayesianRidge, ARD, Lars / LassoLars: means, centered Gram, X'y, residual sums, path errors, leave-one-out rows in blocks | FAST, Apple | x_linear/blocks_gram.mojo (new), x_linear/device.mojo (hook) | no |
| `-D MOJOLEARN_X_LINEAR_FAST_FMA=1` | every x_linear multiply-add in code compiled for an Apple GPU is one fused instruction (FAST used a product and a sum) | FAST, Apple | x_linear/ops.mojo (`xmad`), tops.mojo, sgd.mojo | no (x_linear only) |
| `-D MOJOLEARN_SGD_FAST_TREE=1` | SGD family: the row's folds (row dot, penalty norms, PA's norm) as a `shuffle_xor` warp reduction, 5 steps instead of a chain of 32 | FAST, Apple | x_linear/sgd.mojo | no |
| `-D MOJOLEARN_CD_GRAM_BLOCKS=1` | core Lasso / ElasticNet FAST Gram: upper triangle from 1024-row blocks of 256 threads read straight from x; no (p + 1) x n copy, no 1024-thread block | FAST, Apple | solver/impl/cd.mojo | no |

Not converted (still one block): the SGD family's single problem (the pass
is serial in the rows by definition; only its folds change above) and the
isotonic fit (pool-adjacent-violators, one thread).

Host work in the block fits: the m x m algebra, and one-time O(n) float32
folds over y and the weights (the weight total, the target mean, Quantile's
spread and norm, the folds' row counts). Every pass over X is on the GPU.

## Jobs

| steward id | Mac | what |
|---|---|---|
| 1790627521091-speed-linear-apple3-f4f12a6a3a | m3ultra-b | job 1: the blocks arm builds and runs; first A/B at 100k (no quality run yet) |

## Results

### Job 1: first A/B of the block fits (m3ultra-b, Apple M3 Ultra, steward 1790627521091), FAST, 100k rows

x_linear binding built with and without `-D MOJOLEARN_X_LINEAR_BLOCKS=1` in
the same job at f4f12a6a3; fit s, one run. FAST words change (digests
differ by design); NO quality run in this job, so nothing is flipped on it.

| case | before (one block) | after (blocks) |
|---|---|---|
| poisson | 2.017 | 0.032 |
| gamma | 2.038 | 0.006 |
| tweedie | 0.083 | 0.006 |
| huber | 1.828 | 0.036 |
| quantile | 27.961 | 1.365 |
| logistic-cv | 7.619 | 0.185 |

Quantile takes the same 1.34 s at 20k rows as at 100k: its 5000 ADMM
iterations are bound by the read back and synchronize of each pass (about
0.27 ms), not by the rows. Gamma and Tweedie at 6 ms need the quality run
before they mean anything (an early stop would look the same).
