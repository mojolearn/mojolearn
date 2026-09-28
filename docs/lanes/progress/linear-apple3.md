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

## Jobs

| steward id | Mac | what |
|---|---|---|

## Results

(none yet)
