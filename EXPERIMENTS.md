# Neural speed experiments on this branch: what to test and how

Branch `lane/neural-net-experiment`. Everything here is a RUNTIME TOGGLE,
so ONE build of the three bindings serves every A/B below. Nothing on this
branch has been compiled or measured by its author (a container with no
Mojo toolchain and no GPU); the L40S numbers quoted are the owner's.

The rule for every toggle: it is kept only if the output digest is the
same as the baseline's AND it is faster. `tools/neural_experiments.py`
prints both. A toggle whose digest MOVED is a bug report, not a result.

## Build once

```
bash bindings/build_transformer.sh && bash bindings/build_byte_lm.sh && bash bindings/build_mamba.sh
python tools/transformer_session_check.py            # every group must pass
python tools/transformer_fresh_prefill_check.py      # NVIDIA builds
```

## Run the sweep

```
python tools/neural_experiments.py                   # default set, six lanes, 6 calls each
python tools/neural_experiments.py --set nvidia      # adds the old per-call fresh entry as an arm
python tools/neural_experiments.py --set amd         # adds "legacy everything" (main's paths)
python tools/neural_experiments.py --lane lm-train-step --only speculative_attn,no_layer_sync --calls 12
python tools/neural_experiments.py --lane transformer-forward --gemm-arms shipped,tuned128,half,quarter,kpack,kfoldv
python tools/neural_experiments.py --json results.json
```

Each experiment is a fresh subprocess running `tools/neural_stage_timing.py`
with the toggle's environment. The table gives the median after the first
call, the ratio to baseline, and `same` / `MOVED` for the output digest
(training lanes compare their loss series instead).

To see WHERE a configuration spends its time, run the timing tool directly;
it prints the bindings' stage ticks per call:

```
MOJOLEARN_ATTN_SPECULATIVE=1 python tools/neural_stage_timing.py --lane transformer-forward --calls 10
```

`surface.*` ticks are the binding's own phases (weights, inputs, forward,
backward, downloads); `block.*` and `attn.*` are inside the block; `step.*`
is the byte-LM step; `M3_PHASE` is Mamba-3. For launch and sync COUNTS per
step, rebuild the byte-LM binding once with `-D MOJOLEARN_STEP_PHASE_TIMERS=1`.

## The toggles

| env | default | what it does | where it should help | bits |
|---|---|---|---|---|
| `MOJOLEARN_TRANSFORMER_RETAIN_WEIGHTS=0` | retain (1) | per-call weight upload instead of the exact byte compare + retained device copy | NVIDIA, where the compare cost 0.3 ms per block and the upload was already cheap | same by construction |
| `MOJOLEARN_MAMBA3_RETAIN_WEIGHTS=0` | retain (1) | same for Mamba-3 | NVIDIA | same |
| `MOJOLEARN_TRANSFORMER_SESSION_FRESH=0` | session (1) | stateless forwards take the old per-call `transformer_forward_fresh` (NVIDIA builds) or the state-carrying session path | A/B of the whole session-fresh route | same |
| `MOJOLEARN_TRANSFORMER_LEGACY_SETUP=1`, `MOJOLEARN_MAMBA3_LEGACY_SETUP=1` | off | main's paths: no session at all | the "before" column on any vendor | same |
| `MOJOLEARN_TRANSFORMER_STAGE_RESET=0` | reset (1) | skip the 30 zero-fills a reused workspace gets before each call | every vendor; 30 launches per block call | **must verify**: same only if every cell a call reads it also wrote |
| `MOJOLEARN_ATTN_SPECULATIVE=1` | off | the fused attention's regime scan runs behind the kernels; one host round trip per layer instead of two | LM training and forward: 8 fewer round trips per pass; every vendor, most on AMD and Apple | same by construction (a refused regime discards and reruns eager) |
| `MOJOLEARN_SWIGLU_FUSED=1` | off | SiLU and the gate product in one launch, forward-only entries (block forward, LM logits); one 8 MiB intermediate fewer | transformer, Samba and LM forwards | same by construction; not taken where a backward follows, nor with the trace on |
| `MOJOLEARN_BYTE_LM_LAYER_SYNC=0` | sync (1) | skip the sixteen per-layer host waits in the byte-LM step | LM training, every vendor | same if enqueued frees are stream-ordered (DEVIATION 2520); if Metal misbehaves, leave on there |
| `MOJOLEARN_TRANSFORMER_RETAIN_MB=N` | 512 | retained workspace budget per session (was 64 on main) | AMD; the board shape needs ~87 | same |
| `MOJOLEARN_GEMM_ARM=<name>` | shipped | the identical GEMM's plan (`shipped`, `lfold`, `half`, `half_ks16`, `quarter`, `head`, `half_head`, `ksplit`, `ksplit_leaf`, `tuned128`, `kpack`, `kpack_wide`, `kfoldv`, `kfoldv_leaf`, ... see `gemm_step_arm_parse`) | shape-dependent; sweep on each vendor | same by construction (every arm keeps the fixed fold tree); the gate is still the check |

## What each vendor should try first

NVIDIA (L40S numbers from the owner: transformer 4.4 ms vs 1.8 compiled
torch; LM train 44 vs 19; Samba train 137 vs 40):

1. `--set nvidia`: expect `no_retain_weights` and `legacy_fresh_entry` to
   recover the 0.3 ms the compare costs on the block cells.
2. `speculative_attn` and `no_layer_sync` on `lm-train-step`: the two
   round-trip cuts. If neither moves the step, its cost is in the
   backward kernels and the next lane is a kernel lane, not a toggle.
3. `swiglu_fused` on the three forwards: one launch and 16 MiB of traffic
   per block fewer; small but free.
4. `--gemm-arms` on `transformer-forward`: the projections at M=2048,
   N=384 are one wave of 64x64 tiles on 142 SMs; a smaller tile may fill
   the machine.

AMD (MI325X, pending stock):

1. `--set amd` first: `legacy_everything` is main; the gap between it and
   `baseline` is what the sessions bought on HIP, where allocation and
   pageable copies are the expensive part.
2. Then the same order as NVIDIA. Expect `speculative_attn` and
   `no_layer_sync` to matter more here: a host round trip costs more on
   ROCm than on CUDA.

Apple: `no_layer_sync` is the risky one (see the bits column). Everything
else applies; transfers are free on unified memory so the retention
toggles matter less.

## Reading the result

- Same digest, faster: keep; make it the default in a follow-up commit.
- Same digest, slower or flat: drop; note the number in the progress file.
- MOVED: the toggle is wrong (or, for `swiglu_fused` under a trace, the
  card lost a stage, which is expected and why it is off with the trace
  on). Do not keep it; file the digest pair.
- FAILED: a build or runtime error in that configuration; the subprocess's
  stderr tail is printed above the table.

Record the tables in `docs/lanes/progress/neural-net-experiment.md` with
the vendor, wheel and date; that file is the lane's memory.

## The classical pass (same branch, 2026-09-30)

The worst GPU-versus-GPU cells of the 0.8.25 board were not tuning problems;
each was a serial program on one GPU thread or a per-query host loop. Four
fixes, all "same cells, same order, same bits" by construction, each with an
env that restores the old path for the A/B and the digest check:

| cell (0.8.25) | was | cause | fix | A/B env |
|---|---:|---|---|---|
| AMD lu-factor 8192x8192 | 597 s (torch 0.085) | pivot search: one thread walking a strided column per step | `lu_pivot_block_kernel`: one block, compares only, ties to the lowest row | `MOJOLEARN_XD_LU_PIVOT_SERIAL=1` |
| AMD lu-solve 8192x64 | 616 s (torch 0.083) | one thread for n^2 x nrhs dependent FMAs | `lu_solve_cols_kernel`: one thread per right-hand side | `MOJOLEARN_XD_LU_SOLVE_SERIAL=1` |
| NVIDIA sgd-reg / sgd-clf / sgd-ocsvm istella | 630 / 615 / 124 s (sklearn 55 / 36 / 7) | sequential SGD on ONE GPU thread | the same program on the host (the identical tier's reference) | `MOJOLEARN_X_LINEAR_SGD_HOST=0` |
| NVIDIA lars istella | 15.4 s (cuML 0.064) | the Gram's 24,531 chains on one block | `xg_gram_kernel`: one thread per cell over a grid, same chain per cell | `MOJOLEARN_X_LINEAR_LARS_GRID_GRAM=0` |
| NVIDIA / AMD classical2/ivf istella | 526 / 265 s (Apple 4.9) | the batched scan was Apple-only; a host round trip per query elsewhere | the scan on every vendor, launched and merged on WARP_SIZE | `-D MOJOLEARN_IVF_IDENTICAL_SCAN_OFF` (compile time) |

Build the three bindings (`bindings/build_x_decomp.sh`, `build_x_linear.sh`,
`build_ivf.sh` or the repo's equivalent) and run each lane's identity check
before the board:

```
python tools/bench_board.py --lanes lu-factor,lu-solve,sgd-reg,sgd-clf,lars,ivf ...   # the board's usual form
```

Expected: lu-factor and lu-solve in seconds, not minutes (the trailing
update was already parallel; only the pivot and the solve were serial);
sgd-* near sklearn's time (the same sequential algorithm on a comparable
CPU thread; cuML's 5 s is a different, mini-batch algorithm); lars under a
second; ivf on NVIDIA and AMD near Apple's 5 s. A digest that differs from
the old path on any lane is a bug in that fix, not a speed result.

What this pass does NOT fix: the other one-thread programs in x_linear
(`team_fit` lists the team ones; everything else runs on thread 0 alone),
which the same `_fit_on_host` route can take once measured; and the
neural training step's 162 synchronizes, whose largest sites are the
per-layer waits (16), the attention regime reads (16 + 16 backward) and
the norm gradient GEMMs (16), the first two of which the toggles above
already address.

## The priority-list pass (same branch, 2026-09-30, unmeasured)

Ten more commits after the classical pass, one per item of the priority
list, none measured here. Each keeps the identical tier's bits by
construction (the argument is in each commit message) or is an explicit,
off-by-default, bits-moving experiment marked as such. Every one has an
env that restores the old route, so the A/B is one env per row. The
digest line of `tools/neural_experiments.py` (or the identity gates for
the classical lanes) is the arbiter: a MOVED digest on a row marked "same
bits" is a bug in this branch, not a finding.

| item | commit | what changed | restore the old route with |
|---|---|---|---|
| Samba backward sessions | `3f5dac18a` | Mamba-3 backward on the prefill session: weights retained, the last forward's stages reused when x and the weights are byte for byte the forward's, gradients downloaded straight to the caller. `Mamba3Block.session_info()` reports reuse counts. | `MOJOLEARN_MAMBA3_LEGACY_SETUP=1` (per-call entry); `MOJOLEARN_MAMBA3_RETAIN_STAGES=0` (recompute every backward) |
| LM training step waits | `aa812f164` | the two RMSNorm weight-gradient GEMMs on the block's retained workspace: the last `identical_gemm` on the step path, 2 waits per norm gone (`syncs.gemm.norm_dW`, 16 of 162 on the L40S) | `MOJOLEARN_TRANSFORMER_NORM_DW_OWN_WS=1` |
| gemm-int8 61x | `9b1326101` | the unit plan with a 32 x 32 tile per warp, every fragment feeding two unit tiles, the next step's fragments loaded ahead (NVIDIA IMMA and AMD MFMA). Order-free Int32 sums: cannot move a bit. | build define `-D MOJOLEARN_INT8_MMA_REFERENCE=1` |
| eigh 24,810x / timeout | `262819f9e` | the host-Jacobi and round-robin route envs honoured on every vendor and tier (defaults unchanged: never). Round-robin is NOT the pinned order's bits. | `MOJOLEARN_XD_HOST_EIGH_MAX=0`, `MOJOLEARN_XD_PJ_EIGH_MIN=0`, `MOJOLEARN_XD_PJ_SVD_MIN=0` (the defaults) |
| svd / qr serial | `ddddaaefc` | `numpy.linalg.svd`'s Q sign flips as one kit multiply per column instead of a Python loop over every value (1M rows x 220 columns); `_triu` by row slices | `MOJOLEARN_LINALG_LEGACY_SIGN=1` |
| cholesky serial (x_decomp) | `c94d5bf70` | the kit's Cholesky as a column driver (diagonal chain on one thread, the column below one thread per row) instead of one thread for all of n^3 / 6 | `MOJOLEARN_XD_CHOL_SERIAL=99999` |
| lr-warmup-cosine 2,612x | `5ebf9a802` | the value decided in binary64 with a rigorous error bound (repository Taylor cosine, no libm); the exact rational route only on a rounding boundary. Checked bit-equal on 12,927 values here. | `MOJOLEARN_LR_EXACT_ONLY=1` |
| adafactor 1,852x | `494aa6142` | the two whole-tensor norms folded on the host (the same `sumsq_fold` chain) instead of one GPU thread each | `MOJOLEARN_SEQ_HOST_FOLD=0` |
| clip-grad-norm 43x | `8fed0959e` | the J tensors copied to the device from their own memory and back (`clip_grad_norm_multi`), no packed host copy and unpack | `MOJOLEARN_CLIP_PACKED=1` |
| perceptron / pa-clf / pa-reg / sgd-ocsvm | (no new commit) | all four fit through `_sgd_fit` -> `ALGO_SGD`, so they already take the classical pass's host route (`556daa80b`) | `MOJOLEARN_X_LINEAR_SGD_HOST=0` |

Not done, and why:

* **louvain / pagerank**: a separate project, as the list says; nothing on
  this branch.
* **AMD neural measurement, Apple lm-forward**: measurement items, nothing
  to write; run the sweep in "Run the sweep" on those boxes.
* **eigh at n = 4096 under the pinned cyclic order**: no bit-preserving
  fast route exists on a GPU (one block, n(n-1)/2 serial rotations a
  sweep). The round-robin env is the honest alternative and it moves the
  bits; adopting it means pinning that order, a contract change.
* **qr 'reduced' on taxi (1M x 11, 4.0 s)**: not explained by reading; the
  per-column staged kernels account for tens of milliseconds. Worth a
  `MOJOLEARN_XD_*` timing pass on the box before more code.
* **the cholesky/ lane (the public `Cholesky` class, 557 ms at n = 8192
  on the MI325X, 17x torch)**: a blocked factorization already; not
  touched here. The x_decomp kit's one-thread Cholesky (above) is the
  serial one.

How to test this pass, per row: build the binding the row names (mamba,
transformer/byte_lm, linalg for gemm-int8, x_decomp, x_sequence for
adafactor, training for clip and the schedule), run the lane's board cell
with the row's env unset and then set, and compare the digests / the
identity gate. For the neural rows `tools/neural_experiments.py` already
prints the digest; for the classical rows the lane's own identity check
does.

## The direction pass (same branch, 2026-09-30 evening, unmeasured)

After the L40S ceiling and step measurements (`bench/results/gemm-ceiling-20260930`,
the neural priority run). Four commits: two diagnostic tools that make the
next run answer a question, two fixed15 changes. Plus the measuring lane's
compile fixes cherry-picked (the prefill session bindings I had dropped,
`mut` operands on the device-resident backward).

| question | tool / change | how to run |
|---|---|---|
| gemm-int8: where do the 2.4 s go (the kernel is milliseconds)? | `MOJOLEARN_LOWBIT_TIMING=1` prints the binding's phases; `tools/int8_profile.py` times the Python side around them | `python tools/int8_profile.py --calls 5` |
| Samba step 136 vs 40 ms: which backward stages? | `tools/mamba3_backward_timing.py` (parses `MOJOLEARN_MAMBA_TIMING=1`'s per-stage walls, sorted, with shares) | `python tools/mamba3_backward_timing.py --batch 8 --length 512 --d-model 768` |
| fixed15 mlp_up at 18 TFLOPS: is it the plan? | `MOJOLEARN_INT15_PLAN=<n>` forces the sums kernel's plan (0..10; 4, 5, 6 are the tall blocks) | `MOJOLEARN_INT15_PRICE_ONLY=mlp_up.t512 MOJOLEARN_INT15_PLAN=5 <price binary>` per plan |
| fixed15 conversions (67 -> 24 TFLOPS): the planes quantizer read every operand twice | one block per row, one read one write (`int15_planes_row_block_kernel`); same bits | `MOJOLEARN_INT15_ROW_BLOCK=0` restores the parallel schedule |

What the ceiling run says the fixed15 losses are (qkv.t512, L40S, ms):
tuned product 0.255; planes of X 0.046; planes of W 0.137; the
training operation measured 0.709 -- 0.27 ms more than the sum of its
parts, which is orchestration (waits and allocations between the
conversions and the product), not arithmetic. At mlp_up.t512 the product
alone is 3.31 ms for 3.5x the MACs of qkv (0.255): the plan, not the
conversions (1.5 ms), is the loss there.

## The strides pass (same branch, 2026-09-30 night, unmeasured)

From the direction diagnostics (`bench/results/direction-diag-20260930`).
Four commits, each with its restore-env or build define:

| finding | change | restore with |
|---|---|---|
| gemm-int8: 2.4 s of the 2.46 s cell was Python widening the int8 codes one object at a time | a signed one-byte buffer is the Array's int8 dtype: pre-made codes are a zero-copy view (16M codes: 1,227 ms -> 0.3 ms here) | none needed (a widening request still casts) |
| Samba: s16_s15 63% and s17_operands 24% of the Mamba-3 backward | `mamba3_s16_qk_shared_kernel` (a block per row, operands staged, the two per-j / per-i products formed once); `mamba3_s17_operands_shared_kernel` (a block per row instead of one thread per row) | build defines `MOJOLEARN_MAMBA3_S16_QK_NAIVE=1`, `MOJOLEARN_MAMBA3_S17_OPERANDS_NAIVE=1` |
| fixed15 mlp_up: plan 3 at 34 TFLOPS against the dispatched 18; backward rows prefer plan 0 | a plan table for low-bandwidth NVIDIA boxes, chosen from the device name | `MOJOLEARN_INT15_BOX=high` (the H100 choice), `MOJOLEARN_INT15_PLAN=<n>` |
| fixed15 bwd_dx conversions at 14 TFLOPS with a 45-TFLOPS multiply | the transposing quantizer reads coalesced (absmax thread map; planes through a 32 x 32 tile) | `MOJOLEARN_INT15_TRANSPOSED_TILE=0` |

Run, in this order: `python tools/mamba3_backward_timing.py --batch 2 --length 512
--d-model 384` (the two stages should fall from 23.7 / 9.0 ms to a few
ms; the digest gate says the bits), the Samba board cell, the gemm-int8
board cell (expect ~50 ms), the price harness at bwd_dx and mlp_up (the
transposed A/B and the table), then `mamba/checks` and `gemm/checks` gates.
