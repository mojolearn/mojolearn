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
| `MOJOLEARN_TRANSFORMER_STAGE_RESET=1` | skip (0), default since 2026-09-30 | restore the 30 zero-fills a reused workspace got before each call | the A/B only: skipping them measured 0.89x on L40S transformer-forward, digests unchanged | same (verified by the sweep digests and the session checks) |
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
