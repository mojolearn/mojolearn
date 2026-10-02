# lane/apple-fast-nb: CategoricalNB count table and LDA E-step + statistics under FAST on Apple

Written without a Mojo toolchain (cloud peer); the first M3 builds of `x_prep` (`bindings/build_x_prep.sh`) and
`x_decomp` (`bindings/build_x_decomp.sh`), FAST, are the compile check. Every change is compiled under FAST + Apple
only and defaults OFF; IDENTICAL compiles main's code unchanged. Bench driver: tools/bench_board_algos.py
(`AFC_FAMILY=algos`; lanes `categorical-nb`, `lda`).

| switch | kind | site | what it changes under FAST on Apple |
|---|---|---|---|
| `-D MOJOLEARN_NB_CAT_ATOMIC=1` | define, `NB_CAT_ATOMIC` comptime in `x_prep/fastnb.mojo` | the stage dispatcher of `x_prep/device.mojo` (`run_program_device_ptr`, before the generic `prep_kernel` launch) for ops 133 `cat_hpart` and 134 `cat_hfold` | main folds the category counts on one thread per (2048-row block, feature) into a private K*CMAX histogram (a read-modify-write per row), then sums the blocks: a few thousand threads at 1M rows. The intercept zeroes block 0 of the same histogram scratch (`enqueue_memset`), launches `cat_hist_atomic_kernel` with ONE THREAD PER (row, feature) cell doing an Int32 atomic add into the (feature, class, category) slot, and turns the fold stage into `cat_hist_convert_kernel` (block 0's integer words as float32). Exact integer counts, the same on every run. Only the unweighted form (W < 0 in the stage); `sample_weight` keeps the units. |
| `-D MOJOLEARN_LDA_FUSED_SS=1` | define, `LDA_FUSED_SS` comptime in `x_decomp/lda_fast.mojo` | binding export `x_decomp_dev_lda_estep_ss` (`bindings/_mojolearn_x_decomp.mojo`, guarded); `LatentDirichletAllocation._e_step` / `_fused_estep_ss` (`python/mojolearn/_expansion_decomp.py`, FAST + resident path + export present) | main's `_e_step` with statistics runs `lda_block_kernel` (one 128-thread block per document, ~20 KB threadgroup memory, the topic fold on 16 threads), then `mm(Et, exp_dir)` (n x v), `X / norm_phi` (n x v) and `mm(Et^T, R)` (inner dimension n). The fused kernel runs 4 documents per block on 32 lanes each (~12 KB), persistent blocks (2048) walking document groups ascending and accumulating their own k x v partial of Et^T (X / (Et exp_dir + eps)) in device memory (no atomics); `lda_ss_fold_kernel` sums the partials ascending and applies exp_dir. Per-document sequence as `lda_block_kernel` (Dt, Et same bits); the statistics take a new fixed fold order. Caps k <= 32, v <= 320 (taxi-zones: k = 16, v <= 265); past them the entry returns 0 and Python keeps main's chain. The final `_e_step(cal_sstats=False)` and the perplexity are untouched. |

Risky compile sites (the M3 build is the check): `x_prep/fastnb.mojo` `f.bitcast[Int32]()` on `FP` and
`Atomic.fetch_add(ptr.unsafe_offset(i), Int32(1))` (the spellings of `x_prep/dradix.mojo` and
`hierarchy/checks/nan_guard.mojo`); `x_prep/device.mojo` `df.create_sub_buffer[DType.float32](Int(hq[8]), words)`
inside the `enqueue_memset` (the file's own idiom); `x_decomp/lda_fast.mojo` imports of `_blocks`, `TPB`, `xd_ctx` from
`x_decomp.device` and `_id`, `_n`, `_ptr`, `pool_alloc`, `pool_free` from `x_decomp.resident` (as `kit_device.mojo`
imports them), and the `comptime if LDA_FUSED_SS:` around `m.def_function` in the binding.

Not changed: multinomial-nb / complement-nb on text (289 / 281 ms vs sklearn 266 / 265). The device side is three
coalesced passes over the dense 78k x 4096 count matrix (`colb_part`, `csb_part` x K) plus a tiny `matmul` at predict,
a few ms on the M3; the rest of the time is the 1.3 GB upload of X and the Python staging, which no GPU-only kernel
change moves. No request line for them.
