# Performance continuation, September 10

Ratios below compare our IDENTICAL arm with the opponent's recorded fast arm.
Reuse `bench/OPPONENT_REFERENCE.md` only for its exact hardware, driver,
library, shape, fixture and timing scope. Internal before/after improvements
are separate. Trees remain outside this lane.

| Area | Latest usable evidence | Remaining work |
|---|---|---|
| Apple kNN, measured 400k/4k/d32 k10/k15 | Actual scoped default saves 15.7–25.0% request time in both orders; all outputs match | Expand scope only with additional large evidence; late-window drift is retained |
| NVIDIA kNN, 400k index / 4k queries / d32 | 27.526 ms k10 and 31.861 ms k15; 2.69x / 2.95x cached cuML | Tune NVIDIA distance/selection costs; Apple metadata does not change this ratio |
| H100 dense GEMM, actual Llama t512 | Accepted staging 4.07–4.52x cached FP32 cuBLAS; 9.73–11.28 useful TFLOP/s | Current offline cubin is limited to one block/SM; scalar shared-read trial rejected |
| L40S attention, original HD64 window fixture | Corrected-seam forward 140.3 ms / fwd+bwd 493.7 ms; 3.91x / 4.32x cached SDPA | Backward and operand reuse remain targets; reference is the same model/driver tuple on an earlier physical pod |
| Mamba3, original large grid | Latest two diagnostic processes: narrow visit medians 52.9–58.4 ms, wide 92.0–94.9 ms | Not ordinary prices; historical slow regime did not recur, cause unresolved |
| Transformer, original large grid | Numerical admission remains failed; no qualified opponent ratio | Full H100 attribution implicates attention rounding and amplified QKV/RoPE errors; local o_proj/glue are not the main admission problem |
| UMAP, 100k / 1M rows | Old 100k 5.2x compares L40S with H100; 1M own timing is one cold-inclusive sample | Measure current source at large size before quoting a current ratio; kNN and host graph/init remain candidates |

The old dense 5.2–5.7x and kNN roughly 4x headlines predate later staging
and query-batch work. The old UMAP headline is not a qualified same-GPU
result. Historical captures remain intact and are labeled explicitly.

## Why further kNN work matters

At the recorded NVIDIA target, a hypothetical 25% own-request reduction
would save about 6.9–8.0 ms per call. That is an arithmetic illustration,
not a forecast or a measured NVIDIA gain. Large self-kNN also dominated
the historical 1M-row UMAP capture (~66% of total), so improving that stage
could matter downstream. A 25% improvement of a 66% stage would reduce
that old total by ~16.5%, if other stages and behavior stayed fixed. A
4000-query kNN result does not establish a million-query UMAP result.

## Work completed in this continuation

- Full large Apple kNN metadata trial and scoped default validation are
  recorded by the kNN lane. Promotion requires ordinary request timing,
  both run orders, complete output equality, and the exact arithmetic oracle.
- `gemm_resources_2026-09-10` captures ptxas, SASS, static driver occupancy,
  and the rejected large scalar-load experiment. No new GEMM default.
- `mamba_regime_large_2026-09-10` retains 192 hash-admitted calls, the
  exact newly compiled binary, and host counters; no timing/default claim.
- `transformer_stage_admission_2026-09-10` retains both original full shapes
  and fine-grained FP32/FP64 stage attribution without changing tolerance.

No external opponent timing was rerun. Any future missing opponent tuple
must be measured once and appended with provenance to the opponent table.
Small cases remain correctness/diagnostic controls, not promotion evidence.


## Next NVIDIA kNN experiment

The latest retained phase split is from the earlier 256-query configuration,
not the current 512-query default: distance ~59.4%, selection ~37.5%, merge
~2.2%. Reprofile the current default before reusing those shares. A distinct
next candidate is explicit aligned 128-bit loads of contiguous index values
inside the existing 8x4 distance tile, with scalar ragged fallback, if SASS
shows the compiler currently emits scalar loads. Preserve tile indexing and
ascending feature FMA order. This differs from the already rejected column
remapping. Validate on 400k/ 4000/d32 at both k10 and k15 before any default.
