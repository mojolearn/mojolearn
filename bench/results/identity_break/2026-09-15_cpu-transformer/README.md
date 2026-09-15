# CPU training for the Transformer block lanes, the M4 host column (2026-09-15)

Branch `lane/cpu-training-transformer`. The `transformer` and
`transformer-window` lanes of `tools/identity_break.py` (TransformerBlock at
d_model 32, two heads, one kv head, intermediate 64; full causal and window 8),
run on the CPU through the new transformer host family and diffed with
`tools/identity_break.py --diff --require-columns 4` against the 166-lane
record's three GPU columns (`bench/results/identity_break/2026-09-14_166-lanes/`,
Apple M4, NVIDIA H100 sm_90a, AMD MI325X gfx942), exactly as the CPU identity
gate does.

Each train cell hashes four parts: the stateless forward, the carried-state
prefill (max_tokens 32), one decode step after it, and the zero-state prefill
backward (the input gradient and the nine weight gradients). The infer cell is
the forward on a held-out slab; the batch cell is the forward of each sequence
alone against eight, and every prefix length against 16. There is no model
cell (the block has no save).

| piece | where |
|---|---|
| binding | `bindings/_mojolearn_transformer_host.mojo` (the GPU binding's four entries, same contract) |
| host glue | `transformer/host/transformer_block_host.mojo` (buffers in and out, the rotary table sizes, the window, the KV cache layout conversion) |
| forward | `transformer/checks/transformer_oracle.mojo::transformer_block_oracle` |
| backward | `transformer/checks/transformer_backward_oracle.mojo::transformer_block_backward_oracle` |
| sabotage | `gemm/host/gemm_oracle.mojo::GEMM_ORACLE_HOST_SABOTAGE`, every GEMM leaf walked descending |

Where it ran: the Apple M4, one core, shared machine, bindings built from
commit 82ffcb388 (the column JSONs record it) into fresh output directories.

| file | what |
|---|---|
| `cpu-apple-m4.json` | the CPU column, 2 lanes, 9 fixtures, 2 repeats |
| `diff.four-columns.txt` | `summary: IDENTICAL=18`, `summary (infer/model): IDENTICAL=18, N/A=18`, `summary (batch): IDENTICAL=18`, `require-columns 4 ... : OK` |
| `cpu-apple-m4.sabotage.json` | the same lanes through a `-D MOJOLEARN_HOST_SABOTAGE=1` build |
| `diff.four-columns.sabotage.txt` | `summary: DIVERGENT=18`, `summary (infer/model): DIVERGENT=18, N/A=18`, `summary (batch): DIVERGENT=18`; every train cell differs in all four parts |
| `diff.four-columns.cache-stride-sabotage.txt` | a throwaway build (working tree before the commit, not a shipped arm) whose glue read the carried full causal cache at stride max_tokens instead of cached_tokens: `summary: DIVERGENT=9, IDENTICAL=9`, the nine `transformer` train cells DIVERGENT with only `step` differing, the window lane, infer and batch IDENTICAL, so the step part reaches the cache conversion |

The first run matched; no stage needed bisecting.

Owed: the seven-runner CPU identity gate on the branch (the bit claim on Linux
ARM64, x86-64 and hosted macOS is the gate's, not this directory's).
