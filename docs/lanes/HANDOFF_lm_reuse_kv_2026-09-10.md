# Generalized LM prefill workspace reuse — September 10, 2026

The IDENTICAL trainer now owns one reusable LlamaKVCache workspace. Each
full-sequence layer resets its used length to zero before forward. The append
kernel therefore reads only fresh K/V, and backward retains its original
per-layer stage buffers. Arithmetic, synchronization and attention dispatch
are unchanged. Trees untouched.

Reference: upstream/transformers/src/transformers/models/llama/modeling_llama.py:
402-412 visits decoder layers. Workspace reuse is our storage policy, not an
upstream arithmetic change. Our kv_append_kernel's `j < s_old` branch establishes
that stale workspace contents cannot be read with `s_old == 0`.

Previously each forward allocated and zeroed two buffers per layer. Now two
buffers are allocated once per native trainer. At B8/L2048/KV12/HD64, this
workspace occupies 96 MiB. Twelve layers previously cleared 1.125 GiB across
these allocations per forward; the first forward now requires 96 MiB of
initialization, avoiding 1.03125 GiB. These are source-derived byte counts,
not measured bandwidth or time. The workspace now remains resident through
backward, unlike the old temporary cache. The capacity tool explicitly lists
this workspace among its omissions; its subtotal is not peak memory.

Python still reconstructs the native trainer on each call. This change does
not establish persistent Python/native model or optimizer state.

## Validation

Metal IDENTICAL binding built successfully. All 76 host tests passed. Explicit
fused and eager attention requests each passed four shapes, eight training
steps, eight evaluation invariance checks and the existing wrong-derivative
and negated-gradient oracle controls. Prior fused versus new fused, and new
fused versus new eager, each compare 88 native arrays; all match bit for bit.
These are small correctness fixtures, not performance qualification. No
allocation-specific sabotage or instrumented allocation-count check was run.

Evidence: bench/results/lm_reuse_kv_2026-09-10 contains captures, numerical
verdicts, regression verdicts, source/binary hashes and SHA256SUMS.

## Remaining performance work

Qualified ratios remain unchanged: GEMM 4.07–4.52x FP32 cuBLAS; attention
3.91x forward / 4.32x forward+backward FP32 SDPA; NVIDIA kNN 2.61–2.88x
cuML. Transformer large-shape numerical admission remains unresolved; Mamba3
has no updated qualified ratio. No new opponent measurements or rental.

Next: persistent context/model/Adam state across Python calls with explicit
export/checkpoint and failure semantics; optional full-gradient captures;
bounded vocabulary-loss storage; representative complete-step measurements.
GEMM occupancy and attention remain the largest qualified neural-network
component gaps. kNN selection remains useful separate work, not decoder
training acceleration. There is no established theoretical floor forcing
these current FP32 ratios above 2x or 3x, and no guarantee of reaching them.

Keep experiments at production dimensions with a five-minute execution
deadline each, interleave candidates and controls, retain inconclusive results,
and leave timing gates unchanged without adequate large-shape evidence.
Reuse cached opponents and add any newly measured opponent tuple to the table.
