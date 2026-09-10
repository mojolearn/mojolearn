# Generalized LM lazy attention allocation — September 10, 2026

ByteTrainer now passes `lean=True` to every forward and backward stage
constructor. This wires the allocation mode already implemented by fused
attention. All eight quadratic stage buffers start at one element per layer;
eager execution and diagnostic fallback retain their existing grow-on-demand
behavior. No arithmetic, kernel dispatch, tolerance, or timing gate changed.
The upstream decoder layer loop remains the reference control flow:
`upstream/transformers/src/transformers/models/llama/modeling_llama.py:402-412`.
The lazy stage allocation is our existing identity instrumentation mechanism.

## Evidence

`bench/results/lm_lean_attention_2026-09-10/` retains native captures, FP64
oracle verdicts, per-array regression verdicts, source/binary hashes and six
host allocation reports. Metal binding built successfully. All 76 host tests
passed. Explicit `MOJOLEARN_TRANSFORMER_ATTN_PATH=fused` and `eager` each passed
four configurations, eight training steps and eight evaluation invariance
checks, including the existing wrong-derivative/negated-gradient controls.
Each comparison covers 88 arrays: prior materialized-allocation trainer versus
new fused execution, and new fused versus new eager execution. All match bit
for bit. These small fixtures establish correctness, not a performance gain.
The environment selector requests a path; corner-case fallback remains legal.

For the 162,147,840-parameter configuration (12 layers, DM768, H=KV=12,
HD64, FF2048, V50257), the allocation inventory is now:

| Shape | Fused allocation subtotal | Materialized fallback subtotal |
|---|---:|---:|
| B8, L2048 | 17.75 GiB | 161.75 GiB |
| B1, L2048 | 4.33 GiB | 22.33 GiB |
| B1, L1024 | 3.37 GiB | 7.87 GiB |

Only the eight attention arrays, five vocabulary arrays and flat
parameter/gradient/Adam state are counted. These are host calculations, not
measured peaks. Linear activations, workspaces, duplicate state, host captures
and runtime overhead remain excluded. Fallback may still exceed device memory.
The default capacity report now describes lean allocation; use
`--materialized-attention` to reproduce the eager scenario. Its schema is v2.
Historical v1 reports are retained with their original source hashes.

## Performance targets and limits

Qualified opponent ratios are unchanged: GEMM 4.07–4.52x FP32 cuBLAS,
attention 3.91x forward / 4.32x forward+backward FP32 SDPA, NVIDIA kNN
2.61–2.88x cuML. Transformer admission remains unqualified; no fresh Mamba3
ratio. References remain PERFORMANCE_STATUS_2026-09-10.md and
bench/OPPONENT_REFERENCE.md. No new opponent measurements or rental.

Target: below 2x, with below 3x an intermediate target on qualified production
workloads. No measured theoretical floor has been established above either
threshold. Bitwise identity restricts reduction order and arithmetic choices;
it does not imply a fixed slowdown multiplier. A minimum-time calculation
requires the actual operation count and unavoidable memory traffic:
`time >= max(required FLOPs / allowed peak FLOPs/s, required bytes / bandwidth)`.
Dependencies and launches may impose additional constraints. Dividing that
bound by the shape-matched opponent time gives a bound on the ratio, not a
prediction of attainable performance. Component time fractions are needed to
price whole-step improvement (Amdahl's law).

For the existing GEMM rows, reaching 3x requires approximately 1.36–1.51x
our throughput; reaching 2x requires 2.04–2.26x. This is below the recorded
67-TFLOP/s H100 FP32 non-tensor peak, so that peak does not explain away the
current gap. Register occupancy and ordered-fold scheduling remain open work.
Comparisons to tensor-core mixed precision require a separate denominator and
contract; the current FP32 opponent table does not measure that gap.

## Bounded next measurements and work

User constraint: large refers to production dimensions, not hours of running.
Use a few warmups and interleaved candidate/control samples within a five-minute
execution deadline per experiment; do not launch full token-budget training.
If the deadline cannot provide sufficient samples, retain an inconclusive
result and leave timing gates unchanged. Never substitute small-shape wins.
Report setup and complete request/step time separately, plus peak memory,
path, shapes, correctness and drift. Reuse qualified opponent cache entries;
add every newly measured opponent tuple to the opponent table.

Next training work: persistent Python/native context, model and Adam state;
bounded vocabulary-loss storage; then a short representative complete-step
pilot with a deadline. Full-gradient host capture should become an explicit
option while preserving checkpoint/export and failure semantics. Large
Transformer numerical admission remains required for a qualified ratio.
GEMM occupancy and attention cost remain the principal measured neural-network
component gaps. kNN selection is the next request-level target, but kNN tuning
does not accelerate this decoder trainer. Trees untouched.
