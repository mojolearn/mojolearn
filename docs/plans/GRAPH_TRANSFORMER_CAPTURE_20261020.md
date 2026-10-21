# Faithful captured Transformer block

Internal source handoff. Uncompiled and unqualified at this freeze; no public
support/performance claim, no default change. Lane compilation is prohibited.

The saved complete `transformer-forward` recipe is B1, L2048, D384, six Q/K/V
heads, head width 64 and MLP width 1024. Its immutable 10,229,370-byte input
artifact has SHA256
`7e23ff472e3a21cbfb68ae06dd6c3f271cae9585962bce2da9d3ad688472ab2f`.
The existing specialized block is opaque to OpsProgram optimization.

Four missing native arithmetic boundaries are exposed to captured replay:

* RMSNorm preserves the block's fused sum-of-squares order (or the selected
  shared norm profile), pinned reciprocal square root and separately rounded
  input/weight products. The existing generic RMSNormalization import
  expression computes a different sequence of rounded operations and is not
  silently substituted.
* Full-width Llama rotary embeddings preserve the native inverse-frequency
  constants, absolute positions, portable sine/cosine, split-half pairing and
  three-rounding unfused rotation. The same angle-domain refusal applies;
  no clamping or host-libm table is introduced.
* The private Transformer projection retains the existing neural GEMM
  leaf/balanced-tree profile through its shared `neural_cell` primitive,
  including selected native profile parameters. Generic sequence GEMM has
  its own serial/split-K reduction contract and is not a faithful substitute.
* The private Transformer softmax pins the specialized model's ascending-key
  denominator. Ordinary graph softmax retains its existing `ROW_COOP` fold32
  sum. Both use the existing exponential and single division; neither shared
  arithmetic nor the ordinary graph's numerical contract is changed.

The new opcodes have native first-derivative rules and preserve full-rank
metadata. Recorded higher derivatives remain refused by the existing
admission contract. Existing operations are unchanged. Native owned scratch
lives on the tape; no Python numerical work, callbacks or runtime loops are
introduced. New public RMS/rotary method names are excluded from public docs.

The testing-only whole-block graph in `_verify_graph.py` uses those
primitives plus existing generic layout, attention matmul, SiLU,
masking and residual operations. Its actual K layout has a permutation
followed by the transpose needed for QKᵀ. The existing permutation pass now
composes a mixed explicit permutation/swap only when the explicit
permutation proves the rank. Standalone swaps remain rank-polymorphic;
the exact-rank validator remains after composition, and returned/shared
intermediates are preserved by pruning. No identity node is inserted merely
to manufacture an optimization opportunity.

`tools/bench_graph_optimizer.py prepare-transformer --recipe RECEIPT --out DIR`
loads the complete retained saved recipe and refuses altered dimensions.
Before producing a usable workload manifest it requires exact specialized
versus captured output bits, exact optimized/unoptimized bits, native runtime
audits, a separate float64 complete-block oracle, actual named pass hits and
checkpoint replay. It retains full failed outputs and exception evidence.
The artifact is a complete declared synthetic block workload; it is not a
full language model or a full-corpus training run. Full-block backward is
not claimed by capture preparation.

The installed graph verifier adds `transformer_program_composition` with 13
expected output digests. It checks exact specialized/captured/optimized and
checkpoint forward outputs plus independent directional finite differences
for the input and all nine parameter families. The focused test file adds
21 cases covering RMSNorm on a noncontiguous view, both gradients, rotary
absolute positions/multiple heads, projection first VJP/full-rank metadata,
native dtype/domain refusals and recovery, refusal of scalar protocol hooks
before dispatch, mixed permutation rank and alias
guards, and the whole block. These are correctness fixtures, never substitute
performance workloads.

Stage60 compiled on CPU and Metal, but whole-block exact comparison failed.
A separate native stage probe retained the original arrays and found all
stages through exponentials equal. The ordinary graph's probabilities match
an independent fold32 sum exactly; direct invocation of the existing serial
softmax body matches all 50 specialized probabilities exactly. Nine graph
probabilities differed under the other reduction contract. This established
an adapter error, not an error in either existing softmax implementation.
The new private opcode uses the serial bodies for forward and first VJP and
is registered for replay, inspection and checkpoints. Its three neighboring
length fixtures distinguish both sum contracts, compare a double-precision
derivative oracle, audit public calls and replay saved graphs. Source repair
is pending the next compiled freeze; the exact whole-block assertion remains.
Evidence: `~/mojolearn-evidence/finish-graph-20261020/transformer-stage60-`
`{stage-comparison,softmax-contract-proof}.json`, complete native probe logs,
and retained `transformer-stage60-fixture.npz` and stage-array archives.

Parent must compile changed native closures, run focused and installed
verification on actual targets, then run full saved block preparation. Any
specialized/captured discrepancy is a real failure to diagnose; no tolerance
may replace the required bitwise product comparison. The float64 oracle is
an additional numerical check with its separately stated tolerance. Only a
passed artifact proceeds to installed NVIDIA/AMD full-workload A/B. Other
optimizer-pass workloads, whole-language-model/corpus coverage, physical
multi-GPU qualification and the confirmed upstream ownership blocker remain
separate pending work.
