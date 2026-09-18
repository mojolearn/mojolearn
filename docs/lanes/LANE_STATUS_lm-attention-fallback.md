# Attention fallback: instrumentation first

Branch lane/lm-attention-fallback, from main 3e8dabc37. No optimization yet.

Registered before the first run: H100 80GB HBM3, seed 20260917, pinned R2
enwik8 (100000000 bytes, SHA256
2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8),
700 steps, B1 L2048 DM768 H12 KV12 HD64 FF2048 layers12 V50257.
Prediction: 8400 forward and 8400 backward status observations; at least one
nonzero refusal by step 699. Candidate status 2 is a hypothesis, not a cause.
Zero refusals makes this run INCONCLUSIVE about the reported slowdown;
any status 1 falsifies the hypothesis that every refusal is status 2.
Expect initial eager_bytes 432, eventual 17314086912 if all layers grow;
aexp is separate (2415919104 bytes). Expect head median 0.18–0.26 seconds
(exclude step 0), tail 0.38–0.52 seconds if the transition completes.
Those ranges are predictions, not acceptance requirements or new measurements.

Instrumentation saves the actual forward/backward launcher return codes in
host fields. Session info appends triples per layer after the existing ten
entries: forward status, backward status, materialized. NOT_ATTEMPTED=-1,
FUSED_RAN=0, FUSED_REFUSED_REGIME=1, FUSED_CORNER=2. The materialized flag is
read after backward, which can itself have recomputed forward stages.
No arithmetic or dispatch changes. Historical bindings omit the new fields.

The body refuses missing, wrong-sized or wrong-digest corpus bytes. Run with
MOJOLEARN_STAGE_STRICT=1 and inspect stage.log for successful R2 staging.
Controls at HD64 force eager (-1) and fused (0); the validator deliberately
corrupts coverage, counts, status and sequence and must reject each by name.
No identity claim is made by these metadata checks.
