# lowbit-blocks: the models compute under the profile (inference)

Lane H, branch `lane/lowbit-blocks`, worktree `~/mojolearn-wt/lowbit-blocks`,
forked from `lane/lowbit-int15` (merged again at its d07ecd201). Brief in
force: `~/mojolearn-evidence/lowbit-units/brief_current.md`; the lane's
section of `brief.md` ("Lane H"). Lane files `~/mojolearn-evidence/lowbit-blocks/`.
Nothing here flips the default; `fp32_v1` stays the default and must not move.

## (a) THE PLAN, written before any code

### What computes under `fixed15_v1`, and what does not

| product | shape at SmolLM2-360M (m = B*L tokens) | contracted k | under the profile |
|---|---|---|---|
| q_proj | m x 960 x 960 | 960 | int15 (weight planes packed once, activation quantized per call) |
| k_proj, v_proj | m x 320 x 960 | 960 | int15 |
| o_proj | m x 960 x 960 | 960 | int15 |
| gate_proj, up_proj | m x 2560 x 960 | 960 | int15 |
| down_proj | m x 960 x 2560 | 2560 | int15 |
| Q.K^T (S11), per (batch, head) | L x S x 64 | 64 | int15, BOTH operands quantized per call, one scale per query row and one per key row, each along head_dim |
| head (tied embedding) | m x 49152 x 960 | 960 | int15 (head planes packed once at load) |
| P.V (S19, `attn_context_kernel`) | | S | STAYS on fp32.v1's pinned ascending chain |
| RMSNorm, RoPE, softmax, SiLU, residuals, embedding gather | | | unchanged (fp32 seams) |

Every k is at or below 2560, far under `INT15_MAX_K = 65536`: the profile can
take every product of the block and the head at this model's shapes.
Why decode equals prefill and the batch is invariant BY CONSTRUCTION: every
row's scale is a function of that row's own values (a token's activation row;
a key's head vector; a query's head vector), and the integer sum is exact, so
a cell depends only on its own two rows (W-9, W-8).

### Every call site that changes

`transformer/impl/llama/modeling_llama.mojo` (shared: the transformer and
byte_lm bindings and the transformer checks import it; all are rebuilt):
- the seven projection calls `stages.gemm_workspace.run[False](ctx, c, a, w.w_*, m, n, k, OP_NT)`
  (q, k, v in `llama_attention_forward`, o after the attention, gate, up and
  down, both down forms, in `llama_mlp_forward`) become ONE helper
  `llama_proj(ctx, stages, w, which, c, a, m, n, k)`: with no int15 planes on
  `w` it is exactly the old call (same arguments, same order); with planes it
  quantizes `a` by the PARALLEL quantizer to planes in stage scratch and calls
  `identical_gemm_int15_planes_into` (Lane C's entry point) on the weight's
  planes.
- S11 in `attention_eager_core`: the same switch on the per-head GEMM
  (`qbh`, `kbh` both quantized per call along head_dim).
- `eager_attention_forward`: under the profile the choice is FORCED eager, as
  a softcap forces it today, because the fused kernels compute Q.K^T in fp32
  inside themselves. P.V stays `attn_context_kernel`.
- `LlamaDeviceWeights` gains `int15: Optional[LlamaInt15Weights]` (seven
  weights' `hi`, `lo`, exponents), None in every existing constructor, and one
  new constructor for the profile (the seven float32 projection buffers are
  one-element placeholders there and are never read).
- `LlamaDeviceStages` gains `int15: Optional[LlamaInt15Stage]` (activation
  planes, exponents, the second operand's planes for S11, the parallel
  quantizer's workspace), created on the first call under the profile only,
  so the default path allocates nothing new.

Host oracle `transformer/checks/transformer_oracle.mojo::transformer_block_oracle`:
one optional argument carrying the seven weights' codes; with it, the seven
projections run `gemm_int15_oracle` (activation by `quantize_rows_int15`) and
S11 runs `gemm_int15_from_f32_oracle`. Without it: unchanged.

### How the weights' planes live across calls

- PACKED ONCE AT LOAD: `TransformerBlock(numeric_profile="fixed15_v1")`
  quantizes each of its seven projection weights once at construction with
  `mojolearn.linalg.quantize_int15` (rows along the input features, the
  contracted extent of `Y = X W^T`, W-9) and keeps `(hi, lo, exponents)` as
  private arrays. `CausalLM` does the same for the head.
- GPU: a NEW binding entry `transformer_session_forward_int15` (the fp32
  entries keep their lists word for word). The session uploads the planes on
  its first call and KEEPS THEM ON THE DEVICE for every later call. That is
  sound here and not for the float32 path because the planes are private to
  the block, written once, never exposed or mutated; a caller's float32 array
  is not. The norms and optional tensors are uploaded per call as today.
- CPU: `_mojolearn_neural_host` gains the same entry; the host oracle reads the
  planes (codes = hi * 128 + lo) per call.
- The head: `mojolearn.linalg.matmul_int15(hn, head_planes)` in `CausalLM._run`
  (per call the activation is quantized by the binding's quantizer; the head
  planes are uploaded per call in v1, as the float32 head is today).
- `CausalLM._generate_resident` returns None under the profile (the per-layer
  route runs); the resident session under the profile is owed.

### How the keyword reaches the block

`CausalLM.load(path, numeric_profile="fixed15_v1")` -> `CausalLM.__init__`
passes `numeric_profile=` to the transformer block ONLY when it is not the
default (so a default model builds exactly as on main); a Mamba model under
the profile is refused by name. `TransformerBlock.__init__(..., numeric_profile=None)`
resolves it (`_numeric_profile.resolve`), packs the planes, and its `_call`
routes to the int15 entry; `_call_fresh` and `backward` are not taken under
the profile (backward refused by name: training is out of scope).

OPENING THE GATE IN TESTS ONLY: the lane's test scripts set
`mojolearn._numeric_profile.PROFILES["fixed15_v1"]["inference"] = True` in
their own process before loading. No shipped file reads an environment
variable or a define for it; the orchestrator flips the real field.

### The order of gates

(b) one block, one projection, 4090: device == host oracle bit for bit under
the profile; the default prints main's digests (`check-transformer`'s card
and lines, built from main and from this branch, same box). (c) the whole
block and the head, decode == prefill at 1, 7, L-1, batch invariance, the
sabotage arm (`-D MOJOLEARN_LOWBIT_SABOTAGE=1`) seen failing. (d) SmolLM2-360M
full-logits hashes: 4090, H100, M3 Ultra, M2 Pro, MI325X, CPU. (e) perplexity
on Lane B's two texts against F1-pv32. (f) prefill 512 and decode per token
against fp32_v1 per box.

## Verdicts

### (b) and the block half of (c): GREEN on the 4090 (nvc2), job nvc2-0016, tree synced at commit 013429062

`pixi run check-transformer-int15` (`transformer/checks/transformer_int15_check.mojo`),
logs in `bench/results/lowbit_blocks/2026-09-29/gate_4090/`:
- PROFILE, device against the host oracle (`transformer_block_oracle(..., int15=True)`):
  11 fixture cases (B 1 to 3, L 1 to 64, head_dim 16 and 24, intermediate 64 and 300,
  n_rep 1 and 2, windows 3, 4, 5, 20), 30 of 30 stages bit-identical in every case,
  every projection and S11 included. The device's weight planes came from the
  parallel quantizer on the device; the oracle's codes from `quantize_rows_int15`.
- DECODE == PREFILL: cases base_b3_l16_nrep2 and win4_b1_l16_nrep2, prefix 1, 7
  and 15, every token's block output equals the full prefill's (0 cells) and
  the host oracle's decode (0 cells).
- BATCH: B=3 against each row alone at B=1, 0 cells differ, three rows.
- SABOTAGE (`-D MOJOLEARN_LOWBIT_SABOTAGE=1`, the same program): exit 1, all 11
  profile cases MOVED, the 6 decode comparisons against the oracle failed, 0
  default cases moved (17 failures). Seen failing.
- THE DEFAULT DID NOT MOVE: `transformer/checks/transformer_check.mojo` built
  from this branch and from a clean worktree at the merge base 45464ced2
  (whose block sources, `transformer/`, `gemm/checks/gemm_identical.mojo`,
  `gemm/host/`, `checks/numerics.mojo`, `core/`, `mamba/impl/`, equal
  origin/main's byte for byte), run on the same GPU: identity cards identical
  (31 lines), every output line identical except tcmalloc's mbind warning and
  the card's own path. The check's own default phase: 11 cases fp32 device ==
  fp32 host oracle, digests printed (`int15 default digest`).

## Failures

(none yet)

## Owed

Everything above.
