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

### (d) whole-model identity, SmolLM2-360M, B=2 x L=64 fixed ids, full logits (B x L x 49152 float32)

`tools/lowbit_blocks/model_logits.py` through `CausalLM.load(path, numeric_profile=...)`;
records in `bench/results/lowbit_blocks/2026-09-29/`.

| Box | Job | fixed15_v1 sha256 | fp32_v1 sha256 | decode == prefill (1, 7, 63) | batch |
|---|---|---|---|---|---|
| RTX 4090 (nvc2) | nvc2-0017 | d37c2ea81d13743a... | 833c9a8947bdd619... | EQUAL x3 | EQUAL x2 |
| H100 NVL (nvc3) | nvc3-0035 | d37c2ea81d13743a... | 833c9a8947bdd619... | EQUAL x3 | EQUAL x2 |
| CPU host path (nvc2's x86-64 host, MOJOLEARN_VENDOR=cpu, device cpu) | nvc2-0018 | d37c2ea81d13743a... | 833c9a8947bdd619... | | |
| M2 Pro | steward 1790659101869 | d37c2ea81d13743a... | 833c9a8947bdd619... | EQUAL x3 | EQUAL x2 |
| MI325X | steward 1790660159967 (reference plan), 1790662265717 (tuned plan) | d37c2ea81d13743a... (both plans) | 833c9a8947bdd619... | EQUAL x3 | EQUAL x2 |
| 4090, tuned plan | nvc2-0021 | d37c2ea81d13743a... | | EQUAL x3 | EQUAL x2 |
| H100, tuned plan | nvc3-0043 | d37c2ea81d13743a... | 833c9a8947bdd619... | EQUAL x3 | EQUAL x2 |
| SABOTAGE: 4090, tree copy with `int15_device_value_flip.patch`, linalg and transformer rebuilt | nvc2-0018 | 703b7e2158a0e140... MOVED | 833c9a8947bdd619... unmoved | | |

The M3 Ultra is not in the table: it was released (orchestrator, 2026-09-29).

### (e) quality correspondence (nvc2-0019)

Lane B's protocol and texts; our ids' sha256 equal Lane B's recorded ones
(enwik8 d5be2324..., pile_github 2d337ff1...). Our change is fixed15_v1 over
our own fp32_v1 from the same windows (interval: window means, 1.96 standard
errors, exp(x) - 1, as Lane B).

| Text | our fp32_v1 ppl | our fixed15_v1 ppl | our change (interval) | top-1 agreement | Lane B baseline ppl | Lane B F1-pv32 ppl | Lane B change (interval) |
|---|---|---|---|---|---|---|---|
| enwik8 | 7.937085 | 7.937129 | +0.0006% (-0.0044% to +0.0055%) | 0.9981 | 7.937106 | 7.936947 | -0.0020% (-0.0067% to +0.0027%) |
| pile_github | 3.361989 | 3.362101 | +0.0033% (-0.0010% to +0.0076%) | 0.9987 | 3.361991 | 3.362105 | +0.0034% (-0.0010% to +0.0078%) |

The absolute perplexities agree to five significant digits (7.9371, 3.3620
and 3.3621), which is what two different spellings of the non-product seams
(ours: the block's pinned fp32 seams; Lane B's: a float64-accumulated
simulation) can share. On pile_github the change agrees to two digits;
on enwik8 our change lies inside Lane B's interval and Lane B's inside ours,
and the two differ in sign: the change there is below either run's
resolution. Both are far under the 1 percent bar.

### (f) whole-forward time, SmolLM2-360M, fixed15_v1 against fp32_v1, same box, same job

`tools/lowbit_blocks/model_logits.py --phases time`: prefill of 512 tokens at
B=1 (one untimed call, then the median of 5); decode per token (a 512-token
prefill, then 32 `step` calls; one untimed run, then the median of 5). The
two profiles alternate twice in one job; both medians are given. The job held
every GPU of its box (nvc2: both slots; nvc3: its one; steward: alone).
EVERY TIMED OUTPUT IS CHECKED (rule 10): the sha256 of every timed prefill's
logits and of every decode run's 32 logits rows; within each run they agree,
and ACROSS THE FOUR GPU BOXES they are the same (fixed15_v1 prefill512
c55e4ce6d5e155d9..., decode32 1c2876958e46b7d3...; fp32_v1 da9825e377588fcb...,
2887098d1f90cd1d...), which is also the whole-model identity at 512 tokens.
Reference plan (the default). Records `bench/results/lowbit_blocks/2026-09-29/time_checked/`.

| Box | Job | Prefill 512, fixed15 / fp32 (ms) | over | Decode, fixed15 / fp32 (ms/token) | over |
|---|---|---|---|---|---|
| RTX 4090 | nvc2-0024 | 583.7 / 641.6, 409.9 / 1446.7 | 0.91, 0.28 | 117.2 / 221.2, 89.8 / 303.2 | 0.53, 0.30 |
| H100 NVL | nvc3-0045 | 266.9 / 413.9, 272.0 / 454.7 | 0.64, 0.60 | 133.9 / 281.8, 135.2 / 285.4 | 0.48, 0.47 |
| MI325X | 1790664369046 (3 of 4 runs; the 4th hung, Failures 7) | 232.9 / 266.2, (hung) / 264.6 | 0.87, 0.88 | 103.8 / 160.6, (hung) / 160.4 | 0.65 |
| M2 Pro | 1790664365442 | 4863.8 / 1077.7, 4884.3 / 1074.4 | 4.51, 4.55 | 1470.3 / 579.3, 1505.1 / 574.8 | 2.54, 2.62 |

The 4090's fp32_v1 prefill swings by 2x between the two runs of one job (641.6
and 1446.7 ms; earlier jobs 1055 to 1343 ms); the fixed15_v1 runs are steadier.
The 4090 is a shared pod whose host is busy with other lanes' CPU work; the
H100 and the MI325X are the boxes to judge by.

Earlier runs, kept as columns beside: nvc2-0020 and nvc3-0038 (unchecked
outputs; 4090 0.34 to 0.40 prefill, H100 0.64 to 0.68), nvc2-0021, nvc3-0043
and 1790662265717 (the tuned plan: not faster at the model on any box, so the
reference plan is the default), the M2 Pro's 1790660930020 (UNCHECKED
outputs) and 1790663712754, and nvc2-0023, nvc3-0044, 1790663724684
(CONTAMINATED: the hash of the 100 MB of logits was inside the timed interval;
Failures 4).

WHAT THESE TIMES ARE. The per-layer Python route (`CausalLM.forward`,
`step`), the route every identity run used. On it fp32_v1 uploads and
validates every float32 weight of every layer and the head on every call,
while the profile's planes stay resident in the block's session; the profile
also forces the eager attention path (per (batch, head) launches, one
synchronize per layer) where fp32_v1 runs the fused kernels. So the ratios
mix arithmetic with weight residency and attention scheduling, and decode
per token is mostly per-call overhead on both. A model gains less than a
product does: Lane C's complete 15-bit call over fp32.v1 was 0.44 to 0.56 on
the H100 and 0.19 to 0.30 on the MI325X; the whole forward reads 0.60 to 0.64
on the H100. On the M2 Pro the profile costs 4.5 times fp32_v1's prefill.

## Failures

1. build_byte_lm exited 2 on the 4090 (nvc2-0017): off Apple it needs one
   explicit arch. My job script's fault; the job now passes
   MOJOLEARN_GPU_ARCHS from nvidia-smi or rocminfo. Rebuilt in nvc2-0018.
2. The steward boxes' byte_lm and host builds exited 2, "output already
   exists": those scripts refuse to overwrite an existing .so and the
   steward worktree keeps outputs. The job now removes the stale output
   first. The GPU identity runs there used freshly built linalg and
   transformer bindings (both exit 0), so their verdicts stand.
3. The model was not staged on do-amd (first request exited 3, MODEL NOT
   STAGED) nor on the M2 Pro: staged from R2 on do-amd, and the pinned files
   copied to the M2 Pro (sha256 equal to the manifest).
4. The first checked time runs hashed each timed prefill's 100 MB of logits
   INSIDE the timed interval (nvc2-0023, nvc3-0044, 1790663712754,
   1790663724684): a fixed cost added to every timed prefill. Fixed in
   model_logits.py; all four boxes re-timed.
5. The first gate job read its default gate as "OUTPUT DIFFERS": the only
   differing lines were tcmalloc's mbind warning and the card's own path.
   The filter now drops those two; the cards were identical throughout.
7. MI325X, 1790664369046, the 4th timed run (fixed15_v1) HUNG: the process
   at 100 percent CPU, the GPU at 0 percent use, 80 minutes and counting at
   08:11Z; the three runs before it in the same job finished and agreed.
   Stack (`time_checked/mi325x_1790664369046.HANG_stack.txt`):
   `hipFree`-side `sched_yield` inside `DeviceBuffer::~DeviceBuffer` in
   `core/device_scan.mojo::device_first_nonfinite`, called by
   `llama_refuse_bad_call` in `llama_decoder_layer_forward_planted` under
   `_transformer_run_session_int15`: a buffer release waiting on the stream,
   which never drains. No amdgpu fault for this process in dmesg (faults at
   05:20Z and 05:26Z belong to other lanes' processes). CAUSE NOT
   ESTABLISHED; the same code finished every other MI325X run (identity at
   two commits, three timed runs here, four in 1790663724684). Not
   cancelled (rule 4); the steward's own 3-hour cap ended it (exit 124,
   FAIL). Re-run queued: 1790669531597. Owed: that re-run's verdict, and a stack
   of the stream's last kernels if it recurs.
6. In the progress file I first wrote an invented commit id for the gate
   run; replaced by the real one (013429062).

## What the blocks compute under the profile, and what they do not

UNDER fixed15_v1 (gemm.int15i64.v1, on every box and the CPU): the seven
projections of every transformer block (weights packed ONCE at load by
`quantize_int15`, kept as planes, resident on the device in the block's
session; activations quantized per call by the parallel quantizer), the
score product Q.K^T (one scale per query and per key over head_dim, per
call), and the head (`matmul_int15` on head planes packed once at load).
NOT: P.V (`attn_context_kernel`, fp32.v1's pinned ascending chain, by
decision), RMSNorm, RoPE, the softmax, SiLU, the residuals, the embedding
gather (fp32 seams, unchanged); training (every trainer and
`TransformerBlock.backward` refuse the profile by name); the resident decode
session and `CausalLM.generate`'s resident route (the per-layer route
computes the profile instead); Mamba blocks (refused by name).

The gate is opened in the lane's tests only: `model_logits.py` and `ppl.py`
set `_numeric_profile.PROFILES["fixed15_v1"]["inference"] = True` in their own
process. No shipped file reads an environment variable or define for it.

## Owed

- The orchestrator: flip `inference` for fixed15_v1 and give its `products`
  the families (projections and head, attention_qk: int15i64.v1;
  attention_pv: fp32.v1); a `MEASURED` row from the time table above.
- A resident session under the profile (`TransformerDecodeSession`,
  `CausalLMSession`): generate under the profile runs the per-layer route.
- The head's planes are uploaded per call (as the fp32 head is today); the
  head should keep them on the device as the blocks do.
- The CPU host path quantizes the weights per call (the same codes, by the
  same rule); it should read the planes packed at load.
- The profile forces the eager attention path; a fused Q.K^T under the
  profile would remove the per-(batch, head) launches.
- Apple: 4.5 times fp32_v1 at the model's prefill on the M2 Pro; where the
  time goes is not measured (Lane G's float-unit tuning; the eager path's
  per-head launches; the synchronous sliced launches of clause W-14).
- The repo's harness (`bench/model/harness.py`, `tools/model_leg/`,
  `tools/identity_break.py` lanes) does not carry a profile yet; the lane's
  own scripts produced every number here.
- The block's backward under the profile (training, version 2).
