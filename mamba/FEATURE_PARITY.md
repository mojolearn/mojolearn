# Mamba feature parity, mojolearn vs the reference implementations

Written 2026-09-01, the mamba lane. Decision-grade inventory answering one
question. Should mojolearn's mamba/ carry every user-visible feature the
PyTorch references carry for Mamba-1 and Mamba-2, and for each feature, does
it ship now, ship later on a named trigger, or get refused for a named reason.

**Sources, pinned.** Everything cited below comes from exactly three trees.

| tree | pin | role |
|---|---|---|
| state-spaces/mamba at `~/CascadeProjects/upstream/mamba` | `e9594ce` | the reference blocks, decode, generation |
| huggingface/transformers at `~/CascadeProjects/upstream/transformers` | `d56c55b` | the HF surface, config fields, torch fallbacks |
| this repository's `mamba/` | working tree 2026-09-01 | what ships today |

**What ships today, in one sentence.** One Mamba-1 block, FP32, inference
only, prefill and decode, under profile `mojolearn.identical.mamba1.fp32.v1`
(`mamba/IDENTICAL_MAMBA_CONTRACT.md`), gated bit-identical on three vendors
(IDENTITY_PATHS row 55, card md5 `f072dd22`). Mamba-2 code forms landed
2026-09-01 (one-box gates green per the contract's RUN RECORD; the
three-vendor E-leg still OWED, so no cross-vendor Mamba-2 sentence exists);
Mamba-3 code forms landed the same day, NOTHING RUN (see the section 5 rows).
No training path built (the backward is planned in
`mamba/IDENTICAL_BACKWARD_PLAN.md`, nothing compiled), no model level, no
Python binding. `python/mojolearn/` exports no mamba symbol at all as of
today.

**UPDATE, 2026-09-01 (later the same day), superseding the sentence above
where they disagree.** Two of its clauses closed. (a) The Mamba-2 block
EXISTS: `impl/mamba_ssm/modules/mamba2.mojo` + `ssd_minimal.mojo` under
`IDENTICAL_MAMBA2_CONTRACT.md`, gated on the Apple M4 in the identical
tier (that contract's RUN RECORD; cross-vendor legs still OWED, so the
three-vendor sentence remains Mamba-1's alone). (b) The Python surface is
BUILT, RUNS OWED: `bindings/_mojolearn_mamba.mojo` (the fourteenth
binding, two-list ABI) + `bindings/build_mamba.sh` +
`python/mojolearn/_mamba_impl.py` / `mamba.py`, exporting `Mamba1Block`,
`Mamba2Block` and the explicit state classes from `mojolearn` and
`mojolearn.mamba`; gate `python/mojolearn/tests/test_mamba_surface.py`.
Nothing of (b) has compiled or run anywhere -- every runnable claim is
UNVERIFIED until the build and the gate print. The "Python surface"
section at the end of this document carries the surface's deviations
(791-793) and the owed commands.

**House rules that bind the dispositions.**

- IT IS OK TO ADD CAPABILITY. A feature with no upstream counterpart is
  original, never refused for lacking one.
- A REFUSE names a genuine impossibility or a correct-validation reason,
  never mere effort.
- A SHIP LATER names its trigger.
- Contract design is NOT decided here. A sibling document,
  `IDENTICAL_MAMBA2_CONTRACT.md`, owns every Mamba-2 seam and constant
  decision; rows that depend on it say so and decide nothing.
- Nothing in this document has been run. Every claim about current gate
  status repeats the cited lane documents; anything new is UNVERIFIED,
  RUN OWED.

Dispositions are one of SHIP NOW (in the first shippable Mamba surface the
outside consumer can install), SHIP LATER (deferred, trigger named), REFUSE
(reason named).

---

## 1. Mamba-1 block

Upstream spellings are `mamba_ssm/modules/mamba_simple.py` (ms) at `e9594ce`
and `src/transformers/models/mamba/modeling_mamba.py` (mm) at `d56c55b`.

| feature | upstream spelling | mojolearn today | disposition | why / trigger |
|---|---|---|---|---|
| block forward, prefill, any B and L | ms:119-206 forward; mm:359-483 MambaMixer.forward | SHIPS. `impl/transformers/models/mamba/modeling_mamba.mojo` + `impl/mamba_ssm/ops/selective_scan_interface.mojo`, contract S1-S17 | SHIP NOW | done, three-vendor gated |
| decode step, one token | ms:208-253 step | SHIPS. `impl/mamba_ssm/modules/mamba_simple.mojo`, bit-identical to prefill by construction (contract section 5) | SHIP NOW | done |
| state allocation, zeros | ms:255-266 allocate_inference_cache | SHIPS as zero conv window + zero h | SHIP NOW | done |
| `d_model` free | ms:34 | free; gated at 8 and 16, arithmetic reads no shape | SHIP NOW | widen the gate sweep when the consumer names its sizes; UNVERIFIED beyond gated shapes, RUN OWED |
| `d_state` knob (default 16) | ms:35 | FIXED at 16, a profile constant (contract section 3); changing it is a v2 | SHIP LATER | trigger, a consumer checkpoint with d_state != 16; costs a v2 profile plus regenerated corpus |
| `d_conv` knob (default 4) | ms:36 | FIXED at 4 | SHIP LATER | same trigger shape as d_state |
| `expand` knob (default 2) | ms:37 | FIXED at 2 | SHIP LATER | same trigger shape |
| `dt_rank` knob (default ceil(d_model/16)) | ms:38, :58 | FIXED at the auto rule | SHIP LATER | trigger, a checkpoint that sets it explicitly |
| `conv_bias` (default True) | ms:44 | FIXED True | SHIP LATER | trigger, a checkpoint with conv_bias False; the bias-seed spelling (DEVIATION 721) then needs a zero-seed clause |
| `bias` on in_proj/out_proj (default False) | ms:45, ms:140-141 | FIXED False | SHIP LATER | trigger, a checkpoint with projection bias; gemm v1 bias-add seam decision owed then |
| `dt_min/dt_max/dt_init/dt_scale/dt_init_floor` initializers | ms:39-43, :82-101 | absent; weights arrive as given bits | REFUSE the bitwise-init claim, weights-in ships now | the init consumes torch.rand's Philox stream; bit-reproducing torch's RNG is not a correct validation target for a cross-check library, and the consumer's job is to hand both sides the SAME weights. The corpus hash spec (`corpus/README.md`) is this repository's own reproducible-values story |
| `use_fast_path` fused/fallback switch | ms:46, :145-160; import branches ms:15-28 | ONE arm, deliberately (DEVIATION 732) | REFUSE the knob | correct validation. Upstream's two arms round differently (contract S8; fused kernel vs selective_scan_ref), so a switch that changes bits contradicts the identity claim by construction. The reference rounding IS the profile |
| `layer_idx` + per-layer cache dict | ms:47, :268-294 _get_states_from_cache | absent; caller holds the two state buffers for one block | SHIP LATER | trigger, the multi-block backbone of section 5; a dict keyed by layer index is bookkeeping, not arithmetic |
| `selective_scan_fn` options z, delta_bias, delta_softplus, return_last_state | mamba_ssm/ops/selective_scan_interface.py:127-128 | REFUSED BY NAME at the scan entry (DEVIATION 723); the block computes each exactly once elsewhere | REFUSE (keep as is) | correct validation, a seam with two spellings is a seam with two places to drift |
| `mamba_inner_fn` whole-block fused autograd | selective_scan_interface.py (not ported, derivation map notes) | absent | REFUSE | duplicate spelling of the same block; its rounding is the CUDA kernel's, which the profile already rejected at S8 |
| activation other than silu | ms:74 hard-codes silu, asserts at :171 | silu only, `identical_silu` single-quotient (DEVIATION 744) | REFUSE (matches upstream) | upstream itself accepts only silu/swish |
| NaN/inf refusal, -0.0 preservation, ftz policy, batch-composition invariance gate | no upstream counterpart | SHIPS (contract sections 6 and 8; clause (c) is something upstream cannot state, noted vLLM `supports_batch_invariance()` False for Mamba) | SHIP NOW | ORIGINAL capability, kept; it is the product |

## 2. Mamba-2 block

Upstream spellings are `mamba_ssm/modules/mamba2.py` (m2) at `e9594ce` and
HF `modeling_mamba2.py` (M2) / `configuration_mamba2.py` (cfg) at `d56c55b`.
NOTHING of this section exists in mojolearn today; "mojolearn today" is
absent for every row unless said otherwise. **(Stale as of 2026-09-01
later the same day: the SHIP NOW rows of this table landed in
`impl/mamba_ssm/modules/mamba2.mojo` + `ssd_minimal.mojo` and gated on
Apple -- see the dated update at the top of this document; the SHIP LATER
and REFUSE rows stand unchanged.)** Every constant-freezing and seam
decision below BELONGS TO the sibling `IDENTICAL_MAMBA2_CONTRACT.md`; a row
that says "contract decides" defers to it.

| feature | upstream spelling | mojolearn today | disposition | why / trigger |
|---|---|---|---|---|
| block forward, prefill (torch-fallback arithmetic) | m2:154-276 (non-mem-eff branch :209-276); M2:444-588 torch path | absent | SHIP NOW | the core ask. The reference arithmetic to pin is the unfused branch, exactly as Mamba-1 pinned selective_scan_ref over the CUDA kernel |
| decode step, one token | m2:278-343 step; M2:532-554; selective_state_update torch semantics m2:310-322, M2:192-250 | absent | SHIP NOW | consumer needs single-token decode; prefill == decode gate carries over structurally |
| state allocation + exact state handoff (conv window over the full xBC width, h as [B, nheads, headdim, d_state]) | m2:345-355; M2 cache layers (modeling_mamba2.py:25 Cache/DynamicCache, :491-492) | absent | SHIP NOW | the handoff is the consumer's third requirement; state layout is a contract decision (contract decides the buffer order and dtype) |
| chunked prefill continuation from a carried state | M2:571 `initial_states=recurrent_state`; mamba_chunk_scan_combined `initial_states` (ssd_combined.py:628) | absent | SHIP NOW | resume-from-state prefill is what makes handoff useful beyond L = 1; contract decides whether the profile's scan is sequential or chunked, and this row inherits that |
| scan algorithm choice, sequential recurrence vs chunked SSD (`chunk_size`, default 256; HF pads to a chunk multiple) | m2:59, :244-260; M2:254-341 (pad :282, segment_sum :73-90); ssd_minimal.py:34-78 | absent | SHIP NOW one pinned spelling | CONTRACT DECIDES which spelling is the profile. Named consideration for it, not decided here. The two round differently, and the chunked spelling makes the bits a function of chunk_size and of L mod chunk_size, which touches the batch-composition and length-extension clauses the Mamba-1 profile prizes; the sequential spelling has the Mamba-1 precedent. Whatever it picks, the OTHER spelling is a named sabotage arm, not a second mode |
| `headdim` (m2 default 64, HF head_dim 64), `nheads`, per-head scalar A and dt | m2:45, :80-85; cfg:62-63 | absent | SHIP NOW at contract constants | the head structure IS Mamba-2; constants frozen by the contract, other values a v2 |
| `d_state` (m2 default 128, HF 128) | m2:41; cfg:66 | absent | SHIP NOW at contract constant | ditto |
| `ngroups` (m2 default 1; HF default 8) | m2:47; cfg:74; B/C broadcast M2:219-221 | absent | SHIP NOW at contract constant | the two references DEFAULT DIFFERENTLY, so the contract must name its constant and the checkpoint families it matches; note ms decode torch path asserts ngroups == 1 (m2:311) while the triton/HF paths do not |
| `rmsnorm` gated norm (default True; RMSNormGated, gate inside the norm) | m2:50, :142-145, :269-270; M2:105-120 MambaRMSNormGated | absent | SHIP NOW (True arm) | default-on and present in every stock Mamba-2 checkpoint; the False arm (gate multiplied in the scan instead, m2:252, :321-322) is SHIP LATER, trigger a rmsnorm=False checkpoint |
| `norm_before_gate` (default False; HF hard-codes False, M2:477) | m2:51 | absent | SHIP LATER (False ships as the constant) | trigger, a checkpoint with True; it reorders the norm and the gate product, a different seam list |
| `dt_limit` clamp (default (0, inf)); HF `time_step_limit` | m2:55, :183; cfg:84; M2:276 `torch.clamp` after softplus | absent | SHIP NOW | HF applies the clamp unconditionally in the torch path (no-op at the default), and released checkpoints set it; a clamp is selection-free arithmetic. Spelling (clamp order vs softplus, ftz) is the contract's |
| `d_ssm` partial-SSM + gated-MLP split (z0/x0, d_mlp > 0) | m2:46, :81, :210-215, :271-272, :340-341 | absent | SHIP LATER | trigger, a consumer checkpoint with d_ssm set; default None means d_mlp = 0 and the split vanishes |
| `D_has_hdim` (default False) | m2:49, :139, :191, :251 | absent | SHIP LATER (False ships as the constant) | trigger, a checkpoint with True |
| `A_init_range`, `dt_min/max/init_floor`, `conv_init` initializers | m2:48, :52-54, :43, :114-136 | absent | REFUSE bitwise-init, weights-in ships | same reason as the Mamba-1 initializer row |
| `use_mem_eff_path` fused switch | m2:60, :184-208; mamba_split_conv1d_scan_combined (ssd_combined.py:983) | absent | REFUSE the knob | same DEVIATION-732 reasoning; one profile rounding, the fused arm is a sabotage target not a mode |
| varlen args `cu_seqlens`, `seq_idx`, `causal_conv1d_varlen_states`, `return_varlen_states` | m2:154, :217-228, :255-259; ssd_combined.py:628 | absent | SHIP LATER | see section 6 |
| tensor/sequence parallel (`process_group`, `sequence_parallel`, Column/RowParallelLinear, all_reduce/reduce_scatter) | m2:28-29, :62-63, :97-102, :147-152, :206-208 | absent | REFUSE | genuine impossibility for the identity tier. A cross-rank reduction's fold order is a function of world size and topology, so a bitwise contract over "the same model on any number of devices" is unsatisfiable as stated; mojolearn also ships no multi-device runtime. A single-device run of a TP-trained checkpoint's merged weights is just the ordinary block and needs no TP feature |
| `learnable_init_states` (Mamba2Simple only) | mamba2_simple.py:40 | absent | SHIP LATER | trigger, a checkpoint carrying init_states; note the initial_states PLUMBING ships now via the handoff row, this row is only the learned parameter |
| HF `attention_mask` padding-zeroing | M2:93-102, :456, :524 | absent | SHIP LATER | see section 6 |
| Mamba-2 batch-composition invariance, NaN/inf refusal, -0.0, ftz | no upstream counterpart | absent | SHIP NOW | carry the Mamba-1 contract's sections 6 and 8 forward; ORIGINAL capability, contract decides exact clause wording |

## 3. Decode and inference caching

| feature | upstream spelling | mojolearn today | disposition | why / trigger |
|---|---|---|---|---|
| explicit state buffers in and out, caller-owned | ms:208-253 signature (conv_state, ssm_state in, out) | SHIPS for Mamba-1 device-side | SHIP NOW (and onto the PyPI surface, section "consumer") | this is the exact-state-handoff requirement; keep the states EXPLICIT ARGUMENTS, not hidden in an object |
| `InferenceParams` (seqlen_offset switch between prefill and step, per-layer dict, reset) | generation.py:17-34; ms:127-132 | absent | SHIP LATER | trigger, the multi-block backbone; for one block the two explicit buffers are strictly more testable. The prefill/decode dispatch-by-offset itself is trivially reconstructed by any caller |
| lazy per-layer cache creation | ms:268-294; m2:357-383 | absent | SHIP LATER | with the backbone |
| `selective_state_update` extras, `state_batch_indices` (paged/indexed states), multi-token step | selective_state_update.py:135, :187-188 | absent | SHIP LATER | trigger, a consumer serving loop that pages states or steps more than one sequence set per launch; note this is exactly where batch-composition invariance (already gated for Mamba-1) becomes the selling point |
| CUDA graph decode cache (`cg=True`, DecodingCGCache, capture/replay) | generation.py:270-389 | absent | REFUSE | vendor-specific launch plumbing; CUDA graphs do not exist on Metal or as a portable MAX surface, so a cross-vendor library cannot carry the API. Latency work belongs to FAST, which is unversioned and makes no identity claim, and needs no upstream-shaped API to do it |
| generation loop (greedy/top-k/top-p/min-p/temperature, repetition penalty, eos stop, teacher_outputs, streamer, output_scores) | generation.py:37-117 sampling, :120-243 decode, :246-267 GenerationMixin | absent | SHIP LATER | trigger, the consumer asking for tokens rather than tensors. Two warnings for whoever builds it. Greedy argmax is a float SELECTION, the row-13 tie hazard the block itself deliberately avoids, so an identical-tier generate needs a declared tie rule; and multinomial sampling is RNG-owned by the caller or it is not reproducible. Logits-level cross-checking (SHIP NOW set) already serves the stated consumer needs without this |

## 4. Training and backward

The stance is `mamba/IDENTICAL_BACKWARD_PLAN.md`'s and is not re-litigated
here; Mamba-2 rows are written to be consistent with it.

| feature | upstream spelling | mojolearn today | disposition | why / trigger |
|---|---|---|---|---|
| Mamba-1 backward | SelectiveScanFn autograd + selective_scan_bwd CUDA (cited in the plan) | PLANNED, profile `mojolearn.identical.mamba1.bwd.fp32.v1`, gates MB1-MB10 specified, NOTHING COMPILED | SHIP LATER (already in flight) | trigger is its own lane closing; note the plan's finding that upstream's backward is not reproducible run to run even on one device (five gpuAtomicAdd accumulators, ROCm-different block tables), so ours is capability upstream lacks |
| Mamba-1 backward under a different microbatch schedule; streaming/decode backward | n/a | n/a | REFUSE (already refused by the plan, sections 3.1 and 4.2) | proven impossible there; inherited unchanged |
| Mamba-2 backward | ssd_combined.py `_mamba_chunk_scan_combined_bwd` (:398) and siblings, atomics inside (e.g. `_chunk_scan_chunk_state_bwd_dx` :262) | absent | SHIP LATER | triggers, in order, the Mamba-2 FORWARD profile gated on three vendors AND the Mamba-1 backward gates green. Same shape of claim as the plan's (fixed batch composition, fixed L, separate profile name). Upstream evidence the deterministic ambition is real and theirs is not there yet, mamba_ssm/utils/determinism.py:21-27 gates their own atomics fallbacks behind `MAMBA_DETERMINISTIC` |
| chunked training scan (ssd_minimal listing) | ssd_minimal.py:34-78, segsum :23-32 | absent | SHIP LATER | folds into the Mamba-2 backward row; also the Mamba-1 plan's own note that its serial backward has ONE chunk by construction until a chunked variant is built |
| optimizer/init markers `_no_weight_decay`, `_no_reinit`, `_init_weights` rescale | ms:101, :111, :115; mixer_seq_simple.py:92-121; M2:644-670 | absent | REFUSE | these are annotations for torch's optimizer and init machinery; without torch they denote nothing. Correct validation, a gradient cross-check hands both sides the same weights and compares grads, and that is the SHIP LATER rows above |

## 5. Model level (backbone and LM)

Upstream spellings are `mamba_ssm/modules/block.py` (blk),
`mamba_ssm/models/mixer_seq_simple.py` (seq), `config_mamba.py` at
`e9594ce`, plus HF Mamba2Block/Mamba2Model at `d56c55b`.

| feature | upstream spelling | mojolearn today | disposition | why / trigger |
|---|---|---|---|---|
| residual block wrapper (Add -> Norm -> Mixer) | blk:42-88 unfused arm; mm:505-530 | SHIPS for Mamba-1 (S16 + S1-S4 inside the block) | SHIP NOW | done |
| `fused_add_norm` switch | blk:12, :36-40, :56-66 | absent; the unfused reference spelling is the profile | REFUSE the knob | correct validation, upstream's own comment says the fusion exists "purely for performance"; it is a second rounding of the same seam. FAST may fuse without an API knob |
| `residual_in_fp32` | blk:27, :54-55; cfg residual_in_fp32 | trivially always true, everything is FP32 | SHIP NOW (as a no-op fact, documented) | becomes a real seam only if a reduced-precision tier ships (section 7) |
| multi-block backbone (embedding, n_layer blocks, final norm) | seq:124-218 MixerModel | absent | SHIP NOW | thin composition of certified parts. The embedding lookup is a COPY (bits move untouched), the final RMSNorm is seams S1-S4, the inter-layer adds are S16; per-layer state dict comes with it (section 3 rows). This is what lets the consumer cross-check a model, not just a block. UNVERIFIED, RUN OWED once built |
| LM head + `tie_embeddings` + `pad_vocab_size_multiple` + `num_last_tokens` | seq:221-290; config_mamba.py:17-18 | absent | SHIP NOW (with the backbone) | lm_head is one gemm v1 call; tying is the same buffer twice; padding is an allocation rule; num_last_tokens is a slice, a copy |
| mixed layer types, `ssm_cfg["layer"]` in {Mamba1, Mamba2, Mamba3} per model | seq:54-61 | absent | SHIP NOW for Mamba1-only and, once its contract lands, Mamba2-only stacks; mixed stacks SHIP LATER | trigger for mixed, a real hybrid checkpoint the consumer names |
| MHA hybrid layers (`attn_layer_idx`, attn_cfg) | seq:47-49, :69; modules/mha.py | absent | SHIP LATER, and NOT this lane | attention belongs to the transformer lane; trigger, a hybrid checkpoint plus that lane's owner taking the row. Refusing would be wrong (nothing impossible), but the mamba lane does not own it |
| MLP interleave (`d_intermediate` > 0, GatedMLP, norm2) | seq:73-78; blk:31-35, :69-86; modules/mlp.py | absent | SHIP LATER | trigger, a checkpoint with d_intermediate > 0; stock mamba/mamba2 LM checkpoints use 0 |
| LayerNorm option (`rms_norm=False`) | seq:70-72, :177-179 | absent; RMSNorm only | SHIP LATER | trigger, a LayerNorm checkpoint; needs a mean-subtraction seam the contract does not have |
| `from_pretrained`/`save_pretrained`, HF hub fetch, config json | seq:292-315; utils/hf.py | absent | SHIP LATER (hub fetch); raw-buffer weight intake SHIP NOW | the consumer requirement is "the same weights on both sides", which raw little-endian buffers plus a manifest already deliver (the corpus file format is the precedent). A checkpoint-format importer (torch .bin/safetensors -> our buffers) is a Python-side utility, trigger, first real-checkpoint cross-check |
| Mamba-3 block | mamba_ssm/modules/mamba3.py:43-70 (rope_fraction :53, MIMO :59-60, heavy_tail_activation :27-41, per-token rotary on B/C, trapezoidal discretization inputs :106-108) | BUILT, NOTHING RUN (2026-09-01): contract `IDENTICAL_MAMBA3_CONTRACT.md` + oracle/impl/check code forms landed (`checks/mamba3_*.mojo`, `impl/mamba_ssm/ops/mamba3_siso.mojo`, `impl/mamba_ssm/modules/mamba3.mojo`); every gate RUN OWED, no compile has happened, no claim exists | SHIP LATER, gate-first | the contract sibling now exists and the addendum row below scopes the surface knobs; the trigger for SHIPPING is unchanged (a consumer-named Mamba-3 checkpoint) and the completion claim still lives on the three-vendor E-leg |
| Mamba-3 SURFACE KNOBS (the contract's deferred inventory, addendum to the row above) | mamba3.py:44-70 | pinned or refused, per profile `mojolearn.identical.mamba3.siso.fp32.v1` section 3 | one disposition per knob | PINNED AS PROFILE CONSTANTS: d_state 128, expand 2, headdim 64, ngroups 1, rope_fraction 0.5 (32 angles), A_floor 1e-4, chunk_size 64 (PART OF THE ARITHMETIC, mamba2 DEV 783's standing), B/C norm eps 1e-5. REFUSED BY NAME, structurally (no such knob exists on the surface): is_mimo/mimo_rank, is_outproj_norm, fuse_pregate_headwise_norm, rope_fraction 1.0, ngroups > 1, cu_seqlens/seq_idx varlen, non-Float32 (the shipped bf16 casts, mamba3_siso_combined.py:390-399, refused not reproduced). INITIALIZATION FACTS, out of scope (weights are inputs): dt_min/dt_max/dt_init_floor (:54-56, :111-115), the ones-init B/C biases (:121-122), D init (:140). ABSORBED-KWARG NO-OPS upstream, absent here: dropout, layer_idx, n_layer, device, dtype (:65-69). DEFERRED, not refused (contract section 5): varlen, MIMO, is_outproj_norm arms, multi-block caches |

## 6. Varlen and batching

| feature | upstream spelling | mojolearn today | disposition | why / trigger |
|---|---|---|---|---|
| padded batches + right-padding mask | M2:93-102 apply_mask_to_padding_states, :456, :524 | absent; equal-length batches only (upstream mamba_ssm assumes the same, generation.py:142) | SHIP LATER | trigger, consumer batches ragged sequences in one launch. Note the consumer can already get exact ragged behavior TODAY by splitting the batch, because batch-composition invariance (clause (c)) guarantees the split changes no bits; that is the honest workaround to document |
| packed varlen (`cu_seqlens`, `seq_idx`, varlen conv states, per-seq final states) | m2:217-228; ssd_combined.py:628, :391 (batch must be 1); causal_conv1d_varlen_states | absent | SHIP LATER | same trigger, the throughput shape of the same need; sequence boundaries inside one row make every fold's operand set a function of the packing, so the identity clauses need restating first (contract decides) |
| batch-composition invariance as a stated guarantee | no upstream counterpart (vLLM's batch-invariant mode excludes its Mamba backends) | GATED for Mamba-1, clause (c) | SHIP NOW | ORIGINAL; this is the feature that makes the two rows above deferrable |
| multi-sequence decode with per-slot state indexing | selective_state_update.py:135, :187-188 state_batch_indices | absent | SHIP LATER | duplicate of section 3's row, kept here for the batching reader |

## 7. Dtypes

| feature | upstream spelling | mojolearn today | disposition | why / trigger |
|---|---|---|---|---|
| FP32 everywhere | selective_scan_ref promotes to float (interface :140-146); our contract section 3 "Float32 everywhere" | SHIPS, the certified tier | SHIP NOW | the identity discipline's home |
| bf16/fp16 weights and activations with fp32 islands (A_log kept fp32 ms:109, A exp'd `.float()` ms:143/m2:182; norm computed fp32 M2:112-118; dt/state fp32 inside kernels) | as cited | absent | SHIP LATER | trigger, the consumer cross-checking a reduced-precision deployment. It arrives as a SEPARATE profile (the mixture boundaries, where casts round, are seam decisions), not as a dtype parameter on v1 |
| fp64 reference | corpus `ref64/` (host torch float64) | host-side only, tolerance anchor | REFUSE on device | MAX provides no float64 on the device targets this library certifies; the host corpus already fills the tolerance-reference role, so nothing is lost |
| dtype knobs `device=`/`dtype=` factory kwargs | ms:48-49 etc. | n/a | REFUSE as an API shape | torch-ecosystem plumbing; mojolearn's device and mode selection is its own runtime-parameter surface (three-tier modes), and mirroring torch's kwargs would misdescribe it |

**Recommendation on dtype scope, one paragraph.** The reference
implementations run bf16 or fp16 checkpoints with deliberate fp32 islands
(A, dt, softplus, the norm, the state), so "what PyTorch does" is a MIXTURE
whose cast boundaries differ between mamba_ssm's kernels and HF's torch
fallback, and the two do not agree bitwise with each other even before a
vendor is chosen. The identity discipline here is fp32-first, and that means
the parity claims split cleanly in two. Against OUR OWN pinned oracle the
claim is bitwise, cross-vendor, at fp32, under the named profile, and that
is the only bitwise sentence anyone may write. Against PyTorch the claim is
and remains tolerance-level, anchored by the fp64 corpus (corpus README's
calibration, roughly torch-fp32-width), because torch's fold orders are
library-shaped and were never ours to reproduce. A consumer cross-checking a
bf16 deployment therefore gets, today, an fp32 shadow run that is
bit-reproducible on every vendor plus a tolerance comparison to their bf16
numbers, which is the scientifically defensible offer; a certified bf16
profile is SHIP LATER on an explicit consumer request, ships under its own
version name, and even then its bitwise clause binds our implementations to
each other, never us to torch.

---

## Commercial consumer requirements onto the SHIP NOW set

The consumer is described generically. An external program pip-installs
mojolearn and uses its Mamba surface as an independent cross-check, needing
prefill, single-token decode, exact state handoff between the two, and the
bitwise-identical tier. Mapping each need onto the dispositions above:

| consumer need | SHIP NOW rows that satisfy it | gap today |
|---|---|---|
| prefill | Mamba-1 block forward (ships); Mamba-2 block forward (section 2, to build); multi-block backbone + LM head (section 5, to build) | Mamba-2 and model level do not exist yet |
| decode step | Mamba-1 step (ships); Mamba-2 step (to build) | same |
| exact state handoff | explicit caller-owned state buffers (ships for Mamba-1); Mamba-2 state layout row; chunked-prefill continuation from a carried state | Mamba-2 rows; and the handoff must be BYTE-specified (layout, order, dtype) in the sibling contract so the consumer can round-trip it |
| bitwise-identical tier | profile v1 gated three-vendor for Mamba-1; Mamba-2 inherits the same gate structure from its contract | Mamba-2 gates unbuilt |
| PyPI surface | BUILT 2026-09-01, RUNS OWED. `mojolearn.mamba` (`_mamba_impl.py` over the `_mojolearn_mamba` binding) exposes `Mamba1Block`/`Mamba2Block` forward + step with explicit numpy state in/out and the per-instance `numeric_mode=` parameter, exactly the shape this row asked for | was: "SHIP NOW, the single largest gap". The binding follows the repository's conventions (two-list ABI, DEVIATION 791); nothing has compiled or run -- `bash bindings/build_mamba.sh` per tier, then `tests/test_mamba_surface.py`, are the owed prints. Model-level (backbone/LM head) forward remains unbuilt (section 5's rows stand) |

The SHIP NOW build list, in dependency order: (1) the Python binding for the
existing Mamba-1 surface, since it makes the already-certified work
consumable and de-risks the binding shape cheaply -- **BUILT 2026-09-01,
RUNS OWED** (`bindings/_mojolearn_mamba.mojo` / `build_mamba.sh`,
`python/mojolearn/_mamba_impl.py`, gate `tests/test_mamba_surface.py`;
deviations 791-793 in the "Python surface" section below); (2) the Mamba-2
forward, step, and state handoff under the sibling contract, torch-fallback
arithmetic as the reference, dt_limit included, constants frozen by that
contract -- **BUILT: the Mojo lane landed and gated on Apple (the
contract's RUN RECORD, 2026-09-01 evening; cross-vendor legs OWED), and
its Python surface landed with item (1)'s** (`mamba2_forward` /
`mamba2_decode_step`, the three-piece state as caller-owned buffers),
RUNS OWED for the Python path; (3) the Mamba-1/Mamba-2 backbone + LM head
composition -- still unbuilt; (4) the Mamba-2 corpus -- **DONE** (17 cases
committed at f45fa796, `corpus/README.md`'s run record). Every runnable
claim in that list is UNVERIFIED, RUN OWED until its gate prints on a
device.

## Notable refusals, in one place

use_fast_path / use_mem_eff_path knobs (two roundings, one profile);
bitwise reproduction of torch RNG initializers; tensor/sequence parallel
(cross-rank fold order unsatisfiable as a bitwise contract); CUDA graph
decode cache (not portable, FAST-tier concern); fused_add_norm knob;
device-side fp64; torch factory-kwargs API shape; optimizer/init markers.
Every other absence is a deferral with a trigger, and per the house rule,
"no upstream counterpart" never appears above as a refusal reason.

## Open questions (owed to the sibling contract or to Andrew)

1. Sequential vs chunked spelling for the Mamba-2 profile scan, and if
   chunked, whether chunk_size is a profile constant. Owner, the sibling
   contract.
2. The frozen Mamba-2 constants (d_state 128, headdim 64, ngroups 1 vs
   HF's 8) and which released checkpoint family they must match. Owner, the
   sibling contract, informed by which checkpoints the consumer will
   cross-check.
3. The byte layout of the Mamba-2 handoff state (conv window over the full
   xBC width vs split; h axis order). Owner, the sibling contract.
4. Whether the Python binding exposes per-stage taps (the card stages) or
   only end-to-end outputs plus states. Consumer-facing API choice, owner
   Andrew. **Interim (2026-09-01, the surface's build): end-to-end outputs
   plus states only (plus the Mamba-2 `h_last` report the contract makes
   part of the block's return). Per-stage taps remain OPEN and would be an
   additive change to the addrs lists, not a rework.**
5. Whether a generation loop is wanted at all, given logits-level checking
   covers the stated needs. Owner, Andrew.

---

## Python surface (built 2026-09-01, RUNS OWED)

The consumable surface the consumer table's "PyPI surface" row demanded.
Files: `bindings/_mojolearn_mamba.mojo` (the FOURTEENTH extension; entry
points `mamba1_forward`, `mamba1_decode_step`, `mamba2_forward`,
`mamba2_decode_step`, plus the `mamba_numeric_mode` / `mamba_vendor`
read-backs), `bindings/build_mamba.sh` (three-tier build, sibling of
`build_gp.sh`, AIR floor a placeholder 1 until the first cold build
measures it), `python/mojolearn/_mamba_impl.py` (classes `Mamba1Block`,
`Mamba2Block`, states `Mamba1State`, `Mamba2State`; float32-only with
bf16/fp16/float64 refused BY NAME), `python/mojolearn/mamba.py` (the
public module), exports from `mojolearn/__init__.py`, registration in
`_backend.py` (`_MODULES` + `_build_script`, both, per its DEVIATION 869
header), and the gate `python/mojolearn/tests/test_mamba_surface.py`
(corpus-anchored: `base_b2_l4_d8` and `m2_base_b2_l4_d32` against their
`ref64` at the corpus README's tolerance; decode==prefill and
split-prefill resumption bitwise-ASSERTED under the identical tier,
reported under fast).

NOTHING of it has compiled or run. The owed prints, in dependency order,
per tier (fast, then deterministic and identical with
`MOJOLEARN_NUMERIC_MODE` set): `bash bindings/build_mamba.sh`, then
`cd python && python3 -m mojolearn.tests.test_mamba_surface`. The .so
artifacts carry NO freshness signal, so every tier's binding must be
REBUILT before any Python-surface run is believed.

Deviations 791-793, the surface's own (the 782-789 format; binding and
wrapper headers carry the full text):

- **DEVIATION 791 — the two-list ABI.** Every entry point takes exactly
  two Python lists, `addrs` (buffer addresses, order written in the
  binding docstring and mirrored at the `_mamba_impl.py` call site) and
  `params` (scalars), because `PythonModuleBuilder.def_function` stops
  elaborating above roughly nine arguments — MEASURED 2026-09-01 on the
  gp binding's first ten-positional spelling — and `mamba1_forward`
  carries fourteen addresses. The fold is the tree's own precedent (knn,
  gp). Arrays addressed in the list are bound to Python locals for the
  call's duration; an address in a list keeps nothing alive.
- **DEVIATION 792 — caller-owned explicit state, updated in place.** The
  recurrent state crosses as plain float32 NumPy buffers the caller
  allocates (`allocate_state`), read at entry and written back before
  return: two pieces for Mamba-1 (conv window, h), three for Mamba-2
  (conv window, boundary h, the open-chunk xBC/dt buffer pair) with
  `buffered_tokens` crossing as a params scalar in and the return value
  out. Zeros in = `allocate_inference_cache`; nonzero h at
  buffered_tokens 0 = `initial_states`. Mamba-2's `h_last` REPORT is a
  separate always-written output (contract section 5's distinction). No
  hidden cache object exists on either side — section 3's "explicit
  arguments, not an object" row, made structural.
- **DEVIATION 793 — the refusal split.** The binding refuses only
  boundary disagreements (null address, wrong-length lists, buf_len
  outside [0, 256)); the wrapper refuses dtypes BY NAME (float32 only;
  bfloat16/float16 named as the SHIP LATER profile, float64 named rather
  than downcast — deliberately UNLIKE the classical estimators'
  convenience conversion, because this surface certifies bits), exact
  weight/state shapes, and missing/unknown weight names; everything else
  (non-finite values by flat index, B/L positivity, d_model rules) goes
  down unjudged so the Mojo refusals stay reachable. One duplication is
  accepted and named: the wrapper repeats `Mamba2Dims.of`'s
  multiple-of-32 rule because it cannot size the state buffers without a
  whole `nheads`; the Mojo constructor remains the authority. A binding
  compiled with any sabotage define ABORTS at PyInit rather than serve
  sabotaged bits under a green label.
