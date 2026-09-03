# transformer: the cross-vendor bit-identical FP32 transformer block lane

Opened 2026-08-24. The lane's brief is the "After GEMM: the minimum
transformer path" section of `IDENTICAL_GEMM_PLAN.md` (:153-177) and the
build order in `IDENTICAL_SSM_NOTES.md` (:65-88). DEVIATIONS 800-819 are
this lane's.

**The profile is `mojolearn.identical.transformer.fp32.v1`.** The reference
is a Llama-shaped decoder layer, EAGER attention path only, pinned to
huggingface/transformers `d56c55b`. Changing any seam decision, any frozen
constant or the stage list creates a v2; it does not amend v1.

**Status, 2026-08-28: BUILT, GATED, AND IN PHASE 8 ON THREE COLUMNS.**
`checks/` holds `transformer_check.mojo`, `transformer_oracle.mojo`,
`transformer_fixture.mojo` and the backward triple; the lane has thirteen
sabotage arms and a 30-stage identity card in contract section 9's order.

**Measured on ALL THREE COLUMNS on 2026-08-28, and the three cards are the
same bytes.** `transformer.identical.card` has md5
`8ce661b469681b18fb5cf4d566ad78ff` in all of
`bench/results/e1/2026-08-28_161700-MacBook-Air-1-terrabyte/lanes/`,
`bench/results/e1/2026-08-28_131651-runpod-nvidia/lanes/` and
`bench/results/e1/2026-08-28_203552-mojolearn-e2-amd/lanes/`, 30 records
each. The NVIDIA leg's `bootstrap.log` resolves the nvidia hardware column
(108 SMs, TF32 tensor-core products), so the column is read back and not
taken from the directory name.

- clause (a) PASS on each of the three -- 13 fixture cases, 30/30 stages
  bit-identical to the host oracle on all 262,634 cells, 30/30 card tags.
- clause (d) PASS under IDENTICAL on each of the three -- 4 decode steps
  bit-identical to the prefill on all 11,632 compared cells, with a control
  showing 57 misaligned stage comparisons that DO differ. **It FAILS under FAST** (91 stage-tokens,
  first at token 0 `q_proj.out` on 26 of 32 cells), which is not a defect:
  contract section 7.2 makes decode == prefill true by construction for the
  IDENTICAL profile and FAST promises none of it. `tools/e1_bootstrap.sh`
  therefore runs clause (d) on the identical arm only.
- clauses (b) and (c) are opt-in and not yet run in a round.
- clause (e), the section 8 planted audit, is OFF and it is a defect in the
  clause: it aborts the driver on its first plant because
  `LlamaDeviceWeights` refuses at UPLOAD while the clause's `try` wraps only
  the later forward call. Fix the `try`, then turn it on.

**The PYTHON surface is newer than that record and narrower, 2026-09-02.**
The FIFTEENTH binding `_mojolearn_transformer` compiled for the first time
that day and `python/mojolearn/tests/test_transformer_surface.py` printed
green in all three numeric tiers, **on an Apple M4 and nothing else** -- the
NVIDIA and AMD columns of the SURFACE are OWED even though the lane's own
cards have all three. "The PyPI surface" section below is the run record.

**The paragraph that stood here until 2026-08-28 said "NOT STARTED beyond
this document ... no oracle, no fixture, no device kernel, no gate, no card,
no sabotage and no number", and it was false for days.** It is deleted rather
than annotated. What it cost is worth one line: the lane was reported as
uncovered in a cross-vendor status review while its check driver had honoured
`MOJOLEARN_IDENTITY_TRACE` since DEVIATION 1101 and its own docstring said
`tools/e1_bootstrap.sh` phase 8 sets it. A lane built to be in the round, and
left out of the round, because the file describing it was never re-read.

Every line number in the contract came from reading source on 2026-08-24.

## What this lane is NOT rebuilding

Section 0 of the contract is the inventory, with file and line citations.
The short version. RMSNorm, the residual add and SiLU come from the mamba
lane and the numerics lane. Every projection AND the `q . k^T` product are
`mojolearn.identical.gemm.fp32.v1` cells, certified three-vendor at
`144aa5b`. Every transcendental comes from `checks/numerics.mojo`.
`portable_sinf` for RoPE and `identical_fmax` for softmax's row maximum are
the numerics lane's DEVIATIONS 820 and 825, landed 2026-08-24, cited here and
not written here.

**The genuinely new arithmetic in this lane is RoPE, softmax and the
attention-weighted value sum. Nothing else.**

## The phase ladder

| phase | what | size | needs a GPU? |
|---|---|---|---|
| 0 | the contract. **LANDED**, this directory | done | no |
| 1 | `transformer_fixture.mojo` and `transformer_oracle.mojo`, the NORMATIVE host oracle, built from `identical_mul_add`, `ftz` and the portable primitives so that it IS the contract rather than an opinion about it | moderate | no |
| 2 | the separating fixtures in `transformer_check.mojo`. Every contested decision in contract sections 4 and 5 gets a fixture that refuses to pass unless the two alternatives produce different bits. A random-input hash is insufficient | moderate | no |
| 3 | `impl/transformers/models/llama/modeling_llama.mojo`, the device spelling. One kernel per stage, MAX's `mha_gpu_naive` shape (one thread owns one score, one thread owns one output row), no shared memory and no warp primitive unless a clause names one | **large** | yes |
| 4 | the device gates, clause (a) and clause (b), against the oracle at every gate shape | moderate | yes |
| 5 | the KV cache and the decode path, then clauses (c) and (d) with their negative controls. **Clause (d) is what makes two of the sabotages non-inert; writing it late makes them look pointless** | moderate | yes |
| 6 | the identity card and the sabotage ladder, all thirteen arms, one build each | moderate | yes |
| 7 | `transformer/corpus/`, the independent torch float64 per-stage reference, on the `mamba/corpus/` pattern. The only comparison in this lane whose other side is not our own code. **The directory EXISTS** (committed `82173423`, 2026-08-25): `gen_corpus.py`, a README, and the checker `tools/transformer_corpus_check.py`. What is missing is a RUN and the `ref64/` case data it would write, plus a `pixi.toml` task | moderate | no |
| 8 | the price harness. Wiring, not a published number | small | yes |
| 9 | the three-vendor leg, on `gemm/E1G_RUNBOOK.md`'s pattern. **RUN 2026-08-28 FOR THE FORWARD, and the three cards are the same bytes** (see the status block). Still owed for the BACKWARD profile, and for forward clauses (b), (c), (e) and the sabotage ladder, none of which was in that round | operator | three |

Phases 1 and 2 are host-only and are the whole of the contract's falsifiable
content, so they come before a line of phase 3, exactly as the GEMM lane's
charter clause 5 requires. A kernel written against an unreviewed oracle is
what that charter forbids.

## The commands

The forward gate has run on three columns; these are the commands that ran it.

    # the forward gate, registered 2026-08-31 at `pixi.toml:1091`:
    pixi run check-transformer

    # the backward gate, registered 2026-08-31 at `pixi.toml:1092`, `def main`
    # at :3851. It COMPILED AND RAN for the first time on 2026-09-03, preflight
    # all green, and then REFUSED to certify: its `d_out` fixture cannot
    # separate a fused multiply-add chain from an unfused one, so three
    # sabotage arms are unfalsifiable. DEVIATION 1536 narrows the fixture's
    # binade budget in answer to that and is WRITTEN, NOT RUN.
    # COMPILED AND RUN, NOT YET GATED:
    pixi run check-transformer-backward

    # both modes, the way the other identity gates do it:
    tools/with_build_lock.sh     pixi run check-transformer
    tools/with_identical_mode.sh pixi run check-transformer

**The two tasks above were registered on 2026-08-31.** Until that day the
transformer gates had no `pixi.toml` task at all, and both files carried
their own `def main` the whole time, so the three-column round of 2026-08-28
was driven by path through `tools/e1_bootstrap.sh` phase 8. The sentence that
stood here, that no pixi task was registered and that the intended names were
`check-transformer-block` and `check-transformer-corpus`, is deleted. Those
two names do not exist in `pixi.toml` and never did.

**There is still NO corpus task.** `transformer/corpus/` exists (see the phase
7 row) and `tools/transformer_corpus_check.py` exists, but no `pixi.toml` line
invokes either, and the mamba sibling's `check-mamba-corpus` at `pixi.toml:1103`
is the shape one would take. Registering it is owed.

## What is here

| file | what |
|---|---|
| `IDENTICAL_TRANSFORMER_CONTRACT.md` | **the deliverable.** Twelve sections. The reuse inventory, the pinned reference, what one block call is, the profile constants, all twenty-three seams with their fused-or-unfused decisions, the softmax reduction order, why FlashAttention and SDPA are out of scope, decode equals prefill, the NaN and signed-zero audit, the thirty card stages, the six gated clauses with thirteen named sabotages, what is not claimed, and where this departs from the plan's sketch. |
| `IDENTICAL_BACKWARD_PLAN.md` | the backward profile's specification. Its gate compiled and ran 2026-09-03 and has not yet certified. |
| `checks/transformer_check.mojo` | the forward gate, 3,172 lines, `def main` at :2860. This is what produced the three cards above. |
| `checks/transformer_oracle.mojo` | the NORMATIVE host forward oracle, 1,502 lines. |
| `checks/transformer_fixture.mojo` | the fixture set, 1,169 lines, shared by the forward and backward gates. |
| `checks/transformer_backward.mojo` | the device backward, 3,042 lines. |
| `checks/transformer_backward_oracle.mojo` | the host backward oracle, 1,584 lines. |
| `checks/transformer_backward_check.mojo` | the backward gate, 4,237 lines, `def main` at :3851. COMPILED AND RUN 2026-09-03 on the first attempt, every preflight assertion green, then REFUSED to certify on a `d_out` fixture that cannot separate fused from unfused. DEVIATION 1536 narrows the generator's exponent draw to three binades in answer; WRITTEN, NOT RUN. |
| `corpus/` | `gen_corpus.py` (85 KB) and its README. The generator is written and, per its own README, has not been executed by its author; no case data is on disk. Its checker is `tools/transformer_corpus_check.py`. |
| `__init__.mojo`, `checks/__init__.mojo`, `impl/__init__.mojo` | empty package markers. |

## The PyPI surface (DEVIATION 795, written 2026-09-02, BUILT AND GATED ON APPLE THE SAME DAY)

Until 2026-09-02 the certified block exported no Python symbol at all. The
surface was written, COMPILED and GATED that day -- the binding built on the
first attempt (rc 0) and the surface gate printed green in all three numeric
tiers, 44 checks 0 failed each. **On ONE box and ONE vendor, an Apple M4.**
No NVIDIA and no AMD box has built or run this surface and both columns are
OWED. The run record is at the bottom of this section.

**What is exposed.** `mojolearn.TransformerBlock` (also
`mojolearn.transformer`), through the FIFTEENTH binding
`bindings/_mojolearn_transformer.mojo` and
`python/mojolearn/_transformer_impl.py`. NumPy float32 in and out, no torch.
ONE entry pair, both through `llama_decoder_layer_forward` and nothing else:
`forward` (prefill, and chunked-prefill continuation on a carried state) and
`step` (single-token decode, the same spelling at L = 1 -- contract section
7.2's construction, so there is no second arithmetic path to drift). The
state is EXPLICIT and caller-owned (`TransformerState`): the KV cache as two
flat capacity buffers PACKED AT THE USED STRIDE plus `cached_tokens`, the
bytes round-tripping exactly. DEVIATION 795 (its full block is the binding
file's header) is the surface's departure record: a pointer-ABI surface
where upstream has only torch modules; the packed-at-used-stride cache
crossing whole with `cached_tokens` as the one integer state piece and
`max_tokens` as a params scalar; the rotary table and scale rebuilt per call
from the FROZEN constants (eps 1e-6, theta 10000.0 -- never parameters); and
the refusal split (boundary disagreements refused in the binding, everything
else passed down UNJUDGED so the lane's own by-name refusals stay reachable
from Python).

**What is deliberately NOT exposed.** The BACKWARD / training-step profile
(`checks/transformer_backward_check.mojo` compiled and ran 2026-09-03 but has
NOT CERTIFIED -- a surface over an ungated path would be a green label on
nothing); the corpus (a generator with no case data, the phase 7 row); any
multi-layer backbone, checkpoint/`from_pretrained` import, generation,
FlashAttention/SDPA (contract section 6), biases, dropout, masks beyond
causal, or any reduced-precision dtype (contract section 11). GQA IS exposed
(`n_kv_heads`; DEVIATION 813).

**Certificates and rows.** `IDENTITY_PATHS.md` carries NO transformer block
composition row today (the registry stops at the mamba compositions, rows
92-93); adding one, and any certificate wording, is the orchestrator's call,
not this lane's. The three-column forward record this surface stands on is
the status block at the top of this file, and the surface claims nothing
wider than it.

**RUN RECORD, 2026-09-02, APPLE M4 -- the exact commands, in order, all
GREEN.** There is no `pixi.toml` task for surface gates, deliberately
mirroring the mamba surface (its gate runs as a module, not through pixi and
not through pytest); the smoke inside the build script is a smoke, and
`python/mojolearn/tests/test_transformer_surface.py` is the gate. Each tier
was built FIRST and then gated, and the two upper tiers skip the build-time
AIR gate by design (`MOJOLEARN_SKIP_BUILD_GATE`, the sibling scripts' rule),
so the surface gate is their end-to-end verification.

    # 1. the fast tier: build (the FIRST build ever, rc 0, 15 AIR blobs --
    #    transformer 7, gemm 8, mamba 0, core 0, checks 0 -- so the
    #    placeholder floor transformer:1 is now the measured transformer:4
    #    in bindings/build_transformer.sh; minos 11.0 read back), then the
    #    gate
    bash bindings/build_transformer.sh
    cd python && python3 -m mojolearn.tests.test_transformer_surface

    # 2. the deterministic tier (repeat-run bitwise arm asserted)
    MOJOLEARN_NUMERIC_MODE=deterministic bash bindings/build_transformer.sh
    cd python && MOJOLEARN_NUMERIC_MODE=deterministic \
        python3 -m mojolearn.tests.test_transformer_surface

    # 3. the identical tier (all bitwise arms asserted: decode == prefill,
    #    split == whole, the packed cache round-trip)
    MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_transformer.sh
    cd python && MOJOLEARN_NUMERIC_MODE=identical \
        python3 -m mojolearn.tests.test_transformer_surface

All three printed 44 checks, 0 failed -- fast with the bitwise rows REPORTED,
deterministic with the repeat-call row ASSERTED, identical with the DECODE,
RESUMPTION and DETERMINISM arms all bitwise ASSERTED. The build script's own
smoke passed too (forward B2 L4 plus 4 decode steps, worst |step - prefill|
2.38e-07, float64 refused by name, `n_heads*head_dim` refused by name, a step
past `max_tokens` refused in Mojo by name).

One box, one vendor: a green run of all three closes the PyPI row only, and
says nothing cross-vendor beyond what the lane's own cards already say. **The
NVIDIA and AMD columns of this surface are OWED**, as is the corpus
cross-check below.

**What the first run cost, and the NAMED LIMIT it left.** The first fast-tier
run was RED by exactly one check and THE CHECK WAS WRONG, not the binding. It
asked a transposed `q_proj.weight` to be refused by exact shape, but
`d_model == n_heads*head_dim` is `LlamaDims.validate`'s own rule, so `q_proj`
and its mirror `o_proj` are SQUARE at every legal config and their transpose
is shape-indistinguishable. Every projection was probed -- `k_proj`, `v_proj`,
`gate_proj`, `up_proj` and `down_proj` transposes are all refused BY NAME with
an exact-shape message; `q_proj` and `o_proj` transposes are ACCEPTED. The arm
now asserts the trap on the non-square `k_proj` and records the square pair as
a NAMED LIMIT of shape checking -- no exact-shape check can see it.

**Still owed after this run.** The corpus. `corpus/` holds a generator and no
case data, so the gate's float64 reference arm is the gate's OWN transcription
of the contract, not an independent artifact, and the gate's debt row FAILS
until the corpus is generated and compared. The NVIDIA and AMD columns of this
surface. The checkpoint/resume NVIDIA gap. Everything the status block at the
top of this file already lists as owed for the lane itself.

## Two things a reader should not take from this directory

**"Bit-identical across GPUs" is a measured sentence only for paths that have
run on a second vendor.** The FORWARD path has now run on three, and the three
`transformer.identical.card` files are byte-for-byte the same, so for the
forward the sentence is earned and its artifacts are named at the top of this
file. **It is NOT earned for anything else in this directory.** The BACKWARD
lane compiled and ran on ONE column on 2026-09-03 and its gate has NOT
certified, clauses (b), (c) and (e) of the forward are still skipped on all
three, and no sabotage arm was built in the
2026-08-28 round, so what is closed is clause (a) and clause (d) of the
forward profile and nothing wider.

**Identical does not mean equal to PyTorch.** The profile's fold orders,
transcendentals and division are this repository's. What is claimed is that
they give the same bits on Apple, NVIDIA and AMD, and that they agree with a
float64 reference to a stated tolerance once the corpus in `corpus/` is
actually run. Contract section 11.
