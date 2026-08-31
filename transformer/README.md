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

Measured on the M4:

- clause (a) PASS -- 13 fixture cases, 30/30 stages bit-identical to the host
  oracle on all 262,634 cells, 30/30 card tags.
- clause (d) PASS under IDENTICAL -- 4 decode steps bit-identical to the
  prefill on all 11,632 compared cells, with a control showing 57 misaligned
  stage comparisons that DO differ. **It FAILS under FAST** (91 stage-tokens,
  first at token 0 `q_proj.out` on 26 of 32 cells), which is not a defect:
  contract section 7.2 makes decode == prefill true by construction for the
  IDENTICAL profile and FAST promises none of it. `tools/e1_bootstrap.sh`
  therefore runs clause (d) on the identical arm only.
- clauses (b) and (c) are opt-in and not yet run in a round.
- clause (e), the section 8 planted audit, is OFF and it is a defect in the
  clause: it aborts the driver on its first plant because
  `LlamaDeviceWeights` refuses at UPLOAD while the clause's `try` wraps only
  the later forward call. Fix the `try`, then turn it on.

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
| 7 | `transformer/corpus/`, the independent torch float64 per-stage reference, on the `mamba/corpus/` pattern. The only comparison in this lane whose other side is not our own code | moderate | no |
| 8 | the price harness. Wiring, not a published number | small | yes |
| 9 | the three-vendor leg, on `gemm/E1G_RUNBOOK.md`'s pattern. **Until this runs, "bit-identical across GPUs" is not a sentence this lane may write** | operator | three |

Phases 1 and 2 are host-only and are the whole of the contract's falsifiable
content, so they come before a line of phase 3, exactly as the GEMM lane's
charter clause 5 requires. A kernel written against an unreviewed oracle is
what that charter forbids.

## The commands, once the files exist

None of these run today. They are written down so the lane lands on the same
shape every other identity lane uses.

    # Phases 1-2, host only, no GPU:
    pixi run mojo run -I . transformer/checks/transformer_check.mojo

    # both modes, the way the other identity gates do it:
    tools/with_build_lock.sh     pixi run mojo run -I . transformer/checks/transformer_check.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . transformer/checks/transformer_check.mojo

    # Phases 4-6, the device gates and the card. Needs a GPU and the build lock:
    tools/with_identical_mode.sh pixi run mojo run -I . transformer/checks/transformer_check.mojo

    # Phase 7, the independent cross-check, two steps as the mamba lane does it:
    MOJOLEARN_TRANSFORMER_CORPUS_CASE=1 MOJOLEARN_TRANSFORMER_CORPUS_DUMP=<dir> \
        pixi run check-transformer-block
    MOJOLEARN_TRANSFORMER_CORPUS_DUMP=<dir> pixi run check-transformer-corpus

No pixi task is registered. The orchestrator registers it, and the intended
names are `check-transformer-block` and `check-transformer-corpus`, beside
`check-mamba-block` and `check-mamba-corpus` in `pixi.toml`.

## What is here

| file | what |
|---|---|
| `IDENTICAL_TRANSFORMER_CONTRACT.md` | **the deliverable.** Twelve sections. The reuse inventory, the pinned reference, what one block call is, the profile constants, all twenty-three seams with their fused-or-unfused decisions, the softmax reduction order, why FlashAttention and SDPA are out of scope, decode equals prefill, the NaN and signed-zero audit, the thirty card stages, the six gated clauses with thirteen named sabotages, what is not claimed, and where this departs from the plan's sketch. |
| `__init__.mojo`, `checks/__init__.mojo`, `impl/__init__.mojo` | empty package markers. |

## Two things a reader should not take from this directory

**"Bit-identical across GPUs" is a measured sentence only for paths that have
run on a second vendor.** No path in this lane has run on a first one.

**Identical does not mean equal to PyTorch.** The profile's fold orders,
transcendentals and division are this repository's. What is claimed is that
they give the same bits on Apple, NVIDIA and AMD, and that they agree with a
float64 reference to a stated tolerance once a corpus exists. Contract
section 11.
