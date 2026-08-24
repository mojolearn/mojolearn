# transformer: the cross-vendor bit-identical FP32 transformer block lane

Opened 2026-08-24. The lane's brief is the "After GEMM: the minimum
transformer path" section of `IDENTICAL_GEMM_PLAN.md` (:153-177) and the
build order in `IDENTICAL_SSM_NOTES.md` (:65-88). DEVIATIONS 800-819 are
this lane's.

**The profile is `mojolearn.identical.transformer.fp32.v1`.** The reference
is a Llama-shaped decoder layer, EAGER attention path only, pinned to
huggingface/transformers `d56c55b`. Changing any seam decision, any frozen
constant or the stage list creates a v2; it does not amend v1.

**Status: NOT STARTED beyond this document.** Two files exist,
`IDENTICAL_TRANSFORMER_CONTRACT.md` and this one, plus three empty
`__init__.mojo`. There is no oracle, no fixture, no device kernel, no gate,
no card, no sabotage and no number. Nothing in this lane has been compiled or
run on any device, on any vendor. Every line number in the contract came
from reading source on 2026-08-24.

## What this lane is NOT rebuilding

Section 0 of the contract is the inventory, with file and line citations.
The short version. RMSNorm, the residual add and SiLU come from the mamba
lane and the numerics lane. Every projection AND the `q . k^T` product are
`mojolearn.identical.gemm.fp32.v1` cells, certified three-vendor at
`144aa5b`. Every transcendental comes from `mojo_only/numerics.mojo`.
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
| 3 | `ported/transformers/models/llama/modeling_llama.mojo`, the device spelling. One kernel per stage, MAX's `mha_gpu_naive` shape (one thread owns one score, one thread owns one output row), no shared memory and no warp primitive unless a clause names one | **large** | yes |
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
    pixi run mojo run -I . transformer/mojo_only/transformer_check.mojo

    # both modes, the way the other identity gates do it:
    tools/with_build_lock.sh     pixi run mojo run -I . transformer/mojo_only/transformer_check.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . transformer/mojo_only/transformer_check.mojo

    # Phases 4-6, the device gates and the card. Needs a GPU and the build lock:
    tools/with_identical_mode.sh pixi run mojo run -I . transformer/mojo_only/transformer_check.mojo

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
| `__init__.mojo`, `mojo_only/__init__.mojo`, `ported/__init__.mojo` | empty package markers. |

## Two things a reader should not take from this directory

**"Bit-identical across GPUs" is a measured sentence only for paths that have
run on a second vendor.** No path in this lane has run on a first one.

**Identical does not mean equal to PyTorch.** The profile's fold orders,
transcendentals and division are this repository's. What is claimed is that
they give the same bits on Apple, NVIDIA and AMD, and that they agree with a
float64 reference to a stated tolerance once a corpus exists. Contract
section 11.
