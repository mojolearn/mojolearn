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

    # the backward gate, registered 2026-08-31 at `pixi.toml:1092`. The file
    # exists and carries `def main` at :3787. NO RUN OF IT IS RECORDED anywhere
    # in `bench/results/`, so this command is owed, not reported:
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
| `IDENTICAL_BACKWARD_PLAN.md` | the backward profile's specification. Its gate has not run. |
| `checks/transformer_check.mojo` | the forward gate, 3,172 lines, `def main` at :2860. This is what produced the three cards above. |
| `checks/transformer_oracle.mojo` | the NORMATIVE host forward oracle, 1,502 lines. |
| `checks/transformer_fixture.mojo` | the fixture set, 1,169 lines, shared by the forward and backward gates. |
| `checks/transformer_backward.mojo` | the device backward, 3,042 lines. |
| `checks/transformer_backward_oracle.mojo` | the host backward oracle, 1,584 lines. |
| `checks/transformer_backward_check.mojo` | the backward gate, 4,173 lines, `def main` at :3787. Present and registered; NO RECORDED RUN. |
| `corpus/` | `gen_corpus.py` (85 KB) and its README. The generator is written and, per its own README, has not been executed by its author; no case data is on disk. Its checker is `tools/transformer_corpus_check.py`. |
| `__init__.mojo`, `checks/__init__.mojo`, `impl/__init__.mojo` | empty package markers. |

## Two things a reader should not take from this directory

**"Bit-identical across GPUs" is a measured sentence only for paths that have
run on a second vendor.** The FORWARD path has now run on three, and the three
`transformer.identical.card` files are byte-for-byte the same, so for the
forward the sentence is earned and its artifacts are named at the top of this
file. **It is NOT earned for anything else in this directory.** The BACKWARD
lane has no recorded run on any column, clauses (b), (c) and (e) of the
forward are still skipped on all three, and no sabotage arm was built in the
2026-08-28 round, so what is closed is clause (a) and clause (d) of the
forward profile and nothing wider.

**Identical does not mean equal to PyTorch.** The profile's fold orders,
transcendentals and division are this repository's. What is claimed is that
they give the same bits on Apple, NVIDIA and AMD, and that they agree with a
float64 reference to a stated tolerance once the corpus in `corpus/` is
actually run. Contract section 11.
