# Source handoff

Branch: `lane/neural-identical-ideas-20261006`.
Forked from local `main` at `35b2d4976550a6a9a47c4563ca1c08b6de21670b`.
Worktree: `/Users/andrewhendel/CascadeProjects/mojolearn-neural-identical-ideas`.
The original worktree's uncommitted work was not copied or edited.

**NOT TESTED — NOT COMPILED — NOT MEASURED.** No compilation, tests,
verification scripts, static checking, device runs, measurements or source
generators for host mirrors were executed. There are no new validation logs
or result receipts. Source reads and file-writing commands are not runtime
or qualification evidence. These changes remain unqualified source.

The primary agent wrote the complete 64-card inventory before fanout.
Implementation was delegated to three source lanes. The primary supplied
dense-profile adapters, grouped resident linear backward, bounded retained
scratch, the A/B harness and the older runner's explicit permission for
cross-version bit changes. No enabled production default was promoted.

New source includes independent two-chain GEMM host/device spellings,
transformer activation/statistics and training cache controls, resident
parameter/gradient/token paths, loss fusion, a versioned optimizer arithmetic
seam, Mamba decay/suffix/gradient schedules, embedding gather/reuse, sequence
LayerNorm/gradient profiles, implicit convolution and optimizer reductions.
Each lane's JSON is the authoritative per-card source scope and blocker list.
The inline controls say NOT TESTED — NOT COMPILED — NOT MEASURED and remain
explicitly opt-in. Historical controls/evidence retain their original status.

| Area | Programmed source and explicit limits |
| --- | --- |
| Dense products | G03 resident dA/dW grouped launch with matched separate-launch B; G04 bounded resident caller scratch; G06 named-plan cache adapter; G09 block traversal adapter; V01 two-chain GPU/host profile. G06/G09/V01 still need full caller wiring. G05 reuses an existing component with retained historical losses. |
| Transformer/training | New A03/A04/A05/T03/T04/T05/V06 arms and T10 combination. Existing arms reused separately. Chunked-head settings still require the actual ByteConfig harness adapter; complete step deferral, broad KV reuse and unsupported replay are not silently marked implemented. |
| Mamba/Samba | M03 decay reuse, M04 resource-based tile arm, M07 launch alternatives, M09 suffix seed reuse, M10 gradient fold. Existing M01/M02/M05 arms retained. M06 ownership lease and M08 affine primitive are source building blocks with caller integration still missing. |
| Embedding/sequence/CNN | Eleven source implementations or A/B controls; five existing arms. Includes feature packets, grouping workspace, recurrent/LN profiles, implicit convolution, MoE grouping and optimizer reductions. E03 owning caller, E06 estimator integration, complete S10 Adafactor oracle and broader V02 transformer/Mamba coverage remain incomplete. S02 stays blocked by documented prior quality failure. |

The 64 cards are a thorough experiment inventory, not a claim of 64 fully
integrated or qualified optimizations. Some cards contain several sub-arms;
their scope is preserved even where only one mechanism has source today.

Remaining work is explicit: incomplete caller integrations and ownership or
toolchain prerequisites, followed only under a future request by compilation,
same-version NVIDIA/AMD/Apple/host identity, task quality, full-dataset paired
timing and applicable board-tool updates. Unsupported Mojo facilities must
wait for Modular; no toolchain workarounds are supplied.

The core contract is same-version equality across columns. A changed
candidate profile may differ from the previous version. Preserving task
quality is a separate requirement and has not been established by programming.

Evidence/source paths:

- `docs/plans/NEURAL_IDENTICAL_EXPERIMENTS_2026-10-06.md`
- `experiments/neural_identical_20261006/catalog.json`
- `experiments/neural_identical_20261006/lanes/dense_profiles.json`
- `experiments/neural_identical_20261006/lanes/transformer_training.json`
- `experiments/neural_identical_20261006/lanes/mamba_samba.json`
- `experiments/neural_identical_20261006/lanes/embedding_sequence.json`
- `tools/neural_identical_ab.py`
- `tools/neural_identical_compare.py`
