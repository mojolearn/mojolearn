# Neural experiment source delivery

The [catalog](../../docs/plans/NEURAL_IDENTICAL_EXPERIMENT_IDEAS_2026-10-06.md)
contains 60 neural A/B ideas. Three delegated implementation lanes and the root
graph-neural/dropout lane produced the following source records. These counts
describe candidate subarms, not completed qualification or every proposed variant.

| Source status | GEMM and CNN | Transformer and training | Mamba and sequence | Graph neural and dropout | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Newly wired candidate subarm, unverified | 14 | 11 | 11 | 5 | 41 |
| Existing A/B implementation reused, unverified here | 3 | 6 | 5 | 1 | 15 |
| Broader design pending or unresolved qualification failure | 0 | 0 | 2 | 0 | 2 |
| Rejected after reading the incumbent source | 1 | 1 | 0 | 0 | 2 |
| Ideas accounted for | 18 | 18 | 18 | 6 | 60 |

New source includes bounded GEMM partial streaming and scratch, an alternative
shared GEMM leaf profile, convolution/tape/activation paths, transformer training
fusion and arena views, explicit host/device CE token reduction, retained Samba
forward tapes, Mamba caching and gradient folds, stable MoE packing, Adafactor
statistics, graph message passing and channel-dropout schedules. The second
source pass also wires grouped Q/K/V, bias epilogues, col2im tap staging, native
chunked loss/head paths and decode workspace reuse. The integration pass adds
NI20 shared native-host/device attention forward and backward, its actual model
and threaded Byte host callers, trace stages and numerical-version metadata.
Every new candidate is opt-in. Existing
defaults remain in place; existing route guards also received narrowly documented
repairs. Nothing has measured speed or established quality/identity here.

## Remaining source work

Two broader cards remain explicitly partial; each now also records a separately
selectable implemented variant:

- **NI38:** a bounded-window, state-parallel S alternative is wired; the original
  version-changing affine scan still needs its carry, checkpoint and backward
  contracts. Select the implemented arm as `NI38=state_window`.
- **NI49:** an independent row-serial scan repair candidate is wired, but the
  historical cooperative-scan training-quality failure remains unresolved.
  Neither this candidate nor the dispatch guard repair proves restored quality.
  Select the separate repair arm as `NI49=row_serial_scan`.

Two proposals were rejected from source findings, without a measured performance
claim: **NI13** has no incumbent weight repacking to eliminate; **NI22** already
bounds visible key spans and skips the proposed invisible score work. Their
records retain the reasoning; no fabricated performance result replaces it.

Wired cards may still have broader variants pending. For example, NI25 delivers
the unchanged-statistics output-parallel norm subarm, NI35 changes token-total
loss while parameter-gradient/residual folds remain pending, NI47 reuses decode
workspace while the wider state-format redesign remains pending, and NI57 defines
a new graph-reduction tree but does not yet parallelize its leaves. See each JSON
record's `remaining_work`; a wired primary arm is not a claim of entire-family
coverage. GEMM and training numerical profiles now have API/checkpoint metadata;
any remaining version metadata named in the lane records is still pending.

## Caller and A/B integration

The existing performance and neural experiment tools now share this catalog:
`tools/performance_ideas.py neural` and `tools/neural_experiments.py ideas`.
The selector supplies distinct A/B controls, explicit variants, build plans and
full-workload queue templates; it never executes those plans. The existing frozen
native builder resolves neural selections from the requested commit. The existing
full-operation queue rebinds controls to frozen records and requires actual recipe,
artifact and quality evidence. New plan/queue code is also unexecuted.

`integration.json` maps all 60 ideas to bindings, host companions and affected
full operation groups. Missing dataset hashes, precise dimensions, cap audits and
quality evidence remain explicit blockers in generated templates. Source-based
rejections cannot be selected; partial parent ideas require an implemented variant.
No unchanged-version diagnostic was weakened to admit a versioned candidate.

Concrete source repairs in this pass include CNN dispatch reaching bounded GEMM
streaming, host-generator treatment of the Mamba state-window helper, and shared
CNN/graph numerical-profile metadata. Runtime work remains native Mojo; Python
changes are API metadata, ownership and experiment orchestration.

## Branch and validation

Worktree: `/Users/andrewhendel/CascadeProjects/mojolearn-neural-identical-ideas-v2`.
Branch: `ideas/neural-identical-20261006-v2`.
Fork: `main` at `fd6cf80453a6f18eb02e81566c824e7da106ccf0`.
The owner subsequently requested integration, commit and push of this branch.
This document accompanies that source delivery; use its containing Git commit
as the revision. No merge into `main` or default promotion is part of this delivery.

**Not run at the owner's request:** compilation, tests, lint, syntax checks,
candidate imports/execution, manifest verification, identity/quality checks and
measurements. The commit/push uses the repository's normal policy hooks; those
are separate from model/kernel qualification. No build/test/GPU logs or
successful-run receipts were created.
The source, catalog and lane records are the retained artifacts. Git bookkeeping
and reading/counting source metadata do not establish code correctness.

Same-version host/NVIDIA/AMD/Apple bitwise identity and task quality are still
required before any promotion. Bits may differ across versions. Full-workload
NVIDIA/AMD A/B, Apple/host identity, interactions and all missing recipes remain
pending. Historical evidence stays historical and no board measurement was added.
