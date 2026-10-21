# Captured graph optimization source round — 2026-10-20

Internal implementation and qualification record. This is not a public capability
or performance claim. The candidate is opt-in through existing `optimize=True`
execution/inspection paths; the default remains false.

`autograd/program_optimize.mojo` now applies an ordered native pass pipeline:

1. **Explicit reshape chains.** Two consecutive reshapes with positive, explicit
   dimensions and equal element counts become the final reshape of the original
   input. The final count check remains. Inferred dimensions, squeeze, unsqueeze
   and flatten are not folded, so their validation stays intact.
2. **Permutation composition.** Two valid full permutations of the same rank
   compose their axis metadata. Even an identity retains a permutation operation
   to enforce the original exact-rank check on variable-rank inputs. Two identical
   axis swaps become one identity swap retaining the stricter original axis;
   `_swap` still enforces rank at least two and the original axis bounds.
3. **Dead intermediates.** Remove only intermediates made unnecessary by those
   proven rewrites. Reachability includes every output and every other original
   operation as a root. Thus an unrelated original operation, validation, copy,
   detach or random operation does not disappear merely because its result is
   unused. An intermediate that remains an output or has another consumer stays.
4. **Exact duplicate elimination.** Preserve the existing bitwise attribute
   comparison and random/copy/detach exclusions. No floating point algebra,
   reassociation, fusion or reduction-order transformation is introduced.

The captured program is immutable. Training and VJPs replay that original graph,
preserving gradient accumulation and optimizer order. Optimized inference has
independent numerical/finite-difference comparisons against those derivatives.
Parameter references and all input validation remain part of the program; no
parameter tables or checkpoint formats are rewritten by these passes.

Inspection and profile results expose an ordered pass report: node counts before
and after, rewrite counts, original source-node indices, use counts and logical
last uses. This is logical reference liveness, not a storage-recycling policy.
The tape retains responsibility for pending views and buffer aliases. No memory
reclamation workaround is introduced around the confirmed Mojo PythonObject
ownership blocker. Host profiling intervals are not GPU compute timings.

The verifier gains three required cases covering NaN payloads, signed zero,
infinities/subnormals, retained intermediate/repeated outputs, independent output
buffers, variable ranks, invalid reshape/rank refusals and recovery, copy/detach
boundaries, lazy control flow, bounded loops, serialization and derivatives.
The existing post-optimization training oracle now includes transpose chains.
That optimization source freeze enumerated 20 cases and 88 expected numerical
digests. Cheap controller tests validate the reporting and audit machinery;
they do not qualify native execution. Later frozen execution evidence is
recorded below; these historical case counts must not be added together.

The first native optimizer run exposed a preexisting layout inconsistency: eager
and unoptimized materialized permutations/transposes flushed subnormals through
`ld/st`, while identity views and the optimized identity retained them. The
isolated CPU probe preserved this failure. A separate correctness repair makes
forward rank-N strided copies and transposes copy raw words on every executor.
Their VJP routes retain the existing FTZ behavior, as do arithmetic and reductions.
This deliberately changes affected forward bits within the new version and needs
matching Apple/NVIDIA/AMD evidence before admission. It is not a performance
candidate, an optimizer FTZ emulation, or a relaxation of the exact-bit oracle.
The fixture checks eager, original-program and optimized results against the
same original words, including the sign and payload of special values.

Remaining: the parent integrates and compiles changed sequence source once per
freeze, runs native positive/refusal controls, and qualifies installed packages
and required physical vendors. The known standard-library ownership leak remains
a separate blocker; no unsupported compensation or global completion claim is
permitted. Full-dataset end-to-end NVIDIA+AMD A/B, with preserved identity and
quality, is required before any performance default promotion. No such timing
evidence is claimed for this source round.

## Conditional minibatch training source round

`Program.fit_if` is an excluded, unqualified native source addition. It accepts
three captured components in one session: scalar predicate, true branch and
false branch. Input kinds/ranks must agree and each branch has one output.
The predicate runs without derivatives once per minibatch; only its selected
branch is replayed onto the training tape. A union of parameter identities gives
shared parameters one buffer and optimizer state. Parameters absent from the
selected branch do not advance momentum, weight decay or the optimizer step.
Predicate-only parameters cannot be trained through this discrete decision.
The existing optimizer epoch determines row shuffling and conditional dropout
step keys, with disjoint node ranges for the two branches.

Epochs, batches, selection, tape walks, loss aggregation and updates run in Mojo.
Both ordinary and conditional fit now assign public result attributes natively.
The verifier adds independent scalar SGD, shared-state step counts, inactive
AdamW state, zero-update branches, lazy invalid branches, scalar/finite predicate
refusals, session identity and recovery checks. That source freeze enumerated
23 cases and 109 expected numerical digests; subsequent CPU/Metal execution
evidence is recorded below.
This is component composition for conditional minibatch training, not arbitrary
nested graph regions, loop training, differentiated predicates or distributed
control flow.

The optional v2 control checkpoint suffix now records the native session
optimizer's configuration, schedule epoch, seed, individual parameter step
counts, bias-correction scalars and both supported moment buffers. Parameter
handles remain shared across restored components. Existing parameter-only v1
files retain their original format and load behavior. A new optimizer handle is
created on restore; malformed counters/scalars are rejected before allocation,
and partial restores retire their native state. This covers the existing Session
SGD, Adam and AdamW surface; no extra optimizer family is implied. The independent
resume fixture compares uninterrupted three-epoch training against one epoch,
save/load, and two more epochs, including shuffled batches, branch-specific
dropout, changing learning rate, exact parameter/moment/scalar equality, complete
epoch curves and subsequent inference. It also corrupts a step counter while
repairing the outer hash and checks the semantic refusal and recovery.

The parent compiled the combined conditional fit/checkpoint freeze. Stage31
passes its five affected CPU/Metal cases with 34 matching digests and all 11
ordinary Program regression tests. This is source evidence, not an installed
release qualification.
These source changes do not resolve the confirmed std.python ownership leak,
qualify untested physical vendors, or promote any optimization default.

## Broadcast/ReLU motion candidate

The opt-in native pipeline now has one additional source pass,
`broadcast_relu_hoist`: adjacent `expand -> relu` becomes `relu -> expand`.
Only a single-consumer expand that is not a graph output is eligible. The source
must have an exact shape guaranteed by a fixed parameter or a retained explicit
positive reshape. Captured input dimensions and inferred reshapes do not prove
a dynamic shape and are declined. A static broadcast check retains the original
validity domain; the pass only applies when the expansion increases the element
count. This is a cost rule over every positive shape, not dimension targeting.

Both routes execute the existing ReLU body and existing expand. The pointwise
operation is `F(relu(F(x)))`, where `F` is the existing FTZ normalization;
expand either aliases its input or copies through `F`. Since `F(F(x)) = F(x)`,
the reordered operations have the same input/output words. Neither reductions
nor floating arithmetic are reassociated. Source-node provenance moves with the
operations, while the captured graph, derivatives and optimizer order remain
unchanged. The report gives the numbers of ReLU elements before and after for
rewritten nodes. These are work counts, not timing estimates or a speed claim.

Three verifier cases are added for exact special-value words; observed operation
placement and profiled logical output bytes; declined observable/shared/no-op or
ambiguous dynamic expansions; invalid-shape refusal and recovery; finite
differences and original VJPs; lazy branches and bounded loops; plain/control
checkpoint replay; and unchanged Adam training state after optimized inference.
That source registry enumerated 26 cases and 126 expected output digests.
Source/controller checks do not execute or qualify the candidate. Subsequent
CPU/Metal evidence is recorded below; current installed NVIDIA/AMD evidence and
full-workload timing remain pending, and the default remains off.
The existing upstream PythonObject ownership blocker remains unchanged.

## Integer loop-state correction

An isolated probe of the existing sequence binary confirmed that inference
accepted an integer parameter as floating loop state, and both inference and
VJP accepted an int32 argmax wrapped in a reshape. The direct argmax refusal and
floating argmax controls behaved correctly. Original evidence is retained under
`finish-graph-20261020/integer-loop-probe.{py,json,log}` in the evidence directory.

The separate correctness source repair propagates integer-word kind through
captured input/parameter replay and value-layout/selection operations, including
early-return copy/gather paths. Inference checks the executed result before
copying it into the next state; the existing VJP check now observes the kind.
No word conversion or arithmetic change is made, and zero-iteration wrapped
bodies remain unexecuted. Invalid floating loop state raises ValueError while
normal integer embedding/captured-input/checkpoint paths stay available.
The added verifier case checks direct/parameter/wrapped refusals, recovery,
zero-iteration behavior, floating argmax/VJP, eager and captured integer
embedding, and serialization. Registry: 27 cases, 131 expected output digests.
CPU/Metal compilation and the affected numerical/refusal cases pass in the
stage40 evidence below; installed/current NVIDIA/AMD qualification is pending.

## Current frozen evidence and remaining scope

Evidence paths are relative to `~/mojolearn-evidence/core-completion-20261020/`.
Source overlays and previous installed versions are identified separately.

| Freeze / target | Retained evidence | Result and limit |
|---|---|---|
| Stage40 CPU and Metal, source overlay | `stage40-full-graph-{cpu,metal}-summary.json`, `stage40-graph-comparison.json` | Both sequence bindings compile. Each verifier reports 26 passing cases and one failed Session lifetime case; all 128 digests from passing cases match. Includes ReLU/expand motion, integer loop guards, conditional training/resume, layout special values and diagnostics. These are failed full qualification reports, not 27/27 passes. |
| Stage34 installed AMD, actual HIP gfx942 | `retrieved/amd-round2-stage34/graph.json`, `graph-cpu-comparison.json` | 22 passing cases and the same lifetime failure; all 106 passing digests match the corresponding CPU/Metal source. This earlier binary does not cover the later ReLU/expand and integer-kind changes. |
| Stage41 CPU and Metal, source overlay | `stage41-graph-{cpu,metal}-summary.json`, `stage41-graph-comparison.json` | Each run retains 26 passing cases and the known lifetime failure after constructor/native boundary integration; 128 passing digests match. This does not substitute for installed current-vendor qualification. |

The original failures remain retained, including the pre-repair layout word
mismatch and ownership failure. The independent supported-stdlib reproducer
establishes the PythonObject reference-retention blocker; numerical passes and
whole-operation Python traces do not repair it. No compensating reference-count
code, compiler output rewriting or unsupported build mode is part of this work.

Implemented control is explicit composition of captured component Programs.
Predicates are finite scalar, discrete decisions. Executed-path VJPs cover the
selected branch or bounded loop iterations; conditional minibatch training
covers same-session branches and shared parameters. This does not implement
arbitrary nested control regions inside a captured DAG, loop-based `fit`,
differentiated predicates, symbolic Python tracing or distributed control graphs.
Variable-rank replay remains bounded by each existing operation's shape/kind
contract; it is not arbitrary-shape acceptance by every operator.

Optimization currently covers shape-proven view rewrites, pruning of newly
redundant intermediates, exact duplicate elimination, and the bounded ReLU/expand
motion. General fusion, reduction rewrites, a cost-driven scheduler and actual
storage recycling are absent. The original training graph remains unchanged.
No whole-workload performance benefit has been established; all optimization
switches remain opt-in pending full-dataset NVIDIA/AMD A/B and correctness rules.

Diagnostics include node provenance/errors, logical liveness, numerical gradient
checks and executor allocation/copy/synchronization/launch accounting. Their
host intervals and tracked storage do not provide a whole-process memory map,
GPU hardware event timelines or comprehensive distributed critical-path tracing.
Broader operation/composition/derivative and installed-vendor coverage remains
completion work, independently of the bounded additions already exercised.
