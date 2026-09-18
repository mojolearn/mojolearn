# Attention fallback: exact replay and bounded storage

The NVIDIA default passed paired 700-step training on both pinned R2 corpora.
Every loss and all six FP32/state witnesses at steps 0 and 699 match legacy.

| B1 corpus | Legacy tail seconds | Default tail seconds | Speedup | Device MiB before → after | Backward CORNER before → after |
|---|---:|---:|---:|---:|---:|
| enwik8 | 0.455438880 | 0.197156515 | 2.3100x | 31537 → 15153 | 3697 → 0 |
| Pile GitHub | 0.377133856 | 0.196830695 | 1.9160x | 30257 → 15153 | 1768 → 0 |

Two-corpus geometric-mean speedup: 2.1038x. All 8400 forward observations
per corpus/arm are RAN; default backward is also 8400 RAN per corpus. Zero
regime refusals. Default eager storage is 432 bytes after every step;
aexp stays separate at 2415919104 bytes. Legacy eager storage ends at
17314086912 bytes (English) / 15871246372 bytes (code).

B4 enwik8 completed 2000 steps with finite losses and zero refusals in all
24000 forward and 24000 backward observations. Last-200 median 0.664205971
seconds (12333.5 tokens/s); sampled device peak and tail 38959 MiB. Eager
storage stays 432 bytes; aexp stays 9663676416 bytes. dQ replay is active in
13609 layer-steps; zdot replay is INERT in training.

The separate same-H100 B1/B4 comparison also completed 700 steps per arm.
B4 legacy attention with bounded release and B4 default match every loss and
all six state witnesses at steps 0 and 699. B4 backward CORNER 3172 → 0;
actual dQ replay sites 2666. B4 tail 1.564467359 → 0.659652365 seconds
(2.3717x), sampled device peak 50481 → 38961 MiB. On this SAME device,
default B1 is 0.197901128 seconds (10348.6 tokens/s), default B4 is 12418.7
tokens/s: **1.2000x additional throughput from batch 4**. The registered
batch timing, memory, survival and multiplier predictions passed. Larger
batches were not measured; no saturation or maximum-batch claim is made.

The release-only English control also matches all 700 losses, each layer's
original statuses, and all state hashes: tail 0.461724 seconds, sampled peak
18497 MiB, eager storage 432 bytes after every step. It exercises automatic
returns to fused execution after freeing scratch. It is a storage win, not a
speed win. Default release is INERT when replay prevents all eager calls.

NVIDIA and AMD match at all 130 explicit native repair/preservation sites,
with observed failing controls; both native broad gates pass 15 cases.
Apple has compile and host-mock evidence only, no GPU run. Raw comparisons,
negative controls, source hashes, stage logs and verified teardown records
are under [the evidence directory](../../bench/results/lm_attention_fallback_2026-09-18/).
The numerical implementation tested is source 57cf16d61; the same-device
B4 comparison used f3eeb0c35 with the same numerical implementation. Later
edits are documentation, comparison tools and the probe metadata-label
correction. All rented test pods/VMs have verified teardown records. The
mechanical verdict is [FLIP NVIDIA](../../bench/results/lm_attention_fallback_2026-09-18/default/flip_verdict.log);
its deliberately doubled-time control returned NO FLIP first.

## Initial registration (historical)

Instrumentation branch lane/lm-attention-fallback at 576ee02bf. Main was
3e8dabc37 when first inspected; concurrent checkout activity advanced the
actual parent to 1863520e9 (which contains that main). Continued work is
isolated at ~/mojolearn-wt/lm-attention-fallback on
lane/lm-attention-fallback-work. Keep any eventual merge limited to this
work; unrelated reference-regeneration changes are not part of the task.
No optimization yet.

Registered before the first run: H100 80GB HBM3, seed 20260917, pinned R2
enwik8 (100000000 bytes, SHA256
2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8),
700 steps, B1 L2048 DM768 H12 KV12 HD64 FF2048 layers12 V50257.
Prediction: 8400 forward and 8400 backward status observations; at least one
nonzero refusal by step 699. Candidate status 2 is a hypothesis, not a cause.
Zero refusals makes this run INCONCLUSIVE about the reported slowdown;
any status 1 falsifies the hypothesis that every refusal is status 2.
Expect initial eager_bytes 432, eventual 17314086912 if all layers grow;
aexp is separate (2415919104 bytes). Expect head median 0.18–0.26 seconds
(exclude step 0), tail 0.38–0.52 seconds if the transition completes.
Those ranges are predictions, not acceptance requirements or new measurements.

Instrumentation saves the actual forward/backward launcher return codes in
host fields. Session info appends triples per layer after the existing ten
entries: forward status, backward status, materialized. NOT_ATTEMPTED=-1,
FUSED_RAN=0, FUSED_REFUSED_REGIME=1, FUSED_CORNER=2. The materialized flag is
read after backward, which can itself have recomputed forward stages.
No arithmetic or dispatch changes. Historical bindings omit the new fields.

The body refuses missing, wrong-sized or wrong-digest corpus bytes. Run with
MOJOLEARN_STAGE_STRICT=1 and inspect stage.log for successful R2 staging.
Controls at HD64 force eager (-1) and fused (0); the validator deliberately
corrupts coverage, counts, status and sequence and must reject each by name.
No identity claim is made by these metadata checks.

Local compile: PASS, Mojo one worker (`MOJOLEARN_COMPILE_JOBS=1`), nice 19,
no GPU execution. Output /tmp/mojolearn-attention-trigger-compile.
Host report extraction: MATCH forward [0,1], backward [2,0], materialized
[true,false], eager_bytes 72, aexp_bytes 800. Deliberately changing the
forward status to [2,1] failed the equality check as expected. Historical
ten-field bindings omit the unavailable status fields. This is metadata
validation, not a numerical identity result.

First leg: pod 13velviiqay5hu, 60-minute guarded lease, source 576ee02bf.
R2 stage.log confirms 100000000 bytes and the registered SHA256, one key
staged and one linked. Baseline and forced controls remain owed until the
remote result files complete.

## Trigger measured, step 1 complete

700 completed steps on H100: forward 8400 RAN; backward 4703 RAN and 3697
CORNER; zero REFUSED_REGIME in either direction. Per-layer counts and first
refusals are printed in trigger/verdict.log. First refusal step69; all twelve
layers grown by step456. Head median (steps1–50) 0.198942478s; tail50 median
0.458373688s. Device samples 15151→31535 MiB. Eager 432→17314086912 bytes;
aexp constant 2415919104. Six deliberate corruptions failed the report
validator, and forced HD64 eager/fused controls printed [-1,-1]/[0,0] in
both directions. The completed pod was deleted and verified gone.

## Step 2 trial: exact masked-tail guard, predictions before running

The hd64 r2 joint dk/dv kernel flags every negative zero, even after the
last head with no masked suffix. Trial MOJOLEARN_ATTN_EXACT_TAIL_GUARD
restricts that flag to hi < L-1 OR (a next head exists AND lo > 0).
The initial skipped prefix starts at +0 and cannot change that seed; later
head prefixes may change a preceding -0, so those retain the refusal.
No zero canonicalization, arithmetic, leaf or fold changes. Other kernels
retain their original conservative guards. Do not attribute the measured
CORNERs to this site until the intervention discriminates.

Predict, at the registered 700-step target and seed: all 700 loss values and
six final-step hashes equal the legacy arm; guard runtime witness false vs
true; backward CORNER count falls below 370 (90% of 3697 removed); tail median
below 0.26s if these conservative flags dominate. More than 370 refusals or a
slower tail falsifies the performance prediction; any unequal bit rejects
the trial. A remaining fallback is not proof this site accounts for it.
Adversarial checks must retain real masked-tail refusals and detect deliberately
canonicalizing a final negative-zero dk cell to positive zero. Any inert
sabotage will be reported INERT and the fixture improved before trusting it.

## Step 3 implementation prepared; measurements wait for step 2 result

Trial MOJOLEARN_BYTE_LM_RELEASE_EAGER drops the nine eager-only buffers
immediately after each layer's existing backward completion fence. Replaces
only grown buffers with one-cell placeholders; keeps aexp and all dx/dw.
Reports released_eager_bytes per gradient step plus runtime release_eager.
attn_materialized becomes false after release (buffers no longer valid);
forward/backward statuses still identify what actually ran during that step.

Predictions registered before the release/endurance run: legacy vs release
with the exact-tail guard OFF in both runs must match all 700 loss witnesses
and the six hashes at steps0,699. Refusals must still occur (>0) and released
bytes must be >0, otherwise the memory trial is INERT. At every completed
release step eager_bytes must equal 432, aexp stays 2415919104. Predict tail
device occupancy <20000 MiB (legacy ~31535); >=20000 falsifies the footprint
prediction despite buffer-length success. No throughput gain is promised
for release alone, which re-allocates on every real fallback.

Then the combined trial at B4 must complete 2000 steps with finite losses;
predict tail <0.85s/step and device memory <65000 MiB, eager_bytes=432 at every
step, aexp=9663676416. OOM, missing steps, nonfinite loss or any failed identity
gate rejects qualification. This establishes B4 survival, not the maximum
possible batch; larger batches remain unmeasured.

## Exact-tail guard result: performance prediction FALSIFIED

Second H100 leg, commit305d6a116, podqlbxdtkpfegy6y deleted/verified gone.
700 losses and all six hashes at steps0,699 MATCH. Native sabotage changed
accepted dk from eager 0x80000000 to 0x00000000 and FAILED. Clean negative-zero
fixtures and all 15 fused/eager cases passed. Counter intervention removed
248/3697 refusals (6.7%), leaving3449. Legacy tail0.461326122s vs guarded
0.460533116s, same final31537MiB and17314086912 eager bytes. This does NOT
meet the registered370-refusal/0.26s prediction. Do not promote this guard
for a speed claim. It remains a disabled experimental build option.

## Revised step 2: pre-launch policy from the observed corner

Trial MOJOLEARN_BYTE_LM_STICKY_EAGER: on a layer's first actual FUSED_CORNER,
remember prefer_eager for that resident layer. Subsequent AUTO calls choose
existing eager forward AND backward before either fused launch. Explicit
`attention-path=fused` overrides the preference. No numerical kernel changes.
This is a policy from observed history, NOT a prediction of a corner or an
attribution to an unisolated kernel site. The first refusal still pays both;
each layer pays it at most once per resident session under AUTO.

Three 700-step target arms, same pinned corpus/seed, before B4 endurance:
baseline (both flags off), sticky (release off), released (both flags on).
Exact-tail-guard stays OFF. Predict all700 loss witnesses and six hashes at
steps0,699 match; forward status counts for sticky/released 3012 RAN and
5388 NOT_ATTEMPTED; backward 3000 RAN,12 CORNER,5388 NOT_ATTEMPTED, assuming
all twelve first refusals retain their measured indices. Tail median target
0.18–0.32s; >0.32 falsifies the speed prediction. Any unequal bit rejects.

Release additionally drops dead scores/masked/sbh after the forward fence,
keeping weights+aexp for backward. This avoids holding all layers' forward
scratch when the policy chooses eager before forward. attn_materialized now
means backward weights are valid; it becomes false after backward release.
Predict release eager_bytes=432 after EVERY step, aexp=2415919104, positive
released bytes on exercised steps, and device tail <21000MiB. Higher device
memory falsifies the footprint prediction even if the buffer lengths shrink.

## Policy/release result and registered masked-tail replay trial

H100 policy run completed all three 700-step arms. All losses and six full
checkpoint hashes at steps 0 and 699 match. Baseline/sticky/released tail medians
were 0.457512557 / 0.455645248 / 0.462906826 seconds; device tail was
31537 / 31281 / 21809 MiB. Sticky statuses exactly matched the registered
3012 forward RAN, 3000 backward RAN, 12 backward CORNER, and 5388 skipped
launches in each direction. This disproves the predicted throughput benefit
of sticky routing on this workload. Release returns eager storage to 432 B
every step, but its device prediction <21000 MiB failed (allocator retains
more than live arrays). aexp stays separate at 2415919104 B.

Before running the next experiment: replay omitted masked-tail operations only
when the fused zdot or dq fold ends at negative zero. Preserve the original
ascending RN-FMA then FTZ-multiply chain; stop replay only after +0, which
remaining finite signed-zero terms cannot change. Do not canonicalize signs.
Prediction at the same seed and target shape over 700 steps: 8400 forward and
8400 backward RAN statuses, zero refusals, at least one reported repair site;
eager storage 432 B, aexp 2415919104 B, tail median 0.18–0.27 seconds. Any
refusal falsifies the zero-refusal prediction; no repair observed is INERT.
Any differing loss/checkpoint bit rejects the arm. Independently skip the zdot
and dq replay and require each adversarial native HD64 gate to fail with the
expected differing zero bits before accepting the clean gate. Existing dk/dv
masked suffix and grouped-prefix cases must still refuse. Run baseline and
repair arms for 700 steps each on one strictly R2-staged H100. Sticky and
release remain OFF to isolate this repair.

Before AMD native correctness gates: Hot Aisle availability probe found two
13-core MI300X instances available and zero of two slots occupied. Predict
zdot and all 64 dQ cells equal eager at exactly 0x00000000 in the repair
fixtures and 0x80000000 in the preservation fixtures. Skipping either repair
must fail, and corrupting either preservation output to +0 must fail. Force
the NVIDIA estash schedule through the existing every-column check define
so the AMD broad gate is not an untouched-default comparison. These are
native arithmetic checks, not a short training benchmark or AMD speed claim.

Final qualification, conditional on the replay trial succeeding: paired
700-step legacy/default training on both R2 enwik8 and Pile GitHub, followed
by B4 enwik8 2000-step endurance. Prepared scripts fail their arm witnesses
until the default is explicitly flipped; no current default flip is claimed.

## Replay trial verdict and final predictions (registered before final runs)

The paired H100 trial at 42d225229 completed 700 steps in each arm. Every
loss and all six state hashes at steps 0 and 699 match. Backward statuses:
legacy RAN 4703 / CORNER 3697 / REFUSED_REGIME 0; repaired RAN 8400 /
CORNER 0 / REFUSED_REGIME 0. Repair masks: 4951 none, 3449 dQ, zero zdot.
Thus zdot replay is INERT in this training experiment; its native fixture
is active. Tail median 0.456854569 -> 0.199255138 s (2.293x); device tail
31537 -> 15153 MiB. Eager bytes 17314086912 -> 432; aexp remains
2415919104. Every registered replay prediction passed.

AMD MI300X native gates at 190cebf92 passed all 15 broad cases under the
forced estash schedule, repair and preservation fixtures, and four observed
negative controls. The AMD run is native correctness evidence, not a
700-step training or performance claim. Apple received source compiles only.
Both completed rented VMs were deleted and verified absent.

Enable replay and buffer release in the NVIDIA matrix rows on this branch;
sticky routing stays OFF. Keep explicit legacy defines for repeatable A/Bs.
Before merging, run paired legacy/default 700-step arms on enwik8 and Pile
GitHub. Predict zero default refusals, all exact loss/checkpoint witnesses
matching, 432 eager bytes, 2415919104 aexp bytes, late median 0.18–0.27 s
on EACH corpus. Any bit difference rejects the default; any missing active
repair is INERT for that corpus, not evidence of unchanged repaired bits.
Also run a 700-step legacy-arithmetic/release-only enwik8 arm, sticky OFF:
predict original per-step statuses and bit witnesses, but 432 retained
eager bytes after every step. This exercises return to fused after release.

B4 enwik8 prediction: complete 2000 steps, finite losses and exact step
sequence, 432 eager bytes, 9663676416 aexp bytes; sampled device peak below
70000 MiB and last-200 median 0.55–0.90 s. Predict throughput 1.1–1.5x
B1 default on the same corpus; <=1.1x falsifies that multiplier prediction.
B4 failure or OOM is a failed endurance prediction, never a shorter pass.
The final runs will finish even if an early timing observation loses.

Before the same-H100 batch comparison: run B4 legacy arithmetic with bounded
release for 700 steps, then default B1 and B4 for 700 steps each on one
device. This directly checks B4 bits against eager fallback while keeping
its memory bounded, and avoids attributing a cross-pod timing difference
to batch size. Predict the B4 loss/state hashes match exactly, legacy
release actually executes and keeps 432 retained eager bytes, default B4
has zero refusals and active replay. Timing predictions remain B1 0.18–0.27s,
B4 0.55–0.90s, same-device throughput multiplier 1.1–1.5. A mismatch rejects
the candidate; an INERT arm proves nothing about that mechanism. This
700-step comparison is separate from the already-running 2000-step B4
endurance obligation, which will finish.

## Probe metadata erratum

The historical probe hardcoded `run_metadata.data_schedule.batches` to
"synthetic uniform token ids, no corpus", including when its `--corpus`
argument loaded CorpusBatches. This is a label bug: run_steps passes the
loaded corpus to ids_for, which returns corpus.ids(index); every raw result's
outer `corpus` field holds the actual digest and schedule, and `ids` says
"pinned corpus". Strict R2 stage logs and on-box checksum checks independently
verify those bytes. Preserve the original evidence, including that incorrect
inner label. The probe now passes the corpus description into trainer metadata
at construction. No training inputs or numerical operations change. All
experiments through source f3eeb0c35 used the old label.

Per-layer three-arm attribution now checks the actual tuples, not only their
counts: 248 legacy CORNERs disappear under the dk/dv guard alone, and all
remaining 3449 guarded CORNERs correspond to dQ replay sites in the accepted
repair arm. Zero zdot sites in training: INERT. The checker prints each
matched step/layer, rejects missing dQ activity, invented zdot attribution,
a missing layer, blind head_dim=8, and each corrupted checkpoint hash.
This identifies guards/replay activity, not how many individual zero signs
changed in training. Native fixtures separately prove both sign outcomes.

Host-only compatibility check: 44 byte-LM surface tests pass in 0.89s, using
the freshly compiled Apple binding only for import and mocked training.
No Apple GPU execution.


## Final main integration: registered before the run

Main 5025376d5 independently merged the initial status instrumentation and
three disabled research arms. Preserve those arms and their loaded-binding
witnesses; preserve its safe partial-stage reporting after a failed backward.
The canonical session-info layout now includes both list lengths before the
triples, followed by the qualified policy/replay fields. The actual Python
method passes full/short/empty-list fixtures, after old-offset, missing-bound
and wrong-repair-offset mutations fail. Main's historical scripts explicitly
disable later defaults when reproducing their original arms. Their source
records remain intact. No repair kernel arithmetic or fold order changed.

Prediction: the merged-source default on strict R2 enwik8, seed 20260917,
HD64 target B1, 700 steps, matches all 700 saved legacy losses and six state
hashes at steps 0/699. All 8400 forward/backward statuses remain RAN; the
per-step repair masks match source 57cf16d61 exactly (3449 dQ, zero zdot).
Eager capacity remains 432 B, aexp 2415919104 B. Expected tail median
0.19–0.23 s and sampled device footprint below 16000 MiB. A single different
bit, status or replay-site observation falsifies numerical integration;
exceeding either timing/memory bound falsifies retained-performance prediction.
Commit/push this merged source before renting. Native replay sabotage must
fail for both sites, then the clean native gate and the 700-step run complete.
