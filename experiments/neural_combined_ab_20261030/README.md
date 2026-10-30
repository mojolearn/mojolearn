# Combined native-generation NVIDIA experiment

This experiment measures the eight combinations of resident FP32 sampling,
legacy KV-upload reuse, and the native generation engine. Resident forward is
on in every cell. It covers the saved `mojolearn-162m` and `smollm2-135m`
checkpoints, batch one, prompt 128 and 128 generated tokens, with public reads
of 128 and one token. The optional smoke uses prompt eight and four new tokens;
smoke timings do not qualify the full workload.

Each cell loads the same retained checkpoint, records load time separately,
then consumes one excluded warmup and one scored invocation with a fresh stream.
The scored boundary includes stream creation, all public reads, synchronization
through returned host outputs, and final output concatenation. There is no
opponent comparison, noise estimate, or performance-default promotion. Both
vendors and the existing quality/identity acceptance rules remain required for
promotion. The earlier resident-sampler-only slowdown remains a result for its
original settings; this experiment does not reinterpret it.

## What the switches actually select

`bindings/causal_lm_public.mojo::_read` checks admitted resident FP32 sampling
before the native-engine branch. Therefore setting all three flags does not
compose three independent optimizations: resident sampling can shadow the
other two. The KV-upload reuse switch is consumed by the legacy fallback only.
The native-engine branch uses `transformer/native_generation_engine.mojo`.
All three token loops are already compiled Mojo. The distinction concerns
per-token object handling, provider boundaries, transport and residency, not
whether the source file contains a Python `for` loop.

The legacy loop still accesses admitted Python objects and calls a prepared
forward per token. The native engine owns native token and layer loops, but its
provider endpoint still manages CPython GIL state per call. Direct native
CPython calls are permitted; native imports alone do not establish an entirely
CPython-free execution path.

The resident sampling route calls
`providers/python/transformer_resident_pyapi.mojo::transformer_resident_lm_sampling_read`
with native operation three. It admits and reserves the native session group
once per public read, executes its descriptor with the GIL released, and reaches
`transformer/transformer_binding_impl.mojo::_lm_run`. Thus read128 can amortize a
boundary across 128 tokens, while read1 still crosses it for every token. This
is a source call-path finding, not a timing result or a blanket guarantee about
all possible model/callback admission paths.

## Executed-route and retention evidence

A private opt-in native trace prints only after a successful completed read.
Trace is on during the excluded warmup and off during scoring. The controller
requires the exact number of completed reads, capacities and cumulative token
counts; admission metadata or requested flags alone cannot pass that check.
`retained_kv` in this diagnostic describes the legacy KV-reuse branch only;
a false value on a resident route does not mean resident KV storage is absent.
The trace helper itself adds one environment lookup per public read even when
printing is disabled; these measurements use that same instrumented image in
all arms.

Source bytes are archived through `tools/bench_board_evidence.py` before timing.
Each cell retains inputs, generated tokens, full logs, checkpoint arrays and
untimed teacher-forced logits/NLL. Source and cell archives must pass R2 PUT and
GET hash verification before the next measurement. A retention failure stops
the campaign, preserving completed measurements for retention-only recovery.
Required bytes also return to the Mac before rental teardown. Failed or
unexecuted cells remain explicit; finite model NLL is a limited same-model
check, not independent statistical quality qualification.

The first setup attempt stopped before compilation because its subprocess PATH
omitted installed Python 3.14 and pixi required by the pinned Unicode generator.
Its complete failure archive is retained. The next attempt fixes PATH and
reuses unchanged downloaded artifacts. Compiler, runtime and measurement
outcomes are recorded separately in retained receipts; see the resulting
handoff and coverage records rather than assuming this document is a pass.


## Explicit route follow-up (separate freeze)

`--matrix routes` records 44 requested cells: legacy FP64/FP32 × KV off/on,
resident FP32 × KV off/on, native engine FP64/FP32 × KV off/on, and the complete
auto configuration, each for two models and two read sizes. The saved 162m
architecture has no final norm, while resident admission requires one: its four
explicit resident cells remain unsupported and unexecuted. No norm is inserted.
Native-engine arms explicitly disable resident-forward to satisfy admission;
the complete auto arm records whichever branch actually runs for that model.
Expected supported coverage is therefore 40 cells, separate from 44 requested.

The first frozen matrix exposed a harness issue: forced engine-on could allow
resident sampled generation but reject the separate full-prefix quality call.
The follow-up performs the untimed quality forward with engine off,
resident-forward enabled and route auto, and retains those settings. Original
quality failures are preserved; missing quality can be recovered from retained
output bytes without rerunning completed timings. Descriptor operation counters
are captured immediately before and after timed calls, outside their clocks.

## Final native NVIDIA/AMD observations (2026-10-30)

The frozen `a85b4a21cb3c` bounded campaign now has 500 selected native
observations per vendor and 500 passing NVIDIA/AMD column comparisons. NVIDIA
uses 472 fresh observations, 26 with supported source-equivalence proofs and
two direct current-source CNN observations. AMD uses 429 fresh, 69 proven
reusable and two direct CNN observations. All selected archive bytes and full
comparison evidence are retained on the Mac and R2. See
`results/a85-final-native-500.json` for exact receipts and remaining gates.

This does not qualify Apple, the separately required PTX route, physical
multi-GPU execution, central ledger admission or publication. The native-only
NVIDIA qualification wheel correctly refuses the missing required PTX slot;
no packaging gate was relaxed. Historical numerical failures remain retained.


## Required PTX route (2026-10-30)

The same frozen `a85b4a21cb3c` source now has 500 passing PTX algorithms,
4,500 canonical fixture cells, and 500 passing three-column comparisons with
retained native NVIDIA and AMD observations. Execution used two A40 GPUs
(actual SM86, driver 595.91.07) through the explicit SM80 PTX route. The
changed `x_sequence` PTX group compiled successfully; 32 unchanged groups
were reused with supported source/build proofs. All default identity parts
were collected, with declared N/A coverage preserved. Exact counts, artifacts,
dependency-transition evidence and Mac/R2 receipts are in
`results/a85-ptx-final500.json`.

The common core uses the exact AMD helper bytes after a bounded 402,992-value
C-ABI comparison and the actual PTX/native/AMD campaign. Original native
NVIDIA helper witnesses remain unchanged. Metadata tooling repairs admit the
producer's schema 2 and exact registered lane directories while preserving
partial-run and provenance refusals; 126 admission tests pass. Raw records
and earlier admission failures remain retained.

This completes this PTX identity scope, not the separate installed runtime,
physical multi-GPU, Apple, strict packaging, central ledger or PyPI gates.
The retained candidate wheels remain explicitly non-publishable qualification
artifacts. No performance default changed.
