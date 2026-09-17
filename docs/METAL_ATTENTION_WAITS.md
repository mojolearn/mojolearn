# Attention head-loop synchronization

The eager/materialized attention path queued each head as gather, gather,
GEMM, scatter, with a host wait before GEMM and another after scatter. Both
forward scores and the backward attention-weight gradient used this pattern.
Those fences remained after GEMM scratch became stage-owned.

Each loop now submits all heads on the same in-order context and waits once
at the end. The scatter consumes the current head's scratch before the next
head's gather overwrites it. The stages own all buffers throughout the loop.
GEMM workspace growth and plan-internal lifetime waits remain unchanged.
The backward helper still completes before returning; the forward loop still
completes before scaling its scores.

The default query- and key-gradient kernels also no longer wait individually.
Their outputs remain stage-owned, subsequent operations only enqueue, and the
existing final value-gradient completion wait drains all three. No kernel,
operand, shape, arithmetic order, or validation is changed.

This benefits eager/materialized attention, including per-stage correctness
checks and tracing. A fully fused path that bypasses these loops does not gain
this saving. It is not a general claim about every transformer training step.

## Diagnostic scope

`transformer/checks/attention_wait_check.mojo` runs the existing
`base_b2_l4_nrep2` fixture twice: B=2, L=4, two query heads sharing one KV head.
Distinct batches/heads exercise reused scratch. Every backward stage is compared
bitwise with the host oracle. The first execution disables tracing; the second
records all 37 stages for baseline/candidate card comparison.

A separate counter assertion exercises the four-head backward loop on initialized
scratch, requiring all 16 kernel launches and one completion wait. That assertion
checks scheduling cost; the nonzero, distinct-input oracle fixture checks output
correctness. The baseline measurement copy expects its old eight waits; the
committed candidate check always requires one. Existing GEMM growth/plan waits
are not exercised by this small head-dimension fixture.

Both binaries are compiled serially with two workers and IDENTICAL counters:

```sh
pixi run mojo build -j 2 -D MOJOLEARN_NUMERIC_IDENTICAL=1 \
  -D MOJOLEARN_STEP_PHASE_TIMERS=1 -I . \
  transformer/checks/attention_wait_check.mojo -o /tmp/attention-wait-check
```

Leave `MOJOLEARN_TRANSFORMER_TIMING` unset, because phase timers add waits.
Run one precompiled binary per Metal diagnostic round, with a shared 60-second
queue-plus-execution deadline:

```sh
python3 - <<'PY'
import os, subprocess, time
os.environ['MOJOLEARN_IDENTITY_TRACE'] = '/tmp/attention-wait-check.trace'
os.environ.pop('MOJOLEARN_TRANSFORMER_TIMING', None)
raise SystemExit(subprocess.call([
    'python3', 'tools/mac_slot.py', '--deadline', str(time.monotonic() + 60),
    '--timeout', '60', '--wait-timeout', '60', 'metal', '/tmp/attention-wait-check',
]))
PY
```

The diagnostic is two block forward/backward executions, not optimizer training
or a full matrix. Each comparison arm has one job and one fixture. No fixture
floor is reduced and no broad retry follows a failure. Apple M4 / Metal /
IDENTICAL is the measured scope; CUDA/HIP and other shapes remain unmeasured.


## Measured result: Apple M4, 2026-09-17

Baseline source: `c2fbad2d9`.

| Measurement | Baseline | Candidate |
| --- | ---: | ---: |
| Four-head backward-loop waits | 8 | 1 |
| Four-head backward-loop launches | 16 | 16 |
| Counted waits per complete checked execution | 180 | 164 |
| Counted launches per execution | 206 | 206 |
| Oracle-compared cells per execution | 15,328 | 15,328 |
| Matching backward stages | 37 / 37 | 37 / 37 |
| Untraced execution including oracle, ms | 688.355 | 560.105 |
| Traced execution including oracle, ms | 750.075 | 679.946 |

Both arms passed both executions. The recorded 37-stage cards have the same
SHA-256: `f095b9d302a632e9c39d7386d0f0194b3ded20f7b36d6592eff6aca6275e5f35`.
The reduction is seven waits in each of the two four-head loops, plus two
intermediate gradient waits. Existing scratch-growth and plan-internal waits
are not removed. Trace internals are outside the step-counter instrumentation;
the totals above are counted calls, not every runtime synchronization.

The two paired timings were lower, but one sample of each tracing mode cannot
establish a stable speedup or predict a whole-suite improvement. The exact
counter reduction and unchanged numerical results are the primary evidence.
Each complete job finished within its 60-second queue-plus-execution budget.

Committed raw logs, cards, binary hashes and scheduler receipts:
`bench/results/metal_attention_waits/2026-09-17-apple-m4/`.
Binaries and build logs: `~/mojolearn-evidence/metal-attention-waits-2026-09-17/`.
