# Batched backward-check readback

`backward_device_dump` previously called `_download` once for each of its 37
stages. Each call allocated a host buffer, waited for allocation, copied the
logical device region, and waited for completion: 37 host allocations and
74 counted waits per dump.

The check now retains all 37 logical device views, allocates one packed host
buffer, waits for allocation, queues the same 37 copies, and waits once for
completion. Only then does it copy each host region into the returned stage
lists. Both the views and host buffer are explicitly retained through the
completion fence. Stage order, logical lengths, and bitwise comparisons are
unchanged. No training kernels or production forward/backward code change.

This is a test-overhead improvement wherever `backward_device_dump` is used.
It does not speed a training caller that never downloads these stages. The
tradeoff is a temporary host staging allocation equal to the sum of all
logical stages, rather than one stage at a time. At the measured fixture it
holds 15,328 float32 cells (61,312 bytes). The returned lists already contain
that many cells; they coexist with staging until the function returns.

## Validation

`transformer/checks/backward_readback_check.mojo` runs the existing
`base_b2_l4_nrep2` fixture twice, first untraced and then traced. Every one of
its 37 backward stages is compared bitwise with the unchanged host oracle.

Before each execution, a focused readback check uses the same dimensions with
S capacity 8 and logical S 4. It checks all stage lengths and contents, distinct
sentinels in stages 17 and 36, and negative-zero bits in stage 33. This catches
copies of allocation tails, wrong packed offsets, and floating-point conversion.
It also requires exactly 37 device-to-host copies, two counted waits, one host
allocation, and no kernel launches inside the dump.

The baseline compiles the same probe against `e7a1092ff`, changing only the
expected scheduling counts to its existing 74 waits and 37 host allocations.
The correctness assertions are identical in both arms. Each arm is one
precompiled binary in its own 60-second queue-plus-execution diagnostic round;
compiles are serialized with two workers. No fixture floor or comparison is
removed. Apple M4 / Metal / IDENTICAL is the measured scope; CUDA and HIP are
not measured here.

Build from the relevant checkout (with the compiler admitted through the
shared scheduler):

```sh
python3 tools/mac_slot.py run pixi run mojo build -j 2 \
  -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_STEP_PHASE_TIMERS=1 -I . \
  transformer/checks/backward_readback_check.mojo -o /tmp/backward-readback-check
```

Run the precompiled binary with phase timing unset, since that adds waits:

```sh
python3 - <<'PYRUN'
import os, subprocess, time
os.environ['MOJOLEARN_IDENTITY_TRACE'] = '/tmp/backward-readback-check.trace'
os.environ.pop('MOJOLEARN_TRANSFORMER_TIMING', None)
raise SystemExit(subprocess.call([
    'python3', 'tools/mac_slot.py', '--deadline', str(time.monotonic() + 60),
    '--timeout', '60', '--wait-timeout', '60', 'metal',
    '/tmp/backward-readback-check',
]))
PYRUN
```

## Measured result: Apple M4, 2026-09-17

| Measurement | Baseline | Candidate |
| --- | ---: | ---: |
| Waits inside stage dump | 74 | 2 |
| Host allocations inside stage dump | 37 | 1 |
| Device-to-host copies inside stage dump | 37 | 37 |
| Counted waits per complete checked execution | 164 | 92 |
| Host allocations per checked execution | 70 | 34 |
| Device-to-host copies per checked execution | 58 | 58 |
| Kernel launches per checked execution | 206 | 206 |
| Oracle-compared cells per execution | 15,328 | 15,328 |
| Matching oracle stages | 37 / 37 | 37 / 37 |
| Untraced checked execution, ms | 582.977 | 432.867 |
| Traced checked execution, ms | 696.161 | 528.839 |

Both arms passed both executions. Their 37-stage cards have identical SHA-256:
`f095b9d302a632e9c39d7386d0f0194b3ded20f7b36d6592eff6aca6275e5f35`.
Counters count instrumented calls, not every internal runtime operation.
One sample per tracing mode does not establish a stable speedup or a whole-suite
improvement; traced timing includes diagnostic overhead. The exact wait and
allocation reductions are the primary performance evidence.

Raw logs, cards, scheduler receipts, comparison script, binary hashes and source
provenance are in `bench/results/metal_backward_readback/2026-09-17-apple-m4/`.
Both complete diagnostic jobs finished within their 60-second total budgets.
Build logs and binaries are retained outside Git in
`~/mojolearn-evidence/metal-backward-readback-2026-09-17/`.
