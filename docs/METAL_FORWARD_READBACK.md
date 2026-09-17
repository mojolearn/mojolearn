# Batched forward-check readback

`transformer_check.device_dump` used 30 separate `_download` calls to return
its input, rotary tables and forward stages. Each allocated a host buffer and
waited twice. It now retains 30 logical device views, allocates one packed
host buffer, waits for allocation, enqueues the same 30 copies and waits once
for completion. It then copies each region into the returned lists. Device
views and the host allocation survive through the completion fence.

The stage order and every count expression are unchanged. Attention buffers
are still read at the logical, packed S length, not their allocation capacity.
All 30 host-oracle comparisons remain. Production forward kernels, arithmetic,
and the trace implementation are unchanged; this reduces oracle-check overhead,
not the cost of a training caller that does not download all stages.

The temporary host staging allocation now holds the sum of the logical stages,
rather than one stage at a time. It coexists with the returned lists until the
dump returns. This is the same memory/scheduling tradeoff as the backward dump
in [METAL_BACKWARD_READBACK.md](METAL_BACKWARD_READBACK.md).

## Validation

`transformer/checks/forward_readback_check.mojo` runs the existing
`base_b2_l4_nrep2` fixture twice: first with tracing disabled, then recording
all 30 stages. Both executions compare every stage bitwise with the unchanged
host oracle. A focused check before each execution uses the same dimensions
with cache capacity 8 and logical S 4. It checks attention and cache lengths,
distinct input/score/output sentinels, negative-zero bits, and rotary tables
against independent single-buffer downloads. It requires exactly 30 copies,
two counted waits, one host allocation, and no kernels inside the dump.

Baseline numerical source: `53a3b5154`. The baseline probe changes only the
expected scheduling counts to 60 waits and 30 host allocations; every
correctness assertion is identical to the candidate. Compiles are serialized
through the scheduler with two workers. Each precompiled comparison arm runs
as one job, one fixture, two block executions, with a 60-second total deadline
including queue time. These are forward checks, not optimizer training fits.
No fixture floor or comparison is reduced. The measured scope is Apple M4 /
Metal / IDENTICAL; CUDA and HIP remain unmeasured.

Build from the relevant checkout:

```sh
python3 tools/mac_slot.py run pixi run mojo build -j 2 \
  -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_STEP_PHASE_TIMERS=1 -I . \
  transformer/checks/forward_readback_check.mojo -o /tmp/forward-readback-check
```

Run with a fresh trace path and phase timing unset (it adds waits):

```sh
python3 - <<'PYRUN'
import os, subprocess, time
os.environ['MOJOLEARN_IDENTITY_TRACE'] = '/tmp/forward-readback-check.trace'
os.environ.pop('MOJOLEARN_TRANSFORMER_TIMING', None)
raise SystemExit(subprocess.call([
    'python3', 'tools/mac_slot.py', '--deadline', str(time.monotonic() + 60),
    '--timeout', '60', '--wait-timeout', '60', 'metal',
    '/tmp/forward-readback-check',
]))
PYRUN
```

## Measured result: Apple M4, 2026-09-17

| Measurement | Baseline | Candidate |
| --- | ---: | ---: |
| Waits inside stage dump | 60 | 2 |
| Host allocations inside stage dump | 30 | 1 |
| Device-to-host copies inside stage dump | 30 | 30 |
| Counted waits per complete checked execution | 120 | 62 |
| Host allocations per checked execution | 56 | 27 |
| Device-to-host copies per checked execution | 45 | 45 |
| Kernel launches per checked execution | 90 | 90 |
| Oracle-compared cells per execution | 13,752 | 13,752 |
| Matching oracle stages | 30 / 30 | 30 / 30 |
| Untraced checked execution, ms | 316.016 | 242.124 |
| Traced checked execution, ms | 367.895 | 268.123 |

Both arms passed both executions and the focused readback checks. Their
30-stage cards have the same SHA-256:
`6da67e3255812e6fb624b07fc1a884ee1a7fa41972f1ca71b643c06f35799b60`.
The packed staging allocation is 55,008 bytes for this fixture.

Counters count instrumented calls, not every internal runtime operation.
Both timing samples were lower, but one sample per tracing mode does not
establish a stable speedup or a whole-suite improvement. Traced timing includes
diagnostic overhead. The exact wait and allocation reductions are the primary
performance evidence. Both jobs finished within their 60-second total budgets.

Raw logs, cards, scheduler receipts, comparison script, binary hashes and source
provenance: `bench/results/metal_forward_readback/2026-09-17-apple-m4/`.
Build logs and binaries: `~/mojolearn-evidence/metal-forward-readback-2026-09-17/`.
