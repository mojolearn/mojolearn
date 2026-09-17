# Backward-stage initialization and RMSNorm waits

The retained GEMM workspace change removed per-call scratch lifetime waits,
but `LlamaBackwardStages` still allocated and filled 48 buffers synchronously.
Each fill waited before construction proceeded to the next buffer.

The constructor now queues all 48 fills and waits once before returning.
Every allocation remains owned by a field of `self` through that fence.
All 47 zero fills and the one required ones-vector fill are preserved.
Both full and lean attention-scratch layouts follow this path. Standalone
`_zeros` and `_fill_ones` calls retain their synchronous default, including
capacity growth that replaces existing fields.

IDENTICAL RMSNorm backward also queued three kernels with a host wait after
each, followed by a synchronous GEMM. All buffers are caller-owned, and the
kernels and GEMM use one in-order context without an intervening host read.
Those three intermediate waits are removed. The GEMM's completion wait
remains, preserving completion on return. Other numeric modes retain the
old intermediate fences because their vendor GEMM can return asynchronously.
No arithmetic kernel, operand, launch geometry, or reduction order changes.

## Bounded diagnostic

`transformer/checks/backward_wait_check.mojo` uses one base fixture and two
forward/backward executions. It reads every initialized field back, checking
positive-zero bits and exact-one bits, with full and lean scratch layouts.
It requires one initialization wait and compares all 37 backward stages with
the existing host oracle. Counters are compiled in, but synchronized phase
timing must remain off. These are block forward/backward checks, not complete
optimizer training runs or a certification matrix.

Compile before taking the Metal lease (two compiler workers):

```sh
pixi run mojo build -j 2 -D MOJOLEARN_NUMERIC_IDENTICAL=1 \
  -D MOJOLEARN_STEP_PHASE_TIMERS=1 -I . \
  transformer/checks/backward_wait_check.mojo -o /tmp/backward-wait-check
```

Run one diagnostic job with a 60-second total deadline, including queue time:

```sh
python3 - <<'PY'
import subprocess, time
raise SystemExit(subprocess.call([
    'python3', 'tools/mac_slot.py', '--deadline', str(time.monotonic() + 60),
    '--timeout', '60', '--wait-timeout', '60', 'metal', '/tmp/backward-wait-check',
]))
PY
```

The baseline uses the same probe with only the one-wait assertion relaxed to
allow its old count to be measured. Numerical checks remain active. The
committed regression check has no baseline bypass. Baseline and candidate each
run in a separate single-job, two-execution round with a 60-second deadline;
compilation is finished before either measurement. No broad Apple sweep runs.

Baseline implementation: `dce03921f`. Apple M4 / Metal / IDENTICAL only;
CUDA/HIP validation is pending. The change also benefits callers of the shared
RMSNorm backward helper, but no Samba performance claim follows from this
transformer-only measurement. Binary hashes, scheduler timing/exit records,
and raw diagnostic logs accompany the measured result.


## Apple result, 2026-09-17

| Counter or check | Baseline | Candidate |
| --- | ---: | ---: |
| Backward initialization waits, full / lean | 48 / 48 | 1 / 1 |
| Initialization fill launches | 48 | 48 |
| Counted waits per base forward/backward execution | 225 | 172 |
| Counted kernel launches per execution | 190 | 190 |
| Oracle-compared cells per execution | 13,648 | 13,648 |
| Matching backward stages | 37 / 37 | 37 / 37 |

Both repeats produce the same counters and pass every numerical check.
The 53-wait reduction consists of 47 initialization waits and six intermediate
RMSNorm waits. The latter repeat every block execution; initialization is paid
when stage storage is constructed, not on every resident training step.

Observed forward/backward-plus-oracle times were 631.523 / 561.321 ms before
and 569.366 / 543.947 ms after. Complete diagnostic jobs, including scratch
readback checks, took 1.990 s and 1.898 s, with about 3 ms queue time each.
These are two sequential samples per arm, not a stable speedup estimate. In
particular, the 48 initialization waits only cost a few milliseconds in this
round; multiplying every removed wait by a historical 4 ms would be wrong.
The reliable result is fewer waits with unchanged launches and oracle bits.

Logs, binary hashes and deadline receipts are committed in
`bench/results/metal_backward_waits/2026-09-17-apple-m4/`. Build logs and
binaries remain in `~/mojolearn-evidence/metal-backward-waits-2026-09-17/`.
The optional forward-refusal harness failure documented in
`METAL_GEMM_WORKSPACE.md` is outside this diagnostic and was not rerun.

The next candidates are the per-head gather/GEMM/scatter fences and batching
independent input-validation scans before one host readback. Each needs its
own lifetime/control-flow audit. Refusal order and checks must remain intact;
this change does not defer validation until after state mutation.
