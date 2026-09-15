# The Holt-Winters lanes after DEVIATION 2717, at 420a8ec73 (2026-09-15)

DEVIATION 2717 (f80e5921a, gate report 420a8ec73): when the BFGS line search
stops at `linesearch_iter_limit`, the device kernel
(`holtwinters/impl/internal/hw_optim.mojo`) and the host oracle
(`holtwinters/host/hw_oracle.mojo`, the tsa CPU host binding) store the
lowest-loss trial, not the last one. Trial 0 starts as the best, and a later
trial replaces it only when its loss is strictly lower, so an exact tie keeps
the earliest trial and a NaN loss never replaces anything. The reference line
search (`hw_optim.cuh:484-508`, rapidsai/cuml#888) stores the last trial. A line
search that exits on the Armijo test is unchanged.

## What moved against the 166-lane record: nothing

The identity_break fixtures fit at the default limit of 100, and no series
there reaches it, so the fix cannot select a different trial on them. Every
Holt-Winters cell of the 166-lane record at 1eea14f80
(`bench/results/identity_break/2026-09-14_166-lanes`) is unchanged by the fix
on every column measured:

| column | file | diff | verdict |
|---|---|---|---|
| cpu, AMD EPYC 9754 (RunPod CPU pod) | `cpu-amd-epyc-9754.json` | `diff.166-vs-cpu.txt`, `--require-columns 4`, lanes holtwinters, holtwinters-multiplicative, par-holtwinters, kpss | IDENTICAL=36 (infer IDENTICAL=27, batch IDENTICAL=36), OWED=0, exit 0 |
| amd-mi325x-gfx942 (DigitalOcean) | `amd-mi325x-gfx942.json` | `diff.166-cpu-amd.txt`, `--require-columns 5`, the three Holt-Winters lanes | IDENTICAL=27 (infer IDENTICAL=27, batch IDENTICAL=27), exit 0 |
| nvidia-rtx-4090-sm_89 (RunPod) | `nvidia-rtx-4090-sm_89.json` | `diff.six-columns.txt`, `--require-columns 6` (the three record columns, cpu, amd, nvidia), the three Holt-Winters lanes | IDENTICAL=27 (infer IDENTICAL=27, batch IDENTICAL=27), exit 0 |
| apple-m4 | not recorded for this fix | | not needed: no cell moved, and the 166-lane Apple column already agrees with the cpu, amd and nvidia columns above |

No new record columns are needed and `python/mojolearn/host_surface.py` is not
changed: the CPU gate's comparison against `TRAINING_GPU_COLUMNS` still holds.

The negative control: the tsa and core host families built with
`-D MOJOLEARN_HOST_SABOTAGE=1` give `diff.166-vs-cpu-sabotage.txt`
DIVERGENT=30, IDENTICAL=6 (the four-column diff exits 1), and both CPU
columns pass `tools/cpu_identity_gate_check.py column`.

## Where the fix does change the fit, and by how much

`holtwinters/checks/hw_check.mojo::check_hw_linesearch_limit_keeps_best` runs
one BFGS iteration at `linesearch_iter_limit` 1, 2, 4 and 8 over 18 fixtures
(504 series) and compares the stored point, on the device and in the oracle,
bit for bit with a host replay that records every trial. From the fit's own
SSE at iteration 0, AMD MI325X under IDENTICAL (`legs/amd-mi325x-gfx942.gate.txt`):

- 504 series ran a line search, 331 reached the limit, and on 229 of those the
  last trial is not the best trial; 0 disagree with the replay.
- On all 229 the stored trial has a strictly lower SSE. The last trials' SSE
  totals 42595.94; the fix lowers it by 8626.22 (20.3%). The largest relative
  reduction on one series is 57.3%.

This is a statement about loss, not speed.

## check-holtwinters in every mode, and the sabotage

| mode | box | verdict |
|---|---|---|
| IDENTICAL | NVIDIA RTX 4090 sm_89, AMD MI325X gfx942 | `== hw_check: ALL OK [IDENTICAL] ==` on both (`legs/*.gate.txt`) |
| DETERMINISTIC | Apple M4 | `ALL OK [DETERMINISTIC]` (`legs/apple-m4.check-deterministic.txt`) |
| FAST | Apple M4 | `ALL OK [FAST]` (`legs/apple-m4.check-fast.txt`) |
| IDENTICAL, `-D MOJOLEARN_HW_SABOTAGE_LS_LAST=1` | Apple M4 | FAILS, exit 1: `check_hw_linesearch_limit_keeps_best FAILED (sabotage LS_LAST): 229 series disagree`, exactly the 229 series whose last trial is not the best; the device stored the last trial (`legs/apple-m4.sabotage-ls-last.txt`) |

Under DETERMINISTIC and FAST the device arithmetic is not the replay's pinned
spelling, so the stored point is RECORDED as disagreeing on all 504 series
rather than raised, as the check's docstring specifies. The census (331 limit
series, 229 separating) is the same in every mode. One wording defect for a
follow-up: the check's closing REPORT line in those modes still says "stored
point == replay selection on 504 series" when it has just recorded 504
disagreements.

A CPU pod cannot run this check: its device kernels need a GPU architecture
(`Unknown GPU architecture detected`), so the three modes ran on GPUs only.

## Legs

| column | runner | billed | deletion |
|---|---|---|---|
| cpu | `tools/runpod_cpu_leg.sh --build core,tsa --sabotage-build core,tsa`, `legs/cpu_cmd.sh` | 182 s, $0.0121 | `terminated_verified=1` |
| amd-mi325x-gfx942 | `tools/do_extra_leg.sh amd --skip-gates`, `legs/body_hw.sh` | about 4.5 min at $3.80/h, about $0.29 | DELETE 204, then GET 404 |
| nvidia-rtx-4090-sm_89 | `tools/gemm_remote_leg.sh nvidia --allow-concurrent`, `legs/body_hw.sh` | about 6 min at $0.74/h, about $0.07 | DELETE 204, then GET 404 |

The NVIDIA gate (`legs/nvidia-rtx-4090-sm_89.gate.txt`) prints the same census
and the same loss figures as the AMD gate, digit for digit: 331 limit series,
229 separating, 0 disagreeing, SSE reduced by 8626.223781585693. An earlier CPU
pod run of this leg with a wrong readback call cost $0.009 and is superseded.
