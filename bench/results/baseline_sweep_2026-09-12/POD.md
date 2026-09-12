# The box this board was measured on

**Committed so the pod is findable by someone who is not the session that
rented it.** This run wrote its lease state to `$HOME/mojolearn-evidence/`
(outside the repo) via `TREES_LEG_STATE`, which bypassed
`bench/results/runpod_leases/`. That ledger exists precisely so an orphaned box
can be found and reaped; a pod discoverable only from inside one agent's
context is a single point of failure that bills until a human notices.

| | |
|---|---|
| pod id | `9imx21xh3yuqd8` |
| provider | RunPod, SECURE |
| GPU | NVIDIA H100 80GB HBM3 |
| rate | **$3.49/hr** |
| ssh | `-p 10864 root@103.207.149.101` |
| rented | 2026-09-12T21:10:37Z (billing starts at create) |
| watchdog armed | 2026-09-12T21:11:28Z |
| lease | **240 minutes** |
| expires | **2026-09-13T01:11:28Z** |
| lease file | `bench/results/runpod_leases/9imx21xh3yuqd8.lease` |
| state dir | `$HOME/mojolearn-evidence/baseline-sweep/pod` |
| commit shipped | `06341dbf` |

## The lease bound was chosen before renting

Rentals here default to a **one-hour** dead man and this sweep needs roughly
3.4 hours, so the bound was set deliberately at rent time rather than
discovered at minute 61:

    tools/trees_leg.sh rent --gpu "NVIDIA H100 80GB HBM3" --minutes 240

`tools/trees_leg.sh` accepts a longer bound (`tools/gemm_remote_leg.sh` is the
one that hard-refuses anything over 60). The watchdog runs ON the pod and
DELETEs the pod through the API at the deadline, so the box cannot outlive the
work even if the renting machine disappears.

## To reap it by hand

    MOJOLEARN_RUNPOD_KEY_FILE=~/.mojolearn_runpod_key \
    TREES_LEG_NAME=mojolearn-baseline-sweep \
    TREES_LEG_STATE=$HOME/mojolearn-evidence/baseline-sweep/pod \
      sh tools/trees_leg.sh reap

which DELETEs and then verifies the pod is gone (HTTP 404, 8 attempts). Or
directly:

    sh tools/runpod_guard.sh reap --force 9imx21xh3yuqd8

**Do not reap while cells are owed** (ENGINEERING_RULES section 11): the box is
up because the numbers are owed, and the expensive half of a leg is the front
half -- dataset staging, pip, and the builds -- which the next attempt would
pay again from zero.

## Teardown, verified

    [22:43:26] terminating 9imx21xh3yuqd8
    [22:43:27]   DELETE -> HTTP 204
    [22:43:27] VERIFIED: 9imx21xh3yuqd8 is gone (HTTP 404)

Reaped at **2026-09-12T22:43:27Z**, about 2h28m into a 240-minute lease, after
the box was confirmed idle and after every artifact had been pulled off it and
pushed. `trees_leg.sh reap` does not trust the DELETE: it polls `GET /v1/pods`
up to eight times and only reports success on a 404 or a TERMINATED status,
because "the DELETE returned 200" and "the pod is gone" are different claims.

Nothing was reaped while a cell was owed. The order was: 28/28 cells, then the
two automatic owed items, then the criteo A/B, then the evidence pull, then the
commit and push, and only then the terminate.

The lease file now lives in `bench/results/runpod_leases/reaped.d/`, which is
where this ledger keeps boxes that are confirmed gone.
