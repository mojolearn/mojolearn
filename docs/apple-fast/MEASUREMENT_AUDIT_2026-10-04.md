# M3 maintenance overlap audit — 2026-10-04

The queue ran one job at a time, and A/B arms sequentially. However, manager
filesystem scans and cleanup were started outside that queue. This violates
measurement isolation; a serial benchmark queue alone does not isolate the box.

## Identified scan-window jobs

Manager process observations show the long `du` scan (PID55647) running during
the approximately 08:41–08:46 UTC window, until the manager stopped it. These
job-log creation/modification intervals intersect that window. Mark their speed
results **HOLD-measurement**: do not promote or make a firm speed rejection from
these timings. Raw results remain intact; no candidate was promoted from them.
File timestamps bound harness activity, not the exact GPU timed region, and do
not quantify the performance effect of the scan.

| Tag | Workload | Job-log interval (UTC) |
|---|---|---|
| `dbscantaxi-ab-x` | DBSCAN / taxi | 08:22:52–08:43:42 |
| `dlin-qr-dev-istella-b-x` | QR / istella | 08:43:46–08:44:04 |
| `dlin-lstsq-tiled-istella-b-x` | Least squares / istella | 08:44:05–08:44:11 |
| `dlin-nmf-tiled-istella-b-x` | NMF / istella | 08:44:11–08:44:59 |
| `dlin-fa-qrr-istella-b-x` | Factor analysis / istella | 08:44:59–08:45:27 |
| `dlin-svd-cholqr-chol-istella-b-x` | SVD/Cholesky / istella | 08:45:27–08:46:02 |

DBSCAN's baseline timed out independently; it never established a valid win.
The two initial repaired MCD/eigh source-mismatch failures in that interval did
not run scored arms. MCD capped quality later failed fitted-state checks; that
quality evidence is separate from a speed claim.

## Later results and limits of the audit

- Eigh repaired harness: 08:52:32–08:54:20 UTC.
- Cholesky repaired harness: 08:58:42–08:58:47 UTC.
- Depthwise tree A: 09:02:56–09:03:23; B: 09:03:23–09:03:47 UTC.
- Recorded environment deletions: first group begins 09:01:24–09:01:27;
  next group begins 09:04:45–09:05:17. These records are operation starts,
  not complete duration traces.
- Clean-checkout removals begin 09:09:42–09:09:59 UTC.
- Another long scan (PID73994) was observed around 09:08–09:13 UTC.

The logged deletion starts do not demonstrate overlap with the tree's timed
regions. Smaller manager scans/transfers were not comprehensively timestamped,
so this is not proof those later runs were fully isolated either. Keep their
uncertain speed verdicts provisional; do not assert measured interference where
it has not been established. No new defaults were promoted from these trials.

The promoted ARIMA measurements were around 05:06–05:18 and LabelBinarizer
around 05:26–05:27 UTC, before this session's maintenance. This audit does not
implicate those measurements.

Evidence: process observations in the manager session; M3 `~/mq/out/<tag>.log`
birth/mtime; `~/afc-def/gap26-dwcurrent-taxi/run_{A,B}_1.log`; and
`~/mq/cache-cleanup.jsonl`. Do not delete original results or silently repeat
scored pairs. Follow the current run budget and replay policy for any validation.

Required operating rule: heavy maintenance is a serial queue job, or requires
pausing the runner between jobs and confirming worker exit before starting.
See [EXPERIMENT_PROCESS.md](EXPERIMENT_PROCESS.md).
