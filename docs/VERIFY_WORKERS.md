# Installed verifier execution

`python -m mojolearn verify --all --json-out evidence.json` runs cells
sequentially in batched worker processes: up to 24 cells (`--batch-cells N`)
share one process, so interpreter, import and GPU-runtime startup is paid once
per batch rather than once per cell. Each batch's native context is dropped
with its process, which bounds accumulated GPU state, and the sweep never
runs independent fits on the same GPU at once. Fixture generation, repeats
and enabled probes remain exactly those of the reference protocol; batching
does not shrink input arrays or weaken checks, and a cell's value does not
depend on the execution mode. `--fresh-process-per-cell` runs every cell in
its own process, for diagnosis: comparing that run with a batched one finds
any lane whose result depends on what ran before it.

Before fitting lanes, an isolated comparator self-test must demonstrate that
an untouched input matches and the perturbed input diverges. Portable model
checks and optional smoke checks also use bounded fresh workers. Each cell
has a 120 second limit, including interpreter startup for the first cell of
a batch and cleanup. Override it with `--cell-timeout SECONDS` when
investigating a legitimately longer cell. A timeout is an incomplete
verification, never a successful skipped check.

The parent announces each cell as its result lands, prints a heartbeat every
ten seconds while a cell runs, and judges its parts immediately. Ordinary
divergences remain visible while subsequent cells run. When a batch worker
crashes, times out, reports an unhealthy device or returns a malformed result,
the cells it already answered stand, the stop is listed under
`execution.batch_stops`, and the batch's remaining cells (the failing one
first) each run in a fresh process, listed under `execution.fallback_cells`;
the next batch is batched again. A fresh cell worker that crashes (non-zero
exit or signal, without a timeout or an unhealthy-device signal) refuses its
own cell, is listed under `execution.crashed_cells`, and the sweep continues;
three crashes in a row stop it. A fresh-process timeout, malformed worker
result, explicit unhealthy-device signal or a crash outside the cell sweep
stops the sweep. Outstanding cells become refused, and the overall report
cannot say verified. All child processes in the worker process group are
killed when the worker exits or exceeds its limit.

`execution` also records `mode` (`batched-worker` or
`fresh-process-per-cell`), `batch_cells`, `batches`, `batched_cells` and
`fresh_cells`. Reports from either mode are compared and admitted alike.

With `--json-out evidence.json`, intermediate evidence is available as:

- `evidence.json.cells.jsonl`: one judged part per line, appended after each
  cell (and portable model worker).
- `evidence.json.progress.json`: small atomically replaced progress document,
  including completed/total cells, interruption reason and stage.
- `evidence.json`: the final atomically replaced report, retaining existing
  report format, reference resolution and exact hashes, plus execution data.

The journal and progress document are not comparison inputs and never claim
a completed verification. A progress document with `complete: true` means
the final report was written; consult its exit status and verdict to determine
whether checks passed. Without `--json-out`, progress remains on the terminal.
Full checks cover all applicable recorded fixtures; use `--quick` for a sample.
