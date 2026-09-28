# Installed verifier execution

`python -m mojolearn verify --all --json-out evidence.json` runs cells
sequentially in disposable processes. Each lane/fixture pair has a fresh
native context. This prevents a long sweep from accumulating GPU state and
does not contend with itself by running independent fits on the same GPU.
Fixture generation, repeats and enabled probes remain exactly those of the
reference protocol; isolation does not shrink input arrays or weaken checks.

Before fitting lanes, an isolated comparator self-test must demonstrate that
an untouched input matches and the perturbed input diverges. Portable model
checks and optional smoke checks also use bounded workers. Each worker has a
120 second limit, including interpreter startup and cleanup. Override it with
`--cell-timeout SECONDS` when investigating a legitimately longer cell.
A timeout is an incomplete verification, never a successful skipped check.

The parent announces each cell, prints a heartbeat every ten seconds while
waiting, and judges its parts immediately on completion. Ordinary divergences
remain visible while subsequent cells run in fresh processes. A native crash,
timeout, malformed worker result or explicit unhealthy-device signal stops
the sweep. Outstanding cells become refused, and the overall report cannot
say verified. All child processes in the worker process group are killed
when the worker exits or exceeds its limit.

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
Process startup adds overhead, particularly for very small cells. Full checks
still cover all applicable recorded fixtures; use `--quick` for a sample.
