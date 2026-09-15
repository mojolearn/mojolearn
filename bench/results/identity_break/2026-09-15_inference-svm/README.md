# Saved-model CPU inference: svc-linear, svc-poly, svr, svr-linear

lane/inference-svm, 2026-09-15. Lanes: svc-linear, svc-poly, svr and svr-linear, with svc
as the control. All nine fixtures. No GPU box was rented.

## Files

| file | what it is |
|---|---|
| `apple-m4-metal.json` | identity_break column on the M4 Metal IDENTICAL set (svm and base bindings built from the lane tree), 2 repeats |
| `diff_metal_vs_166-lanes-apple.txt` | that Metal column against the 166-lane Apple column: IDENTICAL=36 train, svc-poly ONE-COLUMN (no record carries it) |
| `classical_host_record_metal.txt` | `tools/classical_host_gate.py record` on the Metal set: `bench/results/classical_host/2026-09-15-apple-m4-svm` (4 lanes, 9 fixtures) |
| `cpu-x86.json` | x86 CPU column, RunPod CPU pod pvd0b6qj0p4qcw: svm, core, linalg host bindings built from the lane commit, 2 repeats |
| `cpu-x86-sabotage.json` | the same families built with `-D MOJOLEARN_HOST_SABOTAGE=1`, 2 repeats |
| `diff_cpu_vs_166-lanes_require4.txt` | `--diff` of the CPU column against the 166-lane Apple, H100 and MI325X columns, `--require-columns 4 --owed-json`. Exit 0. |
| `owed_cells.json` | the 54 owed cell parts: svc-poly train, infer, model and batch on 9 fixtures, and the svr and svr-linear model cells (the new save format) |
| `owed_sabotage_check.txt` | `cpu_identity_gate_check.py owed`: 54 of 54 owed parts moved under sabotage |
| `diff_cpu-sabotage_vs_166-lanes.txt` | the sabotage column against the three GPU columns. Exit 1, as it must. |
| `diff_metal_vs_cpu.txt` | Metal against x86 CPU. Exit 0. |
| `classical_host_check_cpu.txt` | `check` of the Metal recording through `mojolearn.host_model` on the x86 host set, with the three 166-lane columns as `--gpu-column` |
| `classical_host_check_sabotage.txt` | the same check on the sabotage set, `--expect-mismatch` |
| `bindings.txt` | the host bindings the pod built, with sha256 |

## Verdict lines

- CPU vs 166-lane columns, require 4: `summary: IDENTICAL=36, OWED=9`,
  `summary (infer/model): IDENTICAL=54, OWED=36`, `summary (batch): IDENTICAL=36, OWED=9`.
  The svc control is among the IDENTICAL cells, so its recorded hashes did not move.
- Metal vs CPU: `IDENTICAL=45`, infer/model `IDENTICAL=90`, batch `IDENTICAL=45`.
- Sabotage vs 166-lane columns: `DIVERGENT=36`, infer/model `DIVERGENT=54`, batch
  `DIVERGENT=36` (ONE-COLUMN on the parts no GPU column hashes); owed check
  `owed verdict OK (54 of 54 owed cell part(s) moved, 0 failure(s))`.
- Saved-model check on the x86 host set: `gate verdict IDENTICAL (36 fixtures, 3 GPU columns, exit 0)`.
  Every recorded surface EQUAL; the identity hash EQUAL to all three GPU columns for
  svc-linear, svr and svr-linear (27 each); svc-poly ABSENT from every column.
- Saved-model check on the sabotage set: `gate verdict EXPECTED MISMATCH SEEN`,
  identity_hash DIFFER on 9 of 9 fixtures for each of the four lanes, `ties` included.

## Sabotage arms

Before this lane the svm host sabotage was the GEMM leaf walked descending plus the
polynomial kernel arm. The leaf order cannot move an integer-grid fixture, so svc, svc-linear,
svr and svr-linear on `ties` stayed unchanged under it. `svm/host/smo_oracle.mojo` now also adds
half a unit to the fitted intercept (`smo_oracle_fit`) and to every decision value
(`smo_oracle_decision`), so a saved model's file and its predictions move on every fixture.
Both arms are compile-time only.

## Wheel size

The production svm host binding built on the pod from the lane tree, and from the same tree with
both new sabotage arms removed, are both 399,480 bytes (x86-64-v3 Linux). No entry was added to
the binding. The wheel's Python grows by `SVR.save`/`load`, `HostSVR` and the gate table rows.

## Owed to the release record

The NVIDIA and AMD saved-model recordings of these four lanes; the 54 parts in
`owed_cells.json` (every svc-poly GPU cell, and the svr and svr-linear model cells).
