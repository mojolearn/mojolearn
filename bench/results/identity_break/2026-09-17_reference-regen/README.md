# 2026-09-17, the reference regeneration record (lane/reference-regen)

WHY THIS RECORD EXISTS. `python/mojolearn/verify_reference/table.json` is built
from the committed columns under `bench/results/identity_break/`, and five
things were wrong with the shipped one at once:

1. five lanes read `stale reference` and were dropped from every comparison;
2. the table carried no `admission_policy`, so `verify --coverage` reported the
   `legacy` fallback;
3. thirty of the 246 appendix entries had no reference hash of any kind;
4. the four optional properties shipped ZERO hashes;
5. `stepfull` was declared by sixteen lanes and carried by nine.

Two of those were a regeneration that had never been run: `build_table` writes
the `admission_policy` key, and it emits the four `OPTIONAL_PARTS` only under
`--batch-checks`, which no emission had ever passed. The other three needed
columns, which is this record.

## The columns

Every CPU column: `MOJOLEARN_NUMERIC_MODE=identical`, `--repeats 2`, nine
fixtures, all thirty-two host families built from source on the box through
the R2 binding cache (`tools/runpod_cpu_leg.sh`, RunPod CPU pods at $0.24/hr,
8 vCPU). The NVIDIA column: `tools/gemm_remote_leg.sh nvidia` with its ten GPU
bindings built on the box, one device.

| file | box | lanes | opt-in parts | cells |
|---|---|---|---|---|
| `cpu-x86-light1.json` | EPYC 9655 | 115 | `--step-full` | 1035 |
| `cpu-x86-light2.json` | EPYC 9754 | 114 | `--step-full` | 1026 |
| `cpu-x86-shard1.json` | EPYC 9654 | 56 | all four | 504 |
| `cpu-x86-shard2.json` | EPYC 9965 | 55 | all four | 495 |
| `cpu-x86-shard3.json` | EPYC 9655P | 55 | all four | 495 |
| `cpu-x86-shard4.json` | EPYC 9655P | 55 | all four | 495 |
| `cpu-x86-neural.json` | EPYC 7713 | 8 | all four | 72 |
| `nvidia-rtx4090-sm_89-par-lanes.json` | RTX 4090 | 12 `par-*` | none | 108 |

Each file is one column from one box. They are separate columns rather than
one merged file because `--merge` joins the PARTS of one column, not the
LANES of several boxes, and every box here is a different machine.

WHY TWO KINDS OF CPU LEG. `batchscale` on the Mamba-2-family lanes costs 200
to 370 SECONDS per cell on these boxes (measured: `mamba2/base` 366 s,
`mamba2-int8w/base` 277 s, `mamba2-bf16w/base` 209 s) against 8 to 11 s for
the same lanes without it. One all-lanes all-parts sweep fits in no lease. The
`light` legs run every lane cheaply and carry items 1, 3 and 5; the `shard`
legs carry the opt-in tail for item 4.

WHY A SEPARATE NVIDIA COLUMN. The `par-*` multi-GPU drivers and
`gbdt-tensor-ctr-tables` are the only lanes that REFUSE on a CPU-only box, so
no CPU column can ever carry them. Twelve `par-*` lanes were among the
appendix entries with no hash at all. They ran on one device, so
`package.par_devices` is `0` and the column is admissible.

## What was checked BEFORE the table was rebuilt

`refcheck.py`, beside this file, compares every new column against the hashes
the SHIPPED table already carried. A newest-wins rebuild would otherwise
replace a reference four device classes agree on with this record's single
value and say nothing at all.

| column | compared | agree | differ | new |
|---|---|---|---|---|
| `cpu-x86-light1` | 2555 | 2499 | 56 | 613 |
| `cpu-x86-light2` | 2341 | 2304 | 37 | 809 |
| `cpu-x86-neural` | 154 | 61 | 93 | 440 |
| `cpu-x86-shard1` | 1259 | 1259 | 0 | 415 |
| `cpu-x86-shard2` | 1187 | 1187 | 0 | 442 |
| `cpu-x86-shard3` | 1176 | 1176 | 0 | 453 |
| `cpu-x86-shard4` | 1120 | 1120 | 0 | 437 |
| `nvidia-rtx4090-sm_89-par-lanes` | 0 | 0 | 0 | 414 |

**Every differing cell part is in one of the five lanes
`identity_break.LANE_REVISIONS` declares stale**: `transformer`,
`transformer-window`, `mamba3`, `samba-untied-dropout-accum`,
`mamba2-dtlimit`, and nowhere else. That is the divergence appearing exactly
where the harness says the input moved, which is the reason those five were
withheld in the first place. No unexplained divergence, on any column.

THE CHECKER WAS WATCHED TO FAIL FIRST. One STABLE cell of one column was
rewritten to `deadbeefdeadbeef`; `refcheck.py` reported that cell by name
together with the four device classes standing behind the shipped value, and
the clean arm of the same file reported `differ: 0`. A check that has not been
seen to fail is not a check.

## What a CPU column cannot say

These are single-vendor columns for the lanes whose fixture moved. Where a
lane lost every GPU cell to a fixture change, its reference in the regenerated
table has ONE witness, and `host_surface.PUBLIC_PENDING_LANES` holds such a
lane back with the reason `one column` rather than promoting it. The other
three columns come back with the release record.
