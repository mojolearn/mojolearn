# A large block of this library had never been launched on a GPU

Branch `lane/gpu-confirm-never-launched`, 2026-09-20. RunPod NVIDIA, RTX 4090
(sm_89), one and two devices. Every pod DELETEd and verified gone: HTTP 204,
then GET 404, then absent from the listing.

## The columns

| file | lanes | cells | admit() |
|---|---|---|---|
| `...neural-nine-parts-2026-09-20.json` | 5 neural + 2 decode-session | 63 | ADMISSIBLE |
| `...new-since-088-nine-parts-2026-09-20.json` | 13 lanes added since v0.8.8 | 117 | ADMISSIBLE |
| `...stepfull-never-recorded-2026-09-20.json` | `mamba1`, `mamba2` + 5 driver lanes | 63 | ADMISSIBLE |

All at `--repeats 2` on every part, `complete: true`, `par_devices=0`, and
`admit()` asked **at the committed path** rather than the on-box one — the
distinction that caught a leg on 2026-09-19.

`leg6-verify-par/` carries the `verify --par` pair for `par-queries-nn` and
the self-test log; it is a diagnosis, not a column for the table.

## The owed NVIDIA column, at nine parts

`mamba3`, `transformer`, `transformer-window`, `samba` and
`samba-untied-dropout-accum` each read 9/9 parts at their current
`LANE_REVISIONS` entry. What existed before: `mamba3` had twenty NVIDIA
columns and **not one** at `norms-near-one-1`; the other four had exactly two
admissible columns at their current revision, both five of nine parts with no
`stepfull`, both under `bench/results/attention_replay_vendors_2026-09-18/` —
outside `bench/results/identity_break`, which `_verify_all` passes to
`build_table` as its only record root. Invisible to the table however clean.

## The decode sessions, first run on any hardware

`mamba1-decode-session` and `transformer-decode-session` moved `gpu=[]` to
`gpu=['nvidia']`. `mamba1_session_create` and
`transformer_decode_session_create` live only in `bindings/_mojolearn_mamba.mojo`
and `bindings/_mojolearn_transformer.mojo` and each refuses BY NAME on the host
route, so a GPU column is the only place their proposition is stateable.

## `mamba1` and `mamba2` have a `stepfull` hash at last

    mamba1  stepfull = f582474b00117f8e
    mamba2  stepfull = bfd516aa93fe1b12

`stepfull` is `forward(x)` against `allocate_state + step`, position by
position. No committed record on any vendor carried that cell for either lane
before this one, and both are core shipped blocks.

**Five lanes in the same run were never a gap.** `par-samba`,
`par-samba-clip`, `par-byte-lm`, `par-byte-lm-model-pool` and
`par-byte-lm-offload` return
`n/a:driver-lane (the multi-GPU drivers carry no decode state; the
single-device twin lane is asked)`. See
`bench/results/part_set_census_2026-09-20/README.md`: "no column carries this
part" and "this part has never been answered" are different statements, and
only a run distinguishes them.

---

# ONE FACT ABOUT `par-*` LANES, FOUR SYMPTOMS

Read this before diagnosing a fifth.

**No `par-*` lane appears in any family's `training_lanes` or
`inference_lanes` in `python/mojolearn/host_surface.py`.**

Everything below is that one fact wearing different clothes.

**1. 36 `par-*` lanes read `none` in the sabotage census.**
`verification_matrix.lane_rows` fills `define_of` by walking
`FAMILIES[*].training_lanes` and `inference_lanes`. A lane in neither gets no
define, so it cannot read `declared`, and with no committed pair moving it, it
reads `none`. The arm's existence was never the question.

**2. `gap_column_leg.sh` built nothing and the column refused in every cell.**
The body derives its build set from the same table with the lane list in hand.
For an all-`par-*` list both `DECL_GPU` and `DECL_HOST` come back **empty**,
the loop builds `build` alone, and every cell refuses. Measured on pod
`70i7hnr5avagda`.

**3. The refusal named its own cure, and the budget ate it.**

    ImportError: numeric_mode='identical' needs
    python/mojolearn/identical/_mojolearn_gbdt.so, which is not built

The insurance pass then spent 1130 s building exactly that family. `cap()`
shrinks each phase as the lease burns, so the backstop rerun started with
275 s and the two-device column with 61 s; both exited 124. Both halves came
home `complete:false` — 13 cells and 1 — and `--diff` printed "no DIVERGENT or
MOVED cell" over them and exited 0. `admit()` refused both, which is the only
reason nothing was credited.

**4. And it happened again on a mostly-`par` list.** Leg 5's first column read
`stable=18 refused=45`; the backstop built `build_byte_lm` and
`build_transformer` from the refusals' own text and reran to 63/63. Two legs,
same cause, one of them after the cause was already known.

**What was done about it.** `gap_column_leg.sh` now builds every family before
the columns when `MOJOLEARN_GAP_TWO_DEVICE=1`, because for this lane list the
derivation has no information to give. That is a workaround at the leg, not a
fix at the source: the source would be `host_surface.py` describing which
families a `par-*` lane reaches, which nothing in the tree currently does.

## Two lanes no GPU rental can ever close

`language-model-config` and `saved-model-host-infer` are DEGENERATE on
`nvidia-1gpu`: `lane_applicability` reports the lane's arithmetic IS the CPU
host route, so a cell here would measure the box's CPU and say nothing about
cuda. They went to `MOJOLEARN_GAP_DEGENERATE`, and the refusal came home in
`lane_applicability`'s own words, checked on the box. They read `gpu=[]` today
and will keep reading `gpu=[]`. That is a property of the lanes, not a gap.

## What this record does NOT contain

* **A two-device `par-*` sabotage arm watched failing.** The four
  `*_PARALLEL_SABOTAGE` defines were built into the bindings and run on two
  devices, and the run was inconclusive: every column timed out
  `complete:false` and the on-box diff reported `IDENTICAL=9, REFUSED=3` over
  truncated data. That is not "the arms are inert" and must not be read as it.
* **Any AMD or Apple column.** Every column here is NVIDIA.
* **The `par-queries-nn` AMD question.** It passes on two NVIDIA devices
  (`leg6-verify-par/`), which removes two candidate explanations and closes
  nothing; the failure was AMD-only and only an AMD box can settle it.
