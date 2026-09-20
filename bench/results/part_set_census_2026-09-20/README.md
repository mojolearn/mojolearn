# The part-set census: how many of the nine parts each column actually asked

Branch `lane/gpu-confirm-never-launched`, 2026-09-20.

    python3 tools/part_set_census.py > census.txt

`census.txt` in this directory is that command's output at the commit this
README was committed at. Re-run it rather than re-deriving it.

## The finding

`_verify_reference.PARTS` is `train, infer, model, batch, stepfull` and
`OPTIONAL_PARTS` is `batchgrad, batchscale, ragged, rlpair`. Nine. But
`tools/identity_break.py` runs `train/infer/model/batch/rlpair` at its
**defaults** and needs `--step-full`, `--batch-grad`, `--batch-scale` and
`--ragged` to be asked for the other four.

Of **562** admissible, device-classed columns inside
`bench/results/identity_break` — the only tree `build_table` walks — the part
sets are:

| columns | parts | which |
|---:|---:|---|
| 248 | 4 | batch, infer, model, train |
| 151 | 5 | + stepfull |
| 51 | 8 | + batchgrad, batchscale, ragged, stepfull |
| 31 | 6 | + rlpair, stepfull |
| 27 | 5 | + rlpair, **no stepfull** |
| 26 | 3 | infer, model, train |
| 15 | **9** | all of them |
| 3 | 7 | + batchgrad, batchscale, ragged, no stepfull |
| 1 | 8 | + rlpair, no stepfull |
| 1 | 7 | + ragged, rlpair, stepfull |
| 6 | 0 | no cells |
| 2 | 1 | train |

**Only 249 of 562 carry `stepfull` at all. Fifteen carry all nine.**

And the records everyone cites by name — `2026-09-14_118-lanes`,
`2026-09-14_166-lanes`, `2026-09-14_120-lanes-2711flip` — are **themselves
four-part**. "The 166-lane record" means 166 lanes at four of nine parts, and
nothing in that name, or in any tool's output, says so.

## Why nothing catches it

Three separate places, and only the middle one is silent:

* **`admit()` never inspects which parts a column carries.** Its completeness
  check is `j.get("complete") is False` — a run-level "did this run finish"
  flag. A five-part column is admitted exactly like a nine-part one.
* **`build_table` skips an absent part with no trace.** `for part in parts:
  value = _part_value(cell, part, min_repeats=2); if value is None: continue`.
  No log line. The per-column line is `use <path>: class X, commit Y, N cell
  parts`, and a smaller `N` is compared to nothing.
* **The far end does catch it.** `_verify_reference.compare` returns
  `OWED, "no committed record carries this cell part yet"` — by name, per
  part, never IDENTICAL. But `LANE_OWED` is in
  `_verify_all.LANE_STATES_THAT_DO_NOT_GATE`, so an owed part costs a run
  nothing. The hole is **visible and free**, not hidden.

It is the truncated-column shape this repo already names, pointed the other
way: not a column whose hashes agree and credit nothing, but a column whose
hashes agree, are credited, and were never asked four of the nine questions.

## `lacks stepfull` is only a gap where the part applies

`arima` has no decode state and `STEPFULL_DEFAULT` is `n/a:no-decode-state`.
Counting its absent `stepfull` as under-collection would be the same mistake
in the other direction, and the first version of this walk made it: it
reported 679 gaps over 250 lanes, most of which were lanes with no such part.
The census reads the declaring lane set out of the harness (`STEPFULL`,
`BATCHGRAD`, `BATCHSCALE`, `RAGGED`, `RLPAIR`) instead of listing names here.

Restricted that way, for `stepfull`: **27 lanes declare the probe, 83
(lane, class) pairs have a column, and 53 have none carrying it** — 21
distinct lanes.

## What the shipped table rests on

**29 (lane, class) pairs sit in `python/mojolearn/verify_reference/table.json`
for a `stepfull`-declaring lane with no `stepfull` entry, and for every one of
the 29 no committed column carries it either.** Seven lanes are missing it on
**all three** vendor classes:

    mamba1                  amd apple nvidia
    mamba2                  amd apple nvidia
    par-samba               amd apple nvidia
    par-samba-clip          amd apple nvidia
    par-byte-lm             amd apple nvidia
    par-byte-lm-model-pool  amd apple nvidia
    par-byte-lm-offload     amd apple nvidia

plus `mamba1/2/3-bf16w`, `mamba1/2-int8w`, `transformer-bf16w` and
`transformer-int8w` on apple.

`stepfull` is the decode-state part: `forward(x)` against
`allocate_state + step`, position by position. So **decode identity for
Mamba-1 and Mamba-2 — core shipped blocks — has never been verified on any
vendor.** Not divergent. Never asked.

## The bar this sets

**Nine parts or it is not a record.** A record run that takes the harness
defaults produces a four-part column, and a four-part column is not a record
of the nine-part claim however clean every cell in it reads.

`tools/gap_column_leg.sh` grew `MOJOLEARN_GAP_PARTS` for this on 2026-09-20
(a959b73e5 in its pre-rebase form). Leaving it unset keeps the historical
five-part behaviour rather than silently changing the shape of every column
that body has already produced, so a caller has to say what it is asking —
and the body prints `extra_part_flags=` in its gate file either way.
