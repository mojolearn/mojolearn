# rf-reg-gamma-ig/base read MOVED once on the MI325X with five processes sharing the GPU (2026-09-15)

Found while taking the 178-lane identity record at df617c699
(`bench/results/identity_break/2026-09-15_178-lanes/README.md`). Nothing is fixed
here and no kernel was touched. OPEN.

## What was seen

DigitalOcean MI325X (gfx942), leg `bench/results/e1g/2026-09-15_123228-amd-mi325x-do-rec2-a`
(raw archive, not committed; a copy is in `~/mojolearn-evidence/record2/legs/`).
One build of all 44 bindings, then FIVE `tools/identity_break.py` processes at once
on the one GPU, each with its own lane list. Process a5 read:

    MOVED   rf-reg-gamma-ig/base: ['116ecced8bb3be6a', '7b50848144c3a9c2']

Parts of the two fits in that one process:

| fit | gamma | inverse_gaussian |
|---|---|---|
| 1 | 0370566b26cf014c | 1760ee112bb6fddf |
| 2 | 0370566b26cf014c | 39eab6d1f802f5a9 |

Only the second fit's `inverse_gaussian` forest (criterion `inverse_gaussian`,
`max_features=None`, 16 trees, depth 8, `random_state=7`) changed. The other eight
fixtures of the lane read STABLE in that process.

## What it is not (so far)

- Not a column disagreement: the first fit's hash 116ecced8bb3be6a is the value the
  M4, the H100 and the MI325X carry in the 178-lane record, and the value every
  column of the 166-lane record carries.
- Not reproducible alone: the lane reran in a fresh process with nothing else on the
  GPU (leg `2026-09-15_130122-amd-mi325x-do-rec2-m`, the same commit, a separate build
  whose `_mojolearn_rf` digest equals the first build's), and all nine fixtures read
  STABLE with the recorded hashes. The committed MI325X column takes the lane from that
  rerun (`truncated_note` in the dropped part says so).

## Why it matters anyway

A fit whose bytes depend on what else runs on the device is the shape of the 2712
defect (an uninitialized device read in a warm process). The deterministic kernels
should not see another process at all. One MOVED cell in 1602 is not a rate; it is one
observation.

In the same leg a second contention symptom appeared, of a different kind:
`par-feature-freq/denormal` read REFUSED with `EOFError: Ran out of input` (a pooled
worker lost). On a two-MI300X pod with four par2 processes at once
(`2026-09-15_123314-amd-2xmi300x-rec2-par2`, not evidence), most par cells refused with
`RuntimeError: GPU worker failed` or the same EOFError. On a two-H100 pod with six
processes at once, every one-device cell read STABLE and IDENTICAL to the other columns.

## Next steps (owed, not done)

1. On one MI325X (or MI300X), run `rf-reg-gamma-ig --fixtures base --repeats 8` in a
   loop while four other identity processes run other lanes; count MOVED. Then the same
   loop alone. If it moves only under contention, bisect: the `inverse_gaussian`
   criterion's split search on the device versus the host finish.
2. Build with the POISON define (the 2712 recipe, `mamba_zeros` style) and see whether a
   filled-but-unwritten buffer feeds the inverse gaussian split scores.
3. Read the pooled worker's failure text in full (the harness truncates the
   traceback at the worker's `execute` frame) before blaming the device for the refusals.
