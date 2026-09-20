# The last three par-* lanes reach a two-device column

`par-forecast-arima`, `par-forecast-holtwinters` and `par-ivf` -- the three
that had none -- now carry the pair. 2x RTX 4090 sm_89, one box, one commit.

```
one-device (par_devices 0)    complete: true   27 cells   admit() = ADMISSIBLE
two-device (par_devices 0,1)  complete: true   27 cells   admit(par_axis) = ADMISSIBLE
135 part comparisons, 135 agree, 0 disagree
```

Both halves are here because a two-device column alone asserts nothing;
the claim IS the equality (`_par_devices`: "a two-device column ... must
hash equal, cell for cell, to the one-device column of the same commit").
The default `admit` rule REFUSES the two-device half by name
(`par_devices 0,1`) and `par_axis=True` admits it -- that is the designed
path, not a workaround, and only a `par-*` lane may be credited from it.

The 135 agreements were recomputed at home from the two committed files,
part by part. The leg's own `par_diff` also exited 0, and that exit was NOT
what this rests on: it returned in ZERO seconds, the same shape as the
meaningless green in `2026-09-19_par-two-device-gaps`, where an exit 0
covered a diff over one cell.

## THIS TOOK THREE LEGS. THE FIRST TWO FAILED THE SAME WAY, ONE LAYER APART.

Both earlier legs produced a two-device column with `complete: false`, which
`admit` refuses by name ("incomplete identity_break checkpoint") with and
without `par_axis`, so neither credited a single lane however well its
hashes agreed.

1. **Leg one** spent ~50 min building because `bincache` was keyed on
   `MOJOLEARN_GAP_LANES` -- a leg could only hit its own previous lane list.
   `column-two` started with a 60 s cap and exited 124 after one cell.
2. **Leg two** ran with the cache fixed (build 1738 s -> 63 s) and STILL
   timed out at 567 s, which looked like the lease and was not.
3. **The actual cause** is `MOJOLEARN_GAP_BUDGET`, default **2500 s**, an
   internal budget independent of the 60-minute lease. `cap()` hands each
   phase `T0 + BUDGET - now`, so setup plus the 671 s one-device column left
   567 s for a two-device column that needs more. Fixing the build did not
   fix this; it moved the failure one phase later and exposed it.

This leg set `MOJOLEARN_GAP_BUDGET=3300` against a 3600 s lease. Column 394 s,
column-two 550 s, both exit 0.

**If you run this again: raise the budget, not the lease.** A 90-minute
lease is refused outright (one hour is a hard cap); the budget is the knob.
