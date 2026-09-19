# A par-forecast-arima pair that `admit` REFUSES, kept as a diagnostic

CREDITS NOTHING. READ THIS FIRST. Both files here carry
`complete: false`, and `_verify_reference.admit()` refuses each of them by
name -- "incomplete identity_break checkpoint" -- with AND without
`par_axis=True`. So `par_two_device` reads neither, `par-forecast-arima`
still has NO two-device column, and nothing in the matrix may be credited
from this directory. The lane needs its run FINISHED, not redone.

The header of this file used to read "One par-* two-device pair landed"
and opened with WHAT THIS PROVES. That was wrong and is corrected here
rather than quietly rewritten: the observation underneath it is true, and
the conclusion drawn from it was not.

THE TRUE OBSERVATION. `par-forecast-arima/base` on an RTX 4090 pair,
commit 900038b24: the one-device half (`par_devices: 0`) and the
two-device half (`par_devices: 0,1`) agree on every part present --
hashes, infer, model, reload, batch, and the predict/forecast sub-parts.
That equality is what the driver claims (`_par_devices`: "a two-device
column ... must hash equal, cell for cell, to the one-device column of the
same commit").

WHY THAT IS STILL NOT EVIDENCE. The run was cut off by a timeout, so these
are checkpoints, not columns. A truncated column's hashes can be read, but
the admission rule refuses it, and a hash a rule refuses is not evidence
however much it agrees. Reporting bytes that agree as a lane closed is the
same error as reading a green exit from a diff over one cell.

WHAT IT DOES NOT PROVE. The leg was launched for FIVE lanes
(par-border-types, par-forecast-arima, par-forecast-holtwinters, par-ivf,
par-ordered). Only this one produced a cell.

WHY, AND THE TRAP IN THE STATUS FILE. The box spent about 50 minutes
building -- the bincache staged 55 entries and some thirty bindings still
compiled from source -- and the per-phase timeout `cap` scales with the
REMAINING lease. By the time the two-device phase started it was capped at
60 seconds and exited 124.

`par_diff` then exited 0 in ONE SECOND and that zero means nothing: it
diffed a column holding a single cell. A green exit from a diff over an
empty set is the verification that cannot fail. The evidence here is the
pair above, read cell by cell, not that exit code.

WHAT THE NEXT ATTEMPT SHOULD DO DIFFERENTLY. Build and column phases ate
the lease, so either rent for longer than the build takes, warm the
bincache for this partition first, or run the two-device phase before the
optional backstop re-column. The four remaining lanes are unchanged and
still owed.
