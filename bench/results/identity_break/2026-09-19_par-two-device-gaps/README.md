# One par-* two-device pair, and four lanes the lease did not reach

WHAT THIS PROVES. `par-forecast-arima/base` on an RTX 4090 pair, commit
900038b24: the one-device column (`par_devices: 0`) and the two-device
column (`par_devices: 0,1`) agree on EVERY part -- hashes, infer, model,
reload, batch, and the predict/forecast sub-parts. That equality IS the
driver's claim (`_par_devices`: "a two-device column ... must hash equal,
cell for cell, to the one-device column of the same commit"), and a
two-device column alone asserts nothing, which is why both halves are here.

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
