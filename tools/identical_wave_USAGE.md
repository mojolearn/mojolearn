
`prepare --prepare-jobs 4` runs up to four independent builders concurrently;
every Mojo compiler still uses the shared four-slot semaphore and `-j 1`.
All return codes are retained, and any failed builder fails preparation.

The reviewed `be3348860` to `a4d01a130` SGD launch-bound change can reuse
unaffected modules only after recomputing the complete Mojo import graph and
all other tracked compile inputs. The changed GPU linear module is always
rebuilt. Donor graph/input artifacts, exact source pair, and explicit reviewed
noncompiler paths are checked again at import.
