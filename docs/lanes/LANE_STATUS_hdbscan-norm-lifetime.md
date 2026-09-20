# HDBSCAN CPU row-norm lifetime fix, 2026-09-20

`hdbh_mutual_reachability` converted its locally owned `norms` list to an
untracked pointer, then captured only the pointer in its row worker. Mojo could
release the owner before those reads. Explicitly destroying `norms` after the
serial worker or parallel join retains the allocation through every read.

Fresh Linux CPU bindings from a97676ba1 reproduced the release smoke failure
on finite verifier base data: the leaf fit refused 3998 NaN distances out of
4000000. The input's strided-to-contiguous conversion was independently checked
byte-for-byte and was correct. Running the leaf lane first in a fresh process
also failed, excluding contamination from preceding verifier lanes.

A separate overlay build changing only the owner destruction point returned
the established leaf training hash `7e6eda79b757c89f` and EOM training hash
`3c5c76e51b0ae88f`; repeated leaf fits matched. The committed numerical regression
runs both configurations on the same base fixture prefix and passed with one
and two CPU tasks. Production build files were not modified by that diagnostic.

The pre-fix production binary is not suitable for release. Release assembly
must use a binding built with this fix and truthful source provenance.
