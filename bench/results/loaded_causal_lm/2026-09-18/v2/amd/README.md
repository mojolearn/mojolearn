# AMD loaded-model v2 capture

Captured by the existing bounded AMD qualification VM, then fetched before its
scheduled cleanup. No new hardware or build was started to perform this local
comparison.

All24 cases pass their property checks. Exact comparison with each retained
column gives144/144 matched parts against CPU and144/144 against Apple Metal:
all supported checkpoint families, FP32/BF16/int8, full logits, prefill, two
carried-state decode steps, greedy output and state. `comparison.json` lists
every case and part. The standard strict profile comparator was used.

Attribution is deliberately split: baseline Python/native source was
`f771338c7ab68820606b66892e202a45d43c6dda`; five exact proof/loader Python files
from `c9c3b96b7` were overlaid, as listed in `capsule.json`. Each capsule file hash
was independently checked against git before this receipt was committed. The
combined profile-source digest equals the CPU/Metal profile. The archived
capture correctly reports `capture_commit=null`; it is not a clean c9 checkout
or a newly built installed wheel. Actual loaded native binding hashes remain in
the raw capture.

This closes the retained AMD numerical column for the tiny v2 profile. NVIDIA,
rebuilt installed-wheel replay and physical multi-GPU execution remain separate
requirements. No reference-table admission or publication qualification is
inferred; the capture and comparison remain explicitly unqualified.
