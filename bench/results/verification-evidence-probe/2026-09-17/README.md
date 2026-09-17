# Development wheel verification diagnostics

All JSON reports are gzip-compressed. The development wheel combines current
Python and harness code with reused macOS CPU candidate bindings. It is not a
new native release build and carries no GPU bindings. No PyPI upload occurred.
The wheel receipt gives its digest and the exact Python/harness source digests.

`coverage` inspects scope without running algorithms. `ctr` replays both saved
CTR inference lanes over all nine fixtures twice, using only installed assets:
54 IDENTICAL parts, 36 N/A, zero REFUSED/DIVERGENT/OWED.
`ols-a` and `ols-b` are separate real executions on the same CPU. `compare`
confirms their hash agreement, explicitly not independent-vendor evidence.
`compare-changed` compares one real report against a copy whose held-out input
fingerprint was changed while its result hashes were left unchanged. It must
refuse with INCOMPARABLE and exit 4. This is a comparator regression control,
not numerical sabotage of any algorithm.

Commands ran outside the checkout with no PYTHONPATH, external host directory,
external harness or CTR model directory. The probe path excludes this directory
from numerical reference admission. Full logs and the wheel remain under
`~/mojolearn-evidence/verification-evidence-audit/`.
