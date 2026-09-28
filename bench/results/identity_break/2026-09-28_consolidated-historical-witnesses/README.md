# Historical consolidated identity witnesses

These 536 unmodified JSON records were retrieved from completed steward jobs:

- `m4-a`: `~/ev-apple-merged-9f20e20ac/s/clean_0of1`
- `do-amd`: the same directory, plus `~/ev-apple-merged-1c0677d7b/s/clean_0of1`

`provenance.json` records original remote paths, SHA-256 of every raw file,
recorded source commit, bound source fingerprint and enforced backend.
The recorded GPU labels are machine architectures; backend classification
uses the harness's enforced `resume_signature.options.require_backend`,
validated against bound provenance. No vendor labels or numerical values
were rewritten. New tests were not run to obtain these records.

Scope: 172 lanes with all nine fixture inputs and complete applicable parts
corroborated by CPU and GPU device classes, including current-revision GMM
and GMM sampling. Nonapplicable sampler/trainer parts have no numerical
reference. These are historical witnesses, not qualification of newer native
binaries. The original source commits remain visible in every record.

Excluded: the three lanes with historical cross-platform conflicts
(`trees-dart-options`, `x-glm-gamma`, `x-glm-tweedie`), lanes whose batch
property remains undeclared, and the physical multi-GPU claim of `par-gmm`.

Reference generation is a separate, reviewed action. A candidate using these
records contains 1,548 lane/fixture cells and 12,384 parts with zero conflicts;
402 of these raw records are selected by the builder's latest-per-device
policy. The remaining records retain the corroborating collection history.
The shipped reference table and manifest holds are not changed here.
