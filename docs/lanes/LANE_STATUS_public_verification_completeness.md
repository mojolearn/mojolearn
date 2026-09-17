# Public verification and the 246-entry appendix

The MLSys appendix does list **246 algorithms and variants in 12 groups**.
Its source is `mlsys/results/algorithm-variants-2026-09-16.json`, a census of
a dirty source tree at `a2f05dbe8`. It is a different unit from this tree's
229 harness lanes or 220 enumerated Python API entries after integration with
the concurrent low-bit work (215 lanes and 202 entries at this audit's start).
The previous statement
that the inventory was not 246 conflated these counts.

`python/mojolearn/_verification_catalog.py` preserves every appendix title and
its source digest, with explicit lane mappings. Representative portable CPU
model checks cover the two saved-model API entries; their broader dedicated
gates are named separately. New lanes outside the old appendix remain visible.
A mapping is not a certification claim. The appendix's final statement about
every entry running on every backend cannot be established from this audit.
The sister repository was read, not edited.

## Gaps closed

- `verify --coverage` ships in the Python package. It shows every appendix
  mapping, all registered lanes, withheld/unavailable routes, batch contracts,
  optional properties, and reference counts. It fits nothing.
- Missing reference parts now prevent `VERIFIED`. A full request also names
  withheld/stale lanes and fails for incomplete scope. Explicit subsets remain
  subsets. A lane is no longer counted as checked end to end because just one
  of its parts matched.
- Missing/empty portable-model manifests are refusals, not empty successful
  work. A failed comparator self-test cannot leave a passing verdict.
- `verify --batch-checks` exposes the existing gradient, batch-size, ragged,
  sampler/replay and continuous-batching probes. Standard batch invariance and
  step/full checks remain on the ordinary path. Extra probes reuse each fit.
  Each result distinguishes local assertion success from agreement with a
  reference. A failed replay assertion remains a mismatch even without a hash.
- The reference builder can include these optional parts using
  `--all --batch-checks --emit-reference PATH`. It requires matching recorded
  batch/sequence protocols and uses the ordinary admission/revision rules.
  This adds the path to trusted hashes; it does not manufacture missing records.
- Pure-host BPE training and fold construction are now callable in the public
  CPU verifier. They currently report OWED because their bundled hashes are
  missing. Their global batch-invariance exclusions are explicit.
- `select_d` now has a real identity lane and per-series batch check. CPU uses
  the existing KPSS implementation and the device selector's first-stationary
  control flow. Mixed stationary/integrated/twice-integrated series exercise
  orders 0, 1 and the fallback 2. The appendix calls this seasonal-difference
  selection, but the API selects ordinary `d` with caller-supplied seasonal `D`;
  automatic selection of seasonal `D` remains unimplemented.

## Validation and scope

256 focused tests passed, including actual CLI exit codes and verifier/harness
parity on `ols` and `select-d`. CPU work used single-thread numerical libraries
and bounded scheduler jobs; no compiler or more-than-three-core work was used.
The same 256 tests passed again after merging the concurrent Transformer and
low-bit changes. The inventory automatically includes the 14 new low-bit lanes.

`select-d` passed all nine CPU fixtures twice, with all batch parts stable.
Metal base passed twice within a 60-second total budget; the final run took
1.59 seconds including queue time. Both computation (`b03785e1fe9435fe`) and
batch (`6fc89ade43696902`) hashes agree with CPU. A harness batch sabotage
produced `BATCH_MOVED` and exit 1; this is a negative control of the probe,
not a sabotaged native KPSS build.

An external development wheel was built through setuptools and installed in a
venv. It contains the current Python/harness and the prebuilt CPU host bindings
from the earlier macOS candidate, without GPU bindings. All checks run outside
the source checkout with no PYTHONPATH or external host directory:

- 246-entry inventory: successful inspection, no algorithm execution.
- Four portable models plus OLS: VERIFIED, exit 0.
- `select-d`, nine fixtures twice: local batch checks pass, references OWED.
- BPE training and fold construction, nine fixtures twice: stable computations,
  references OWED.
- OLS extended batch-size check: local assertion passes, reference OWED.
- Mamba-1 base twice: four standard parts IDENTICAL, one N/A; gradient,
  batch-size, ragged and sampler/replay assertions all pass locally, all four
  references OWED. Overall exit 5, not a false certificate.

An earlier source diagnostic used older host binaries lacking
`mamba1_forward`: it correctly returned six REFUSED parts and exit 4. The
wheel-host run resolved those refusals. Neither result is interchangeable
with a newly built final release wheel.

Evidence is retained under
`bench/results/public-verification-probe/2026-09-17/`; the probe path prevents
these development diagnostics from entering the reference table. Full installed
reports are gzip-compressed JSON. The development wheel is not published.
Its receipt describes this audit before the concurrent Transformer/low-bit
integration; it is not installed-wheel qualification of the combined tree.

## Still owed before claiming comprehensive 0.8.7 verification

Freeze the final source, build and qualify the actual release wheels, record
all applicable fixtures/properties on the required CPU/GPU and multi-GPU
hardware, repair missing saved-model CTR fixtures, and regenerate/admit the
reference payload. Replay that payload from the final wheels. These changes
make missing coverage visible and executable; they do not establish the
outstanding cross-vendor results. No PyPI publication was performed.
