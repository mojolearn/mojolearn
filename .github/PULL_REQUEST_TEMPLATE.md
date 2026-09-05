## What changed

<!-- State the user-visible outcome and the boundary of this change. -->

## Provenance

- [ ] Original `original` work; the file names the call or need it serves.
- [ ] Derived/ported work; exact upstream file, commit and license are below,
      and `DERIVATION_MAP.tsv` / `NOT_IMPLEMENTED.tsv` are updated.
- [ ] No implementation code changed.

Upstream details:

## Numerical impact

- FAST impact:
- IDENTICAL impact:
- Identity profile clause or `IDENTITY_PATHS.md` row:
- [ ] Bit-inert evidence is attached.
- [ ] A separating fixture and profile-version decision are attached.
- [ ] No operation capable of moving IDENTICAL bits changed.

## Evidence

- Tests added or changed:
- Commands run:
- Hardware actually exercised:
- Columns still pending:
- Performance before/after, if relevant:

<!-- External PRs receive automatic hosted CPU checks and a separate metadata
admission report. Unrun GPU columns remain GPU_PENDING, even if CPU CI is green.
The workflow does not currently rent GPUs or auto-merge. Existing implementation
optimizations may enter the future isolated GPU queue; infrastructure, dependency,
test, contract and public-API changes require review. No second GPU is required
from contributors. See CONTRIBUTING.md for the exact boundary and pending setup. -->

- Existing feature/path being optimized (if applicable):
- Reproducible benchmark shape, mode, seed and timing scope:
- Legacy certified bytes expected to remain unchanged:

## Public surface

- [ ] Documentation and support matrix are updated where required.
- [ ] Unsupported behavior refuses by name rather than silently falling back.
- [ ] No credentials, generated provider state, private data or machine-specific binaries are included.
- [ ] Licensing and attribution were reviewed.
