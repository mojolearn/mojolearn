# Missing parallel references on the published 0.8.7 binaries

148 previously N/A reference entries now have measured numerical hashes. The
update covers 12 lanes, nine existing fixtures and two repeats, using 35 complete
records from source `4e1828f90384c4b2bb3e4bdc025f4931b7553dfb` in identical mode.
No existing numerical reference changed; cells outside these 12 lanes are retained.

Eleven lanes agree on Apple M4, NVIDIA H100 and AMD MI325X: 396 numerical cell
parts. Parallel resampling agrees on Apple and AMD: 18 numerical parts and
18 inapplicable parts. The partial NVIDIA resampling capture is not admitted.
The optional step/full checks were not part of this reference update.

The exact published wheels were checked before capture:

- macOS: `25c728aa4321b011c14b801b123905aed04f48f6139aef8ea81c33abb50f1d97`
- Linux: `5b1c2db8d4350aff672f155c804d309148311b0cc4a4e36efadef998ebbae470`

Each JSON retains fixture witnesses, repeats, protocols, source and binding
provenance. The per-lane diffs require three numerical columns, except resampling,
which requires two. `summary.json` records the scope and raw-record hashes;
`admission.log` records strict reference admission. Generation uses
`_verify_reference.build_table(..., parts=('train','infer','model','batch'))` and
`merge_reference_lanes` for exactly the lanes in the summary.

The initial diagnostic looked at 32 GPU parallel lanes. Existing evidence was
retained for lanes without missing references; the remaining NVIDIA work was
narrowed to gaps. AMD resampling alone needed a longer bounded retry. Full raw
attempts, including timeouts and interruptions, are retained in the release's
external evidence archive. No failed or partial record supplied a reference.

This is not a new CPU implementation, a physical multi-GPU certificate, or a
universal library-wide matrix. Native fault injection is not newly claimed here;
changed-verifier mismatch/refusal tests and final-wheel light smoke are separate.
