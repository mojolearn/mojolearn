# Release evidence

This directory holds the evidence of the checks run for one Mojolearn release: the release named in
`manifest.json`. Each release replaces the whole directory, so it only ever holds one release's evidence.

Start with `SUMMARY.md`: what was released, which wheel bytes each identity column was recorded from, whether
the columns agree, the multi-GPU and measurement checks, how to run the same check on your own machine, and
where the full logs are.

What is here:

- `SUMMARY.md`: the human-readable account of this release's evidence, written from the files below.
- `manifest.json`: the release version, the source commit the wheels were built from, every wheel's file name
  and sha256, the tools used, and for every file below its sha256, the release step that produced it and the
  wheels it was checked against. It also lists what was not copied here (logs, the wheels, the full identity
  table, build receipts, box job results) by path or storage tag plus sha256.
- `identity/`: the one identity verification round: the check at the frozen source, the per-column summary of
  the identity table that shipped, and what each column recorded.
- `gates/`: the verdicts of the installed-wheel checks on each GPU vendor and on Apple and the CPU, including
  the multi-device shard check, the measurement smoke and the facts of the machine each ran on.
- `publish/`: the TestPyPI rehearsal before publishing and the byte checks of what PyPI serves after it.
- `checks/`, `compile/`: the release's tool tests and which compiled bindings were built or reused.
- the top-level JSON files: the published wheels' manifests and admissions, the release provenance and the
  binding identities the next release reuses from.

To see an earlier release's evidence, read it from that release's tag:

    git show v0.8.67:bench/results/release_evidence/manifest.json
    git ls-tree -r --name-only v0.8.67 -- bench/results/release_evidence/

Releases up to 0.8.65 kept one dated directory each under `bench/results/release_verification/`; those are
in git history (for example `git log --format=%H -n 1 -- bench/results/release_verification/` names the
commit that removed them; read them at its parent).
