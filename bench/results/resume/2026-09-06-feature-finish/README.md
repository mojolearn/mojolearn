# Apple completion checks, 2026-09-06

Source starts at `2e53699e`. Only the main operator ran checks, serially.
Extension builds use two compiler workers; BLAS/OpenMP use one thread.

## Source Mamba checks

All three rebuilt Apple Mamba bindings pass all 102 Python surface checks
per mode. FAST/DETERMINISTIC report continuation byte equality; IDENTICAL
asserts it. Build and test transcripts and the loaded source-binding hashes
are retained here. These runs are not installed-wheel evidence.

## Corrected Apple native backward

`apple-backward-baseline/results.tsv` passes all five cases: Mamba1, Mamba2,
Mamba3, Mamba2 L257, and Mamba2 incoming-state gradients. The complete native
baseline certificate retains 54 public gradient tensors.

`apple-backward-long/results.tsv` passes Mamba1 L64 and **fails Mamba3 L65**.
The latter fails `partial.qkdot.dt` against the independently generated
PyTorch float32 reference at flat cell 87:

- Native: `0.0794239342212677`.
- Apple reference: `0.07942590862512589`.
- Allowed error: `1.7942590862512587e-6`.

All 93 available native arrays shared with the older retained AMD operand
capture match by bytes; two newly captured operands are absent in that older
capture. The float64 check for this gradient passes, as does comparison to
the older AMD float32 reference. The Apple and AMD float32 references have
different hashes. See `apple-amd-long-diagnostic.json` and
`apple-long-reference-diagnostic.json`. Both manifests report Python 3.14.7
and PyTorch 2.13.0 with CPU reference execution. This localizes the refusal to a
platform-sensitive reference comparison; it does not change the contract or
turn the failed certificate into a pass. The prior AMD capture has different
source provenance, so this is a diagnostic, not a new cross-vendor certificate.

## Release gate changes

The macOS wheel verifier now refuses missing/failed interpreter environments,
ambiguous candidates, empty or invalid mode lists, and unknown arguments.
Its final line identifies import-only runs as device-untested. It also uses
`printf` to retain literal backslashes in JSON evidence. Nine synthetic
shell regression tests and five existing OrderedRMSE orchestration tests pass.

The installed smoke additionally runs the full Mamba and Transformer surface
gates in each interpreter/mode job. Independent corpora are loaded from the
checkout while implementation imports resolve to the installed wheel.

The preexisting `python/dist` contents are preserved in `preexisting-dist/`.
## Qualified Apple candidate

The full45 Apple 0.6.0 wheel passes all fifteen installed Python 3.10–3.14 /
FAST, DETERMINISTIC, IDENTICAL jobs. Each job includes UMAP transform,
OrderedRMSE (including save/load across a changed process default), all 102
Mamba surface checks and all 44 Transformer surface checks.

Artifact: `python/dist/mojolearn-0.6.0-py3-none-macosx_11_0_arm64.whl`.
SHA256: `061d9f47acdce1f6f4d3fa451fc03a7b55ea8909f8e84a318e02723a41739027`.
`apple-wheel-qualification.json` retains all 45 binding hashes, harness hashes,
the fifteen job identities and parsed OrderedRMSE model/prediction records.
`apple-wheel-verify.log` is the successful final transcript; the verifier
exited zero and its wheel/harness hashes were checked unchanged afterward.

The initial build produced this same wheel and all fifteen test jobs passed,
but a live verifier edit interrupted its final shell step (exit 2). That
superseded transcript is retained as `apple-wheel-build-first-verification.log`;
it is not the successful qualification. The separate clean verifier rerun
above is the qualifying result.

This candidate is not published. Linux GPU qualification, the independent
long Mamba3 backward refusal, and the Python backward API remain open.
