# Ordered multi-GPU cloud evidence — 2026-09-14

The final comparison [passed](h100/parallel-out/cross-architecture.json) between
two RTX 4090s and two H100s. This is NVIDIA cross-architecture evidence for the
fixtures below, not qualification across all algorithms or vendors. No local
builds, tests, or model executions were performed.

Both rented pods were terminated after receipts were fetched; the API confirmed
HTTP 404 for each. See [4090 termination](rtx4090-termination.log) and
[H100 termination](h100-termination.log).

| Path | Evidence compared across hosts |
| --- | --- |
| Byte LM | Ordered gradients and parameter/optimizer state over three steps |
| MLP | Ordered gradients and complete model state over three steps |
| Samba, Mamba-only | Gradients and complete state over two steps |
| Samba, Mamba/attention with dropout 0.1 | Gradients and complete state over two steps |
| RandomForest/ExtraTrees, classifier/regressor | Complete tree arrays for all four estimators |
| KMeans | Centers, labels, inertia, counts and scales for nine fixtures |

Within-host gates also check one-device replay, checkpoint migration, refusal
and rollback behavior, forest equality with serial fits, and KMeans equality
with the original path. See the individual JSON reports for exact coverage.
The native ordered-add cancellation/zero/FTZ gate and existing byte-LM session
gate passed on both hosts.

Final receipts:

- [H100 build and gate statuses](h100/parallel-out/status.tsv),
  [final verification](h100/parallel-out/verify.tsv).
- [Frozen-source 4090 build and gate statuses](rtx4090-frozen/parallel-frozen-out/status.tsv),
  [final verification](rtx4090-frozen/parallel-frozen-out/verify.tsv).
- Each host directory contains GPU metadata, toolchain version, binary hashes,
  source manifest, snapshot hash, base commit, gate logs, and model hashes.
- [Exact source archive](source/frozen-source.tgz) used for the final comparison.

The working repository advanced during the original cloud runs. Their model
hashes matched, but their source manifests differed; that rejected comparison
is retained as `h100/parallel-out/initial-source-mismatch.json`. The 4090 build
was then repeated from the exact H100 source archive. The final comparison
above requires source manifests and every model hash to match. The archive
also preserves source directories outside the narrower manifest. The original
4090 logs remain under `rtx4090/`; they include development failures and are
not the final same-source qualification. Earlier comparison logs likewise
remain historical; `cross-architecture.json` is the final comparison result.

Corpus: R2 `corpus/enwik8/input.txt`, 100,000,000 bytes, SHA256
`2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8`.
The source archive excludes corpus data, environments, compiled extensions
and benchmark outputs.

Boosting, remaining classical estimators, distributed individual tree splits,
AMD/Apple qualification, eight-device scaling, and throughput claims are not
covered. Byte-LM gradient bodies currently execute serially across resident
replicas. See [implementation scope](../../../../docs/multi_gpu/README.md).
