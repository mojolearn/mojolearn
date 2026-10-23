# Five CNN public-method coverage gaps

The original changed-method audit at `857c0c176` credited existing `x-cnn-*`
lanes with five operations those lanes do not call. The audit after the
22-method coverage addition (`7fecf2c84`) reports these five as unmapped.
The public wrappers still use `_XCNN`; sharing that native dispatch entry does
not establish execution of each operation.

| Public operation | Existing lane calls | Added lane and consumed answer |
| --- | --- | --- |
| `Conv1d.fit` | `forward`, `backward`, `transform` | `public-cnn-layer-fit`: returned identity, feature count, unchanged weights and subsequent transform |
| `Conv2d.fit` | `forward`, `backward`, `transform` | `public-cnn-layer-fit`: returned identity, feature count, unchanged weights and subsequent transform |
| `BatchNorm1d.fit` | `forward`, `backward`, `transform` | `public-cnn-layer-fit`: returned identity, running statistics, batch count, restored evaluation mode and subsequent transform |
| `BatchNorm2d.fit` | `forward`, `backward`, `transform` | `public-cnn-layer-fit`: returned identity, running statistics, batch count, restored evaluation mode and subsequent transform |
| `CNNClassifier.predict` | `fit`, `predict_proba` | `public-cnn-classifier-predict`: actual returned class words, checked against the probability argmax and hashed on training/held-out inputs |

The existing `tools/identity_lanes/cnn.py` is unchanged, preserving admitted
payloads. Both added lanes are **OWED**: source reachability is not execution,
cross-vendor agreement, or reference admission. There are no binding or
numerical source changes and no mapping whitelist or exemption.

Eight public regression cases in
`python/mojolearn/tests/test_public_cnn_verifier_methods.py` check convolution
fit's shape/weight contract, both batch-normalization classes' single-batch
updates against a float64 oracle in both starting modes, and classifier label
gathering for row/image inputs with nonconsecutive class labels. These tests
were authored without running product code in the source lane.

The exact reason for the historical over-credit is not established. Read-only
source analysis found no CNN name change when removing/readding the seven new
public lanes (555 versus562 registered lanes); no helper crossed the shared
helper threshold. Scoped name-closure traces on both frozen Python source trees
also omit the five absent operations, including the `probe` seed introduced by
the shared harness result wrapper. This does not reproduce a nondeterminism
bug, so no speculative auditor change was made.

The complete retained audit comparison and source-only traces are under
`~/mojolearn-evidence/codex-recovery-20261023/cnn-coverage-audit/`:
`method-comparison.json`, `lane-names.json`, `seeds.json`,
`closure-origins.json`, and `closure-with-probe.json`, with their source scripts
and complete logs. Root qualification must rerun the changed-method audit,
the public regression file, and both new verifier lanes on the frozen native
build before claiming their coverage green.
