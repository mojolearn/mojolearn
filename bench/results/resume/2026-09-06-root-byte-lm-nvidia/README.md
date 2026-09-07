# Two-block byte LM NVIDIA validation — September 6

First attempt: source `659ce86da4932bd2f6442a8faea1ad9cee5e979f`, NVIDIA RTX
4090, explicit `sm_89`, IDENTICAL. Setup, serial-guard checks and comparator
fixtures passed. Compilation failed because two optimizer-config assignments
needed explicit `.copy()` under Mojo ownership rules. No model executed and
there is no learning, gradient or cross-vendor result from this attempt.

Root fixed both assignments without changing arithmetic. Retained first-attempt evidence:
[statuses](run1/remote/byte-lm-validation/results.tsv),
[compiler log](run1/remote/byte-lm-validation/byte-build.log),
[controller](run1/controller.log), and [frozen source](run1/source.tar.gz).

Root alone executed the campaign, under two-core/thread guards, memory limits
and deadlines. Pod `fqrhv7jmdkf3ru` was deleted with HTTP 204 and absence
verified with HTTP 404, with 43 minutes left on its 60-minute lease. The
15-minute source upload is being addressed separately; it was not GPU time.

## Corrected run: single-vendor admission passed

Run 2 used frozen source `d921eade0da5454e39e0cd103b9a1799286b41b9`,
RTX 4090 / `sm_89`, IDENTICAL. All 12 serial jobs passed, and root's fetched
[file-only admission](run2/admission.json) passed. The independent first-step
FP64 gradient/AdamW oracle and controls passed before the separate 128-step
training run. The first step matched the full run byte for byte.

The two-block, 34,944-parameter next-byte model trained on pinned real text.
Held-out loss fell from 5.5412986278533936 to 2.8436418771743774, ratio
0.5131724651836634, passing the predeclared maximum ratio 0.9. Evaluation
left training state unchanged. This is a tiny-model single-vendor learning
result, not cross-vendor identity, resume, useful generation or performance
certification.

Retained [job statuses](run2/remote/byte-lm-validation/results.tsv),
[oracle](run2/remote/byte-lm-validation/gradient-oracle.json),
[full capture summary](run2/remote/byte-lm-validation/full128/summary.json),
[controller](run2/controller.log), and [frozen source](run2/source.tar.gz).
Root alone executed all jobs with two-core/thread guards, memory bounds and
deadlines. Pod `g0by6c4ijuuwdb` was deleted (HTTP 204) and absence verified
(HTTP 404), with 57 minutes remaining on the 60-minute lease.
