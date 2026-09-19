# NVIDIA hardware coverage follow-up

The eleven newer parallel lanes now have complete two-L40S captures: nine default
fixtures, two repeats, and 378 numerical fixture-parts matching the shipped
references. These ran the published 0.8.8 Linux wheel (SHA and source commits in
`new-lanes-summary.json`). GPU identities are retained in `nvidia-gpu-inventory.csv`.
No existing reference digest changed.

The first installed distributed/CV attempts exposed a receipt serialization bug:
`environment()` returns a ModeReport object, which JSON cannot serialize. Commit
c3b5783cfee741673e6dd17fcb34f377ea50c6f5 adds an explicit JSON environment snapshot.
The two verifiers then passed on a private candidate containing that fix:
30 distributed numerical cases and ten transport controls; twelve CV runs across
one/two/reversed devices and eight comparator controls. Both retain distinct
worker/device inventory. Kernel execution traces and native arithmetic fault
controls are not claimed by these receipts.

The candidate shares the version label 0.8.8 but is NOT the published wheel.
Its distinct SHA is recorded in `receipt-candidate-audit.json`; all 123 native and
runtime files are byte-identical to the published wheel. Only the three Python
verification/reporting modules changed. This candidate has not been published.
36 targeted software tests passed, including preservation of mode-mismatch
information through JSON serialization.

These records close the listed NVIDIA gaps; they do not certify all algorithms
on every device. Historical records exist for 39 other parallel lanes, but some
omit later model/batch checks, and their UMAP revision is outdated. A separate
bounded capture is working on those missing parts. AMD execution of the newer
lanes remains pending: RunPod reports no two-MI300X capacity, and DigitalOcean
rejected an eight-MI325X request because it exceeds the account GPU quota.
No AMD droplet was created.

Full capture snapshots, saved CV models, initial failures and lifecycle receipts:
`/Users/andrewhendel/mojolearn-evidence/hardware-gaps-088/`.

## Additional model-file evidence

`model-gaps-summary.json` and the `nvidia-model-*` captures close ten older
model-file gaps, with nine fixtures and two repeats each. All270 captured
train/infer/model parts match the current references, including90 newly recorded
two-device model witnesses. Existing batch checks were omitted from these jobs.
Four scaler fixtures also completed before the bounded job expired; they remain
in the external partial capture and are excluded from the next run's requests.

The first test pod was deleted and verified absent via HTTP404. A second bounded
NVIDIA leg is targeting only the remaining168 reference-parts plus loaded-model
layer distribution against the retained24-case single-device baseline. AMD
capacity/quota remains unresolved. No release tag or PyPI artifact was changed.
