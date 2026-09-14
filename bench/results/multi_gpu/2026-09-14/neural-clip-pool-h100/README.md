# Neural clipping tensors — two H100s

The final production and injected-failure builds pass on RunPod
`smqlvlvt7exixd`, two H100 80GB HBM3 GPUs, driver 580.126.09, Mojo/MAX 26.5,
2026-09-14. Builds and tests ran only in the cloud.

Whole registry tensors remain on their assigned GPU through the norm and
scale phases. Each original tensor sum-of-squares kernel retains its complete
contraction length. Only the small canonical scalar vector goes to the first
GPU for the original cross-tensor norm/coefficient calculation. Host staging
publishes scaled gradients and both information scalars after all owners
succeed. One tensor and workspace must fit one GPU. This removes the full
root gradient allocation during clipping; it is not an end-to-end model
capacity or throughput measurement.

## Results

- 45 exact one/two-device comparisons cover five registries, three value
  distributions and three clipping thresholds. Registries include 129 tensors
  and unaligned lengths through 65537; values include signed zero/subnormals.
  Every scaled gradient byte and norm/coefficient byte agrees.
- Nonfinite/overflowed norms and invalid admission refuse without publishing
  gradients or information scalars. Separate fault builds inject after scaling
  on either owner; caller canaries survive and subsequent valid calls recover
  exactly. Production/fault receipt hashes agree.
- Sixty optimizer configurations over three steps, 30 accumulation cases,
  MLP, clipped Mamba Samba and clipped attention/dropout Samba pass.
- Full optimizer, accumulation, MLP and clipped attention/dropout JSON receipts
  equal the preceding gradient-pool phase. This checks unchanged numerical
  behavior for those fixtures, not new RTX 5090 or AMD/Apple execution.

`out/` retains JSON reports, build/test logs, hardware/corpus identities, binary
hashes and exact job scripts/return codes. `out/comparison.log` records the
successful final comparison. Source is `out/source.tgz` (SHA256
`70976048bac2a20ca7e941308ef59551fe222e9f85ac318124f894aa90fb77da`)
with `training/clip_multi_gpu.mojo` replaced by `out/clip-fixed.mojo` (SHA256
`1dd4be29f4801a1b3afe04535a32aeb12de55967ad5a9268de1b9e849a624061`).

The initial build passed, but its first pooled clipping test refused a NaN
norm. The host sum-of-squares List was not retained through its asynchronous
upload. The fix explicitly retains it until synchronization and retains final
publication sources through their copies. `out/initial/`, the original source
archive and failed job retain that defect's evidence; the successful retry
rebuilt production and fault binaries from the corrected source.

The pod remains leased for subsequent IsolationForest/checkpoint work. A
termination claim is not part of this intermediate receipt.
