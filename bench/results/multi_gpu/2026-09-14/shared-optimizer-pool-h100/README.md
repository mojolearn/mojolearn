# Shared optimizer pooling: two H100s

RunPod `rbtojh7e0esekh`, IDENTICAL, sm_90a. All compilation and model
execution occurred on RunPod with the existing R2 enwik8 corpus.

The optimizer gate passes 60 configurations, three steps each, comparing every
parameter, first/second moment, gradient, momentum flag and clip-info bit.
It covers SGD without momentum, SGD with dampening, Nesterov SGD, Adam and
AdamW; no clipping, active clipping and a clip coefficient of one; four
registries including single-cell state and tensors spanning device ownership
boundaries. Seven refusal checks leave all caller bytes unchanged: NaNs in
each scanned input, invalid configured device counts and a third physical
device requested on the two-device pod.

Full MLP and Samba training match one-GPU ordered replay and checkpoint
migration. Samba covers Mamba3 alone and Mamba3 plus attention/dropout, with
and without global clipping. Original MLP, Samba and attention/dropout hashes
all match the earlier continued-H100 receipt (`golden-comparison.json`).
The models job exits zero, including both new clipped cases.

Initial native runs crashed because raw worker pointers did not retain device
context owners. `88fe55c45` keeps contexts and host scratch alive through the
join; the native gate then passed. The MLP phase of that job stopped because
the new pod lacked its linalg binding. After building that dependency, the
models job passes. The concurrent initial Samba job had used the earlier
binary and its worker-crash log is retained. No failing run is discarded.

`optimizer-pool-qualified-source.tgz` captures final source through
`a8b158e91`. It overlays the qualified byte-LM source, QR/Gram and pointwise
changes on the initial `eaec62839` pod checkout. Training was rebuilt from
`88fe55c45`; the subsequent commit adds only the clipped model gate. Mamba,
Transformer and linalg dependencies use the pod's existing source chain.
All source archives, build logs, binary hashes and job scripts are retained.

Parameters/gradients/moments are distributed only for the update phase and
staged in host memory between calls. Global clipping still stages a complete
gradient on the first GPU, and gradient workers still need full model weights.
This is not persistent residency, full pooled model capacity, new cross-vendor
qualification, or a speedup claim. No local builds or tests ran.

Final lifecycle: all owed jobs completed and evidence was downloaded before
RunPod `rbtojh7e0esekh` was deleted on 2026-09-14 at 18:11 UTC; GET returned
404. See `../pooled-source-freeze/h100-termination.log`. The exact final source
tree was frozen for subsequent RTX 5090 qualification.
