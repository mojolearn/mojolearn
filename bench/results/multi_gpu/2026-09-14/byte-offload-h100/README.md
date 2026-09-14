# One-GPU byte-LM replay with host offload

Cloud-only implementation/qualification on RunPod pod `6ewmfb4taf9u1q`, two
H100 80GB HBM3 GPUs (81559 MiB each), driver 580.126.09, IDENTICAL, `sm_90`.
Offloaded training selects one physical GPU. The other GPU is used by the
comparison model-pool driver. No local build, test or model execution ran.

The new native driver keeps canonical P/M/V, gradient sums and saved layer
inputs on the host. GPU work owns one decoder layer or optimizer chunk at a
time, plus persistent embedding/head buffers. Backward recomputes the original
layer forward stages. Ordered gradient additions and AdamW use the existing
kernels; no CPU floating-point training arithmetic is introduced. All chunk
updates are staged before publishing the new host state, with a retained host
snapshot for post-native result-publication rollback.

## Sources

Implementation commit `568546615`; capacity gate `3ded9dfe5`; independent
process memory sampling `af172e5b6`. `out/source-final.tgz` is the authoritative
frozen executable-source archive, SHA256:

`e0b861d017676c021be0b095059eebb84fe22063d8265f8c649024531d5f40fb`

It overlays these files on the previous qualified model-pool archive:
`training/byte_lm_offload.mojo`, `python/mojolearn/offload_training.py`,
`bindings/_mojolearn_byte_lm.mojo`, the native/public offload checks and the
capacity check. The historical `commit.txt` inside the base archive is not
its complete identity. Later merges added identity receipts, workflow metadata
and documentation; they did not change this training arithmetic. No binaries
or corpus bytes are committed; binary hashes and the corpus SHA are retained.
R2 supplied the pinned 100,000,000-byte enwik8 corpus.

## Checks and limits

- Four distinct native model shapes, including sequence length 33, compare
  every loss, P/M/V value and gradient against original single-device ordered
  replay; invalid final tokens, rollback and re-execution are checked.
- Twelve public shape/logical-count groups (one/two/three layers;
  one/three/five/eight logical microbatches) compare offload with two-GPU model pooling and original
  replay. They include nonzero moments, step 7, weight decay, checkpoint
  continuation on the other GPU, migration to the replica driver, and failed
  result-publication recovery.
- The separate fault binary injects nonfinite gradients, optimizer refusal,
  nonfinite updated moments and negative updated moments at a later canonical
  chunk. Earlier staged updates must not escape into published host state.
- The 958,746,624-parameter fixture runs two consecutive updates with three
  logical microbatches each. Per-step hashes cover all P/M/V/flags, gradients
  and losses. It has uniform initialization and is a capacity/identity fixture,
  not a learning-quality or throughput result.

The first pooled capacity process had already loaded the thread-based sampler
when the process-based replacement arrived. Its state/gradient results remain
valid, but its sampler could not run during native calls holding the Python
GIL. `capacity-pooled-thread-sampler.*` and `initial-thread-sampler.py` preserve
that record. The final pooled measurement repeats with the same external
`nvidia-smi` process sampler as offload and retains raw `.memory.csv` samples.
Sampling is periodic; reported peaks are observed samples, not allocation
traces or formal upper bounds.

`out/reference-5090/` contains earlier RTX5090 pooled-model and original-byte
receipts. Matching them is a check against established outputs, not evidence
that this offload implementation itself ran on RTX5090. Three RunPod requests
(one/two GPUs) found no RTX5090 stock; the refusals are retained. RTX5090 and
AMD/Apple offload qualification, eight physical devices, and full pooling for
the remaining estimators are still owed. Each individual decoder layer and
its training buffers, an optimizer chunk, and the persistent embedding/head
must fit the selected GPU. Full canonical state and transaction copies must
fit host RAM. A single oversized layer is not partitioned by this driver.

## Result

All five named jobs (`offload-build`, `offload-fault-build`, `offload-public`,
`offload-final`, `offload-eight`) completed with return code zero. The native
checks, twelve public groups, three fault groups and original byte-LM regression
passed. `out/comparison.json` verifies nine complete state/gradient/loss hash
groups against the earlier RTX5090 pooled receipts, three fault-recovery groups,
the unchanged original byte-LM receipt, and both oversized updates against
pooled training. The three additional eight-logical-shard groups compare all
bytes directly within their gate; eight physical GPUs were not exercised.

For 958,746,624 parameters and two updates of three logical microbatches each,
all P/M/V/flags, gradient and loss hashes match. External process sampling
observed 21551/21551 MiB for the pooled H100 run and 2621 MiB on the selected
H100 for offload. The other device's observed maximum was 531 MiB; the offload
driver itself opens only device 0. The offload run's selected-device samples
stay below 32 GiB; this does not substitute for a run on an actual RTX5090.
The earlier actual one-RTX5090 out-of-memory baseline is documented in
`../byte-model-pool-rtx5090/`.

All jobs finished and receipts were downloaded before pod termination.
`termination.log` records DELETE 204 followed by GET 404.
