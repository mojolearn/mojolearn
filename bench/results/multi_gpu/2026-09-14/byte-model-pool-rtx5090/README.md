# Layer-owned byte-LM model training and capacity: RTX 5090

2026-09-14, RunPod `cvisjmryfcmz6g`, two RTX 5090 GPUs, 32607 MiB each,
driver 580.126.09, IDENTICAL sm_120. All builds and tests ran on RunPod.

Implementation commits: `375fad66c` (complete model driver/binding/Python),
`8d971d9ba` (separate head owner, including one decoder layer on two GPUs),
`780b7b2b4` (complete hash receipts and same-driver capacity baseline).
The earlier layer component is `67efe046b`. The exact final source archive
is `model-pool-v2-out/source-780b7b2b4.tgz`, SHA256:
`cd53edd9ba9d8973d490ce47a777c68d30d91a2f16b0448caf920fb4d31e9094`.
The archive includes the qualified frozen source and these overlays; an old
`commit.txt` inside it is not its complete source identity. The same archive
was built and checked on H100 (the neighboring receipt directory).

## What is pooled

Decoder layers have one owner. Their weights, saved forward/backward stages,
packed gradients and ordered sums stay on that GPU. The first device owns
embedding/head. Canonical parameters, AdamW moments, gradients and rollback
copies are partitioned by those same owners. Transfers copy activations and
cotangents; every existing arithmetic kernel retains its original shape.
Logical gradients use the original sequential FP32 add. All chunks snapshot
before any chunk updates, and rollback attempts every chunk.

The public class is `mojolearn.model_pool_training.PooledByteLanguageModelTrainer`.
It supports one or more logical microbatches independently of physical GPU
count, at most `min(64, n_layers + 1)` GPUs, and canonical checkpoint migration
to/from `ParallelByteLanguageModelTrainer`. Ownership reports describe actual
canonical buffer allocations, not total VRAM. The layer schedule is sequential.

## Exactness and recovery

The final production gate covers 1/2/3 decoder layers with 1/3/5 logical
microbatches: nine fixtures, three steps each and a checkpoint continuation.
Every loss, gradient, parameter, moment, flag and step matches one-GPU ordered
replay. Gates include reversed device order after restore, pooled-to-replica
checkpoint migration, invalid final tokens, malformed post-native results,
and actual allocation lengths. The native gate additionally covers length 33,
four layers and every layer's forward/backward path. `model-pool-final/` holds
the final loss/gradient/state hash receipts.

The separate fault build plants bad summed gradients, refused moments,
nonfinite updated moments and negative updated moments at an owner on device 1.
Earlier chunks have already updated in the optimizer-failure cases. All state
is restored bit for bit and replay succeeds. Production and fault final state,
gradient and loss hashes match. The original byte-LM regression receipt remains
identical. Production was restored after the fault gate. Binary hashes, build
logs, gate logs and job return codes are retained.

## Measured capacity

One fixed 958,746,624-parameter model (28 layers, d_model 1536, FF 6144,
24 attention heads, 6 KV heads, head_dim 64, vocab 256, B1/L1) uses the same
initial state and first two bytes of the R2 enwik8 corpus in all arms.

- Existing one-GPU replica trainer: actual CUDA_ERROR_OUT_OF_MEMORY.
- New model-pooling driver with one GPU: actual CUDA_ERROR_OUT_OF_MEMORY.
- The same new driver with two GPUs: completed an optimizer step, with finite
  state scans; 21766 MiB and 21732 MiB resident usage afterward.

The final capacity records are `model-pool-v2-out/capacity-pooled*.json`.
The original baseline and first implementation's successful capacity leg are
also retained under `model-pool-out/`. This is a capacity fixture, not a learning
or speed benchmark. It retains complete host initialization/checkpoint arrays;
an individual layer and embedding/head must fit their owner. Single-GPU replay
of this oversized model needs additional host offload work.

Initial gate compile errors (invalid test references) and the first capacity
setup error (missing data_schedule argument) are preserved. Later gates and
all owed jobs completed before pod termination; see `job-status.txt` and
`termination.log`. This receipt does not establish AMD/Apple identity or full
pooling for other estimators.
