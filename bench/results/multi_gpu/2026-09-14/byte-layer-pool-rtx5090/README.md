# Byte-LM layer ownership, RTX 5090

2026-09-14, RunPod cvisjmryfcmz6g, two RTX 5090 GPUs. Source parent
576c908d6 plus the layer-pool implementation and gate in this commit.
No local compilation or tests. `native.log` passed, job exit 0.
The initial gate compile failure is retained.

Each decoder layer's weights, saved forward/backward stages, packed gradient
and ordered sum live on exactly one device. Forward activations and backward
cotangents are copied between owners. All arithmetic uses the original
shape-preserving decoder kernels and ordered FP32 addition kernel.

Four fixtures compare every layer's forward and backward outputs, transferred
endpoints, and canonical accumulated gradients against the existing one-device
ByteTrainer graph: 2 layers/d16/L7 on one and two GPUs with three logical
microbatches; 3 layers/d24/L7 on reversed devices [1,0] with five microbatches;
4 layers/d32/L33 on two GPUs with two microbatches. Resident weight lengths
sum to exactly the decoder registry and each GPU owns a proper subset on the
two-GPU cases. Duplicate accumulation refuses without changing the sum.

This receipt qualifies the internal layer component on RTX 5090. It does not
yet qualify a complete model-pooling trainer, beyond-one-GPU capacity,
throughput, or cross-vendor identity. The embedding/head and atomic optimizer
integration are separate work. The pod remains leased for that work.

Final lifecycle: the shared RTX pod completed the later model-pooling jobs
and was terminated. See `../byte-model-pool-rtx5090/termination.log` for
DELETE/GET verification. The earlier statement that it remains leased is superseded.
