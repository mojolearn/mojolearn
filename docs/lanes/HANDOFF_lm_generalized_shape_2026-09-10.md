# Configured language-model trainer, September 10

The training implementation now takes nine runtime fields: batch, length,
d_model, n_heads, n_kv, head_dim, intermediate, n_layers and vocab_size.
The same Mojo forward/backward loop serves the original fixture and every
admitted configuration. There is no separate two-block trainer implementation.

```python
from mojolearn import LanguageModelConfig, LanguageModelTrainer

shape = LanguageModelConfig(
    batch=1, length=2048, d_model=768,
    n_heads=12, n_kv=12, head_dim=64, intermediate=2048,
    n_layers=12, vocab_size=50257,
)
print(shape.n_total)       # 162147840; host sizing, no model allocation
print(shape.n_tensors)     # 110
registry = LanguageModelTrainer.parameter_registry(shape)
# Supply initialized FP32 parameters and actual int32 token IDs separately.
# trainer = LanguageModelTrainer(parameters, shape=shape, data_schedule=descriptor)
```

This is the existing RMSNorm/RoPE/SwiGLU architecture with untied embedding and
head, no final norm, biases or dropout. It is not an exact GPT-3 implementation.
Its actual registry count, not a GPT-3 parameter label, must drive budgeting.
The large example was admitted by Python and the native host checks; no large
model allocation, training run, memory-fit or throughput claim follows from it.

`SmallByteLanguageModelTrainer` and `ByteLanguageModelConfig` remain compatible
names for the same implementation. Default v1 and prior seven-field v2 profiles
remain unchanged. Native seven-field inputs default to two layers and 256 tokens;
nine-field inputs specify both. Old seven-field checkpoint shape dictionaries
remain readable. New layer/vocabulary profiles carry all dimensions and v3.
The native v1 ABI delegates directly to the generalized trainer.

Registry names, offsets, gradient dictionaries, optimizer flags, loss/embedding
sizes, token bounds and native checkpoint descriptors all use the configuration.
The public JSON checkpoint remains bounded at 2 MiB; configured state_dict
export/import works independently of that codec. This change does not qualify
large checkpoint I/O or add a production-size checkpoint format.

Evidence: `bench/results/lm_generalized_shape_2026-09-10/`.
76 host tests passed. Native host-only checks include the 162.1M registry.
Metal execution passed four shapes (one, two and three layers, vocabularies
256, 257 and 513), eight gradient/AdamW steps and eight evaluation state-invariance
checks against the existing independent FP64 oracle and unchanged tolerances.
Every configured layer's wrong-SiLU-derivative control was detected.
Four prior retained default/alternate captures match every native array byte,
including losses, full gradients, parameters, moments and flags over two steps.
These are local numerical/regression checks, not new cross-vendor certification.

The layer construction and iteration reference was read in upstream Transformers
`src/transformers/models/llama/modeling_llama.py:354-356,402-412`. The current
stages are moved out of the list temporarily to make their mutable borrow disjoint
from the preceding layer's residual. No activation-copy workaround was added.
Numerical kernels and within-kernel fold order are unchanged.

Next work toward a practical long run:

1. Expose an owned native training session to Python. Today each call constructs
   a DeviceContext/trainer, uploads state and returns full captures. The Mojo
   trainer owns reusable buffers, but its capture API still downloads full state
   before and after every step and transfers cotangents through host lists.
2. Qualify training memory at the chosen large configuration. Forward/backward
   attention stages remain quadratic; loss also retains several token-by-vocab
   arrays. Configuration admission is not memory planning.
3. Add streamed binary checkpoints with shape/registry integrity and exercise
   save/resume at hundreds of MB, then gradient accumulation and schedule support.
4. Measure full pilot steps before promoting GEMM/attention/fusion defaults.
   Large representative runs decide performance gates; these small fixtures
   establish correctness only. Reuse exact cached opponent tuples.

No new performance measurement or opponent timing was made in this change.
Existing kNN/GEMM/attention performance status remains in PERFORMANCE_STATUS.
Trees were untouched.

Continuation: [device gradients and branch audit](HANDOFF_lm_device_gradients_2026-09-10.md).
