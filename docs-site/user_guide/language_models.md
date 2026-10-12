# Language models

mojolearn trains and runs decoder language models in FP32. The trainer runs on the GPU; inference and a
reference training step also run on the CPU. Tokenization is part of the model and is chosen before the
first step. The names are listed in the [language model API reference](../api/language_models.rst).

| Class | Runs on | Purpose |
|---|---|---|
| `LanguageModelConfig` | - | Model shape: batch, length, width, heads, layers, vocabulary |
| `LanguageModelTrainer` | GPU | Forward, backward and AdamW update; logits, greedy next tokens, checkpoints |
| `LanguageModelInference` | CPU | Forward only: `logits`, `loss`, `loss_bits`, greedy `next_bytes` |
| `LanguageModelHostTrainer` | CPU | One training step on the CPU (reference path) |
| `BpeTokenizer` | CPU | Byte-level BPE `encode` and `decode` in the GPT-2 format |
| `lm_corpus` | CPU | Tokenize a corpus once and serve training batches |
| `models.CausalLM` | GPU or CPU | Experimental loader for Hugging Face-format checkpoints |

`ByteLanguageModelConfig` and `SmallByteLanguageModelTrainer` are older names for `LanguageModelConfig`
and `LanguageModelTrainer`.

## The model

The decoder uses Llama-style blocks: RMSNorm before attention and before the MLP, rotary positions,
grouped-query attention and a SwiGLU MLP. It has no biases, no dropout, no final norm, and untied input
and output embeddings. There is no hidden initialization, padding or random number generator: you
supply the parameters.

## Train a model

```python
import numpy as np
import mojolearn

rng = np.random.default_rng(0)
shape = mojolearn.LanguageModelConfig(batch=2, length=32, d_model=32, n_heads=4, n_kv=2,
                                      head_dim=8, intermediate=64, n_layers=2, vocab_size=256)

registry = mojolearn.LanguageModelTrainer.parameter_registry(shape)  # name, shape, offset, size
n_params = registry[-1]["offset"] + registry[-1]["size"]
params = (0.02 * rng.standard_normal(n_params)).astype(np.float32)   # your initialization

trainer = mojolearn.LanguageModelTrainer(
    params, shape=shape, lr=3e-4, weight_decay=0.1,
    data_schedule={"corpus_sha256": "...", "batch": 2, "length": 32},
)
ids = rng.integers(0, 256, size=(2, 33)).astype(np.int32)   # (batch, length + 1)
result = trainer.train_step(ids)
logits = trainer.logits(ids[:, :32])                         # (batch, length, vocab_size)
```

`train_step` takes int32 ids of shape `(batch, length + 1)`: the first `length` positions predict the
next `length`, and the loss is the mean over all targets. `parameters` may be a flat FP32 array laid out
by `parameter_registry(shape)`, or the named tensors it lists.

`data_schedule` is a JSON object you write, such as the corpus hash, sample order and batch offsets. It is
stored with the step counter in every checkpoint. The trainer does not fetch or reorder data and does not
check the descriptor against the batches you pass; after a restore, you feed the next batch.

`resident=True` keeps the model and optimizer on the GPU between calls; the state is copied out by
`state_dict()`, `export_gradients()` and the checkpoint methods.

## Checkpoints

| Method | Format | Use |
|---|---|---|
| `export_checkpoint(path)` / `from_checkpoint(path)` | JSON, at most 2 MiB | Small models |
| `export_checkpoint_binary(path)` / `from_checkpoint_binary(path)` | Binary `mojolearn.byte-lm-stream.v1` | Any size; returns the file's SHA-256 |
| `state_dict()` / `load_state_dict(state)` | Python dict | In-process handoff, multi-GPU trainers |

A checkpoint holds the parameters, both AdamW moments, the step count and the data schedule, so
training resumes where it stopped. In `identical` mode a checkpoint written on one supported GPU resumes
on another vendor's GPU with the same bits; see [Cross-vendor training](cross_vendor_training.md).

## Inference on the CPU

`LanguageModelInference` runs the forward pass on a CPU, with no GPU:

```python
model = mojolearn.LanguageModelInference.from_checkpoint("final.checkpoint.json")
logits = model.logits(ids[:, :32])              # float32 (batch, length, vocab_size)
next_ids = model.next_bytes(ids[:, :32])        # greedy; ties go to the lowest id
logits = model.logits(ids[:, :32], lengths=[32, 17])   # ragged, right-padded rows
```

`LanguageModelInference(parameters, shape=shape)` builds the same model from a parameter array.
By default calls split batch rows across threads (`threaded=True`); `threaded=False` runs the
single-threaded reference path. Both give the same bits. The CPUs on which the forward pass has been checked
against the recorded GPU bytes are listed in
[BYTE_LM_CPU_INFERENCE.md](https://github.com/mojolearn/mojolearn/blob/main/docs/BYTE_LM_CPU_INFERENCE.md).

`LanguageModelHostTrainer(parameters, m=..., v=..., shape=shape, completed_steps=...)` runs one forward,
backward and AdamW step on the CPU. It has no threaded path. What its gate has shown is recorded in
[BYTE_LM_CPU_TRAINING.md](https://github.com/mojolearn/mojolearn/blob/main/docs/BYTE_LM_CPU_TRAINING.md):
on seven CI runners it reproduced every recorded step of the gate's small model from that step's own
state. Treat it as a reference path, not a fast trainer.

## Tokenizers and corpora

`BpeTokenizer` encodes and decodes byte-level BPE in the GPT-2 format. Load your own vocabulary:

```python
tok = mojolearn.BpeTokenizer.from_ranks_file("ranks.tsv")
# or: mojolearn.BpeTokenizer.from_files("encoder.json", "vocab.bpe")
ids = tok.encode("The water cycle describes how water")
text = tok.decode(ids)
```

`<|endoftext|>` is the last id and is recognized only with `encode(..., allow_endoftext=True)`.
mojolearn ships no vocabulary and no corpus.

`lm_corpus.prepare` tokenizes a text file once and caches the ids (under `$MOJOLEARN_LM_CACHE`, else
`~/.cache/mojolearn/lm`). With no `vocab`, it trains a vocabulary on the corpus's training range with
`mojolearn.tokenizer.BpeVocabularyTrainer`; the same corpus and settings give the same vocabulary bytes on
any machine. Pass `vocab=` to use your own token map instead:

```python
from mojolearn import lm_corpus

corpus = lm_corpus.prepare("corpus/input.txt", vocab="ranks.tsv")
batches = corpus.batches(batch=4, length=2048)
shape = mojolearn.LanguageModelConfig(batch=4, length=2048, vocab_size=corpus.n_vocab)
trainer = mojolearn.LanguageModelTrainer(params, shape=shape, data_schedule=batches.data_schedule())
trainer.train_step(batches.ids(0))
```

Batches read only the corpus's training range, so no step reads a validation or test token.

## The published 162M model

[mojolearn-ai/mojolearn-162m](https://huggingface.co/mojolearn-ai/mojolearn-162m) is a 162,147,840-parameter
decoder (12 layers, width 768, 12 heads of 64, context 2,048, vocabulary 50,257) trained in FP32 in
`identical` mode on 2.62 billion FineWeb-Edu tokens. It is a research artifact for reproducible training,
not a production model. Download `checkpoint.blm` and `tokenizer.json`, then:

```python
import numpy as np
from tokenizers import Tokenizer
from mojolearn import ByteLanguageModelConfig, LanguageModelInference
from mojolearn import _byte_lm_checkpoint as ck

state = ck.load("checkpoint.blm")            # verifies every array digest
shape = ByteLanguageModelConfig(**{**state["model_shape"], "batch": 1, "length": 128})
model = LanguageModelInference(state["parameters"], shape=shape)
tok = Tokenizer.from_file("tokenizer.json")

ids = tok.encode("The water cycle describes how water").ids
for _ in range(24):
    ids.append(int(model.next_bytes(np.array([ids[-128:]], dtype=np.int32))[0]))
print(tok.decode(ids))
```

This uses the Hugging Face `tokenizers` package (`pip install tokenizers numpy`). Use the model's own
`tokenizer.json`; do not substitute the GPT-2 tokenizer. The model was trained without end-of-text
separators and does not stop on its own. `LanguageModelTrainer.from_checkpoint_binary("checkpoint.blm")`
restores the full training state, AdamW moments included. File formats are described in
[ARTIFACTS.md](https://github.com/mojolearn/mojolearn/blob/main/release/gpt3-small-162m/ARTIFACTS.md).

## Hugging Face checkpoints (experimental)

`mojolearn.models.CausalLM.load` builds a model from a directory holding `config.json` and
`model.safetensors` (or a sharded index). `models.Tokenizer.from_pretrained` reads `tokenizer.json` for
byte-level BPE families.

```python
from mojolearn import Array
from mojolearn.models import CausalLM, Tokenizer

lm = CausalLM.load("path/to/model", weight_format="bfloat16")
tok = Tokenizer.from_pretrained("path/to/model")
out = lm.generate(Array.from_list([tok.encode("The capital of France is")], "<i4"), 8)
```

`generate` is greedy and produces exactly `max_new_tokens` per row. `forward`, `allocate_state` and `step`
are also available. `weight_format` is `"float32"`, `"bfloat16"` or `"int8"` weight storage.

| `model_type` | Loader support |
|---|---|
| `llama` (Llama 3, TinyLlama, SmolLM, SmolLM2), `qwen2`, `qwen3`, `mamba`, `mamba2` | Config and weights mapped; byte-level BPE tokenizer |
| `mistral` | Config and weights mapped (sliding window); v3 (tekken) byte-level BPE tokenizer; SentencePiece refused |
| `phi3` | Weights mapped; its SentencePiece tokenizer is refused |
| `gemma`, `gemma2` | Refused |

Arithmetic is FP32 (`fp32_v1`) by default. `numeric_profile="fixed15_v1"` opts transformer models into
15-bit integer matrix products; it gives different bits from `fp32_v1`, and it is slower on Apple.
SmolLM2-360M is the published model whose full logits are recorded through this loader on NVIDIA, AMD,
Apple and the CPU. The other families are covered by the name mapping and refusals, not by recorded logits.
See [CHANGELOG 0.8.29](https://github.com/mojolearn/mojolearn/blob/main/CHANGELOG.md).

## Related pages

- [Neural networks](neural_networks.md): the blocks the language model is built from.
- [Multi-GPU training](multi_gpu.md): train one language model on several GPUs.
- [Cross-vendor training](cross_vendor_training.md): move a run between vendors.
