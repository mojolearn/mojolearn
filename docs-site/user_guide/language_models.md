# Language models

mojolearn trains and runs decoder language models in FP32. The trainer runs on the GPU; inference also runs
on the CPU. Tokenization is part of the model and is chosen before the
first step. The names are listed in the [language model API reference](../api/language_models.rst).

| Class | Runs on | Purpose |
|---|---|---|
| `LanguageModelConfig` | - | Model shape: batch, length, width, heads, layers, vocabulary |
| `LanguageModelTrainer` | GPU | Forward, backward and AdamW update; logits, greedy next tokens, checkpoints |
| `LanguageModelInference` | CPU | Forward only: `logits`, `loss`, `loss_bits`, greedy `next_bytes` |
| `BpeTokenizer` | CPU | Byte-level BPE `encode` and `decode` in the GPT-2 format |
| `lm_corpus` | CPU | Tokenize a corpus once and serve training batches |
| `models.CausalLM` | GPU or CPU | Loader for Hugging Face-format checkpoints |

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
| `state_dict()` / `load_state_dict(state)` | Python dict | In-process handoff |

A checkpoint holds the parameters, both AdamW moments, the step count and the data schedule, so
training resumes where it stopped. In `identical` mode a checkpoint written on one supported GPU resumes
on another vendor's GPU with the same bits; see [Moving training between vendors](cross_vendor_training.md).

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
single-threaded reference path. Both give the same bits.

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
registry = mojolearn.LanguageModelTrainer.parameter_registry(shape)   # sized for this vocabulary
n_params = registry[-1]["offset"] + registry[-1]["size"]
params = (0.02 * rng.standard_normal(n_params)).astype(np.float32)
trainer = mojolearn.LanguageModelTrainer(params, shape=shape, data_schedule=batches.data_schedule())
trainer.train_step(batches.ids(0))
```

Batches read only the corpus's training range, so no step reads a validation or test token.

## Hugging Face checkpoints

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
SmolLM2-360M logits through this loader are recorded equal across NVIDIA, AMD, Apple and the CPU.

## Related pages

- [Neural networks](neural_networks.md): the blocks the language model is built from.
- [Moving training between vendors](cross_vendor_training.md): resume a run on another vendor's GPU.
