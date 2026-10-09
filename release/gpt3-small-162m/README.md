---
language:
- en
license: apache-2.0
library_name: mojolearn
pipeline_tag: text-generation
datasets:
- HuggingFaceFW/fineweb-edu
tags:
- mojolearn
- reproducibility
- bitwise-identical
- cross-vendor
- fp32
- research
---

# MojoLearn 162M: one model, two hardware routes

A 162M-parameter decoder trained twice from the same initialization on the same
2.62 billion FineWeb-Edu tokens, once along each of two different routes through
NVIDIA, AMD and Apple GPUs. The two runs are **bitwise identical at every one of the
5,000 optimizer steps**: training state, summed gradients, all 64 per-shard losses
and the learning rate. Their final checkpoint files have the same SHA-256. This
repository distributes that one final model with the evidence from both routes.

The model is GPT-3 Small-sized and trained with GPT-3 Small's hyperparameters, but
its blocks are modern Llama-style blocks; see the comparison below. It is a research
artifact for reproducible training, not a production language model.

## Compared with GPT-3 Small

| | GPT-3 Small (Brown et al., 2020) | This model |
|---|---|---|
| Layers / width / heads × head dim | 12 / 768 / 12 × 64 | same |
| Context | 2,048 | same |
| Batch | 0.5M tokens | 524,288 tokens (64 shards × 4 × 2,048) |
| Peak learning rate | 6.0e-4 | same |
| Optimizer | Adam β=(0.9, 0.95), weight decay 0.1 | AdamW, same β and decay, eps 1e-8 |
| Schedule | warmup, cosine decay to 10% | 250-step warmup, cosine to 10% |
| Positions | learned table | rotary (θ = 10,000) |
| Norm | LayerNorm | RMSNorm (eps 1e-6), pre-norm, **no final norm** |
| MLP | GELU, 3,072 wide | SwiGLU, 2,048 wide (same parameter count) |
| Biases | yes | none |
| Input / output embeddings | tied | untied |
| Parameters | 125M | 162,147,840 (the extra 38.6M are the untied input table) |
| Tokenizer | GPT-2 BPE | MojoLearn BPE, 50,257 ids, trained on FineWeb-Edu |
| Training data | 300B tokens, GPT-3 mix | 2.62B tokens, FineWeb-Edu |
| Arithmetic | mixed precision | FP32 throughout |

Per token, the matrix-multiply work is the same as GPT-3 Small's: SwiGLU's three
768 × 2,048 matrices hold exactly as many weights as GELU's two 768 × 3,072 ones,
attention is identical, and both models multiply by a 50,257 × 768 output matrix.
The untied input table is a lookup with no arithmetic, but it adds 38.6M parameters
to every optimizer update and to every checkpoint whose bits must agree.

The token budget is Chinchilla-scale: about 21 tokens per parameter of a
125M-parameter GPT-3 Small, or about 16 per parameter of this 162M model. It
is about 1/114 of GPT-3 Small's 300B tokens, so this model is not expected to match
GPT-3 Small's quality, and the comparison below is for orientation only.

## Results

All results are from the single distributed checkpoint (step 5,000).

| Evaluation | Result |
|---|---|
| Held-out FineWeb-Edu loss | 3.265 nats/token (perplexity 26.2) |
| Held-out bits per UTF-8 byte | 0.976 |
| HellaSwag, 0-shot, validation (10,042) | acc 28.0%, acc_norm 29.6% |
| Final training loss (step 5,000, mean of 64 shards) | 3.140 nats/token |

- **Held-out loss** is computed by MojoLearn's own CPU forward pass
  (`LanguageModelInference.loss`) on 128 windows of 2,049 tokens
  (262,144 scored tokens), one every 721,773 tokens from the start of the corpus's
  held-out shard (FineWeb-Edu `sample-10BT` file `013_00000.parquet`, token range
  2,926,502,182–3,110,556,447, which no training step read); the windows span its
  first half, tokens 2,926,502,182–3,018,169,402. Bits per byte divide the summed loss
  by the UTF-8 length of the scored tokens and do not depend on the tokenizer.
  `tools/fetch_heldout.py` (which reads the token stream from the project's R2
  store; it is not public) and `tools/heldout_loss.py` reproduce it; details in
  `results/heldout-loss.json`.
- **HellaSwag** uses the EleutherAI lm-evaluation-harness formulation (acc_norm
  normalizes by the choice's UTF-8 length) through `reference_torch.py`. GPT-3 Small
  reported 33.7% zero-shot after 300B tokens. Details: `results/hellaswag.json`.
- **Training loss** comes from the chain records both routes share. It is not a
  held-out metric.
- **Generations** (greedy, 48 new tokens, MojoLearn CPU forward pass) are in
  `results/generations.json`. They are fluent and on topic, repeat themselves as
  greedy decoding at this scale does, and state facts unreliably. For example:

  > **Photosynthesis is the process by which** plants convert sunlight into energy.
  > The process of photosynthesis is the process by which plants convert sunlight into energy. …

## Using the model

### With MojoLearn (the training library)

```sh
pip install mojolearn tokenizers numpy
```

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

`LanguageModelInference` runs MojoLearn's FP32 forward pass on the CPU, with
`logits`, `loss` and greedy `next_bytes`. It was tested with mojolearn 0.8.35 from
PyPI on macOS arm64 (`tools/generate_mojolearn.py`, `results/generations.json`).
`LanguageModelTrainer.from_checkpoint_binary("checkpoint.blm")` restores the full
training state, AdamW moments included, for continued training (not exercised in
this package).

### Anywhere else: safetensors + PyTorch

`model.safetensors` holds the same weights as `checkpoint.blm`'s parameter array,
byte for byte (`tools/export_safetensors.py` writes and checks it), under MojoLearn's
tensor names. Linear weights are stored `[out_features, in_features]`.
`reference_torch.py` is a plain PyTorch implementation of the forward pass:

```sh
pip install torch safetensors tokenizers
python3 reference_torch.py "The water cycle describes how water" --tokens 40
```

On held-out tokens its logits agree with MojoLearn's to within
3.8e-05 (largest absolute difference) and it picks the same greedy token at
every position checked (`tools/compare_reference.py`, `results/reference-agreement.json`).
It is a convenience, not part of the bitwise claim: PyTorch sums in its own order.

The weights do **not** load into Transformers' `LlamaForCausalLM` unchanged, because
this model has no final norm before the output head.

### Tokenizer

`tokenizer.json` is a Hugging Face `tokenizers` file (byte-level BPE with GPT-2's
pre-tokenizer pattern). Its vocabulary is identical to `ranks.tsv`, the file whose SHA-256
the training recipe pins. Do not substitute the GPT-2 or GPT-3 tokenizer. Training
documents were concatenated **without** `<|endoftext|>` separators, so the model
never learned to emit id 50,256 and will not stop on its own.

## Training

| | |
|---|---|
| Initialization | NumPy `default_rng(93261)`, N(0, 0.02); norm weights 1 + N(0, 0.02). Both routes start from the same step-0 file (SHA-256 `2b03554d…`) |
| Steps | 5,000 per route |
| Tokens | 524,288 per step; 2,621,440,000 per route |
| Gradient | 64 logical shards, each a mean over 4 × 2,048 tokens; shard gradients **summed** in a fixed order |
| Optimizer | AdamW, β = (0.9, 0.95), eps 1e-8, weight decay 0.1, no gradient clipping |
| Learning rate | 250-step linear warmup to 6e-4, cosine decay to 6e-5; an exact FP32 table of all 5,000 values is in the recipe |
| Arithmetic | FP32 everywhere; MojoLearn IDENTICAL mode |
| Data | FineWeb-Edu `sample-10BT` files 000–003 for training (2,926,502,182 tokens); file 013 held out |

The complete recipe is
[supplement/gpt3-six-segment-2026-09-28/recipe.json](supplement/gpt3-six-segment-2026-09-28/recipe.json).

## Two routes, one checkpoint

| Steps | Route A | Route B |
|---|---|---|
| 1–1,000 | 2 × NVIDIA H100 | 1 × AMD MI325X |
| 1,001–2,000 | 2 × NVIDIA H100 | 2 × AMD MI300X |
| 2,001–2,400 | H100 + MI325X jointly, in separate machines | 1 × NVIDIA H100 |
| 2,401–3,900 | 2 × AMD MI300X | 2 × NVIDIA H100 |
| 3,901–4,900 | 2 × NVIDIA H100 | 2 × AMD MI300X |
| 4,901–5,000 | Apple M3 Ultra | NVIDIA L40S |

Each route continues from its own checkpoints. The segments also ran different
MojoLearn releases: route A used 0.8.15, 0.8.15, 0.8.17, 0.8.18, 0.8.19 and 0.8.24;
route B used 0.8.18 and then 0.8.22. The bits still agree at every step. All 56
common saved checkpoints match. The final checkpoint SHA-256, recorded by both routes
and re-hashed from both stored copies before packaging, is:

```
4129921e99ed404db1ca4b7b94ebc43c0bee18e7711deb326b31ec89e0b402c8
```

`checkpoint.blm` is that original file: parameters, both AdamW moment arrays,
initialization flags and the training metadata (format in [ARTIFACTS.md](ARTIFACTS.md)).

## Verifying

```sh
python3 tools/verify_release.py        # Python 3.11+; file hashes, checkpoint header and array digests
python3 tools/check_gpt3_evidence.py   # both routes' training records
```

Neither needs a GPU or a network connection. The evidence checker confirms that the
retained records are complete and agree; it does not recompute training. Its exact
scope is in the [supplement README](supplement/gpt3-six-segment-2026-09-28/README.md).
Matching bits establish reproducibility, not model quality; the quality results are
reported separately above.

## Limitations

- A small research model trained on 2.6B tokens: it is not instruction tuned, it
  hallucinates, repeats itself under greedy decoding and does not stop on its own.
- English educational web text only (FineWeb-Edu); it inherits that corpus's biases.
- The MojoLearn forward pass used here runs on the CPU in FP32 and is slow; it is a
  reference path, not a serving stack.

## License, data and citation

The weights, tokenizer and code in this repository are released under the Apache
License 2.0, the license of MojoLearn. The training data is FineWeb-Edu
(HuggingFaceFW/fineweb-edu), released under ODC-By 1.0; its terms and the terms of
the underlying Common Crawl data apply to the data, not to this repository's files.

Source: [github.com/mojolearn/mojolearn](https://github.com/mojolearn/mojolearn), where
this model card, its tools and evidence live in `release/gpt3-small-162m/`. Cite
MojoLearn with its concept DOI
[10.5281/zenodo.22068632](https://doi.org/10.5281/zenodo.22068632).
