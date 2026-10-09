# Artifact format and usage

| File | What it is |
|---|---|
| `checkpoint.blm` | The original step-5,000 training checkpoint, unmodified (SHA-256 `4129921e…02c8`) |
| `model.safetensors` | The checkpoint's parameter array alone, cut into 110 named tensors, byte for byte |
| `config.json` | Architecture settings (MojoLearn format, not a Transformers config) |
| `architecture.json`, `checkpoint-header.json` | The checkpoint's tensor registry and its full JSON header |
| `tokenizer.json`, `ranks.tsv` | The FineWeb-Edu BPE vocabulary, as a `tokenizers` file and as the pinned ranks |
| `reference_torch.py` | A plain PyTorch forward pass for `model.safetensors` |
| `tools/` | Verification, export, evaluation and generation scripts |
| `results/` | Evaluation results and verification receipts |
| `supplement/` | Both routes' training evidence and controls |

## checkpoint.blm

The schema is `mojolearn.byte-lm-stream.v1`: the magic bytes
`MOJOLEARN-BYTE-LM\x01\n`, a little-endian uint64 JSON-header length, the header, its
32-byte SHA-256 digest, then the arrays in header order: `parameters`, `m` and `v`
(each 162,147,840 little-endian float32) and `flags` (110 int32). Each array
descriptor binds dtype, length and SHA-256. The parameters array's SHA-256 is
`c50fd76eb831a5d9754c661780ece20e10fe3b42c89f8b3d4650e2aba215a8ea`.

Readers: `mojolearn._byte_lm_checkpoint.load(path)` (verifies every digest) and
`LanguageModelTrainer.from_checkpoint_binary(path)` (restores the trainer).
`LanguageModelInference.from_checkpoint(path)` reads a different, JSON checkpoint
format; for this file, pass `load(path)["parameters"]` to `LanguageModelInference`
as shown in the README.

## model.safetensors

Tensor names are MojoLearn's: `embed` [50257, 768]; for each `block0`–`block11`,
`norm1_w` [768], `w_q`, `w_k`, `w_v`, `w_o` [768, 768], `norm2_w` [768], `w_gate`,
`w_up` [2048, 768], `w_down` [768, 2048]; and `lm_head` [50257, 768]. Linear weights
are `[out_features, in_features]`. Concatenating the tensors in that order reproduces
the checkpoint's parameter bytes and their SHA-256; `tools/export_safetensors.py`
writes the file and checks this. The file's metadata records the parameters SHA-256.

## Data

The training token stream is identified by SHA-256 in the recipe and is not bundled.
It is FineWeb-Edu `sample-10BT` files 000–003 (training) and 013 (held out), each
document encoded alone with no separator token.
