# `mojolearn.models`: the checkpoint loader (lane/model-loader, 2026-09-17)

A generalizable loader for Hugging Face-shaped causal language models:
`safetensors.py` reads the format, `config.py` holds the option matrix,
`causal_lm.py` assembles embedding, blocks, final norm and head from the
library's own block classes and the Samba stack's three primitives, and
`tokenizer.py` loads `tokenizer.json` for the byte-level BPE families with
the pre-tokenization pattern as a parameter.

**NO REAL CHECKPOINT HAS BEEN LOADED THROUGH THIS PACKAGE YET.** Every test
in `python/mojolearn/tests/test_models_loader.py` writes a tiny synthetic
checkpoint to a temp directory and loads that. Loading a published model,
comparing its logits, and recording the result is lane C's run. Until it is
recorded, this package claims the format reader, the name mapping, the
refusals by name, and that a synthetic Llama-shaped and Mamba-shaped
checkpoint assemble and decode deterministically; it claims no model's
output.

## The option matrix (`config.FAMILIES`, `plan_for`)

The block interface coded against (lane B1 is adding the second and later
lines in parallel; today only the first line exists):

    TransformerBlock(weights, *, n_heads, n_kv_heads=None, head_dim=None, window=0,
        rope_theta=10000.0, rope_scaling=None, rope_dim=None, max_positions=8192,
        qkv_bias=False, o_bias=False, norm="rmsnorm", norm_eps=1e-5, norm_bias=False,
        mlp="swiglu", mlp_bias=False, qk_norm=False, attn_softcap=None)

The loader reads the LIVE signature, passes a keyword only when the plan's
value differs from the interface default, and refuses by name any option the
live block does not accept at a value other than what today's block fixes
(`FIXED_TODAY`: rope_theta 10000, norm_eps 1e-6, no biases, no rope
scaling, no qk_norm, no softcap). So today a Llama 3 config is refused at
`norm_eps=1e-5` and `rope_theta=500000.0`, and it is honored the day B1's
signature lands, with no change here.

| model_type | kind | fields honored (config -> option) | fields refused BY NAME | tensor names | tokenizer |
| --- | --- | --- | --- | --- | --- |
| `llama` (Llama 2/3, TinyLlama, SmolLM/SmolLM2 which ship `model_type: llama`; alias `smollm`) | transformer | `hidden_size`, `num_attention_heads` -> `n_heads`, `num_key_value_heads` -> `n_kv_heads`, `head_dim`, `intermediate_size`, `num_hidden_layers`, `vocab_size`, `rms_norm_eps` -> `norm_eps`, `rope_theta`, `sliding_window` -> `window`, `attention_bias` -> `qkv_bias`, `mlp_bias`, `tie_word_embeddings`, `max_position_embeddings` (capped at 8192) | `rope_scaling` (any non-null: linear, dynamic, yarn, longrope, llama3), `hidden_act` other than silu, `num_local_experts`/`num_experts` (MoE), `attention_dropout` nonzero, `use_sliding_window`/`layer_types` alternation | `model.embed_tokens.weight`, `model.layers.N.input_layernorm.weight`, `.post_attention_layernorm.weight`, `.self_attn.{q,k,v,o}_proj.weight`, `.mlp.{gate,up,down}_proj.weight`, `model.norm.weight`, `lm_head.weight` (absent when tied) | Llama 3 / SmolLM2: byte-level BPE, pattern `llama3`, loads. Llama 2: SentencePiece, refused by name |
| `mistral` | transformer | as llama; `sliding_window` -> `window` (null = full causal) | as llama | as llama | v1/v2 SentencePiece refused; v3 "tekken" is byte-level BPE with the Llama-3-style pattern if its regex matches a known spelling, else refused by name |
| `qwen2` (Qwen 2, 2.5) | transformer | as llama; `qkv_bias` True by default (the family's q/k/v biases, `.self_attn.{q,k,v}_proj.bias` -> `q_proj.bias` etc.) | as llama; `use_sliding_window` True (alternating `max_window_layers`) | llama names plus the three biases | byte-level BPE, pattern `qwen2`, NFC normalizer, loads |
| `qwen3` | transformer | as llama; `qk_norm` True (`.self_attn.{q,k}_norm.weight` -> `q_norm.weight`, `k_norm.weight`), `head_dim` explicit | as llama | llama names plus the two norms | as qwen2 |
| `gemma` | transformer | (shapes are read) | REFUSED at `hidden_activation`/`hidden_act` (GeGLU gelu_pytorch_tanh), the (1 + weight) RMSNorm and the sqrt(hidden_size) embedding scale: no interface value names them | llama names | SentencePiece, refused |
| `gemma2` | transformer | (shapes are read) | REFUSED at `attn_logit_softcapping`, `final_logit_softcapping`, `query_pre_attn_scalar`, then everything gemma refuses; also its pre/post feed-forward norms and alternating window | llama names plus `pre_feedforward_layernorm`, `post_feedforward_layernorm` | SentencePiece, refused |
| `phi3` | transformer | as llama; the fused `self_attn.qkv_proj.weight` is SPLIT by rows into q (n_heads*head_dim), k, v (n_kv_heads*head_dim each) and `mlp.gate_up_proj.weight` into gate then up (intermediate rows each); a row slice copies bytes, no arithmetic | as llama; `rope_scaling` (longrope) and `original_max_position_embeddings` | `model.layers.N.self_attn.qkv_proj.weight`, `.self_attn.o_proj.weight`, `.mlp.gate_up_proj.weight`, `.mlp.down_proj.weight`, the two norms, `model.embed_tokens.weight`, `model.norm.weight`, `lm_head.weight` | SentencePiece, refused |
| `mamba` (state-spaces/mamba-*-hf) | mamba1 | `hidden_size`, `num_hidden_layers`, `vocab_size`, `tie_word_embeddings`, `layer_norm_epsilon` (must be 1e-5, the profile's) | `state_size` != 16, `conv_kernel` != 4, `expand` != 2, `time_step_rank` != auto/ceil(d/16), `use_bias` True, `use_conv_bias` False, `layer_norm_epsilon` != 1e-5, `hidden_act` != silu (each a profile constant of `mamba/IDENTICAL_MAMBA_CONTRACT.md`) | `backbone.embeddings.weight`, `backbone.layers.N.norm.weight`, `.mixer.{in_proj,x_proj,dt_proj,out_proj}.weight`, `.mixer.conv1d.{weight,bias}`, `.mixer.dt_proj.bias`, `.mixer.A_log`, `.mixer.D`, `backbone.norm_f.weight`, `lm_head.weight` | GPT-NeoX byte-level BPE (GPT-2 pattern), loads through the compiled binding |
| `mamba2` | mamba2 | `hidden_size` (multiple of 32), `num_hidden_layers`, `vocab_size`, `tie_word_embeddings`, `time_step_limit` -> `dt_limit`, `layer_norm_epsilon` (1e-5) | `state_size` != 128, `head_dim` != 64, `expand` != 2, `n_groups` != 1, `chunk_size` != 256, `conv_kernel` != 4, `use_bias`, not `use_conv_bias`, `rms_norm` False, `norm_before_gate` True, `num_heads` inconsistent | `backbone.layers.N.norm.weight` -> `block_norm.weight`, `.mixer.{in_proj,out_proj}.weight`, `.mixer.conv1d.{weight,bias}`, `.mixer.dt_bias`, `.mixer.A_log`, `.mixer.D`, `.mixer.norm.weight` -> `norm.weight`, plus the embedding/final-norm/head names above | as mamba |
| anything else (`mixtral`, `gpt2`, `gpt_neox`, `falcon`, `bloom`, `opt`, ...) | | | REFUSED: not in the matrix (the message lists the known types) | | |

Common to every transformer row: `vocab_size` must equal the embedding
table's rows and the head's rows (refused by name otherwise); ids must fit
int32; the blocks' own shape refusals (`d_model == n_heads*head_dim`,
`n_heads % n_kv_heads == 0`, even `head_dim`, and the Mojo constructor's
copies) stay theirs. No vocabulary ceiling was found in
`bindings/_mojolearn_training.mojo` or `bindings/_mojolearn_neural_host.mojo`
(the head is one GEMM at `[n, vocab, d]` and the embedding one gather), so
the loader states none.

Positions: the transformer block refuses absolute positions at or above
8192 in Mojo (DEVIATION 812); `CausalLM.load(max_positions=...)` defaults to
min(`max_position_embeddings`, 8192), refuses more, and checks every call's
length itself before a kernel runs. Mamba models carry no position.

## Weight formats

`CausalLM.load(path, weight_format="float32" | "bfloat16" | "int8")`. The
projection matrices of each block (attention and MLP; Mamba's `in_proj`,
`x_proj`, `dt_proj`, `out_proj`) are handed to the block as
`lowbit.BF16Weight` (the checkpoint's own BF16 bits when it is stored BF16,
no conversion; `lowbit.pack_one` otherwise) or `lowbit.Int8Weight`
(`pack_one(..., "int8")` of the materialized float32). `A_log`, norms,
biases, convolution taps, the embedding table and the head stay float32
(the head GEMM and the gather take float32). Every block reports
`weight_format` equal to what was asked, and the block materializes exactly
and runs its fp32 path (`lowbit.py`'s contract).

## Where the pre-tokenization pattern lives now

| pattern | families | where it runs |
| --- | --- | --- |
| `gpt2` | GPT-2, GPT-NeoX, Pythia, the Mamba checkpoints | the compiled binding (`tokenizer/impl/pretokenize.mojo`) through `GPT2Tokenizer` |
| `llama3` | Llama 3.x, SmolLM2 | Python, `models/tokenizer.py::_pretoken_end_llama` over the byte codes, then the existing Python BPE merge `_tokenizer_synthetic._bpe` (the second implementation `check-tokenizer` holds the Mojo merge to) |
| `qwen2` | Qwen 2, 2.5, 3 | as `llama3` with one digit per pre-token |

The Mojo side could not take a second pattern in this lane (no binding build
and no run on this box, and the binding's `encode` always cuts with the
GPT-2 pattern first, so it cannot be handed pre-cut Llama-3 pieces). Porting
the two patterns into `pretokenize.mojo` beside `pretoken_end` is owed to a
lane that can build the binding; the Python spelling here is its oracle.
`\p{L}`/`\p{N}` are `unicodedata`'s categories at the running Python's
Unicode version (`Tokenizer.unicode_version`), `\s` the White_Space list.
DEVIATION 2960 records that the pattern is a parameter and that the
`llama3` and `qwen2` cuts change tokens.

## Tokenizer status per family

| family | status |
| --- | --- |
| Llama 3, SmolLM2, Qwen 2/2.5/3, GPT-2, GPT-NeoX/Pythia, Mamba | loads (`tokenizer.json` byte-level BPE; NFC normalizer honored; template BOS/EOS added by `encode`) |
| Llama 2, Mistral v1/v2, Gemma, Phi-3 | refused by name (SentencePiece: `tokenizer.model`, `model.type: Unigram`, `byte_fallback`, Metaspace, `▁` spellings) |
| any other pre-tokenizer regex | refused by name with the regex in the message |

## Owed

- Lane C: load one published checkpoint per loading family, compare logits
  against the reference, record. Nothing here is certified until then.
- The `llama3`/`qwen2` patterns in Mojo, with cases, beside `pretoken_end`.
- `tools/identity_break.py` lanes for a whole loaded model (the Samba lanes
  cover the primitives; a loaded model adds only the assembly).
