# 2026-09-19 models-namespace: the three `mojolearn.models` lanes

Apple M4, CPU host bindings (a CPU-only install: the package tree with no GPU
`.so`, so `vendor()` is `cpu` and `CausalLM`'s `device="auto"` resolves to the
host route), one core, `nice -n 19`, `--repeats 2`, **all nine fixtures**,
lanes `hf-checkpoint`, `hf-tokenizer` and `hf-causal-lm`, on
lane/lm-attention-fallback. `run_arm.sh` is the exact command per arm.

Twenty-seven cells per arm; every arm read `stable=27 moved=0 refused=0`.

| file | arm | the lane it is the control for |
|---|---|---|
| `m4.clean.json` | clean | - |
| `m4.replay.json` | clean again | IDENTICAL=27 |
| `m4.linalg-host.sabotage.json` | linalg host built `-D MOJOLEARN_HOST_SABOTAGE=1` | **none: INERT, see below** |
| `m4.linalg-convert.sabotage.json` | linalg host built `-D MOJOLEARN_HOST_SABOTAGE=1 -D MOJOLEARN_LOWBIT_CONVERT_SABOTAGE=1` | `hf-checkpoint`, 9/9 |
| `m4.tokenizer-encoder.sabotage.json` | tokenizer host built `-D MOJOLEARN_HOST_SABOTAGE=1 -D MOJOLEARN_TOKENIZER_HOST_SABOTAGE=1` | `hf-tokenizer`, 9/9 |
| `m4.tokenizer-trainer.sabotage.json` | tokenizer host built `-D MOJOLEARN_HOST_SABOTAGE=1 -D MOJOLEARN_BPE_TRAINER_SABOTAGE=1` | `hf-tokenizer`, 9/9 |
| `m4.neural-host.sabotage.json` | neural host built `-D MOJOLEARN_HOST_SABOTAGE=1` | `hf-causal-lm`, 9/9 |

Each arm rebuilds ONE family into a fresh directory and links the other
thirty-one clean bindings beside it, so an arm that moves a lane other than
its own would be a mis-attribution and is visible here: none does.

## What moved, per part, on all nine fixtures

| arm | lane | parts moved | parts unmoved |
|---|---|---|---|
| linalg host | *(all three)* | — | *(everything)* |
| linalg convert | `hf-checkpoint` | `dtypes`, `sharded` | `bits`, `infos`, `plan`, `refusals`, `caught`, `flags` |
| tokenizer encoder | `hf-tokenizer` | `gpt2_ids`, `text_ids`, `json_ids`, `specials`, `decoded`, `flags` | `bounds`, `names`, `llama3_ids`, `qwen2_ids` |
| tokenizer trainer | `hf-tokenizer` | `gpt2_ids`, `llama3_ids`, `qwen2_ids`, `json_ids`, `specials` | `bounds`, `names`, `text_ids`, `decoded`, `flags` |
| neural host | `hf-causal-lm` | `every`, `logits`, `step` | `generate`, `params`, `flags` |

`base` cell hashes: `hf-checkpoint` 2e4e083eaea46d80 -> fb18c7b40e708e5b
(convert arm); `hf-tokenizer` a66a961b530381ae -> 4bb26860b4f41795 (encoder)
and -> dc2af787603953e6 (trainer); `hf-causal-lm` 13d9b4ba89461093 ->
9b87625cbfe914f6 (neural).

## THE INERT ARM IS THE POINT OF THE THIRD ROW

`-D MOJOLEARN_HOST_SABOTAGE=1` on the linalg family alone leaves
`hf-checkpoint` EXACTLY where it found it, on every fixture and in every one
of its eight parts. That arm is `gemm_oracle`'s descending leaf and the
factorizations; the only native call `hf-checkpoint` makes is
`lowbit.widen_bf16` -> `from_bf16`, a bit shift with no accumulation order to
perturb. It is recorded here rather than left out because a set built with
the generic define alone is not a negative control for this lane, and a
column total cannot tell an arm that did not reach a lane from one that did
and found nothing. `MOJOLEARN_LOWBIT_CONVERT_SABOTAGE` is the arm that
reaches it, and `host_surface.GATE_SABOTAGE_OWN_DEFINES["linalg"]` names it.

Which parts move and why each unmoved part cannot is in each lane's docstring
in `tools/identity_break.py`.
