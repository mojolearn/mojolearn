# lane/lowbit-quality: progress

Brief: `~/mojolearn-evidence/lowbit-units/brief.md`, section Lane B. Plan:
`docs/lanes/LOWBIT_UNITS_PLAN.md`. Lane files: `~/mojolearn-evidence/lowbit-quality/`.
Box: shared pod nvc1 (2x RTX 4090), lane `lowbit-quality`. Nothing runs on the laptop.

## State

| Step | State |
|---|---|
| Model and text staged from R2 onto nvc1, pins checked on the box | done 2026-09-29T02:29Z (`stage.log`) |
| Host binding `_mojolearn_linalg_host` built on the pod (what `lowbit.pack` runs through) | done, `/root/lowbit-quality-work/host` |
| Reference arithmetic, forward, evaluation, quantizer cross-check written | done, `bench/lowbit_quality/` |
| Smoke job (cross-check, forward against transformers, 8 windows) | queued |
| Inference table | owed |
| Training table | owed |

## Setup facts

- SmolLM2-360M's safetensors hold bf16 (290 tensors, all BF16). They widen to
  float32 exactly, so arm b (bf16 weights, fp32 activations) rounds nothing
  on this model.
- The pod's pixi env has no torch. The jobs run in a venv over the pod's own
  torch 2.4.1+cu124 with tokenizers 0.20.3, safetensors 0.8.0 and
  transformers 4.46.3 from pip.
