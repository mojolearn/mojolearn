# Loaded-CausalLM proof and distribution

2026-09-18, branch `lane/causal-lm-distributed-proof`.

Implemented a dependency-free installed-package capture command:

```
python -m mojolearn._verify_causal_lm --device cpu --formats float32 bfloat16 int8 --output cpu.json
python -m mojolearn._verify_causal_lm --device gpu --formats float32 bfloat16 int8 --output gpu.json
```

Nine nontrivial two-layer checkpoint cases: Llama tied/untied and Mamba1 tied,
FP32/BF16/int8. Checks full logits against prefill plus two carried-state steps,
row-batch invariance, reload, reset, repeated greedy output, tensor mapping,
and a composition fault that removes one layer. Captures include logits,
decode, greedy and state bytes, checkpoint hashes and binding hashes. CPU
native fault flag is read from the binding. A capture is explicitly unqualified;
no automatic reference admission. Compare via `_verify_causal_lm.compare`.

`CausalLM.reset_state(state)` now replaces caches with fresh zero state. States
carry model ownership; same-shape states from different models are refused.

Local retained clean CPU versus Metal: all 54 part hashes match across nine
cases. 61 loader/proof tests pass. A retained file named training-sabotage was
also tried; its neural binding reports sabotage=False and produces clean bytes,
so it is explicitly NOT accepted as a negative control. Composition controls
are observed, native neural fault proof remains owed.

Still owed: Mamba2 fixtures, real supported checkpoint smoke, NVIDIA/AMD,
installed rebuilt wheel, native fault build, ragged/low-level cache faults and
physical multi-GPU evidence. No layer-parallel capability claimed yet: native
Transformer resident sessions exist, but ordinary state APIs keep host-backed
caches. Distribution must state that memory/transport limitation honestly.
