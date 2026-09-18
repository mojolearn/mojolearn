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

Second checkpoint: `models.ParallelCausalLM.load(path, layer_devices=(0,1))`
implements explicit process-owned layers, canonical block kernels, owner-local
state and activation transfer via host RPC. Only assigned blocks are constructed
in each isolated worker. Embedding runs on first owner, norm/head on last owner;
tied weights retain the canonical values. Full parent-RAM checkpoint
materialization remains required. No resident KV/capacity benchmark claim.
Supports forward, state allocation/step/reset, greedy generation, parameters,
close and context management. CUDA/HIP only, no CPU parallel user API.

The capture accepts `--device gpu --layer-devices 0 1` for physical qualification.
Comparator now enforces all requested architecture/format cases, two independent
loads, exact check/part schemas, SHA256 syntax and profile-source equality;
two identically truncated reports are refused. Sources, bindings, backend,
platform and capture commit are recorded. No reference qualification implied.

69 tests pass including real CPU arithmetic through pickle-isolated mock workers:
full forward, prefill/decode, exact state, reset, greedy, parameter mappings and
reversed owner ordering. This validates orchestration, NOT physical GPU use.
Physical two-device NVIDIA/AMD runs, vendor identity and execution traces remain
owed; host transfer cost, memory capacity and resident cache are not certified.

Native-fault followup COMPLETE for nine fixtures: paired clean/fault neural host
bindings compiled from a2061f8af, readbackFalse/True; freshclean matches54/54
Metal parts; faultmoves45/45 floating parts, greedyIDs unchanged9/9. Evidence in
`bench/results/loaded_causal_lm/2026-09-18/native-fault-comparison.json` and paired
captures/buildwitness. Earlier nativefault OWED statements are superseded for
this exact scope. NVIDIA/AMD, installedwheel and physicalmultiGPU remain owed.
