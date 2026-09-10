# Transformer discarded-cache prefill candidate

Base main1b3f53d6; candidate913a2cca. IDENTICAL/NVIDIA default after the first complete H100 A/B gate; `MOJOLEARN_TRANSFORMER_LEGACY_FRESH_PREFILL=1` retains the baseline. `MOJOLEARN_TRANSFORMER_FRESH_PREFILL=1` explicitly forces the new entry for Apple validation, while Apple default remains unchanged. Public `TransformerBlock.forward(x, state=None)` explicitly discards final state; there is no optional public cache/report return to preserve in this branch. Stateful forward, step, linear/ring continuation, and backward retain their existing paths. Other numeric modes retain their original behavior.

The existing stateless route allocates two host-zero caches, allocates device-zero caches, overwrites the device copies from the host, runs the block, downloads both caches, and discards them. Each original large GQA grid cache is 16 MiB, so avoiding two uploads and two downloads removes 64 MiB of request transfers plus host cache allocation. The dedicated fresh entry keeps the existing LlamaKVCache constructor, metadata/capacity guards, original zero device contents, full decoder execution and output download. It uses smax=L, s0=0, and the same configured sliding window. The compile-time discard-cache branch never reads the two zero sentinel cache addresses. It returns only y through the existing synchronous public semantics; no weight or cache survives on device beyond the call.

The Python frame holds x, nine weight owners and y until the native call returns. Weights are uploaded and refused in their existing order on every call, so caller mutations remain observable. The existing explicit state path is also the fresh correctness oracle by passing allocate_state(B,L).

`tools/transformer_fresh_prefill_check.py` requires that the new entry exists and that fresh calls do not allocate host state. It compares complete output bits to explicit zero-state forward for HD64/HD128, linear and two ring sizes, multiple sequence lengths, ordered mutable-weight poison refusals, and the exact absolute-position capacity refusal at L=8193. The existing `tools/transformer_transfer_check.py` adds 82 full-array cross-build witnesses covering explicit state, unused cache bytes, decode, ring continuation and all backward gradients.

Root-owned Apple reproduction: `/tmp/mojolearn-transformer-fresh-apple.sh`. Isolated NVIDIA reproduction: `tools/transformer_fresh_prefill_leg.sh`; set checkout/results variables and choose image `/usr/bin/python3` consistently across arms. It activates Pixi for Mojo, preflights both fixture generators, and installs the matched archived original public harness. Both original large output files must match complete SHA256; all raw timing rounds are retained. No opponent is timed; original transformer Torch admission remains unqualified and no ratio should be claimed.

Apple and NVIDIA identity gates passed, including 82 maps and complete large output SHA witnesses. First H100 seven-round baseline/fresh medians: 211.354129/204.134570 ms narrow and 212.392121/187.488862 ms wide. Final default and reverse-order validation passed: narrow baseline/default 210.893420/204.793565 ms, wide 214.515634/188.905248 ms (2.9%/11.9% less). The final 82-array maps and original large output hashes remain identical. [Lane evidence](../../bench/results/transformer_fresh_2026-09-10/README.md) links the parent final-performance artifact containing all samples and source manifests. Root owns all local builds, GPU allocation and timing serialization.

## Full original-grid numerical attribution continuation

The original B8/L4096/D512 and B8/L1024/D2048 H100 diagnostics now run
through 20 local/cumulative stage comparisons, with shared production RoPE
constants. Prior final Torch FP32-versus-FP64 statistics reproduce exactly.
Attention-core rounding and amplification of Q/K/V projection/RoPE error
both contribute; o_proj, same-input norm2, and residual additions have zero
violations. The wide MLP has additional local error, but propagated error
dominates. Full counts, hashes, scope, and reproduction are in
[stage admission evidence](../../bench/results/transformer_stage_admission_2026-09-10/README.md).
No tolerance or production algorithm changed, no new opponent timing was
measured, and the original Torch opponent ratio remains unqualified.

`tools/transformer_admission_diagnose.py --reference-only --reference64`
can diagnose the original comparator without rebuilding the native Mojo
binding or requiring an own-output NPY. Ordinary own-output comparison still
requires `--output`. Reports now distinguish the worst tolerance witness
from the largest absolute-error witness. Six host tests pass, alongside both
complete full-grid CUDA diagnostic runs. Next numerical attribution: split
projection versus RoPE, then attention logits/softmax; this evidence alone
does not warrant changing IDENTICAL arithmetic to match Torch.
