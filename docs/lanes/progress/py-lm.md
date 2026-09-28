# lane/py-lm progress

Brief: ~/mojolearn-evidence/py_work_brief.md; findings: python_work_audit.md
(causal LM decode rows 1 to 3, byte LM rows 1, 2 and 5, training wrappers rows 1 and 3).
Base: lane/apple2-merged 0a11b50c7.

## Changes (branch lane/py-lm)

| item | commit | what | bits |
|---|---|---|---|
| byte LM validation | d700761bc | `_validate_state`, token admission, flag checks and the CPU adapter's per-element `isfinite` / `x < 0` / `x not in (0, 1)` scans now run through native `all_finite` and `Array.min()/max()` (reduce_stat, DEVIATION 3101). Python loops remain only to name the first bad id. | booleans only; no float moves |
| Samba training head | 3786a7145 | `samba_head_loss` (training binding): head GEMM, CE loss with gradient and head backward in ONE call; logits cross once (the loss's own refusal scan), dlogits never leave the device (was four crossings per microbatch). `identical_ce_loss_host` split into its refusals and a device core `identical_ce_loss_resident`, one spelling for both callers. Target count via `bytes.count`, exact in the admitted range. The three-call route stays as the reference (`MOJOLEARN_HOTPATH=python`, or a binding without the entry). | same kernels, operands, M and order |
| CausalLM generate | 749ae14ab, be4dbe871 | `causal_lm_session_*` (transformer binding): embedding, final norm and head resident; one call runs the stack's open `TransformerDecodeSession`s (DEVIATION 2940) for the prompt and every greedy token and returns ids. Per-layer route kept as the reference and for CPU and Mamba stacks, whose argmax moved to the native `argmax_rows_f32` (same strict `>` rule). | same kernels, operands and M (B*L at the prompt, B per token) |

## Verification (LIGHT, one job)

`tools/py_lm/job.sh` on the shared NVIDIA pod: builds base (0a11b50c7 sources for the two changed
GPU bindings) and lane in one tree, runs `tools/py_lm/witness.py` for base and lane on GPU and CPU
(CausalLM 8 families x 3 weight formats x generate 40 tokens + prefill/step logits; Samba 3 steps
with ignored targets and clipping, accumulation 2; byte LM stateless and resident 3 steps), compares
base == lane per device, then the identity lanes byte-lm, byte-lm-resident, byte-lm-host-train, samba,
samba-untied-dropout-accum, transformer, transformer-decode-session, mamba1-decode-session.

| machine | mode | item | before s | after s | digests | job |
|---|---|---|---|---|---|---|
| (pending) | | | | | | |

## DEVIATION changes

- `causal_lm.py` header "the argmax is a host selection": now native `argmax_rows_f32`, same rule (audit verdict: retire). Header text still to update once proven.
- 2940 (resident decode session): now the default route of `CausalLM.generate` on a GPU transformer stack.
- 2680 adapter text: the refusals stay; the per-element scans behind them are native (docstring updated).

## Unproven / owed

- NOTHING ABOVE IS BUILT OR RUN YET. The shared NVIDIA pod went down at ~19:20Z before the first job
  could be submitted; lanes cannot provision (orchestrator only).
- SambaStack.step / SambaInference.step / CausalLM.step still per layer per call (caller-owned state
  semantics); Mamba and mixed stacks still per layer in generate (their decode sessions live in the
  mamba binding, which cannot hand its session type to the transformer binding).
- Samba training: weights still uploaded per op and optimizer moves 7n per step (needs a resident
  Samba step session).
- Byte LM stateless step still snapshots 9n floats (memcpy speed) and GPU `next_bytes` downloads [B, L, V].
