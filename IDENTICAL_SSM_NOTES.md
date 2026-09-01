# What a bit-identical Mamba (selective SSM) block would need, and whether it is worth it

Written 2026-08-23 by the orchestrator, answering Andrew's question of that
morning: *is there anything in the Mamba direction we can build now, and is
going into neural-network operators worth doing at all?* The sources were
read that day; every claim below names its file. **Nothing here is in scope
until promoted**; Andrew's same-day decision was classical ML first (cuML
mirrors under the identical discipline), and this file exists so the
decision about the neural layer is made later against a written list.

## Short answers

1. **The neural layer does NOT inherit the ACCESS thesis.** PyTorch on MPS and
   MLX both reach the Apple GPU for Mamba today (HF `modeling_mamba.py`
   `slow_forward` is a Python loop of MPS ops; mlx-lm `models/mamba.py`
   `_process_sequence` loops MLX ops; mlx-lm Mamba-2 decode has a custom
   Metal `ssm_kernel`). Neither is a fused multi-token scan, but the device is
   reached. So for neural operators the only thing this repository would be
   selling is **IDENTITY** (bit-identical across vendors, batch-invariant,
   launch-invariant). That is a different thesis from `ROADMAP.md`'s and it
   must be stated as such wherever it is claimed.
2. **The Mamba-specific piece is exactly one primitive: a pinned selective
   scan.** Everything else in the block is GEMM (have, `gemm/` v1), a 4-tap
   causal depthwise conv (trivial, serial), RMSNorm (missing, small), and
   elementwise transcendentals (SiLU/sigmoid, softplus, exp, rsqrt: LANDED
   2026-08-23 as `portable_sigmoidf`/`portable_siluf`/`portable_softplusf`/
   `portable_rsqrtf`/`portable_log1pf` plus the division pin `portable_divf`,
   `checks/numerics.mojo`, IDENTITY_PATHS rows 49-54, Apple-gated only). There is **no pinned prefix scan or linear recurrence of
   any kind, in any dtype, anywhere in this tree** (every existing scan is
   integer-typed: `core/block_scan.mojo`, `core/scan_by_key.mojo`,
   `gbdt/gpu_util/kernel/scan.mojo`), and MAX ships no device cumsum
   (`VENDOR_LIBS.md` rows for `nn.cumsum`).
3. **A Mamba-1 block is a CHEAPER first one-block identity demonstration than
   a transformer block.** The transformer path in `IDENTICAL_GEMM_PLAN.md`
   needs softmax (a pinned max with the row-13 signed-zero hazard, exp, a
   pinned sum, a pinned division), RoPE (portable sin AND cos with argument
   reduction; BOTH exist since DEVIATION 820's shared
   `_cephes_sincosf_core`, domain |x| < 8192 — this line said only cos
   existed and was stale, corrected 2026-09-01 by the M3 recon), and an
   attention kernel. Mamba-1 needs none of those. Roughly two new primitives
   (scan, RMSNorm) against four.
4. **Nobody has published a bitwise or batch-invariant SSM scan.** Thinking
   Machines' *Defeating Nondeterminism in LLM Inference* covers RMSNorm,
   matmul and attention only; vLLM's `VLLM_BATCH_INVARIANT=1` RAISES for the
   Mamba1/Mamba2/GDN attention backends (`supports_batch_invariance()` is
   False there; issue #42960); `mamba_ssm/utils/determinism.py` (NVIDIA PR
   #827, Dec 2025) is run-to-run determinism of the Mamba-2 BACKWARD only.
   That is a concrete hole, and it is the same hole the GEMM lane's
   `check_pinned_gemm_is_batch_invariant` fills one layer down.
5. **Worth doing?** Yes, *if* the identity thesis is the product, and only
   AFTER the three-vendor GEMM run has closed (`gemm/E1G_RUNBOOK.md`), because
   a neural block built on an uncertified GEMM is construction on construction.
   None of the primitives below needs a rented box to BUILD and Apple-gate;
   they need one to CERTIFY, like everything else.

## Where vendor dependence enters the reference implementations

| step | reference spelling | why it is not bit-stable | what a pinned version does |
|---|---|---|---|
| selective scan over L | `csrc/selective_scan/selective_scan_fwd_kernel.cuh`: `cub::BlockScan<float2, kNThreads, BLOCK_SCAN_WARP_SCANS>` with `SSMScanOp (a,b)∘(a',b') = (a'a, a'b+b')`, `kChunkSize = kNThreads*kNItems`, traits picked by seqlen bucket and DIFFERENT on the ROCm branch (`<64,*>` vs `<32,*>`) | the combine tree is a function of kNItems, kNThreads, warp width (32 vs 64) and the cub algorithm; the affine op re-associates under rounding | the order is a pure function of L and a profile chunk constant: serial inside a chunk, one fixed tree across chunks (the GEMM v1 leaf/fold split, applied to an affine op). **MAX's own `state_space.selective_scan_fwd_gpu` is one thread per (batch, dim), serial over L and over N=16**; that design is already order-fixed along L, so the cheapest identical scan is MAX's shape plus pinned arithmetic, and the chunked one is the priced follow-on |
| compiler policy | `setup.py` builds nvcc with `--use_fast_math`; HIP gets only `-fgpu-flush-denormals-to-zero` | same source, two exp/division/denormal policies | `identical_exp`, `ftz` at the seams, one division policy (`identical_div`, row 49, characterized per class on Apple by `check-division`) |
| softplus, sigmoid/SiLU | `log1pf(expf(delta))` with a `delta <= 20` guard; `z/(1+expf(-z))` | vendor `expf`/`log1pf` | `identical_softplus` / `identical_sigmoid` / `identical_silu` (rows 52-54; silu keeps the reference's ONE-division spelling) |
| C·h dot over N=16 | serial over `state_idx` in the CUDA kernel; `einsum` in the reference | einsum order is torch-defined | serial ascending, fixed |
| RMSNorm | `rms_norm_ref`: `1/sqrt(mean(x²)+eps)`; Triton `tl.sum` | `tl.sum`'s tree is Triton's | pinned sum-of-squares tree (`pinned_block_sum` + fixed fold), `identical_rsqrt` (row 50: `1/portable_sqrtf`, the reference's spelling) |
| Mamba-2 / SSD | `ssd_combined.py`: chunk cumsum, `tl.dot` intra-chunk, a SERIAL inter-chunk state pass, `chunk_size` default 256 | every reduction boundary moves with `chunk_size`; `tl.dot` is a vendor GEMM | GEMM-shaped: intra-chunk through GEMM v1, inter-chunk serial, the chunk size a PROFILE constant |

## If and when it is promoted: build order

1. `numerics.mojo` rows (identity lane's file; request, do not edit):
   DONE 2026-08-23 except `portable_sinf` (not asked for by the Mamba-1
   block): `identical_div` (division characterized per class, row 49),
   `identical_rsqrt`, `identical_log1p`, `identical_sigmoid`,
   `identical_silu`, `identical_softplus` (rows 50-54), gates
   `check-division` and `check-portable-nn`, Apple-gated; the H100/MI325X
   certificate re-prints are owed with step 5.
2. A pinned FP32 SCAN profile (`mojolearn.identical.scan.fp32.v1`): the
   affine-recurrence scan as its own directory, contract first, serial oracle,
   chunked oracle, distinguishing fixtures (serial vs chunked vs two chunk
   sizes; fused vs unfused; FTZ), device kernel, launch/batch-invariance
   gates, sabotages — the GEMM lane's phases 0-4, verbatim in shape.
3. RMSNorm and the causal depthwise conv as small pinned kernels with cards.
4. ONE Mamba-1 block, reference quality, stage-carded after every op
   (`core/identity_trace.mojo`), Apple-gated; then the transformer block.
5. Certify on three vendors, same runbook shape as `gemm/E1G_RUNBOOK.md`.

Upstream for the port, in this repository's order of preference: cuML/RAFT
have nothing here, so the rule in `IDENTICAL_GEMM_PLAN.md`'s charter applies
(switch to PyTorch's algorithms and mirror those): `mamba_ssm`'s
`selective_scan_ref` / `mamba_inner_ref` are the normative reference, MAX's
`state_space` package is the Mojo spelling to mirror for the kernels.

## What is NOT claimed

- "Bit-identical AI inference" is not claimed; one block is.
- Portability is inherited from Mojo, never novelty.
- Nothing here has run on any device. This is a reading of other people's
  source, dated, and it will go stale.
