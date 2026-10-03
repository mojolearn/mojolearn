# afn-lm notes: the byte LM train step and forward on Apple FAST

Lane afn-lm, branch lane/apple-fast-neural-lm, 2026-10-03. This file covers code
reading and compiles only. Nothing here was run or measured. The board shape is
`full` = [batch 1, length 2048, d_model 384, heads 6, kv 6, head_dim 64, ff 1024,
8 layers, vocab 8192]: M = 2048 tokens, about 20.4M parameters (82 MB), and
64 MB of logits.

## Item 0: the FAST byte LM
- `bindings/build_byte_lm.sh` accepts `MOJOLEARN_NUMERIC_MODE=fast`. FAST adds no
  mode define and goes to `python/mojolearn/` by default. IDENTICAL keeps
  `-D MOJOLEARN_NUMERIC_IDENTICAL=1` and goes to `python/mojolearn/identical/`.
  Deterministic is refused.
- `MOJOLEARN_BYTE_LM_OUTDIR` overrides the output directory. A fresh directory is
  required, because an existing output is refused as before.
- `MOJOLEARN_SKIP_BUILD_GATE=1` is accepted. The script never runs, imports or
  smokes the binding, so there is no gate to skip.
- `python/mojolearn/_backend.py` lists `_mojolearn_byte_lm` in `_CLASSICAL_FAST`.
- `python/mojolearn/_byte_lm_impl.py` `_mode()` admits the process tier
  (identical or fast) and loads that tier. It checks `byte_lm_numeric_mode()`
  against it (1 or 0). A state's `numeric_mode` records the tier it was trained
  in, and either tier's state loads.
- Native refusals (`_require_profile`, `_require_binding_profile`, the run entry)
  admit FAST only when `BYTE_LM_FAST_APPLE`. A FAST build on NVIDIA or AMD still
  refuses at run time.
- FAST off is NOT IDENTICAL's schedule. The block backward kept three host waits
  per RMSNorm backward on non-IDENTICAL tiers. It also routed FAST around the fused
  SiLU-gate and norm2+residual kernels. See BWD_NOSYNC and BWD_FUSE.

## Profile of one resident train step (main, Apple column, layer sync off)
Counts come from the code paths. Attention internals belong to afn-attn and are not
counted.

| Phase | Launches | Host waits (FAST off) | Waits (IDENTICAL) |
|---|---|---|---|
| ids upload (2 staged host buffers) | 2 copies | 2 | 2 |
| unpack weights (`byte_block_copy`, 1 per layer) | 8 | 0 | 0 |
| embedding forward | a few | 0 | 0 |
| 8 x block forward (modeling_llama, about 30 stages each) | about 200 | 0 | 0 |
| head GEMM [2048 x 8192 x 384] | GEMM | 0 | 0 |
| CE forward: 64 MB logits scan, targets download, kernels | about 6 | 3 | 3 |
| loss download | 1 copy | 1 | 1 |
| CE backward (weights, dlogits) | 2 | 0 | 0 |
| head dA, dB GEMMs | 2 GEMMs | 0 | 0 |
| 8 x block backward (about 48 stages each) | about 300 | 48 (norm fences) | 0 |
| embedding backward | a few | 0 | 0 |
| pack grads (1 per layer, plus 2 copies) | 10 | 0 | 0 |
| gradient finite scan | 1 | 1 | 1 |
| shadow copy (3 x 82 MB) | 3 copies | 0 | 0 |
| optimizer: fused refuse scan, Adam kernel | 2 | 2 | 2 |
| validate-after: 4 scans | 4 | 4 | 4 |
| binding final sync | 0 | 1 | 1 |

Device reads spent only on validation: about 740 MB per step (9 scans of 82 MB),
plus the 64 MB logits scan. Allocations per step: none on the steady path. The
arena (core/device_arena.mojo) and the stage pools hold everything. The 144
per-layer weight and weight-gradient buffers stay live as separate allocations,
which raises every Metal launch's cost.

## Candidates (each `BYTE_LM_FAST_APPLE and -D MOJOLEARN_AFN_LM_<NAME>`, or `_ALL`)
1. **NOSYNC** (training/byte_lm.mojo `_afn_byte_step_nosync`, `_byte_forward_loss[deferred]`,
   `byte_gradient_device[deferred]`; training/byte_lm_afn.mojo `afn_upload_ids`,
   `afn_step_status_kernel`, `afn_step_finish`, `afn_reset_kernel`).
   - It removes the upload, loss, gradient-scan, optimizer and validate-after waits:
     10 waits become 1.
   - All nine validation scans become one status launch.
   - The CE forward's own 3 waits remain unless HEAD_FUSE is also on.
2. **BWD_NOSYNC** (transformer/checks/transformer_backward.mojo `_bwd_rms_norm_kernels`).
   It removes the 48 FAST-only waits.
3. **BWD_FUSE** (transformer_backward.mojo: the norm-kernel arm gate, `fuse_gated_silu`,
   `fuse_norm2_residual`). FAST takes IDENTICAL's fused Apple routes.
4. **PARAM_VIEWS** (training/byte_lm.mojo `_unpack_block`, `_afn_bind_grad_views`, the pack
   loop bound). The weights and weight gradients become views of the flat buffers. It
   removes 16 launches per step and 8 per forward, and frees 144 live buffers.
5. **HEAD_FUSE** (training/byte_lm_afn.mojo `afn_ce_fused_kernel`; training/byte_lm.mojo
   CE gates). Softmax, CE, mean loss and dlogits run in one row kernel plus a reset. It
   removes 3 waits and the logits scan.
6. **ALL**: all of the above.

With ALL, a train step makes 2 host waits: the status readback and the binding's final
wait on an idle queue. FAST off makes about 62, and IDENTICAL about 14.

## Brief items not delivered, and why
- **WGRAD_SPLIT** (split-K weight-gradient GEMMs with f32 atomics). The weight-gradient
  GEMMs go through `GemmWorkspace.run` -> `identical_gemm_into` (gemm/, owned by afn-gemm).
  - A split-K route needs a GEMM tile kernel. The only one this lane could write
    without the gemm lane's simdgroup tiles would be a naive kernel that would lose
    to the tuned one. So it is left out rather than shipped half-written.
  - Request for afn-gemm: a split-K `OP_NT`/`OP_NN` dB entry for K = tokens
    (2048) with 384x384 and 384x1024 outputs. `_route_b` in transformer_backward.mojo
    is the single call site to switch.
- **LAYER_PIPE**. The layer loops already run in Mojo with no Python per layer. The
  per-layer host wait is already off on Apple (`_byte_layer_sync`). The remaining
  per-layer host work is List pop/insert of the stage structs and a prefix String.
  No device mechanism is left to take, so no define was added.
- **BWD_FUSE, deeper.** The brief target was 48 stages down to about 20 launches,
  with the elementwise VJPs folded into matmul epilogues and prologues. That needs
  GEMM epilogue hooks in gemm/ (afn-gemm) and fused-attention backward changes
  (afn-attn). This lane takes every fused kernel that already exists.
- **HEAD_FUSE with the logits GEMM inside.** At V = 8192 a per-row dot kernel would
  read the 12.6 MB head weight once per row. The head GEMM stays, and the fusion
  starts at its output.
- **lm-forward**: 64 MB of logits go to the host per call (about 21 ms on Apple by the
  measured D2H rate). The API returns the logits, so this lane leaves it.

## Compile record
On 2026-10-03 the orchestrator stopped local compiles: the M3 manager peer compiles
everything. The one FAST build this lane queued waited for a slot and was killed
before it started, so no Mojo build compiled here. The Python files passed
`python3 -m py_compile`.

| Build | rc here |
|---|---|
| `py_compile` of `_byte_lm_impl.py` and `_backend.py` | 0 |
| `sh -n bindings/build_byte_lm.sh` (syntax) | 0 |
| FAST, no define (item 0) | UNCOMPILED (peer compiles) |
| FAST `-D MOJOLEARN_AFN_LM_NOSYNC` | UNCOMPILED (peer compiles) |
| FAST `-D MOJOLEARN_AFN_LM_BWD_NOSYNC` | UNCOMPILED (peer compiles) |
| FAST `-D MOJOLEARN_AFN_LM_BWD_FUSE` | UNCOMPILED (peer compiles) |
| FAST `-D MOJOLEARN_AFN_LM_PARAM_VIEWS` | UNCOMPILED (peer compiles) |
| FAST `-D MOJOLEARN_AFN_LM_HEAD_FUSE` | UNCOMPILED (peer compiles) |
| FAST `-D MOJOLEARN_AFN_LM_ALL` | UNCOMPILED (peer compiles) |
| IDENTICAL (default) | UNCOMPILED (peer compiles) |

Build command (from the worktree root):

    MOJOLEARN_COMPILE_JOBS=1 MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_SKIP_BUILD_GATE=1 \
      MOJOLEARN_BYTE_LM_OUTDIR=<fresh dir> MOJOLEARN_MOJO_BUILD_FLAGS="-D MOJOLEARN_AFN_LM_<X>" \
      sh bindings/build_byte_lm.sh

Compile risks to check first (code that was written without a compiler):
- `Atomic[DType.int32].min` and the f32 `Atomic.fetch_add` in training/byte_lm_afn.mojo.
  Both are used elsewhere on Apple (x_decomp/graph_device.mojo,
  ensemble/checks/atomic_width_probe.mojo).
- The `return` inside a `comptime if` that has code after it: `_unpack_block`,
  `_byte_step_device`, and the deferred loss return in `_byte_forward_loss`.
- `training/byte_lm_host_kernels.mojo` asserts IDENTICAL. The device binding does not
  import it (grep: only comments mention it in byte_lm_config/byte_lm_logits), so FAST
  is not expected to reach that assert.
