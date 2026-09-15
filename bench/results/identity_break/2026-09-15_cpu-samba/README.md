# CPU training for the Samba stack lanes, the M4 host column (2026-09-15)

Branch `lane/cpu-training-samba`. The `samba` and `samba-untied-dropout-accum`
lanes of `tools/identity_break.py` (SambaStack, one Mamba-3 layer and one
attention layer at d_model 32 over 256 bytes, weights from the stack's seeded
generator, three AdamW steps; the second lane with untied embeddings, dropout
0.1, four accumulation microbatches, a global-norm clip and a warmup-cosine
schedule), run on the CPU and diffed with `tools/identity_break.py --diff
--require-columns 4` against the 166-lane record's three GPU columns
(`bench/results/identity_break/2026-09-14_166-lanes/`: Apple M4, NVIDIA H100
sm_90a, AMD MI325X gfx942), as the CPU identity gate does.

SambaStack (`python/mojolearn/_samba_impl.py`) holds no numerics. Every step
is a call into three bindings, and on a CPU-only install each routes to its
host family:

| binding | host family | what the stack calls |
|---|---|---|
| `_mojolearn_training` | training (`bindings/_mojolearn_training_host.mojo`) | embedding, RMSNorm and head forward and backward, cross-entropy, accumulate, clip, AdamW, and `neural_rng` (NEW on this branch) |
| `_mojolearn_mamba` | mamba (lane/cpu-training-mamba) | Mamba3Block forward and zero-state prefill backward |
| `_mojolearn_transformer` | transformer (lane/cpu-training-transformer) | TransformerBlock forward and zero-state prefill backward |

The one operation the stack reached with no host entry was the neural RNG
(the initializers draw uniform and normal, the dropout lane draws the mask
forward and replays it backward). `neural_rng` is
`core/philox_neural.mojo::neural_rng_host` over `core/philox.mojo`'s Philox
block, written out for the host by `tools/mamba_host_gen.py`
(`mamba/host/gen/philox_neural.mojo`, `mamba/host/gen/philox.mojo`): the
kernel body unchanged, its launch a serial loop over the grid. `--check`
fails (`STALE mamba/host/gen/philox_neural.mojo`, exit 1) when
`core/philox_neural.mojo` is edited and not regenerated.

Where it ran: the Apple M4, one core, shared machine. The core, training,
mamba and transformer host bindings were built with `-j 1` from the working
tree at 4cc3609e2 (the JSONs record it) into fresh directories.

| file | what |
|---|---|
| `cpu-apple-m4.json` | the CPU column, 2 lanes, 9 fixtures, 2 repeats (130 s) |
| `diff.four-columns.txt` | `summary: IDENTICAL=18`, `summary (infer/model): IDENTICAL=36`, `summary (batch): IDENTICAL=18`, `require-columns 4 over ['samba', 'samba-untied-dropout-accum']: OK` |
| `cpu-apple-m4.sabotage.json` | the same lanes through the four families built with `-D MOJOLEARN_HOST_SABOTAGE=1` |
| `diff.four-columns.sabotage.txt` | `summary: DIVERGENT=18`, `summary (infer/model): DIVERGENT=36`, `summary (batch): DIVERGENT=18`; loss, logits and params differ in every train cell |
| `diff.four-columns.dropout-mask-arm.txt` | a throwaway training host binding (never committed) whose generated dropout arm kept an element where its coin was at least p/2 instead of p: `summary: DIVERGENT=9, IDENTICAL=9`; every samba-untied-dropout-accum cell DIVERGENT, every samba cell (no dropout) IDENTICAL x4. The mask the lane draws is the host `neural_rng`'s |

Owed: the seven-runner CPU identity gate on the branch.
