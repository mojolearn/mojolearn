# The four lanes that had no CPU route, and now have one

lane/cpu-routes-gpu-only-four, 2026-09-20, Apple M4, ONE CORE, `nice -n 19`,
CPU host bindings only. No Metal job ran and no GPU was rented: `vendor` reads
`cpu-apple-m4` in every column here and `column=cpu` in every header. Every
column is `tools/identity_break.py --repeats 2 --step-full --batch-scale
--ragged --batch-grad` over all nine fixtures, recorded at this lane's own
commit (`commit` reads the same twelve characters, `2a2d3491d472`, in all
seven files).

`--repeats 2` AND NOT 1, ON PURPOSE. `stable_digest()` refuses a part with
fewer than two repeats, so a sabotage cell reading STABLE beside a plainly
different hash falls through and the move is thrown away. Every column here
carries two fits per cell.

## What was wrong with the reason these four carried

lane/unlaned-public-algorithms gave six unlaned public algorithms a lane the
day before and closed two of them. It declared the other four unclosable, with
the reason `no cpu route` -- "this box is the wrong box, and always will be" --
and recorded each refusal verbatim in
`bench/results/identity_break/2026-09-20_unlaned-public-algorithms/
gpu-only-refusals-probe.json.errors.txt`. Every one of those four refusals was
real, by name, on a CPU-only install. Every one was read as a statement about
the lane's ARITHMETIC. Not one of them was.

| lane | what refused | what it was about |
| --- | --- | --- |
| `mamba1-decode-session` | `mamba1_session_create` is in the GPU binding only | RESIDENCY. The session's BYTES are `mamba_step`, the block at L = 1, which the host binding exports as `mamba1_decode_step`. |
| `transformer-decode-session` | `transformer_decode_session_create` is in the GPU binding only | RESIDENCY, the same. `step`/`forward` are `transformer_decode_step`/`transformer_forward`, which the host binding exports over `transformer/host/transformer_block_host.mojo`. |
| `par-causal-lm` | `_backend.vendor() not in ('cuda', 'hip')` | THE VENDOR. The partition is layers and the merge is `CausalLM._run`'s hand-off, both the driver's own Python; the shard is the neural host family's. |
| `par-cross-val` | `_backend.vendor() not in ('cuda', 'hip')` | THE VENDOR, and a DEVICE INVENTORY. The fold split, the wave dispatch and the merge are the serial API's own code; the inventory is the one piece with no CPU counterpart. |

The decode sessions now carry a HOST ARM -- the ownership wrapper over the
per-call entry, with no new arithmetic and NO speed claim, because on the host
there is nothing to keep resident. The two drivers now take the CPU route,
where a device index is one WORKER PROCESS. Metal is still refused by name in
both drivers: `DevicePool` gives an Apple group no visibility mask, so "one
device per owner" would be a sentence with nothing behind it.

## The columns

| file | host set | what it is |
| --- | --- | --- |
| `clean.json` | the built host set, unmodified | 36 cells (4 lanes x 9 fixtures), every one STABLE, 0 REFUSED; `infer` STABLE on 18, `stepfull` STABLE on 18, `batch` STABLE on 9 |
| `par-two-process.json` | the same | the two `par-*` lanes again at `MOJOLEARN_PAR_DEVICES=0,1`: 18 cells, every one STABLE, and IDENTICAL to `clean.json` cell for cell |
| `sabotage-mamba.json` | the mamba host binding built `-D MOJOLEARN_HOST_SABOTAGE=1` | `mamba1-decode-session`, 9 cells |
| `sabotage-transformer.json` | the transformer host binding, same define | `transformer-decode-session`, 9 cells |
| `sabotage-neural.json` | the neural host binding, same define | `par-causal-lm`, 9 cells |
| `sabotage-gbdt.json` | the gbdt host binding, same define | `par-cross-val`, 9 cells |
| `sabotage-neural-transformer-session.json` | the neural host binding, same define | `transformer-decode-session` again, for the ONE part the transformer define does not reach |

`admit()`, printed rather than assumed:

    clean.json                    complete=True  cells=36  par_devices='0'
      admit(default)  = None            <- ADMISSIBLE
      admit(par_axis) = None
    par-two-process.json          complete=True  cells=18  par_devices='0,1'
      admit(default)  = 'par_devices 0,1'
      admit(par_axis) = None            <- ADMISSIBLE on the par axis, which is
                                           the only axis it claims
    sabotage-mamba.json           admit = 'sabotage, partial, probe, unfixed or smoke run (by name)'
    sabotage-transformer.json     admit = 'sabotage, partial, probe, unfixed or smoke run (by name)'
    sabotage-neural.json          admit = 'sabotage, partial, probe, unfixed or smoke run (by name)'
    sabotage-gbdt.json            admit = 'sabotage, partial, probe, unfixed or smoke run (by name)'
    sabotage-neural-transformer-session.json
                                  admit = 'sabotage, partial, probe, unfixed or smoke run (by name)'

## The sabotage arms, and that they are really sabotage builds

A .so DIGEST DOES NOT PROVE A DEFINE: two builds of identical source always
differ. Each host binding exports its own predicate, and the loader refuses a
sabotage build BY NAME unless the gate opts in, so both facts were read back:

    mamba_host_sabotage()       = False   (the clean set)
    ... and the sabotage set, without MOJOLEARN_HOST_ALLOW_SABOTAGE=1:
    ImportError: ... is a SABOTAGE build and computes wrong answers on purpose
    ... and with it:            = True

and the same three lines for `transformer_host_sabotage`,
`neural_host_sabotage` and `gbdt_host_sabotage`. Each sabotage set is the
CLEAN set with exactly ONE .so replaced; every other binding in the directory
is a symlink to the clean one, so the only difference between the two columns
is the one define. Each was built from THIS branch's source through
`bindings/build_<family>_host.sh`, one core, `nice -n 19`, into its own fresh
directory -- never into the main checkout's binding directories.

## The moves, seen

`--diff clean.json sabotage-<family>.json --lanes <lane>`, all nine fixtures:

| lane | arm | train | infer | batch | stepfull |
| --- | --- | --- | --- | --- | --- |
| `mamba1-decode-session` | mamba | DIVERGENT=9 | DIVERGENT=9 | n/a | DIVERGENT=9 |
| `transformer-decode-session` | transformer | DIVERGENT=9 | **IDENTICAL=9** | n/a | DIVERGENT=9 |
| `transformer-decode-session` | neural | IDENTICAL=9 | **DIVERGENT=9** | n/a | IDENTICAL=9 |
| `par-causal-lm` | neural | DIVERGENT=9 | n/a | DIVERGENT=9 | n/a |
| `par-cross-val` | gbdt | DIVERGENT=9 | n/a | n/a | n/a |

The base and wide fixtures, hash for hash:

    mamba1-decode-session      arm: mamba host
      base/cell     036f89b10477d901  ->  045ddafec4027839
      base/step     bfe92dd1b46862d2  ->  ea33ec8021c1652e   moved
      base/state    0800dac84123170c  ->  c4a7bc5e82f2348e   moved
      base/flags    cd5e26e9d1168a65  ->  cd5e26e9d1168a65   SAME
      base/infer    97e86b9db80719a7  ->  6753cf2714f371cf   moved
      base/stepfull 7d79ddc6d8ec34e4  ->  05873f52a4bc6dcc   moved
      wide/cell     f765fff13969ae65  ->  24bafbb54ae3e497
      wide/step     8da81e4e3c9aa155  ->  24f533e36817994d   moved
      wide/state    d7263806a5402f63  ->  75425dd7812d0593   moved
      wide/infer    905c98cf2a276d1d  ->  7753d640b0858e33   moved
      wide/stepfull 07799d0f4bc09ccf  ->  8c10f60f635ff26c   moved

    transformer-decode-session arm: transformer host
      base/cell     44ed8cdc5dc0e0b9  ->  6657236ab4e13f46
      base/step     168d9413b27ce4c8  ->  8bdbd89c40392439   moved
      base/prefill  3c87773c8b7148a3  ->  adf5f57ed904e344   moved
      base/state    272bab078083396a  ->  6a91e5ba540b157b   moved
      base/flags    cd5e26e9d1168a65  ->  cd5e26e9d1168a65   SAME
      base/infer    80ab5cc2b56af36f  ->  80ab5cc2b56af36f   SAME
      base/stepfull 2bd2002e82de9749  ->  a793fff57a1c597b   moved
      wide/cell     a52a0d87addfd3a4  ->  fc5891a0fa5df1bf
      wide/step     2068c28786c88b24  ->  2adacbe84afc0eff   moved
      wide/prefill  9ee52efb54714083  ->  c99deac89ec97522   moved
      wide/state    46a7f63520cce1a4  ->  8be53340f42de475   moved
      wide/infer    472ac906ecc50310  ->  472ac906ecc50310   SAME
      wide/stepfull 4d8b013d3ce03bec  ->  0d12b44678452afd   moved

    transformer-decode-session arm: NEURAL host, the held-out infer cell
      base/cell     44ed8cdc5dc0e0b9  ->  44ed8cdc5dc0e0b9   SAME
      base/infer    80ab5cc2b56af36f  ->  e82a788d8147d378   moved
      wide/cell     a52a0d87addfd3a4  ->  a52a0d87addfd3a4   SAME
      wide/infer    472ac906ecc50310  ->  3a0df2d623821363   moved

    par-causal-lm              arm: neural host
      base/cell     5171c3ffeb9dd8f8  ->  7d27c6550969e43a
      base/logits   71b13186e4fa95f3  ->  fee752ce131b3283   moved
      base/params   bb88c3e24c41da72  ->  bb88c3e24c41da72   SAME
      base/flags    1b93738cd9ef30df  ->  1b93738cd9ef30df   SAME
      base/batch    2d58207bcc498e7c  ->  cb2983a3f99060b6   moved
      wide/cell     f711a5aa202b3e41  ->  e25dae3a4ebeec63
      wide/logits   5deeb2b77b89c972  ->  387259c70eb496a0   moved
      wide/batch    97051f4a7819532e  ->  05b0e4bf3b664121   moved

    par-cross-val              arm: gbdt host
      base/cell     b42f6f50bae5b03c  ->  172bf04639539090
      base/scores   9336dc0965e2ba0f  ->  c2080afcb1cb8d59   moved
      base/flags    4a69320a0d048338  ->  4a69320a0d048338   SAME
      wide/cell     40c2b054e9a24cdc  ->  a39bf03cb76a13fa
      wide/scores   8641f26de10ebdb7  ->  5a8d814c3c2821fc   moved

## What does NOT move, and why it is written down

**`flags` never moves, on any lane, under any arm.** A refusal is not
arithmetic: a build that computes wrong answers still refuses a closed
session, a state that already belongs to one, a row count that is not the
session's, a foreign estimator's FAST numeric mode and a layer map of the
wrong length, and refuses them with the same words. The part is hashed anyway,
because a driver that STOPS refusing has changed its contract and nothing else
in the cell would see it.

**`par-causal-lm`'s `params` never moves.** It is the weights the layer owners
hold, read back through `parameters()`; a wrong GEMM does not change the bytes
that were sent to the worker. It is in the cell because the RPC round-trip of
the weights is exactly what a layer-owner bug would corrupt.

**`mamba1-decode-session/negative`'s `step` does not move** under the mamba
arm, though its `state` does. It is the only one of the nine fixtures where
that happens, and the diff says so by name (`parts differ: state; agree:
step,flags`). The cell is still DIVERGENT, which is why the state part is
hashed beside the step part.

**`transformer-decode-session`'s held-out `infer` cell does not move under the
transformer arm at all, on any of the nine fixtures.** This is the finding the
lane would have been weakest without. That part is not the block: it is
`_neural_inference(ml, "transformer", e)`, which on a CPU column builds
`TransformerBlockInference` -- the NEURAL family's binding, not the
transformer family's. So the transformer define cannot reach it and never
could. `sabotage-neural-transformer-session.json` is the second arm that does
reach it, and it is the mirror image: `infer` moves 9/9 and the train cell and
`stepfull` do not move at all. Two arms, neither redundant, each watching what
the other cannot.

## The device axis, and what this column does not say

`par-causal-lm` and `par-cross-val` are `par-*` lanes and the CPU column is the
DEGENERATE case of their device axis: one process per index, no visibility
mask, no hidden activation and no fold crossing a device boundary. That is
what `identity_break._par_devices`'s docstring says of every `par-*` lane.

`par-cross-val` is the one lane where a one-index column would also be a
different DISPATCH -- the driver sends waves of `len(pool.devices)` and calls a
worker-placement check before the first fold -- so the width-2 dispatch was
taken too, and on this route it costs nothing, because an index is a process.
`par-two-process.json` is `MOJOLEARN_PAR_DEVICES=0,1` with `cv=3`: one full
wave of two folds and one partial wave of one. It hashes EQUAL to the
one-process column, cell for cell, on both lanes and all nine fixtures
(`--diff` reads `IDENTICAL=18`, and `batch` `IDENTICAL=9`), which is the
drivers' whole claim.

The placement check itself is NOT carried over and is not faked. On CUDA/HIP
`cross_val_score` calls `require_distinct_workers` on `visible_gpu_inventory`:
one visible physical GPU per worker at local ordinal zero, no repeated UUID or
PCI id. There is no such question on a host box, so the CPU route asks
`require_distinct_processes` over `worker_process_inventory` instead -- one
distinct worker PROCESS per index, under its own record kind, in a SEPARATE
function so the GPU check's bar is untouched. It admits the dispatch and
nothing about hardware.

IDENTITY IS NOT CORRECTNESS. Every cell here is AGREEMENT: the session against
the per-call step on the same block and state, the layer-owned model against a
plain in-process `CausalLM`, the parallel folds against the serial
`cross_val_score`. If `mamba_step` were wrong, the session and the per-call
step would be wrong together and every column on every vendor would read
IDENTICAL. What these cells rule out is a defect in the wrapper, the
partition, the merge and the ordering -- which is what the lanes are for.

## What is still owed

A committed record that carries a GPU hash for each of the four, so an
installed `verify --all` has a cell to read; that is why all four sit at
`no reference` in `host_surface.PUBLIC_PENDING_LANES` rather than being
promoted. For the two `par-*` lanes that record is a TWO-DEVICE CUDA or HIP
column, and nothing here substitutes for it:

    MOJOLEARN_PAR_DEVICES=0,1 MOJOLEARN_NUMERIC_MODE=identical python3 \
      tools/identity_break.py --lanes par-causal-lm,par-cross-val --repeats 2 \
      --step-full --batch-scale --ragged --batch-grad --json <column>.2gpu.json

    MOJOLEARN_NUMERIC_MODE=identical python3 tools/identity_break.py \
      --lanes mamba1-decode-session,transformer-decode-session --repeats 2 \
      --step-full --batch-scale --ragged --batch-grad --json <column>.json

No GPU work and no rental was done for this record.
