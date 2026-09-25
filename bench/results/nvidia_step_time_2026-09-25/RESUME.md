# RESUME: lane/nvidia-step-time (NVIDIA optimizer step time at the GPT-3 Small shape)

Read this first after a restart. Branch `lane/nvidia-step-time` (from
origin/main f2293183c, Release 0.8.18), worktree
`/Users/andrewhendel/CascadeProjects/mojolearn/.claude/worktrees/agent-ad52900fc55db0c7d`.
Never merge or push main from this lane. Never touch the paused T3 run
(`~/mojolearn-wt/gpt3-tooling`, `~/mojolearn-evidence/gpt3-run/`, R2
`runs/t3/2026-09-22/` except read-only presigned GETs) nor the release
worktree `~/mojolearn-wt/release-0814` and its boxes. Kill no process and
delete no box this lane did not create.

## Goal

One optimizer step at the T3 shape (162,147,840 parameters, batch 4, length
2048, K = 64 shards of 8,192 tokens) on one H100: 39.9 s (T1, 0.8.17
kernels). Make it faster with identical bits (rule: no floating-point
operation, operand, rounding or order may change; tensor cores only with a
bit-equality proof against the scalar chain).

## Bit proofs every change must pass

- Chain replays (`tools/nvidia_step_time/session.sh replay`): steps 101..103
  from `runs/t3/2026-09-22/A/1/ckpt_00000100.blm` against A/1's chain, and
  1999..2000 from `A/2/ckpt_00001998.blm` against A/2's chain. State digests
  101 abc8b816b5c3fb15, 102 a9421f91b947f82c, 103 fcdb48b8ab51f2ef,
  1999 dcb05e4e668a81e1, 2000 0e39ed2bfe9bcbae (checked in the chain files
  2026-09-24).
- GEMM A/B at the T3 shapes (`bench/gemm_excp_ab_main.mojo`, 28 cases)
  identical to the branch reference build.
- Identity lanes (`python -m mojolearn verify --lanes ...`) 0 DIVERGENT.

## How a leg runs

Inputs staged in the scratchpad (regenerate if lost):
`python3 tools/amd_step_time_urls.py <dir>/urls runs/t3/2026-09-22/A/1/ckpt_00000100.blm runs/t3/2026-09-22/A/2/ckpt_00001998.blm`,
A-1 chain from `~/mojolearn-evidence/gpt3-run/witness/A-1.chain.partial.jsonl`
(copy, read only), A-2 chain by a presigned GET of
`runs/t3/2026-09-22/A/2/chain.jsonl`, and `base0817.tgz` =
`git archive 4795b0d54 checks/kernel_matrix.mojo checks/numerics.mojo gemm/checks/gemm_identical.mojo`.

    MOJOLEARN_RUNPOD_KEY_FILE=~/.mojolearn_runpod_key \
    MOJOLEARN_GEMM_LEG_EXTRA=tools/nvidia_step_time/legN.sh \
    MOJOLEARN_GEMM_LEG_OUT=~/mojolearn-evidence/nvidia-step-time/<stamp>-legN \
    sh tools/gemm_remote_leg.sh nvidia --rent --allow-concurrent \
       --segment-lease 90 --dollar-cap 7 --gpu "NVIDIA H100 80GB HBM3"

After the pod is up, push `urls/`, `amd_in/` and `base0817.tgz` to `/root/`
over the ssh target the runner prints; the body waits for them. The body
holds at the end until `/root/nv_step_done` exists (touch it over ssh when
the interactive work is over, or the lease runs out and the runner fetches).
Evidence is copied from the leg OUT into this directory (small files only).

## State (update every session)

- 2026-09-24 (lane day 0): branch created; tools written
  (`tools/nvidia_step_time/{session.sh,leg1.sh,probe_gemm_ptx.mojo,nsys_fold.py}`).
  Leg 1 (profile) about to rent.

## Findings so far (source reading, no NVIDIA measurement yet)

- The AMD lane's changes reaching NVIDIA: only `GEMM_LAUNCH_BOUND` (256,
  `.maxntid 256`). Leaf split, MFMA, class ftz and the `_bswz` default are
  AMD rows. `-D MOJOLEARN_GEMM_NO_LAUNCH_BOUND=1` now means a bound of 1024
  (which on NVIDIA caps registers at 64 a thread), NOT 0.8.17's unbounded
  kernel, so the 0.8.17 baseline is built from its own three files.
- NVIDIA GEMM: `kpack_hg` body on every TUNED call, ksplit groups at S = 132
  on long k, seam `fma.rn` + `mul.rn.ftz` by one.
- Tensor cores: sm_90 has no instruction that forms one fp32 product plus
  an fp32 accumulator rounded once to fp32 (TF32 mma truncates operands;
  DMMA rounds to fp64 and the second rounding to fp32 is a double rounding;
  f32-accumulating mma/wgmma sum several products per instruction). Not
  pursued unless a probe says otherwise.

## Owed / open

- Everything above: the profile, the changes, the proofs.

## Costs

(none yet)
