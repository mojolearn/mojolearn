# neural-cpu: progress (the neural family's CPU-speed lane)

Lane opened 2026-09-28. Branches: `lane/neural-cpu` (single-thread work,
off origin/main) and `lane/neural-cpu-threads` (the thread splits; it carries
a merge of `origin/lane/cpu` for `core/host_parallel.mojo` and merges to main
only after lane/cpu lands that file there). Pod `neural-cpu` (RunPod H100
box, Xeon Platinum 8480+, cgroup quota 23.8 CPUs). Evidence and scripts:
`~/mojolearn-evidence/neural-cpu/`.

## Scope and tier

Every neural CPU route is a host binding (`_mojolearn_{training,mamba,
transformer,embedding,neural,byte_lm}_host`), and every host binding builds
IDENTICAL only (`bindings/build_host_family.sh` refuses any other tier). So
on the CPU the neural family has ONE tier; a FAST CPU tier would be a new host
build tier (packaging, `_backend` routing), not a speed change, and is not
started here. Every change below keeps the IDENTICAL bits.

## What changed (all bit-identical; proven as noted)

Single thread (`lane/neural-cpu`):
1. `gemm/host/gemm_host_rows.mojo`: `gemm_oracle`'s bits at CPU speed
   (flushed packed operands, right operand in GHR_G-column panels, SIMD cells
   down p, deferred flush with an exact subnormal fallback, in-place balanced
   fold); `gemm_host_rows_right_zero_padded` likewise. Every neural host
   caller of `gemm_oracle` / `gemm_oracle_right_zero_padded` (transformer
   forward/backward oracles, Mamba-1/2/3 oracles and backward oracle, the
   Mamba host shim, samba ops, loss and optimizer oracles, the neural host
   binding's MLP, the byte LM host step's head) now calls it. The oracle
   itself is unchanged and stays the reference.
   Proof: `pixi run check-gemm-host-rows` (5,403 cases: every op, one and
   many leaves, the carry, ragged n, subnormal / signed-zero / planted /
   ReLU fixtures, forced fallback, the padded door at real_k 0..k with and
   without garbage past real_k) PASS; its sabotage
   (`check-gemm-host-rows-sabotage`, no fallback) FAILS.
2. `core/host_lanes.mojo`: the byte LM's lane seams (`ftz_lanes`,
   `expf_lanes`, `silu_lanes`, `fmax_fold_span`) moved out of
   `training/byte_lm_host_kernels.mojo` (re-exported there, byte LM bits
   unchanged: `byte_lm_host_exp_check` all 2^32 patterns 0 mismatches and
   `byte_lm_host_kernels_check` PASS after the move) plus span helpers
   (scale, exp-shift, div, fmax fold, SiLU, product, add), each naming the
   scalar statement it equals. `pixi run check-host-lanes` PASS; the
   `span_div` reciprocal sabotage FAILS it.
3. Transformer oracle: S19 value sum as head_dim lanes; S12-S18 (scale,
   mask, max, exp, denominator, division), SiLU, gated product and the
   second residual as lanes over preallocated stage lists (stage order and
   plants unchanged).

Threads (`lane/neural-cpu-threads`, every split through
`core/host_parallel.mojo::host_parallelize`, task count from
`core/host_predict_threads.mojo`):
4. `gemm_host_rows`: packs and cells split over tasks (rows, or panels when
   m is small); an unpacked small-m path (m <= 4: decode steps) with columns
   over tasks.
5. `tools/mamba_host_gen.py`: every generated launch evaluates its arguments
   once and splits its grid over tasks (`device_shim.host_launch`); all
   generated files regenerated.
6. `mamba/host/mamba3_s16_host.mojo` (substituted by the generator for two
   launches): the Mamba-3 S16 q/k/v backward and the S17 reverse-state
   backward as lanes (over n / p) with the p-independent products computed
   once, rows over tasks.
7. Transformer oracle softmax rows and S19 rows over tasks; transformer
   backward oracle d_q_rope / d_k_cache / d_v_cache chains as head_dim lanes,
   rows over tasks.
8. Host refusals (`refuse_nonfinite` in the transformer, Mamba, optimizer,
   loss, embedding and samba-ops oracles): a lane bit-scan fast path
   (`all_finite`); transformer host weights moved instead of copied.
   `mamba2/3` oracles' `_zeros` is one sized List.

Sabotage patches (each must make the lane check DISAGREE):
`gemm/checks/sabotage/gemm_host_rows_descending.patch`,
`transformer/checks/sabotage/transformer_host_value_sum_descending.patch`,
`transformer/checks/sabotage/host_lanes_span_div_reciprocal.patch`,
`transformer/checks/sabotage/transformer_host_bwd_kv_descending.patch`,
`mamba/checks/sabotage/mamba3_s16_host_dk_descending.patch`,
`mamba/checks/sabotage/mamba3_s17_host_direct_descending.patch`.

## Speed, before -> after (CPU, pod neural-cpu, same box, same inputs)

Measured while a lane check was compiling on the same box, so absolute
numbers are noisy; every "after" read the same digest as its "before".
bench = `tools/bench_board_neural.py race --arms ours --shape full` (CPU lanes,
L capped at 512); train = the same inputs through the public class.

| route | before (main) | after, 1 thread | after, 24 threads |
|---|---|---|---|
| TransformerBlockInference.forward (B1 L512 DM384) | 9,856 ms | 155 ms | 78 ms |
| Mamba1BlockInference.forward | 5,487 ms | 314 ms (single-thread branch) | 132 ms |
| Mamba2BlockInference.forward | 6,793 ms | 350 ms | 182 ms |
| Mamba3BlockInference.forward | 5,650 ms | 379 ms | 124 ms |
| SambaInference.forward (B2 L512) | 63,467 ms | 1,208 ms | 827 ms |
| MLPInference.predict_logits (256 rows) | 0.49 ms | 0.34 ms | - |
| SambaStack.train_step on CPU (full) | 142,000 ms | - | 3,900 ms |
| SambaStack.train_step on CPU (small) | 1,420 ms (1.7 s) | 144 ms | 85 ms |
| SambaInference.step (decode, B2) | 112-174 ms | 25 ms | 26 ms |
| LanguageModelHostTrainer.train_step (B1 L256 DM256, 2 layers) | 6,911 ms | 534 ms | 443 ms |

## Gate status

- `lane/neural-cpu` (items 1-3): lane check on the 47 lanes
  `lane_select --changed-since origin/main` names
  (`~/mojolearn-evidence/neural-cpu/gate1_lanes.txt`): NOT RUN TO A VERDICT.
  Pod 4hzekcstjwydl4 vanished before the gate or the sabotage runs
  (`sab_runs.sh`) reported (RunPod GET 404, 04:53Z Sep 28, lease still
  open to 05:55Z); nothing was pulled. Re-renting failed: RunPod
  `Your account balance is too low to rent a pod`, and the account lists
  ZERO pods (every lane's RunPod box is gone). Branch merged with
  origin/main (clean) and pushed. Coordinator (Sep 28 ~05Z): RunPod balance
  negative, do not retry renting.
- AMD central box instead (gfx942 vs CPU), QUEUED 05:00Z Sep 28 for a slot
  (both held by decomp and sequence): `/root/ev-neural-cpu/amd_gate.sh` on the
  box (copy in `~/mojolearn-evidence/neural-cpu/amd_gate.sh`) runs the three
  sabotages (gemm panel chain, S19 value lanes, span_div reciprocal; each must
  read DISAGREE) and then the 47-lane check; verdicts land in
  `/root/ev-neural-cpu/summary.txt`, the Mac-side waiter logs to
  `~/mojolearn-evidence/neural-cpu/amd_gate_local.log`. Fetch with
  `tools/amd_central.sh fetch neural-cpu /root/ev-neural-cpu
  ~/mojolearn-evidence/neural-cpu/amd-results`.

### Owed pod steps (need a RunPod NVIDIA pod `neural-cpu`)

1. NVIDIA + CPU gate of `lane/neural-cpu`: `algos_lane_check.sh` over
   `gate1_lanes.txt` (AGREE) and the three sabotages (`sab_runs.sh`, bite).
2. test_host_surface on the branch; then merge to main and push.
3. `lane/neural-cpu-threads` (after core/host_parallel.mojo is on main):
   merge main, gate at MOJOLEARN_CPU_THREADS=1, 3 and default, its three
   extra sabotages (bwd kv, S16 dk, S17 direct), test_host_surface,
   test_lane_select, merge.
4. Before/after timing re-read on one quiet box (the table above was taken
   while a lane check compiled).
- `lane/neural-cpu-threads`: waits for core/host_parallel.mojo on main.

## Findings for other lanes

- `core/host_predict_threads.mojo` defaults to `num_physical_cores()` (112
  on this pod) while the container's cgroup quota is 23.8 CPUs: the default
  oversubscribes and measured slower than MOJOLEARN_CPU_THREADS=24 on the
  GEMM. A cgroup-aware default is the cpu lane's call.

## Next

1. Merge `lane/neural-cpu` once its gate passes (and the sabotage arms bite).
2. When core/host_parallel.mojo is on main: merge main into
   `lane/neural-cpu-threads`, gate the lanes it selects (NVIDIA + CPU at
   MOJOLEARN_CPU_THREADS=1, 3 and default), sabotage arms, test_host_surface,
   test_lane_select (Mojo imports changed), merge.
3. Remaining hot spots: the other naive Mamba-3 backward kernels (S17
   operands), the Mamba-3 forward's serial stages, decode-step overheads
   (per-call weight reads and state copies).
4. Apple / AMD identity: one batched steward request after each merge.
