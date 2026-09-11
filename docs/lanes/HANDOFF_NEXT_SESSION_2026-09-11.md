# Handoff, 2026-09-11 02:15 EDT

Everything below is on main (070da847 or earlier) with evidence under
bench/results/e1g/. Read the briefs before touching any of it.

## Shipped

- mojolearn 0.8.0 on PyPI (macOS 02:46Z, Linux 02:52Z, 2026-09-11), the
  tree-only tier rule (DEVIATION 2490). CHANGELOG has the commits and tags.
- Parallel extension builds on every wheel builder (DEVIATION 2501,
  MOJOLEARN_BUILD_JOBS default 4); first timed parallel release OWED.

## AI/classical IDENTICAL handoff, executed

1. Byte-LM lifetime hang (priority 1): CLOSED. Cause was our teardown
   destroying a DeviceContext with stream-ordered buffer frees in flight,
   which leaves the MAX runtime allocator lock held on an RTX 4090 pod;
   the next context's first enqueueCreateBuffer blocks forever (native
   backtrace in bench/results/e1g/2026-09-11_010801-nvidia). Fix DEVIATION
   2520: synchronize after releasing the trainer and before the context on
   both paths. 18 of 19 harness cases pass, none hang. Upstream report OWED
   (docs/lanes/BRIEF_byte_lm_lifetime_2026-09-10.md, run 6).
2. Device-owned LM step (priorities 1 and 2): DONE, DEVIATION 2514, design
   docs/lanes/DESIGN_lm_device_owned_step_2026-09-11.md. Target-shape step
   (162,147,840 params, L2048, V50257) on the H100: 45 s to 0.565 s, every
   step bit-equal to the previous trainer, six failure controls roll back.
   step_result defaults to 'lean' for resident trainers. What remains per
   step is device compute: blocks 0.50 s of which attention backward 262 ms.
3. GEMM/attention (priority 3): NOT STARTED, and now the whole target-shape
   step. Start from docs/lanes/HANDOFF_speed_gemm_2026-09-10.md and the lean
   timing step's phase list (bench/results/e1g/2026-09-11_004220-nvidia/
   remote/lm-step-memory/target-lean-timing). Attention backward first.
4. kNN selection (priority 4): the premise (insertion chain dominates
   through its admissions) was WRONG in a specific way: the chain issued on
   90 to 96 percent of warp-steps because SOME lane admits, so bounds,
   deferred insertion and CAP = K were all neutral or negative. Measured
   on the H100 with timing-only arms: rank 0.4 to 0.75 ms, tile reads 3.5
   ms, chain 6.7 ms (k10) and 10.8 ms (k15). WIN (DEVIATION 2523, flipped
   for NVIDIA in checks/kernel_matrix.mojo): the warp bound behind a
   ballot branch, admit rate 0.46, requests 30.8 to 27.8 ms (k10) and 36.0
   to 31.0 ms (k15), bit-equal. OWED: Apple and AMD timing (identity
   passes), a paired cuML run at the qualified boundary, and the next
   lever (a tighter bound now that the branch exists). Read
   docs/lanes/BRIEF_knn_selection_2026-09-10.md Steps 4 to 9 first.
5. Training readiness (priority 5): NOT STARTED.

## Running the machinery

- Every NVIDIA measurement: `MOJOLEARN_GEMM_LEG_EXTRA=<sh file>` with
  `sh tools/gemm_remote_leg.sh nvidia --payload gemm --gpu "<name>"
  --local-card bench/results/e1/2026-08-28_131651-runpod-nvidia/lanes/gemm.identical.card
  --allow-concurrent --rent --minutes 60`; it ships the committed tree
  (refuses a dirty tree; run from a `git worktree` at HEAD if lanes are
  editing), the extra body runs on the pod, output comes home under
  `<leg>/remote/`. Delete `remote/tools-venv` before committing evidence
  (now gitignored).
- Byte LM leg bodies must build `bindings/build.sh` (identical) before the
  byte LM binding: the NumPy-free Python layer takes its host helpers from it.
- The kNN gate: arms via MOJOLEARN_KNN_SELECTION_ARMS, timing-only arms via
  MOJOLEARN_KNN_SELECTION_TIMING_ONLY_ARMS, phase timers via
  MOJOLEARN_KNN_SELECTION_PHASE_TIMERS=1 (serializes requests: mechanism
  verdicts only; promotion needs an unserialized run, all eight cells).
- Rules that bit tonight: every literal count or spelling in a release gate
  cost a rental pass; mojolearn.Array has no __setitem__, .copy() or fancy
  indexing (view through np.asarray in harnesses); the trees lane merges
  its branches into main every hour (expect non-fast-forward pushes).
