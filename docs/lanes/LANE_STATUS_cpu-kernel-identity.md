# CPU kernel identity completion

Worktree `~/mojolearn-wt/cpu-kernel-identity`, branch `lane/cpu-kernel-identity`.
Implements plan A3: polynomial, sigmoid and Laplacian KernelRidge/Nystroem CPU
arithmetic and saved-model inference. Degree now travels through the existing
binding params contract (previously ignored on CPU), bounded 0..32. The host
restates the device's FMA/polynomial/tanh/L1-exp operations, with no GPU imports.
The Laplacian host sabotage arm reverses the ascending feature sum.

Six new harness lanes have held-out, save/reload and batch probes. They are
explicitly pending (`no reference`) and do not enter default public verification.
Their saved-model recording debt is named in SAVED_MODEL_INFERENCE_OWED rather
than hidden by the older registry being empty. The new kernels do not imply
precomputed-kernel support or public CPU training.

Validation so far: 164 source/manifest checks passed. Native clean/sabotage
builds are queued under mac_slot after the running Apple capture. External
logs/builds: `~/mojolearn-evidence/next-wheel-coverage/cpu-kernel-identity/`.
Build queue session 3015; 900-second execution cap, one compiler job. Pinned
compiler and identical pixi.lock from release-087-final; outputs are isolated,
and no frozen release file/binary is changed. Runtime tests and numerical
reference comparisons must pass before merging this implementation to main.

Still required: native compilation, new CPU runtime tests, full nine-fixture
repeated CPU/Apple comparisons, observed sabotage detection, and independent
NVIDIA/AMD records plus installed-wheel qualification before default promotion.
Plan and user direction also require merging completed work and removing only
finished, clean, merged worktrees with no active dependent jobs. Preserve the
release worktrees and unrelated owners' active work.
