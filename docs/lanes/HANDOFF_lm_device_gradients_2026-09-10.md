# Language-model device gradients and branch audit — September 10, 2026

Starting remote main: a66765f0, including generalized model commit 72f79a18.
The tree lane had independently published after that commit. This work takes
remote main as its base and changes no tree implementation.

## Branch audit

A branch not being an ancestor is not evidence of missing implementation.
`git log --cherry-pick --right-only --no-merges origin/main...BRANCH` has no
remaining commits for byte-runtime, byte-numerical, fused-attention,
fused-attention-regblock, gemm-identical, gemm-splitk, the recent Mamba lanes,
UMAP pow64, embedding sort, transformer admission, kNN width/throughput and
IVF width lanes. The detached knn-next, mamba-next and transformer-admission
worktrees also have no remaining patch-unique commits.

Three older branches need squash-aware interpretation:

- `lane/knn-selector`: its neighbor implementation is exactly the one in
  squash b3abed5c (`git diff lane/knn-selector b3abed5c -- neighbors` is empty).
  Main has subsequent fixes and tuning. Do not restore its old dispatch table.
- `lane/knn-next-pass`: its neighbor implementation and kernel matrix exactly
  match 06e6c93d. Two intermediate experimental commits appear unique because
  their rejected pieces were removed in the integrated result.
- `lane/mamba3-statepass`: integrated by 16b650a6. The ops differences at that
  integration point are an explicitly REJECTED hardware-FTZ probe and comments;
  the production state kernels were incorporated. Main has further Mamba work.

There is no pending ready implementation from these lanes to merge. Existing
branch names are retained as historical evidence. Unrelated certification,
grid, NumPy and tree branches are outside this audit's integration scope.
The old Mamba statepass worktree also has uncommitted kernel-matrix/GEMM edits;
they predate main's current GEMM changes (including the rounded-FMA repair and
wide split plan already on main) and include tree dispatch edits. They are
retained in place, not merged over newer code.
The old detached kNN checkout contains an untracked oracle text file; it was
not promoted as implementation or qualified evidence.

## Training change

The shared transformer backward now accepts a caller-owned device gradient.
The old host-list API uploads once and delegates to the same implementation.
The generalized LM traverses all layers in reverse using the successor's
`d_x` directly. Removing a layer's stage from the owning list permits disjoint
mutable borrows; reinsertion preserves layer order. The one-layer path reads
the head gradient directly too.

Upstream tensor graph: transformers/models/llama/modeling_llama.py:402–412.
No arithmetic kernel, rounding, reduction order or acceptance tolerance changes.
The device nonfinite scan remains; an invalid gradient downloads only its first
bad scalar to preserve NaN versus infinity diagnostics. Existing synchronization
keeps borrowed buffer lifetimes valid. This does not remove the other stage
synchronizations or the Python binding's per-call trainer construction.

At 32,768 tokens × width 768 × 12 layers, eliminating both directions removes
2.25 GiB of explicit inter-layer staging per training step (byte-count estimate,
not a measured speedup). Small correctness checks cannot establish training
throughput. Large pilot timing remains required; no opponent row or dispatch
performance gate is changed here.

## Validation and next work

Both bindings compiled on Apple. The 76 host tests, eight native training steps,
eight evaluations and public Transformer surface suite pass. All 88 native
arrays in eight captures match the prior generalized-model implementation bit
for bit. Evidence: `bench/results/lm_device_gradients_2026-09-10/`.
No new opponent measurement, rental or performance claim.

Next: resident Python/native training session, reduced stage materialization,
memory-efficient attention/loss, then a large pilot with complete step timing.
GEMM and attention still need large-shape tuning; Transformer opponent admission
is still unqualified. The small surface suite does not change that status.

Continuation: HANDOFF_lm_capacity_2026-09-10.md records current opponent gaps
and a live large-shape allocation blocker, with a reproducible host inventory.
