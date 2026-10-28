# Lane: peer2-preprocessing (Normalizer/prep coherent freeze + PCA fit/fit_transform)
Read COMMON.md. Worktrees: ~/mojolearn-wt/peer2-preprocessing-20261027 (lane/peer2-preprocessing-20261027 @ 7f8297cfe5, 11 dirty)
and ~/mojolearn-wt/peer2-decomposition-20261027 (@ 3bd993dec5). Parent branch lane/peer2-preprocessing-decomposition-manifold-20261027
holds batch 7/8 history; read experiments/peer2_native_boundary_20261027/HANDOFF_20261028.txt, compiled_batch_{7,8}.json,
BATCH8_SOURCE_REVIEW.txt and the root handoff Peer2 section. Evidence: ~/mojolearn-evidence/peer2-preprocessing-20261028/.
Known state: batch 7 prep compiled at 71cba19d87 but the late semantic/error-order guard 9608dee4e5 (object-input refusal before
Normalizer width checks) was NOT in that compile; prep e8a10* excluded from batch 8 (no caller coverage); Normalizer/PCA fit still held.
Goal: (1) finish the dirty prep draft so Normalizer and the other prep paths own validation/preparation/result natively, including
9608dee4e5 semantics, with narrow prep_stateless_c/_kind provider/context type/descriptor admission (Peer2 owns it); freeze, compile the
prep root (entry per batch-7 receipt) and push a coherent green head. (2) Then PCA fit/fit_transform native ownership in the
decomposition worktree; compile its root; push. Shared-ownership limits from the handoff: host_sparse_native narrow sparse snapshot
helpers, native_scalar_decode captured-dtype helper and training_scalar_native dtype reuse are Peer2's; do not touch Peer1 ANN files
beyond conflict resolution. Merge origin/main into the lanes before the freeze.
