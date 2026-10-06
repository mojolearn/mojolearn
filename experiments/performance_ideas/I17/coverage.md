# I17 implementation coverage

Mode: IDENTICAL. New source mechanisms are default off.

Implemented: Actual Depthwise/Lossguide complete tree fits with partition/frontier controls, changing leaf capacity, feature packing and repeated-fit split/leaf/weight fingerprints; existing complete boosting gate is declared.

Remaining original-card scope: New default-off bounded16 MiB IDENTICAL frontier-stat residency retains scored node words on device across rounds and downloads once at the existing final wait. Complete SymmetricTree/Depthwise/Lossguide callers with numeric and actual feature-frequency CTR input now fingerprint all tree fields, borders, losses and public predictions across repeated fits. Physical partitions, row indices, offset/size planes and histogram/statistic storage are already TTreeWorkspace/TDepthwiseWorkspace-owned and remain resident through the tree; NS_INHERIT_PARTITION independently attributes reuse into the next tree. Current host selection/termination/planning needs leaf size metadata, and build_non_symmetric_tree consumes host result_paths/result_weights/result_values. Moving final model assembly would require a new device tree-builder API and downstream model consumer; an unused device mirror would add copies without extending the live operation. The bounded statistics candidate covers the supported new residency comparison. Whole-boosting identity/performance validation remains owed.

Qualification: compile checks only on the development machine. Same-version host/NVIDIA/AMD/Apple output identity and NVIDIA+AMD full-operation performance acceptance have not been demonstrated. Apple IDENTICAL is an identity witness only. No speed claim or production promotion is made.

Merge admission: new source candidates remain explicit default-off opt-ins.
Qualification remains pending. `native_arms.json` lists independently compiled
incumbent/candidate and available rollback arms; compilation never promotes a
switch or supplies performance evidence. Existing promoted defaults remain unchanged.
