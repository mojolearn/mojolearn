# I16 implementation coverage

Mode: IDENTICAL. New source mechanisms are default off.

Implemented: New compact long-list chunk tasks with device prefix descriptors, native-warp exact selection and variable-query exact partial merge; three-way flat/staged/task fixture fixes selected lists and distance bits on empty/uniform/giant/skewed lists and tail dimensions.

Remaining original-card scope: Short-list compact batches use the selected-list task descriptor mechanism. Scratch, descriptor preparation and the task-total readback remain included in whole-query timing; no recall/index-training changes.

Qualification: compile checks only on the development machine. Same-version host/NVIDIA/AMD/Apple output identity and NVIDIA+AMD full-operation performance acceptance have not been demonstrated. Apple IDENTICAL is an identity witness only. No speed claim or production promotion is made.

Merge admission: new source candidates remain explicit default-off opt-ins.
Qualification remains pending. `native_arms.json` lists independently compiled
incumbent/candidate and available rollback arms; compilation never promotes a
switch or supplies performance evidence. Existing promoted defaults remain unchanged.
