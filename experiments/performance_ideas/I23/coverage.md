# I23 implementation coverage

Mode: IDENTICAL. New source mechanisms are default off.

Implemented: Actual independent batched ARIMA likelihood/gradient, LBFGS fit, batch-composition/launch invariance, refusal and public forecast gates; concrete pointer-origin compiler repair does not alter arithmetic.

Remaining original-card scope: A versioned affine recurrence/filter scan is not implemented. No Kalman covariance rank-one approximation or changed initialization/differencing is introduced.

Qualification: compile checks only on the development machine. Same-version host/NVIDIA/AMD/Apple output identity and NVIDIA+AMD full-operation performance acceptance have not been demonstrated. Apple IDENTICAL is an identity witness only. No speed claim or production promotion is made.

Merge admission: new source candidates remain explicit default-off opt-ins.
Qualification remains pending. `native_arms.json` lists independently compiled
incumbent/candidate and available rollback arms; compilation never promotes a
switch or supplies performance evidence. Existing promoted defaults remain unchanged.
