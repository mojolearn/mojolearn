# RF/ET shared arithmetic extraction — 2026-09-10

Base `7620749b`, local Apple M4 / Mojo 1.0.0 ed45d567.
RF and ET previously had identical generic logarithm wrapper bodies. They now
import `core.tree_math.tree_log` under their existing `_log_seam` name.

Both existing independent `objectives_check.mojo` programs passed in FAST,
DETERMINISTIC and IDENTICAL under the shared build lock. Each of the six logs
records the exact command and exit code 0. They exercise arithmetic anchors,
GPU objective paths and their existing negative controls. This is bounded
local verification, not cross-vendor identity, whole-forest timing or a wheel
publication. No remote work was run.
