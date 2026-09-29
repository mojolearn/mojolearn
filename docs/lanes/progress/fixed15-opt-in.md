# FP32 default restoration

User request 2026-09-29: fixed15 must be opt-in again. Isolated branch
fix/fixed15-opt-in, based on origin/main 55381675a. DEFAULT restored to
fp32_v1; fixed15 remains inference-enabled but experimental. Explicit
keyword, environment and process opt-ins remain. Saved-profile adoption
and training refusals are unchanged. No native arithmetic changed.

Selector-only tests on the remote M3 Ultra CPU: 26 passed, 22 deselected.
Command: pytest --noconftest tests/test_numeric_profile.py -q -k
'not package and not trainer and not bare_block and not loader and not every_family and not fresh_process'.
Temporary isolated venv /tmp/fixed15-opt-in.2yyFqr/venv, no GPU use.
Native/model tests remain pending; the separate fair-comparison branch
inherits the restoration and is acquiring an authorized NVIDIA pod.
## Merge decision

User explicitly requested merging this restoration. The only runtime change
is DEFAULT=fp32_v1 plus the optional profile's status/note metadata; no
native kernel or FP32 arithmetic changes. Existing NVIDIA FP32 reference
and resident gates at 55381675a remain applicable to that unchanged path.
The 26 selector tests check the changed default and retained opt-ins.

Additional model evidence: AMD matched-comparison request
1790684366433-speed-profile-fair-compare-94ddca292b completed PASS. Its source
inherits this restored selector: unnamed FP32 and explicit FP32/fixed15
passed resident versus per-layer full-logit, decode and generation checks.
Its experimental private AMD dispatch is NOT part of this restoration.
Fresh NVIDIA comparison nvc1-0001 is still building, not claimed as passed.

Merge this restoration only; no wheel upload or publication accompanies it.
Main and 0.8.25 then share FP32 default arithmetic, not identical source,
feature sets or wheel bytes. Explicit fixed15 states keep their profile.
