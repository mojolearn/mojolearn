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
This branch has NOT yet been merged to main or published in a wheel.
