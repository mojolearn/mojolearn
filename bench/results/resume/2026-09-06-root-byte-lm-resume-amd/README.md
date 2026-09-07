# AMD foreign checkpoint continuation

Root-only serialized campaign, two CPU cores/threads, exact retained AMD
binding and successful 1 GiB HIP runtime pool settings.

## resume1: interpreter prerequisite failure

Source `3d255241d84c5ba856a3e5eabab942887d88b9ae`. Transport setup and compact
preflight passed. The Conda Python at `/opt/conda/envs/py_3.10/bin/python`
lacks `os.memfd_create`, so the frozen driver stopped before loading the
NVIDIA checkpoint or running resumed training. This is not a numerical
mismatch. No missing-moments control ran and no resume admission is asserted.

Original logs: [resume128](resume1/remote/byte-lm-resume/resume128.log),
[controller](resume1-controller.log). Pod `4452jcpnumdud4` was deleted 204 and
verified absent 404 with 56 minutes left. Next setup explicitly selects
Ubuntu system Python and checks the sealed-memory API before capture;
if needed, its guarded setup installs python3-venv. No capture/numerical
source or exact retained native binary is changed by that transport fix.

## resume2: foreign continuation and control passed

Transport snapshot `d5893e2ae270236c629b00a198c2e017c26cb5af`, exact retained
AMD binding and numerical inventory. Ubuntu system Python exposes memfd/seal
APIs; guarded setup installed python3-venv. Setup and all five compact jobs
passed: preflight, resume128, verify-resume, zero-moments65 and verify-control.

The final [root all-raw comparator](../2026-09-06-root-byte-lm-cross-vendor/nv-to-amd-resume128-comparison.json)
admits bounded identity and learning with no missing prerequisites. It verifies
actual NVIDIA step-64 checkpoint consumption and exact continuation against
the same-device continuous run. Resetting moments changes the three expected
post-update arrays; the control is effective. This proves one direction only.

Pod `ocshz624wm2axa` was deleted with HTTP 204 and verified absent with HTTP 404.
All failed attempts remain retained separately. No Apple model or speed claim.

## head1: reverse-direction head passed

A fresh AMD head64 campaign at transport snapshot `d5893e2a` passed setup,
preflight, 64 training steps and remote verification. Root then admitted the
full AMD baseline and compared every raw head step before authoring the foreign
handoff. Its checkpoint SHA is identical to the NVIDIA head checkpoint:
`264f189179b5a3b8a3cbbceb0f54209fb657ff7dacf701b4f8a760c693da8ff0`.

[Root author record](../2026-09-06-root-byte-lm-cross-vendor/handoff-amd-head64-author.json).
Pod `670mgud0t4oh6l` was deleted 204 and verified absent 404. The subsequent NVIDIA continuation now passed its final all-raw comparator;
both directions are admitted in the cross-vendor evidence index.
