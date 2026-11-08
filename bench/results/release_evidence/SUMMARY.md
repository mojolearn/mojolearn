# Mojolearn 0.8.66: release evidence summary

Written by the release's record step from the files in this directory; nothing was run to write it. Every number below links to the file it comes from.

| | |
|---|---|
| version | 0.8.66 |
| tag | `v0.8.66` |
| frozen source commit | `43b5f7c320e0942bdef8c212917b440d1ab1d513` |
| release tooling commit | `12a5af7d89844e557c9d10f7783b854a745a6f85` |
| date | 2026-11-05 |
| published platforms | macos, linux, nvidia, amd, linux-meta |

## Soft failures (fix lanes owed)

| vendor | unit | reason |
|---|---|---|
| nvidia | measuresmoke | measuresmoke: no measurement-smoke summary: it is produced only by the measurement smoke's own box job (tools/ops/release_gate.py <vendor> <version> --job measuresmoke, launched by the gate-<vendor> step); a vendor composed on the Mac under --reuse-unchanged has none. To pass, rerun the gate-<vendor |
| nvidia | runtime | runtime qualifier left no report at /Users/andrewhendel/mojolearn-evidence/release/0.8.66/43b5f7c320e0/gate-nvidia/c92469d260a8b3a5f70d-release-light-v1/runtime-nvidia/runtime-report.json |
| amd | measuresmoke | measuresmoke: no measurement-smoke summary: it is produced only by the measurement smoke's own box job (tools/ops/release_gate.py <vendor> <version> --job measuresmoke, launched by the gate-<vendor> step); a vendor composed on the Mac under --reuse-unchanged has none. To pass, rerun the gate-<vendor |
| amd | runtime | runtime qualifier left no report at /Users/andrewhendel/mojolearn-evidence/release/0.8.66/43b5f7c320e0/gate-amd/9feb99a3181f7fe126eb-release-light-v1/runtime-amd/runtime-report.json |
| amd | autograd/autograd-gradients | units: unit autograd/autograd-gradients did not pass |
| amd | autograd/autograd-higher | units: unit autograd/autograd-higher did not pass |
| amd | autograd/autograd-import-ops | units: unit autograd/autograd-import-ops did not pass |
| amd | autograd/autograd-lm-tape | units: unit autograd/autograd-lm-tape did not pass |
| amd | autograd/autograd-mamba-tape | units: unit autograd/autograd-mamba-tape did not pass |
| amd | autograd/autograd-mlp-identity | units: unit autograd/autograd-mlp-identity did not pass |
| amd | autograd/autograd-public | units: unit autograd/autograd-public did not pass |
| amd | autograd/autograd-samba-tape | units: unit autograd/autograd-samba-tape did not pass |
| amd | autograd/autograd-small-mlp-tape | units: unit autograd/autograd-small-mlp-tape did not pass |
| amd | autograd/import-torch-smoke | units: unit autograd/import-torch-smoke did not pass |
| amd | autograd/real-model-smoke | units: unit autograd/real-model-smoke did not pass |
| amd | runtime | runtime qualifier status failed in /Users/andrewhendel/mojolearn-evidence/release/0.8.66/43b5f7c320e0/gate-amd/9feb99a3181f7fe126eb-release-light-v1/runtime-amd/runtime-report.json:  |
| amd | box-exit | the amd release gate box job exited 1; log /Users/andrewhendel/mojolearn-evidence/release/0.8.66/43b5f7c320e0/gates/gate-amd.log (its HARD parts passed) |
| amd | measuresmoke | measuresmoke: 37 of 653 recipes failed: algos/ann-refine/istella/rows=2000 (ours: status not-scored); algos/ann-refine/taxi/rows=2000 (ours: status not-scored); algos/avgpool1d/synthetic/rows=2000 (-: status failed; -: BOARD-PARAMS NOT CHECKED (the driver printed no BOARD-PARAMS line); ours: status  |
| nvidia | autograd/autograd-public | units: unit autograd/autograd-public did not pass |
| nvidia | autograd/import-torch-smoke | units: unit autograd/import-torch-smoke did not pass |
| nvidia | autograd/real-model-smoke | units: unit autograd/real-model-smoke did not pass |
| nvidia | runtime | runtime qualifier status failed in /Users/andrewhendel/mojolearn-evidence/release/0.8.66/43b5f7c320e0/gate-nvidia/c92469d260a8b3a5f70d-release-light-v1/runtime-nvidia/runtime-report.json:  |
| nvidia | box-exit | the nvidia release gate box job exited 1; log /Users/andrewhendel/mojolearn-evidence/release/0.8.66/43b5f7c320e0/gates/gate-nvidia.log (its HARD parts passed) |
| testpypi | rehearsal/secondary | pip install mojolearn-core[max]==0.8.66 on manylinux_2_35_x86_64 py3.10: pip could not resolve it (exit 1; /Users/andrewhendel/mojolearn-evidence/release/0.8.66/43b5f7c320e0/testpypi-rehearsal/resolve/mojolearn_core_max_manylinux_2_35_x86_64_py3.10.log) |
| testpypi | rehearsal/secondary | pip install mojolearn-core[max]==0.8.66 on manylinux_2_35_x86_64 py3.11: pip could not resolve it (exit 1; /Users/andrewhendel/mojolearn-evidence/release/0.8.66/43b5f7c320e0/testpypi-rehearsal/resolve/mojolearn_core_max_manylinux_2_35_x86_64_py3.11.log) |
| testpypi | rehearsal/secondary | pip install mojolearn-core[max]==0.8.66 on manylinux_2_35_x86_64 py3.12: pip could not resolve it (exit 1; /Users/andrewhendel/mojolearn-evidence/release/0.8.66/43b5f7c320e0/testpypi-rehearsal/resolve/mojolearn_core_max_manylinux_2_35_x86_64_py3.12.log) |
| testpypi | rehearsal/secondary | pip install mojolearn-core[max]==0.8.66 on manylinux_2_35_x86_64 py3.13: pip could not resolve it (exit 1; /Users/andrewhendel/mojolearn-evidence/release/0.8.66/43b5f7c320e0/testpypi-rehearsal/resolve/mojolearn_core_max_manylinux_2_35_x86_64_py3.13.log) |
| testpypi | rehearsal/secondary | pip install mojolearn-core[max]==0.8.66 on manylinux_2_35_x86_64 py3.14: pip could not resolve it (exit 1; /Users/andrewhendel/mojolearn-evidence/release/0.8.66/43b5f7c320e0/testpypi-rehearsal/resolve/mojolearn_core_max_manylinux_2_35_x86_64_py3.14.log) |
| testpypi | rehearsal/secondary | pip install mojolearn-core[numpy]==0.8.66 on manylinux_2_35_x86_64 py3.10: pip could not resolve it (exit 1; /Users/andrewhendel/mojolearn-evidence/release/0.8.66/43b5f7c320e0/testpypi-rehearsal/resolve/mojolearn_core_numpy_manylinux_2_35_x86_64_py3.10.log) |
| testpypi | rehearsal/secondary | pip install mojolearn-core[numpy]==0.8.66 on manylinux_2_35_x86_64 py3.11: pip could not resolve it (exit 1; /Users/andrewhendel/mojolearn-evidence/release/0.8.66/43b5f7c320e0/testpypi-rehearsal/resolve/mojolearn_core_numpy_manylinux_2_35_x86_64_py3.11.log) |
| testpypi | rehearsal/secondary | pip install mojolearn-core[numpy]==0.8.66 on manylinux_2_35_x86_64 py3.12: pip could not resolve it (exit 1; /Users/andrewhendel/mojolearn-evidence/release/0.8.66/43b5f7c320e0/testpypi-rehearsal/resolve/mojolearn_core_numpy_manylinux_2_35_x86_64_py3.12.log) |
| testpypi | rehearsal/secondary | pip install mojolearn-core[numpy]==0.8.66 on manylinux_2_35_x86_64 py3.13: pip could not resolve it (exit 1; /Users/andrewhendel/mojolearn-evidence/release/0.8.66/43b5f7c320e0/testpypi-rehearsal/resolve/mojolearn_core_numpy_manylinux_2_35_x86_64_py3.13.log) |
| testpypi | rehearsal/secondary | pip install mojolearn-core[numpy]==0.8.66 on manylinux_2_35_x86_64 py3.14: pip could not resolve it (exit 1; /Users/andrewhendel/mojolearn-evidence/release/0.8.66/43b5f7c320e0/testpypi-rehearsal/resolve/mojolearn_core_numpy_manylinux_2_35_x86_64_py3.14.log) |
| testpypi | rehearsal/secondary | pip install mojolearn-core[torch]==0.8.66 on manylinux_2_35_x86_64 py3.10: pip could not resolve it (exit 1; /Users/andrewhendel/mojolearn-evidence/release/0.8.66/43b5f7c320e0/testpypi-rehearsal/resolve/mojolearn_core_torch_manylinux_2_35_x86_64_py3.10.log) |
| testpypi | rehearsal/secondary | pip install mojolearn-core[torch]==0.8.66 on manylinux_2_35_x86_64 py3.11: pip could not resolve it (exit 1; /Users/andrewhendel/mojolearn-evidence/release/0.8.66/43b5f7c320e0/testpypi-rehearsal/resolve/mojolearn_core_torch_manylinux_2_35_x86_64_py3.11.log) |
| testpypi | rehearsal/secondary | pip install mojolearn-core[torch]==0.8.66 on manylinux_2_35_x86_64 py3.12: pip could not resolve it (exit 1; /Users/andrewhendel/mojolearn-evidence/release/0.8.66/43b5f7c320e0/testpypi-rehearsal/resolve/mojolearn_core_torch_manylinux_2_35_x86_64_py3.12.log) |
| testpypi | rehearsal/secondary | pip install mojolearn-core[torch]==0.8.66 on manylinux_2_35_x86_64 py3.13: pip could not resolve it (exit 1; /Users/andrewhendel/mojolearn-evidence/release/0.8.66/43b5f7c320e0/testpypi-rehearsal/resolve/mojolearn_core_torch_manylinux_2_35_x86_64_py3.13.log) |
| testpypi | rehearsal/secondary | pip install mojolearn-core[torch]==0.8.66 on manylinux_2_35_x86_64 py3.14: pip could not resolve it (exit 1; /Users/andrewhendel/mojolearn-evidence/release/0.8.66/43b5f7c320e0/testpypi-rehearsal/resolve/mojolearn_core_torch_manylinux_2_35_x86_64_py3.14.log) |
| testpypi | rehearsal/secondary | pip install mojolearn-core[verify]==0.8.66 on manylinux_2_35_x86_64 py3.10: pip could not resolve it (exit 1; /Users/andrewhendel/mojolearn-evidence/release/0.8.66/43b5f7c320e0/testpypi-rehearsal/resolve/mojolearn_core_verify_manylinux_2_35_x86_64_py3.10.log) |
| testpypi | rehearsal/secondary | pip install mojolearn-core[verify]==0.8.66 on manylinux_2_35_x86_64 py3.11: pip could not resolve it (exit 1; /Users/andrewhendel/mojolearn-evidence/release/0.8.66/43b5f7c320e0/testpypi-rehearsal/resolve/mojolearn_core_verify_manylinux_2_35_x86_64_py3.11.log) |
| testpypi | rehearsal/secondary | pip install mojolearn-core[verify]==0.8.66 on manylinux_2_35_x86_64 py3.12: pip could not resolve it (exit 1; /Users/andrewhendel/mojolearn-evidence/release/0.8.66/43b5f7c320e0/testpypi-rehearsal/resolve/mojolearn_core_verify_manylinux_2_35_x86_64_py3.12.log) |
| testpypi | rehearsal/secondary | pip install mojolearn-core[verify]==0.8.66 on manylinux_2_35_x86_64 py3.13: pip could not resolve it (exit 1; /Users/andrewhendel/mojolearn-evidence/release/0.8.66/43b5f7c320e0/testpypi-rehearsal/resolve/mojolearn_core_verify_manylinux_2_35_x86_64_py3.13.log) |
| testpypi | rehearsal/secondary | pip install mojolearn-core[verify]==0.8.66 on manylinux_2_35_x86_64 py3.14: pip could not resolve it (exit 1; /Users/andrewhendel/mojolearn-evidence/release/0.8.66/43b5f7c320e0/testpypi-rehearsal/resolve/mojolearn_core_verify_manylinux_2_35_x86_64_py3.14.log) |
| testpypi | rehearsal/secondary | pip install mojolearn[max]==0.8.66 on manylinux_2_35_x86_64 py3.10: pip could not resolve it (exit 1; /Users/andrewhendel/mojolearn-evidence/release/0.8.66/43b5f7c320e0/testpypi-rehearsal/resolve/mojolearn_max_manylinux_2_35_x86_64_py3.10.log) |
| testpypi | rehearsal/secondary | pip install mojolearn[max]==0.8.66 on manylinux_2_35_x86_64 py3.11: pip could not resolve it (exit 1; /Users/andrewhendel/mojolearn-evidence/release/0.8.66/43b5f7c320e0/testpypi-rehearsal/resolve/mojolearn_max_manylinux_2_35_x86_64_py3.11.log) |
| testpypi | rehearsal/secondary | pip install mojolearn[max]==0.8.66 on manylinux_2_35_x86_64 py3.12: pip could not resolve it (exit 1; /Users/andrewhendel/mojolearn-evidence/release/0.8.66/43b5f7c320e0/testpypi-rehearsal/resolve/mojolearn_max_manylinux_2_35_x86_64_py3.12.log) |
| testpypi | rehearsal/secondary | pip install mojolearn[max]==0.8.66 on manylinux_2_35_x86_64 py3.13: pip could not resolve it (exit 1; /Users/andrewhendel/mojolearn-evidence/release/0.8.66/43b5f7c320e0/testpypi-rehearsal/resolve/mojolearn_max_manylinux_2_35_x86_64_py3.13.log) |
| testpypi | rehearsal/secondary | pip install mojolearn[max]==0.8.66 on manylinux_2_35_x86_64 py3.14: pip could not resolve it (exit 1; /Users/andrewhendel/mojolearn-evidence/release/0.8.66/43b5f7c320e0/testpypi-rehearsal/resolve/mojolearn_max_manylinux_2_35_x86_64_py3.14.log) |
| testpypi | rehearsal/secondary | pip install mojolearn[numpy]==0.8.66 on manylinux_2_35_x86_64 py3.10: pip could not resolve it (exit 1; /Users/andrewhendel/mojolearn-evidence/release/0.8.66/43b5f7c320e0/testpypi-rehearsal/resolve/mojolearn_numpy_manylinux_2_35_x86_64_py3.10.log) |
| testpypi | rehearsal/secondary | pip install mojolearn[numpy]==0.8.66 on manylinux_2_35_x86_64 py3.11: pip could not resolve it (exit 1; /Users/andrewhendel/mojolearn-evidence/release/0.8.66/43b5f7c320e0/testpypi-rehearsal/resolve/mojolearn_numpy_manylinux_2_35_x86_64_py3.11.log) |
| testpypi | rehearsal/secondary | pip install mojolearn[numpy]==0.8.66 on manylinux_2_35_x86_64 py3.12: pip could not resolve it (exit 1; /Users/andrewhendel/mojolearn-evidence/release/0.8.66/43b5f7c320e0/testpypi-rehearsal/resolve/mojolearn_numpy_manylinux_2_35_x86_64_py3.12.log) |
| testpypi | rehearsal/secondary | pip install mojolearn[numpy]==0.8.66 on manylinux_2_35_x86_64 py3.13: pip could not resolve it (exit 1; /Users/andrewhendel/mojolearn-evidence/release/0.8.66/43b5f7c320e0/testpypi-rehearsal/resolve/mojolearn_numpy_manylinux_2_35_x86_64_py3.13.log) |
| testpypi | rehearsal/secondary | pip install mojolearn[numpy]==0.8.66 on manylinux_2_35_x86_64 py3.14: pip could not resolve it (exit 1; /Users/andrewhendel/mojolearn-evidence/release/0.8.66/43b5f7c320e0/testpypi-rehearsal/resolve/mojolearn_numpy_manylinux_2_35_x86_64_py3.14.log) |
| testpypi | rehearsal/secondary | pip install mojolearn[torch]==0.8.66 on manylinux_2_35_x86_64 py3.10: pip could not resolve it (exit 1; /Users/andrewhendel/mojolearn-evidence/release/0.8.66/43b5f7c320e0/testpypi-rehearsal/resolve/mojolearn_torch_manylinux_2_35_x86_64_py3.10.log) |
| testpypi | rehearsal/secondary | pip install mojolearn[torch]==0.8.66 on manylinux_2_35_x86_64 py3.11: pip could not resolve it (exit 1; /Users/andrewhendel/mojolearn-evidence/release/0.8.66/43b5f7c320e0/testpypi-rehearsal/resolve/mojolearn_torch_manylinux_2_35_x86_64_py3.11.log) |
| testpypi | rehearsal/secondary | pip install mojolearn[torch]==0.8.66 on manylinux_2_35_x86_64 py3.12: pip could not resolve it (exit 1; /Users/andrewhendel/mojolearn-evidence/release/0.8.66/43b5f7c320e0/testpypi-rehearsal/resolve/mojolearn_torch_manylinux_2_35_x86_64_py3.12.log) |
| testpypi | rehearsal/secondary | pip install mojolearn[torch]==0.8.66 on manylinux_2_35_x86_64 py3.13: pip could not resolve it (exit 1; /Users/andrewhendel/mojolearn-evidence/release/0.8.66/43b5f7c320e0/testpypi-rehearsal/resolve/mojolearn_torch_manylinux_2_35_x86_64_py3.13.log) |
| testpypi | rehearsal/secondary | pip install mojolearn[torch]==0.8.66 on manylinux_2_35_x86_64 py3.14: pip could not resolve it (exit 1; /Users/andrewhendel/mojolearn-evidence/release/0.8.66/43b5f7c320e0/testpypi-rehearsal/resolve/mojolearn_torch_manylinux_2_35_x86_64_py3.14.log) |
| testpypi | rehearsal/secondary | pip install mojolearn[verify]==0.8.66 on manylinux_2_35_x86_64 py3.10: pip could not resolve it (exit 1; /Users/andrewhendel/mojolearn-evidence/release/0.8.66/43b5f7c320e0/testpypi-rehearsal/resolve/mojolearn_verify_manylinux_2_35_x86_64_py3.10.log) |
| testpypi | rehearsal/secondary | pip install mojolearn[verify]==0.8.66 on manylinux_2_35_x86_64 py3.11: pip could not resolve it (exit 1; /Users/andrewhendel/mojolearn-evidence/release/0.8.66/43b5f7c320e0/testpypi-rehearsal/resolve/mojolearn_verify_manylinux_2_35_x86_64_py3.11.log) |
| testpypi | rehearsal/secondary | pip install mojolearn[verify]==0.8.66 on manylinux_2_35_x86_64 py3.12: pip could not resolve it (exit 1; /Users/andrewhendel/mojolearn-evidence/release/0.8.66/43b5f7c320e0/testpypi-rehearsal/resolve/mojolearn_verify_manylinux_2_35_x86_64_py3.12.log) |
| testpypi | rehearsal/secondary | pip install mojolearn[verify]==0.8.66 on manylinux_2_35_x86_64 py3.13: pip could not resolve it (exit 1; /Users/andrewhendel/mojolearn-evidence/release/0.8.66/43b5f7c320e0/testpypi-rehearsal/resolve/mojolearn_verify_manylinux_2_35_x86_64_py3.13.log) |
| testpypi | rehearsal/secondary | pip install mojolearn[verify]==0.8.66 on manylinux_2_35_x86_64 py3.14: pip could not resolve it (exit 1; /Users/andrewhendel/mojolearn-evidence/release/0.8.66/43b5f7c320e0/testpypi-rehearsal/resolve/mojolearn_verify_manylinux_2_35_x86_64_py3.14.log) |

## Identity columns

Each column installed the exact wheel files below (sha256 checked before install) and recorded the identity lanes whose code changed since the last good record; the other lanes' rows are reused.

| column | label | machine | lanes run / reused | lane seconds | start (UTC) | end (UTC) | run |
|---|---|---|---|---|---|---|---|
| NVIDIA native | `nvidia-l40s` | NVIDIA GeForce RTX 4090, 24564 MiB, 595.91.07; AMD EPYC 7402 24-Core Processor; 48 cores | 513 / 0 | 7m 03s | 2026-11-08T17:36:14Z | 2026-11-08T17:56:56Z | ok |
| NVIDIA PTX | `nvidia-ptx-l40s-sm80` | NVIDIA GeForce RTX 4090, 24564 MiB, 595.91.07; AMD EPYC 7402 24-Core Processor; 48 cores | 513 / 0 | 9m 28s | 2026-11-08T17:57:07Z | 2026-11-08T18:20:19Z | ok |
| AMD | `amd-mi325x` | GPU[0]		: Card Series: 		AMD Instinct MI300X VF, GPU[0]		: Card SKU: 		M3000108; INTEL(R) XEON(R) PLATINUM 8568Y+; 20 cores | 513 / 0 | 10m 14s | 2026-11-08T17:36:56Z | 2026-11-08T17:39:27Z | ok |
| Apple | `apple-m4` | Apple M4 | 513 / 0 | 27m 19s | 2026-11-08T13:37:46Z | 2026-11-08T13:50:29Z | ok |
| CPU | `cpu` | AMD EPYC 7763 64-Core Processor, AMD EPYC 9V45 96-Core Processor | 510 / 0 | 9m 46s | 2026-11-08T15:55:33Z | 2026-11-08T15:59:49Z | ok |

Recorded from (wheel file and sha256, as each column installed it):

- NVIDIA native: `mojolearn_core-0.8.66-py3-none-manylinux_2_35_x86_64.whl c8775ebcd0cc8095806b33b6cc57667dd310147a701e4629fded608384f6c12c`; `mojolearn_nvidia-0.8.66-py3-none-manylinux_2_35_x86_64.whl 9f18a484053facae329e3f606cef3f5c42966e98d6c1bcbc4850c34e4379e725`
- NVIDIA PTX: `mojolearn_core-0.8.66-py3-none-manylinux_2_35_x86_64.whl c8775ebcd0cc8095806b33b6cc57667dd310147a701e4629fded608384f6c12c`; `mojolearn_nvidia-0.8.66-py3-none-manylinux_2_35_x86_64.whl 9f18a484053facae329e3f606cef3f5c42966e98d6c1bcbc4850c34e4379e725`
- AMD: `mojolearn_amd-0.8.66-py3-none-manylinux_2_35_x86_64.whl 1cd7b953b9afcb5868137a5a23c6aa4f461929f888327164ec2612670b9ca9ed`; `mojolearn_core-0.8.66-py3-none-manylinux_2_35_x86_64.whl c8775ebcd0cc8095806b33b6cc57667dd310147a701e4629fded608384f6c12c`
- Apple: `mojolearn-0.8.66-py3-none-macosx_26_0_arm64.whl 72ae6ae0007c6a0fe2adf59ddd9a8be44ce055f27c795df593e9f9067d524ac1`
- CPU: `mojolearn_core-0.8.66-py3-none-manylinux_2_35_x86_64.whl c8775ebcd0cc8095806b33b6cc57667dd310147a701e4629fded608384f6c12c`

Sources: [`identity/amd-identity-wheels.json`](identity/amd-identity-wheels.json), [`identity/apple-identity-wheels.json`](identity/apple-identity-wheels.json), [`identity/cpu-identity-wheels.json`](identity/cpu-identity-wheels.json), [`identity/nvidia-identity-wheels.json`](identity/nvidia-identity-wheels.json).

## Admission

All columns agree: **no** (1125 part(s) not on every column)

- admitted at commit `9d70418446aa876bfd10ea5e7c19d31199610939`, table sha256 `8efc5b051f9382e2b2b147184af6be1a7d5a671990d635a20ad425cf515d3fe5` ([`identity/admitted.json`](identity/admitted.json))
- shipped table `9d70418446aa876bfd10ea5e7c19d31199610939:python/mojolearn/verify_reference/table.json`, sha256 `8efc5b051f9382e2b2b147184af6be1a7d5a671990d635a20ad425cf515d3fe5`: 16245 parts, 15120 equal on every column, 0 PTX-divergent ([`identity/table-summary.json`](identity/table-summary.json))
- table columns: `amd` 572 lanes, 16245 parts; `apple` 572 lanes, 16245 parts; `cpu` 536 lanes, 15156 parts; `nvidia` 572 lanes, 16245 parts; `ptx` 569 lanes, 16164 parts

## Wheels

| platform | file | sha256 | published |
|---|---|---|---|
| amd | `mojolearn_amd-0.8.66-py3-none-manylinux_2_35_x86_64.whl` | `1cd7b953b9afcb5868137a5a23c6aa4f461929f888327164ec2612670b9ca9ed` | yes |
| linux | `mojolearn_core-0.8.66-py3-none-manylinux_2_35_x86_64.whl` | `c8775ebcd0cc8095806b33b6cc57667dd310147a701e4629fded608384f6c12c` | yes |
| linux-meta | `mojolearn-0.8.66-py3-none-manylinux_2_35_x86_64.whl` | `8bdfdf1afce524cda26d9caa775adf674549fe31a786d719fb404e9a7e9e3d54` | yes |
| macos | `mojolearn-0.8.66-py3-none-macosx_26_0_arm64.whl` | `72ae6ae0007c6a0fe2adf59ddd9a8be44ce055f27c795df593e9f9067d524ac1` | yes |
| nvidia | `mojolearn_nvidia-0.8.66-py3-none-manylinux_2_35_x86_64.whl` | `9f18a484053facae329e3f606cef3f5c42966e98d6c1bcbc4850c34e4379e725` | yes |

Per-platform manifests and file admissions: [`alpha-manifest-amd.json`](alpha-manifest-amd.json), [`alpha-manifest-linux-meta.json`](alpha-manifest-linux-meta.json), [`alpha-manifest-linux.json`](alpha-manifest-linux.json), [`alpha-manifest-macos.json`](alpha-manifest-macos.json), [`alpha-manifest-nvidia.json`](alpha-manifest-nvidia.json), [`file-admission-amd.json`](file-admission-amd.json), [`file-admission-linux-meta.json`](file-admission-linux-meta.json), [`file-admission-linux.json`](file-admission-linux.json), [`file-admission-macos.json`](file-admission-macos.json), [`file-admission-nvidia.json`](file-admission-nvidia.json).

## Device checks

- amd multi-GPU check: **PASS**, 15 of 15 lanes MATCH across device counts [1, 2, 3] ([`gates/amd/shardcheck.json`](gates/amd/shardcheck.json))
- nvidia multi-GPU check: **PASS**, 15 of 15 lanes MATCH across device counts [1, 2, 3] ([`gates/nvidia/shardcheck.json`](gates/nvidia/shardcheck.json))
- every other check's verdict: [`checks/rehearsal.json`](checks/rehearsal.json), [`compile/bindings.json`](compile/bindings.json), [`compile/cross-compile.json`](compile/cross-compile.json), [`gates/amd/box-facts.json`](gates/amd/box-facts.json), [`gates/amd/check-units.json`](gates/amd/check-units.json), [`gates/amd/job-status.json`](gates/amd/job-status.json), [`gates/amd/runtime-report.json`](gates/amd/runtime-report.json), [`gates/apple/adapter-smoke.json`](gates/apple/adapter-smoke.json), [`gates/apple/runtime-report.json`](gates/apple/runtime-report.json), [`gates/cpu/checks.json`](gates/cpu/checks.json), [`gates/nvidia/box-facts.json`](gates/nvidia/box-facts.json), [`gates/nvidia/check-units.json`](gates/nvidia/check-units.json), [`gates/nvidia/job-status.json`](gates/nvidia/job-status.json), [`gates/nvidia/runtime-report.json`](gates/nvidia/runtime-report.json), [`publish/finish-line.json`](publish/finish-line.json), [`publish/published-amd.json`](publish/published-amd.json), [`publish/published-apple.json`](publish/published-apple.json), [`publish/published-nvidia.json`](publish/published-nvidia.json), [`publish/testpypi-rehearsal.json`](publish/testpypi-rehearsal.json)

## Check it on your own machine

    pip install "mojolearn[verify]==0.8.66"
    python -m mojolearn verify

The wheel ships the admitted reference table above; `verify` runs every applicable routine lane on your machine in `identical` mode and checks its output hashes against that table (`--quick` for representative checks in seconds to minutes; see the user guide, Verify on your hardware).

## Where the time went

Seconds each unit recorded for itself; the previous release's for the same unit next to it.

Release steps, longest first (21478 s in all):

| step | seconds | 0.8.65 | change |
|---|---|---|---|
| gate-nvidia | 8091.5 | - | - |
| testpypi-rehearsal | 2586.7 | - | - |
| gate-amd | 2565.7 | - | - |
| cross-compile | 2253.7 | - | - |
| gate-apple | 1740.9 | - | - |
| identity-cpu | 1346.3 | - | - |
| gate-boxes | 1075.9 | - | - |
| macos-build | 308.3 | - | - |
| boxes-down | 262.7 | - | - |
| publish-meta-linux | 198.7 | - | - |
| publish-macos | 179.4 | - | - |
| publish-nvidia | 164.3 | - | - |
| publish-core-linux | 160.2 | - | - |
| publish-amd | 158.2 | - | - |
| identity-admit | 98.7 | - | - |
| reuse-plan | 91.8 | - | - |
| linux-pack | 91.5 | - | - |
| finish-line | 41.5 | - | - |
| linux-builds | 31.6 | - | - |
| published-nvidia | 11.4 | - | - |
| rehearsal | 8.5 | - | - |
| published-amd | 8.2 | - | - |
| linux-wait | 1.6 | - | - |
| identity-at-freeze | 0.9 | - | - |
| freeze-changelog | 0.0 | - | - |
| freeze-commit | 0.0 | - | - |
| freeze-docs-facts | 0.0 | - | - |
| freeze-version | 0.0 | - | - |
| gpu-column-amd | 0.0 | - | - |
| gpu-column-nvidia | 0.0 | - | - |
| gpu-column-nvidia-hopper | 0.0 | - | - |
| linux-assemble | 0.0 | - | - |
| macos-smoke | 0.0 | - | - |
| readiness | 0.0 | - | - |
| record | 0.0 | - | - |

Longest path (each step started when the one before it on the path ended; start = end - seconds), 7h 25m 50s from first start to last end:

- rehearsal: 2026-11-08T13:30:46Z to 2026-11-08T13:30:55Z, 0m 08s
- reuse-plan: 2026-11-08T13:30:55Z to 2026-11-08T13:32:27Z, 1m 32s
- macos-build: 2026-11-08T13:32:26Z to 2026-11-08T13:37:35Z, 5m 08s
- gate-apple: 2026-11-08T13:37:35Z to 2026-11-08T14:06:36Z, 29m 01s
- linux-builds: 2026-11-08T15:15:33Z to 2026-11-08T15:16:05Z, 0m 32s, after 1h 08m 57s idle
- cross-compile: 2026-11-08T15:15:33Z to 2026-11-08T15:53:07Z, 37m 34s
- linux-wait: 2026-11-08T15:53:07Z to 2026-11-08T15:53:09Z, 0m 02s
- linux-pack: 2026-11-08T15:53:08Z to 2026-11-08T15:54:40Z, 1m 32s
- identity-cpu: 2026-11-08T15:54:40Z to 2026-11-08T16:17:07Z, 22m 26s
- gate-boxes: 2026-11-08T17:16:21Z to 2026-11-08T17:34:17Z, 17m 56s, after 59m 14s idle
- gate-nvidia: 2026-11-08T17:34:16Z to 2026-11-08T19:49:08Z, 2h 14m 52s
- boxes-down: 2026-11-08T19:49:08Z to 2026-11-08T19:53:31Z, 4m 23s
- testpypi-rehearsal: 2026-11-08T19:55:53Z to 2026-11-08T20:39:00Z, 43m 07s, after 2m 22s idle
- publish-core-linux: 2026-11-08T20:38:59Z to 2026-11-08T20:41:40Z, 2m 40s
- publish-nvidia: 2026-11-08T20:41:40Z to 2026-11-08T20:44:25Z, 2m 44s
- publish-amd: 2026-11-08T20:44:24Z to 2026-11-08T20:47:03Z, 2m 38s
- publish-meta-linux: 2026-11-08T20:47:03Z to 2026-11-08T20:50:22Z, 3m 19s
- publish-macos: 2026-11-08T20:50:21Z to 2026-11-08T20:53:21Z, 2m 59s
- finish-line: 2026-11-08T20:53:21Z to 2026-11-08T20:54:03Z, 0m 42s
- published-amd: 2026-11-08T20:56:24Z to 2026-11-08T20:56:33Z, 0m 08s, after 2m 22s idle
- published-nvidia: 2026-11-08T20:56:25Z to 2026-11-08T20:56:37Z, 0m 11s

Slowest build groups (compile cache provenance; the longest build of each binding):

| where | binding | seconds | 0.8.65 | change |
|---|---|---|---|---|
| hip-gfx942 | `_mojolearn_native_sequence.so` | 732.7 | 722.3 | +10.4 (+1%) |
| cuda-sm_80 | `_mojolearn_native_classical_trees_host.so` | 652.0 | new | - |
| hip-gfx942 | `_mojolearn_native_classical_family_host.so` | 633.3 | new | - |
| hip-gfx942 | `_mojolearn_native_trees.so` | 614.8 | 572.5 | +42.3 (+7%) |
| cuda-sm_80 | `_mojolearn_native_trees.so` | 608.8 | 580.3 | +28.5 (+5%) |
| cuda-sm_89 | `_mojolearn_native_trees.so` | 606.1 | 655.6 | -49.5 (-8%) |
| cuda-sm_80 | `_mojolearn_native_sequence.so` | 599.3 | 713.8 | -114.5 (-16%) |
| hip-gfx942 | `_mojolearn_native_neural_host.so` | 588.3 | 459.1 | +129.2 (+28%) |

Linux leg wall seconds: cuda-sm_80 5217 (0.8.65 2712); cuda-sm_89 5253 (0.8.65 3263); hip-gfx942 5513 (0.8.65 3892)

Identity columns, total lane seconds and the 5 slowest lanes:
