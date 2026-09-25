# Hot Aisle as a release AMD provider (2026-09-25)

The release can build its gfx942 bindings and run its AMD column on Hot Aisle
when DigitalOcean is unavailable. Branch `lane/release-hotaisle`.

## What was added

- `tools/hotaisle_vm_lib.sh`: one guarded Hot Aisle MI300X VM for the release.
  The slots are the ones `tools/hotaisle_leg.sh` uses. The whole horizon is
  priced live, then checked against a dollar cap and against the prepaid
  balance plus $5. A Mac dead-man is armed before the create, and the
  description is PATCHed. An on-box watchdog is verified from two ssh
  sessions. The teardown sends DELETE ?force=true, then waits for GET 404 or
  for the VM to be absent from the listing.
- `tools/release_wheel_smoke.sh --vendor hip --provider hotaisle`: the AMD
  column. `--provider auto` walks runpod, then hotaisle, then do. Hot Aisle
  takes the 1x VM, or the 2x VM pinned to GPU 0
  (`ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0`) when no 1x VM is in stock.
- `tools/hotaisle_release_leg.sh`: the AMD build leg. It runs
  `tools/do_release061_leg.sh`'s build on a 1x MI300X: the same archive, the
  same host preparation, the same pinned Ubuntu 22.04 container,
  `release061_remote_build.sh`, the same read-backs and provenance. It writes
  the same output tree.
- `tools/release.py`: the AMD build leg goes to Hot Aisle when DigitalOcean has
  a GPU droplet live or no token. `--amd-build-provider` or
  `MOJOLEARN_AMD_PROVIDER` pins the provider. `--amd-provider` also accepts
  hotaisle. linux-wait walks a live-droplet refusal to Hot Aisle.
- Tests (no rentals): `tools/tests/test_hotaisle_release_shim.py` (16),
  `tools/test_release.py` (AMD route, 5 new), `tools/test_release_wheel_smoke.py`
  (Hot Aisle refusals and the auto walk to DigitalOcean).

## The rehearsal: the AMD column of the PUBLISHED 0.8.18 wheel on Hot Aisle

`column-0818/`. The wheel was `mojolearn-0.8.18-py3-none-manylinux_2_35_x86_64.whl`,
sha256 `c160fb6d...3836`, the same digest PyPI serves. The selection was 0.8.18's
`selection-hip.json`, with 209 lanes, the base, denormal and odd fixtures, and
one fit per cell.

| | |
|---|---|
| box | Hot Aisle 2x MI300X VM, `enc1-gpuvm005`, 26 cores. No 1x VM was in stock. The column ran on GPU 0 only. |
| host | Ubuntu 24.04, kernel 6.8.0-124, Python 3.12.3. The wheel ran natively as root, the same way as on the DigitalOcean droplet. |
| install, self-test | install_exit=0, `vendor hip`, selftest_exit=0 |
| cells | 627 cells ran. Against the CPU column of f2293183c: IDENTICAL=618, DIVERGENT=0 |
| same cells, other GPU | Against 0.8.18's recorded DigitalOcean MI325X column: IDENTICAL=627, DIVERGENT=0 |
| time | create 11:07:59Z, verified gone 11:13:26Z (5 min 27 s; the column itself took 4.5 min) |
| cost | $5.98: balance 4882 -> 4284 cents. The 2x offering bills a 60-minute minimum. |
| teardown | DELETE 204. GET 404 1 s later. The dead-man was cancelled and the slot released. |

The CPU diff reads `ONE-COLUMN=27`. Those are the cells the CPU column holds and
the AMD selection leaves out. 0.8.18's own DigitalOcean column shows the same
count. The full `column-hip.json` (sha256 `88088a94...86a5`) and both full
diffs are outside the repository, under
`~/mojolearn-evidence/release-hotaisle/2026-09-25/column-0818/`.

## The build leg: not run for real

Neither probe found a 1x MI300X in stock. The 2x VM was the only offering, and
by the end of the column run nothing was offered. The build leg takes only the
1x VM, because `tools/amd_serial_guard.py` requires exactly one visible render
GPU. So no build ran and no binding digests were compared. The leg is
shim-tested and its dry run is GREEN. It is now a step of the release
rehearsal (`amd-hotaisle-leg-dry-run`).

Measured lease basis: 0.8.18's DigitalOcean leg took 11 minutes from create to
verified destroy, and the build itself took 469 s. The Hot Aisle leg's default
lease is 60 minutes and its horizon cap is $10. At $2.99/h the leg costs at most
$3.99.

To run it when a 1x VM is in stock, from a clean checkout at f2293183c (with
the two new scripts copied in):
`bash tools/hotaisle_release_leg.sh f2293183c729059331cb31afcb8319f528143e55 --rent`.
Then compare the sha256 of every `.so` under `release-build/build/sets/hip/gfx942/`
with the 0.8.18 DigitalOcean leg's tree and with the published wheel's
`mojolearn/hip/gfx942/` members. Every digest must be equal.
