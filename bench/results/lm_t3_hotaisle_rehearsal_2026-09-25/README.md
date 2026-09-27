# T3 on Hot Aisle: merge, driver dry run, rehearsal not run (2026-09-25)

Branch `lane/hotaisle-segments-2` (from origin/main 75d3c3ad4, merged to main
as 5a9257d10 by the coordinator).

## The merge

`origin/lane/hotaisle-segments` merged onto main (8571c2cbe). The only
conflicts were in `tools/lm_run_driver.py`:

- module docstring: the branch's Hot Aisle provider text, with main's
  sentence on the driver's R2 use kept, plus one line on resumed segments.
- `render`: main's resume path and `_chain_key` kept, with the branch's
  `suffix` parameter added. A second rendering (the Hot Aisle body with
  `hotaisle_devices`) resumes the same way as the first: its own checkpoint,
  the remaining steps, the partial chain as `--expect-key`. It does not log
  the resume again and does not upload the partial chain again (952f08ed4).
- `_land_resumed` (main) and the branch's per-provider `_vendor_class`, both
  kept.

`tools/hotaisle_leg.sh` now sources `tools/hotaisle_vm_lib.sh` (also copied
into its immutable snapshot) for the settings every Hot Aisle runner shares:
API, team, key file, ssh key and fingerprint, slot prefix, create lock, slot
count, balance floor. It also uses the lib's `ha_dollars` and a new
`ha_lease_cents`, which the lib's own `ha_rent` now uses too. The leg's own
guard functions stay as they were.

New test: `tools/tests/test_lm_run_driver_resume.py::ResumeOnHotaisle`. A/4
stops at 2600 with provider hotaisle. Both renderings get `--from
ckpt_00002600.blm --steps 1300` and the own-checkpoint sha/key, `--expect-key
.../A/4/partial.chain.jsonl`, and no `--replay`. The devices are `0` and
`0,1`. There is one presign-put, the leg is `hotaisle_leg.sh ... --one-body
--segment-lease 2130 --dollar-cap 140`, and there is one RESUMING log line.

Tests: `pytest tools/tests -q -k "driver or lm_run or hotaisle or controls or
file_evidence or release"`: 134 passed, 113 deselected.

## The rehearsal: NOT RUN, nothing rented

The body was rendered with `lm_segment_leg.py controls --arm amd --devices 0,1
--steps 3 --controls "" --wheel 0.8.18`. It starts from
`runs/t3/2026-09-22/A/1/ckpt_00000100.blm` and uses the chain lines from
`witness/A-1.chain.partial.jsonl`: steps 100 to 103, state digests
abc8b816.., a9421f91.., fcdb48b8.. for 101 to 103. It uses GETs only. The
dry run of `hotaisle_leg.sh amd --skip-gates --spec 2gpu --one-body
--segment-lease 70 --dollar-cap 9` was GREEN. `--rent` then took slot 1 and
waited 62 minutes (08:44 to 09:45 ET). Every minute the 2gpu offering read
`none, quantity 0`, and the team's available list was empty for every spec.
The coordinator stopped the wait. A TERM ran the leg's trap, which released
the slot. No dead-man was armed and no create was sent. `hotaisle_leg.sh
status` afterward showed 0 VMs. The balance was $42.84 before and after.
Files are in `leg/` (leg.txt, teardown.txt, leg_log_excerpt.txt). The body
copy with presigned URLs was removed.

There is no seconds-a-step figure on two MI300X, and no gpu.txt.

## The driver dry run (the command the person will run)

`driver_dry_run.py` (output: `driver_dry_run.out.txt`) makes a scratch copy
of `t3_spec.json` with `"provider": "hotaisle"` on A/4. Its scratch ledger
holds the real A/1 to A/3 entries. The A/4 partial entry is written by the
driver's own `record_partial` from copies of `t3/legs/A-4/hang-attempt-1`:
resume_from 2600, checkpoints 2500 `374ec7a1..` and 2600 `756b0c9f..`,
chain 2401 to 2699, arrival PASS. In the scratch spec only,
`resume_check_r2` is false, so no R2 read happens. `_start` then runs with
`subprocess.run` mocked:

    MOJOLEARN_HOTAISLE_SPEC=2gpu MOJOLEARN_HOTAISLE_GPU_ONLY=1 MOJOLEARN_GPU_ARCHS=gfx942 \
    MOJOLEARN_HOTAISLE_LANE=lm-A-4 MOJOLEARN_HOTAISLE_STOCK_WAIT_MINUTES=5 MOJOLEARN_HOTAISLE_SLOT_WAIT_MINUTES=5 \
    MOJOLEARN_GEMM_LEG_EXTRA=<out>/bodies/A-4-hotaisle.sh MOJOLEARN_GEMM_LEG_OUT=<out>/legs/A-4/leg-hotaisle-1 \
    bash tools/hotaisle_leg.sh amd --rent --skip-gates --spec 2gpu --one-body --segment-lease 2130 --dollar-cap 140

The body (`A-4-hotaisle.sh`) is rendered with `--from ckpt_00002600.blm
--steps 1300 --devices 0,1 --from-sha 756b0c9f0fbc0a8c... --from-key
runs/t3/2026-09-22/A/4/ckpt_00002600.blm --expect-key
runs/t3/2026-09-22/A/4/partial.chain.jsonl --boundary 3900`, with no arrival
replay.

## What will refuse on the real start

- The lease is priced for the whole segment (2100 + 30 minutes), not for the
  1300 remaining steps. The leg refuses a lease above `--dollar-cap 140`. At
  any 2x MI300X price above about $3.94/h, 2130 minutes is over $140, so the
  leg refuses it ("above the --dollar-cap"). The driver then skips Hot Aisle
  instead of halting. To avoid that, set `hotaisle_dollar_cap`, or lower A/4's
  `lease_minutes` for the resumed start.
- The balance must hold the whole lease plus $5. At $42.84, a lease of more
  than about $37.84 is refused on the balance, which covers roughly 5 to 10
  hours of the 2x VM, not 35.
