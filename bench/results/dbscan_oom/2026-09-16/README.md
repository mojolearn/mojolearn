# The DBSCAN hipErrorOutOfMemory on a "192 GB card" (lane/amd-dbscan-oom)

Three boxes, 2026-09-16, all terminated and VERIFIED gone (HTTP 404). The
reading is `docs/lanes/LANE_STATUS_lane-amd-dbscan-oom.md`; these are the
numbers it stands on, and every one of them can be re-derived from the JSON.

| file | box | what it shows |
|---|---|---|
| `mi300x-runpod.card-was-full.json` | RunPod MI300X, pod o3paueazuvo87t | `used_boot` 196231.6 MiB of 196592.0, read BEFORE `import mojolearn`, so **360.4 MiB free**; and the first fit's `used_after` LOWER than its `used_before` by 12031.2 MiB, which a fit of ours cannot do |
| `mi300x-runpod.card-was-full.txt` | the same pod | the hand transcript: eight cards in `/sys/class/drm`, the owned one is `card33`, `rocm-smi --showpids` naming a kfd pid outside this container holding 178.6 GiB |
| `mi300x-runpod.identity-six-lanes.json` | the same pod | the original column's own six lanes reproduced: `moved=0`, and `dbscan/base` model plus `dbscan-brute-l1/negative` batch `REFUSED` with `hipErrorOutOfMemory` |
| `mi325x-do.repeat-base.json` | DigitalOcean MI325X, droplet 601157435 | 24 fits of ONE shape: three warm-up fits of +179 MiB, then **+2 MiB every second fit exactly**, so +1.000 MiB per fit, with no plateau |
| `mi325x-do.arms-base.json` | the same droplet | the same +1.000 MiB per fit with `prediction_data` on and off, brute and rbc, weighted and not |
| `mi325x-do.seq.json` | the same droplet | the 54 fits the original column ran, `refused=0` |
| `rtx4090-runpod.*.json` | RunPod RTX 4090, pod iphu8b1kxgrg4d | 24564 MiB total, **1.0 MiB used before `import mojolearn`**, self check moved it +397.0 MiB, and then ONE distinct `used_after` value of 398.0 MiB across 55, 33 and 25 fits with `refused=0` |

Read them with `tools/dbscan_memory_probe.py`'s own vocabulary: `used_boot` is
the device before this library was imported, `used_before` and `used_after`
bracket one fit, and `card_already_in_use` is set when less than half the
device was free at the start.

The flat NVIDIA line is not an unproved flat line. `self_check_moved_mib` is
397.0 in every one of those files, and no arm is allowed to run until the
reader has been seen to move.
