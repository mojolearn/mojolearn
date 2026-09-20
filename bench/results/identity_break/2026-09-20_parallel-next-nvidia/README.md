# NVIDIA parallel reference completion

NVIDIA RTX 4090 (sm_89), seven parallel-driver lanes × nine fixtures, full parts, one repeat. All 63 cells completed; all 234 numeric reference comparisons agree exactly, supplying the 126 planned NVIDIA values. One repeat does not establish repeated-run stability. One physical GPU was used; this is not physical multi-GPU qualification.

Python/harness source: `14944378ff9d7a733cd39779c6bdb24b7d041e76`. Native binaries come from the published 0.8.9 wheel built at `819a47ae48166e91951f54f54e64ee173658e32a`; its SHA256 is in `provenance.json`.

The initial 900-second process guard stopped at 49 completed cells. The identical protocol resumed, paused at 50 completed cells to qualify the exact 0.8.10 release wheel, then resumed to completion under an 1,800-second guard. Interrupted cells were rerun; completed checkpoints were retained. The two-CPU and 12 GiB RSS limits, fixtures and part settings remained unchanged.

The lease watchdog was renewed explicitly. After evidence was fetched, the rental was deleted and verified absent (HTTP 404); `teardown.log` retains that receipt. Full runtime logs are retained locally in `mojolearn-evidence/parallel-next/nvidia149`.
