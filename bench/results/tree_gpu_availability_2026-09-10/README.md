# Dedicated GPU availability — 2026-09-10

The user requested AMD/NVIDIA evidence rather than performance decisions from
the local MacBook. Read-only account discovery found:

- `sh tools/trees_leg.sh pods`: one running pod, named
  `samba-train-w2048_s16_seed0-20260909_163920`. This is the protected training
  workload. No tree jobs, environment changes or SSH reconfiguration were run.
- DigitalOcean `GET /v2/droplets?per_page=200`, authenticated using the existing
  task credential without exposing it: zero droplets, no next page.
- Local repository connection state was historical, including AMD droplets
  recorded as destroyed. No stale IP was probed, and no usable SSH target
  configuration was found.

An idle endpoint or a separate temporary-rental spending limit was requested
from the user. No new resource was created. Preparing and checking local source
continues, but no AMD/NVIDIA timing or cross-device validation has run for this
slice. MacBook timing remains exploratory, not a basis for promotion.
