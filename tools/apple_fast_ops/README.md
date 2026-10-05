# Apple FAST manager ops (laptop -> M2 builds, M3 queue)

Copies of the scripts the Apple FAST manager runs. The live copies are in ~/mojolearn-evidence/ (laptop), ~/m2-arms/ (M2) and ~/mq/ (M3).

- `apple_watch.sh` (laptop): watcher tick. M3 queue position/runner/disk, M2 build state, alerts, M2/M3 bare main <-> GitHub sync, opp_to_end.
- `opp_to_end.py` (M3 ~/mq): keeps opponent-only `opp*` jobs behind all FAST work. It stops a running opp job when FAST work is waiting and requeues its unfinished races. `python3 ~/mq/opp_to_end.py [new_lines.json]` appends jobs safely.
- `ab_extract.py` (M3 ~/mq): `python3 ~/mq/ab_extract.py '^rab3-'` prints one line per A/B: A ms, B ms, delta, both qualities, digest same/diff.
- `m2_build_ab.py` + `m2_bq.sh` (M2 ~/m2-arms): serial build queue. Drop `[{"sha","binding","A","B"?,"mode"?}]` JSON into ~/m2-arms/bq/NN-name.json; logs go to bq/done/. Binding `core` = bindings/build.sh.
- `ident_to_afb.py`: IDENTICAL sweep race.txt -> AFB lines (see tools/af_board_ident_update.py for the board fold-in).

Verdict rule (CLAUDE.md): FAST flips on when it's faster and quality doesn't go down (no material drop vs FAST main, >= best opponent). Bits are free.
Board: tools/af_board_apply.py (quality-gated), tools/af_board_quality_audit.py.

## Source of truth (2026-10-05)
These files in git ARE the source of truth. ~/mojolearn-evidence is for disposable outputs only: on 2026-10-05 ~23:46 every top-level file there was deleted
(scripts and handoffs lost; no backup). Restore a live copy with: cp tools/apple_fast_ops/<file> ~/mojolearn-evidence/ (laptop),
scp it to M3 ~/mq/ (opp_to_end.py, ab_extract.py) or M2 ~/m2-arms/ (m2_build_ab.py, m2_bq.sh). Edit here first, then copy out.
