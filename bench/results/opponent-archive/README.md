# Opponent archive (append-only)

Every opponent measurement the bench board has made, per box. Opponents are measured once and reused
(`tools/bench_board.py --opponent-store`); our arms are re-measured on every new wheel. Nothing here is overwritten:
a new board run appends, and this archive is refreshed by adding records, never by replacing them.

| box | `opponent-store.jsonl.gz` (store records) | `opponent-cells-all.jsonl.gz` (every opponent cell) |
|---|---:|---:|
| nvidia-l40s | 868 | 2,868 |
| amd-mi325x | 748 | 2,441 |
| m3-ultra | 688 | 2,227 |
| m2-pro | 618 | 1,937 |

- `opponent-store.jsonl.gz`: the union of every board root's `opponent-store.jsonl` on that box (deduplicated by
  record). It seeds a new board root, so a rerun on a new wheel re-times only ours. Its `key` holds the box hash,
  OS, library version, data, parameter and settings hashes; a record is reused only for an exact key match.
- `opponent-cells-all.jsonl.gz`: every non-ours cell (fit and inference) from the stores and from every board.json
  on that box (board roots and the Sep 29 full boards), with `_source` naming the file it came from.

Raw copies of each store and board.json: `~/mojolearn-evidence/opponent-archive/<box>/<root>/` on the
orchestrating laptop.
