# Six-lane Oct 6 campaign measurements (moved out of the tree)

The 979 files that were here (`20261006/`: the published board, receipts, publication inputs, comparisons and
audits of the 2026-10-06 six-lane full A/B campaign, 305,012,956 bytes) left the repository on 2026-11-25. They are
kept, not deleted:

- local copy: `~/mojolearn-evidence/repo-data/experiments/six_lane_integration/measurements/`
  (`$MOJOLEARN_EVIDENCE_ROOT/repo-data/...` when that variable is set);
- R2: key `repo-data/experiments/six_lane_integration/measurements.tar` (one deterministic tar), pinned in
  `bench/results/dataset_store/manifest.tsv`;
- per-file pins (path, bytes, sha256), tracked here in git:
  `bench/results/dataset_store/repo-data/experiments@six_lane_integration@measurements.tsv`.

Every path that cites `experiments/six_lane_integration/measurements/<file>` names the same relative path under
the local copy. Tools resolve it with `tools/repo_data.py` (`resolve()` in code); by hand:

```sh
python3 tools/repo_data.py path experiments/six_lane_integration/measurements/20261006/BOARD.md
python3 tools/repo_data.py fetch experiments/six_lane_integration/measurements   # when the local copy is absent
```

The last commit that held the files in the tree is the parent of the commit that added this README.
