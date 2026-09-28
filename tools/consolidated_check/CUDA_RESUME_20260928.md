# Prepared CUDA continuation

This recipe allocates nothing and submits no remote job. Run it only on an
already provided Linux CUDA host after that host's usage is authorized. It does
not implement a dollar budget or estimate completion cost.

The pinned manifest covers all 445 structurally comparable lanes exactly once:

| Source commit | Lanes | Fixtures |
|---|---:|---|
| `308878e80679` | 337 unaffected lanes | base |
| `9d64cb98b284` | 97 repaired/prep/property lanes | base |
| `9d64cb98b284` | 4 GLM + 2 SHAP lanes | all nine |
| `ad1680bbeaaf` | resample-bca, resample-unpaired, x-decomp-nmf | all nine |
| `64f7d941396d` | sequence-layernorm | all nine |
| `2c81a1293390` | x-metrics-regression | all nine |

The five corrected lanes are excluded from the older 97-lane group.
LayerNorm uses its fitted feature width, including the 17-column odd fixture,
with the revised batch protocol. Regression metrics use the corrected undefined
variance-weighted score semantics (`metrics-undefined-weighted-score-1`).
No old failed probe is deliberately repeated. The base fixture run is a
cross-vendor diagnostic; it does not establish all-nine-fixture reference
coverage. The first group also runs the consolidated radix public check.

Review without creating directories, accessing a GPU, or running checks:

```sh
python3 tools/consolidated_check/cuda_resume.py \
  --repo /path/to/mojolearn --workspace /path/to/cuda-continuation --gpu-arch sm_90
```

Replace `sm_90` with the provided GPU's architecture. The existing repository
must already contain the five full commit objects listed in the JSON manifest.
After reviewing the plan and obtaining any required host-usage approval, append
`--execute` to that same command. The script checks the hardware architecture,
creates detached worktrees, installs each pinned Pixi environment, and runs the
seven groups sequentially with a 120-second limit per GPU/CPU arm. It never
checks out or edits the supplied clone's active branch. Use a dedicated workspace
and do not run other GPU jobs on that device concurrently.

Each commit gets its own source directory and each group its own evidence
directory. A process lock prevents a second invocation from sharing the workspace.
Native libraries are copied through the existing source-closure-validated
`steward_build` store, scoped to CPU/GPU architecture. Missing or stale objects
are rebuilt by the ordinary consolidated build check. Pixi's package cache and
compiler caches remain available, but no Python package directory or native
output is symlinked between worktrees. Reuse this workspace only on the same
host/toolchain/GPU target; do not copy its native store onto another machine.

Rerunning the identical command uses the consolidated driver's existing resume
checks: source commit, native hashes, architecture, lane selection, fixture
selection and settings must match. Existing failed rows remain failures; this
script does not delete evidence or silently retry failures under new sources.
Each group continues independently after a check failure, and the final exit
status fails if any group failed. Build/setup failures abort execution.

Evidence lives under `WORKSPACE/evidence/COMMIT-GROUP/clean_0of1`. Supply all seven
record directories as CUDA overlays to `compare.py`, alongside matching Apple
and AMD records. The comparator checks matching per-lane commits and protocols;
this script does not admit references or publish a package.
