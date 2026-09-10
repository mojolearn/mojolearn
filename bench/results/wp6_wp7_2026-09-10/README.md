# WP6 / WP7 qualification — 2026-09-10

Non-tree host-copy and duplicate-upload changes, measured in IDENTICAL.
Classical output compatibility also ran in DETERMINISTIC and FAST. Trees,
GBDT, isolation forest and WP8 are outside this change.

## Results

- 21 complete output captures contain **956 recorded arrays**. All match
  before/after, as do 36 primary stage traces and two additional cards.
- All IDENTICAL estimator suites pass. The integrated current-main source
  passes 88 host tests and reproduces all 13 controlled IDENTICAL captures.
- The shared copy helper passed 54,944 bit-cell checks on the Apple M4 host.
- RTX 4090 kNN classification at 400k rows / 4k queries / 32 features,
  k=10/15, uniform/distance weights: **2.2–4.8% lower median native request
  time**. Every one of the 20 timed pairs favored the candidate; all 48
  calls (including warmups) passed complete selected-output comparisons.
- GP native mean prediction from a synthetic preconstructed 20k-row, one-feature
  model, four queries, `return_std=False`: **4.073 s → 1.514 s median**
  (62.8% less time). The factor buffer is a 1.6 GB identity matrix. The qualified repeat took 48 s,
  used three warmup pairs plus five timed pairs, and matched every output.
  Minimum times are 3.924 s → 1.423 s; baseline spread is 5.1%.

These are complete native calls, including their host/device transfers.
The timing boundary excludes Python estimator preparation and model fitting.
Neither benchmark measures an opponent, and the kNN result is classification,
not the bare-search opponent-table workload. No opponent-cache row changed.

`summary.json` contains derived numbers; `knn-large.json` and
`gp-repeat-qualified.json` retain every sample. The earlier `gp-large.json`
window failed the 20% baseline-spread gate (29.8%) and is not a qualified
speed claim. Its samples remain intact. The GP repeat ran the same two
admitted binaries on a different RTX 4090 allocation with a fresh paired
baseline; absolute times from different allocations are not combined.

## Scope and existing failures

The DETERMINISTIC and FAST kernel-method sabotage suites reach their card
checks, then fail because `POLY_VIA_POW` changes no fixture bit. That failure
already occurs on the unchanged baseline. Both versions emit the same stage
fingerprints and the same failure. The two retained failure markers are
included in the 61 file comparisons; they are **not passing-test counts**.
No numerical tolerance or product arithmetic was weakened to pass them.

The unchanged NVIDIA baseline hangs when a new trainer context is created
after a stateless Byte-LM call. A resident-only process passes two training
steps and evaluation; that sequence and separate one-call lifetime gates
match before/after. Context recreation remains an open issue. Large-model
training speed and Apple/AMD estimator execution were not measured here.

One early ARIMA trace inherited records from the original nested classical
capture runner. The exact ARIMA prefix already matched the entire candidate
trace. The original combined file and its classical suffix remain in the
archive; `trace-repair.json` records the split. Dedicated classical captures
are compared independently. Initial fixture, oracle, build-wrapper and
runtime-staging failures are retained as diagnostics, not counted as passes.

## Evidence and reproduction

- `receipt.json`: distinct before/after native binary hashes and source pins.
- `comparisons.json`: all 61 byte-comparison outcomes, including their kinds.
- `capture-inventory.json`: frame counts and SHA-256 hashes for output captures.
- `integration-comparisons.json`: controlled candidate versus integrated main.
- `raw/`: public text traces, final check/surface logs and recorded exit states.
  Published logs have trailing whitespace normalized, recorded in
  `log-whitespace-normalization.json`; original logs remain in the external
  archive. Trace/card bytes and array-capture hashes are unchanged.
- `artifacts.json`: SHA-256 hashes and absolute locations for the complete raw
  capture archive and the four native benchmark binaries, retained outside the
  repository at `/Users/andrewhendel/CascadeProjects/mojolearn-evidence/wp6_wp7_2026-09-10/`.
  The archive includes raw array captures, synthetic corpus fixtures, build logs,
  failed runs, staged drivers and the original source patch. Repository policy
  forbids tarballs and fixture dumps under `bench/results/`.
  `/tmp/wp67-rental2/integration-source.tar` retains the integration source archive.
- `repeat-runtime-closure.json` and `repeat-runtime-packages.json`: the ELF
  closure staged from the exact lockfile-pinned MAX/Mojo packages for the
  clean-node GP repeat. The first clean-node attempt lacked those libraries;
  `gp-repeat-missing-runtime.log` preserves that setup failure.
- `cleanup.json`: scoped rental deletions and provider absence verification.

The controlled baseline is d330a49d. The original candidate snapshot is
4b5063e6, supplemented by the retained kNN source from 5528df19 and the test
repairs. Integration is 53fb90b2, rebased onto the current main changes;
subsequent commits in this WP6/WP7 lane change only harness/evidence
documentation. Later unrelated main changes have their own qualification. The archive
records the actual staged sources and scripts so rewritten local commit IDs
do not become the sole provenance.

`tools/wp67_remote_qualify.sh` stages independent drivers, saves candidate
sources, reverses the supplied `/root/wp67.patch`, and compares both arms.
A clean rental also needs the archived synthetic corpus extracted at the
source root. Independent estimator checks run two at a time; repeated runs
of a given check must not overlap because their control-card paths are fixed.
Preflight binaries are reused only after matching every Mojo source byte and
checking that both trees use the same compiler environment. The large timing
programs run after compilation and correctness checks finish.
