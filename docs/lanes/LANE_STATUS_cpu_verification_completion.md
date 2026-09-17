# CPU verification completion — active checkpoint

Updated 2026-09-17. User asks to finish coverage of every exposed algorithm
and expose honest verification in the installed wheel. Standing workflow:
**use a separate worktree, commit coherent verified batches, push and merge
them into main**. This preference persists for future work.

Worktree: `~/mojolearn-wt/cpu-verification-completion`, branch
`fix/cpu-verification-completion`. External evidence:
`~/mojolearn-evidence/cpu-verification-completion/`.
No PyPI upload or final release qualification is claimed.

## Current counts

Baseline: 128 CPU lanes available, 51 withheld, 50 parallel exclusions.
Now: **162 available, 17 withheld, 50 excluded**, out of 229 harness lanes.
The appendix has 246 entries; source API entries are a different inventory.

All **176 numerical non-parallel lanes** now have historical native training
controls on all nine fixtures. This excludes the Python BPE/fold functions
and kmeans-cosine's expected-refusal contract. It does not claim every optional
property, one-fault localization, current-wheel sabotage or GPU qualification.

Remaining CPU holds:

- Five ordinary neural lanes: mamba3, transformer, transformer-window, samba,
  samba-untied-dropout-accum. Qualifying current references are CPU-only;
  mamba3, samba and samba-untied-dropout-accum also have missing parts. Preserve
  the holds until complete independent device-class references are recorded.
- Twelve vendor-held lanes: gp-optimize, gp-optimize-restarts, svc-poly,
  gbdt-query-rmse, gmm-random-init-sample, gmm-sample, gp-normalize-y,
  gp-sample-y, gp-sample-y-normalize, gpc, gpc-multiclass, ivf-extend.
  CPU clean/native-control records pass. NVIDIA/AMD evidence remains required.

Public CPU training outside verification still refuses. Physical device-use
attestation, multi-GPU pairs, other interpreters, older macOS and full frozen
release qualification remain owed. Never flip release_qualified to hide gaps.

## Installed artifact evidence

All 32 Mac host families were freshly built from 7f5b786ae with Mojo 1.0.0,
then runtime-staged and checked for every manifest export. Build receipts and
warnings are retained in probe fresh-host-build. A missing-NumPy run is
retained; the verifier now explains the optional dependency rather than
printing a traceback.

The initial 150-lane installed replay completed: 121 passed, 29 OWED, zero
DIVERGENT/REFUSED. Full reports reconstruct byte-for-byte from probe
fresh-cpu-wheel-initial. Strict source admission repaired 249 numeric gaps:
all equal the independent frozen Mac wheel, no existing numeric reference
changed. Five new complete clean source records supplement older records.

Installed candidate a44a6e2f9, same exact 32 native bindings, all nine fixtures
twice with default core and step/full checks:

- Twelve low-bit lanes: 396 IDENTICAL, 144 N/A.
- Twenty-nine repaired lanes: 954 IDENTICAL, 351 N/A.

Both groups have zero DIVERGENT/OWED/REFUSED, pass the verifier self-test,
execute outside the checkout from site-packages without path overrides, and
are included in default CPU selection. Probe installed-162 contains full
compressed reports, scripts and receipts. This proves the 41 changed lanes
on one updated wheel; a single all-162 final-artifact replay is not claimed.
Binaries are archived externally under wheel-artifacts/<sha256>/.

## Native-control audit and retained failures

The 99-lane Linux audit at 763565275 built all 32 production and 32 sabotage
families, used two single-thread workers, all nine fixtures, two repeats and
saved CTR assets. Every clean property is STABLE/N/A. Original batch: 90 pass.
Corrected exit classification: 93 pass. Both byte-LM inference lanes and Adam
clip already changed training bytes on all nine fixtures; repeated explicit
RLPAIR_MOVED/BATCH_MOVED properly explains their negative-arm exit 1.
Original and reevaluated results remain separate. The validator still rejects
refusals, exceptions, repeat instability, missing samples and unexplained exits.

Five real numerical gaps were fixed: DBSCAN's core-threshold fault missed
all-noise fixtures; it now increments the first actual native label, including
noise. MinMaxScaler's offset sign fault missed zero minima; it now corrupts
that fitted offset too. Only compile-time sabotage branches change.
The frozen-source repair pair at 544605965 passes all 45 training controls;
all 180 clean core parts equal the installed-wheel references. An earlier
pair rejected for mismatched commit witnesses is preserved outside admission.
Kmeans-cosine's unchanged expected refusal is retained, not called a numerical
native fault. Combined native faults do not localize individual operations.

Records: identity_break/2026-09-17_cpu-native-nine and
identity_break/2026-09-17_cpu-native-control-repairs. The earlier classical,
neural and complete-dependency batches retain all failures, including omitted
transitive dependencies and inert faults. Concurrent reference-regen and
sabotage-sweep branches are integrated; our measured ordered-sum zero fault
and explicit numerical-error CI gate remain active.

## Resources and next steps

**No cloud rental remains active from this lane.** All four pods verified
deleted: 0krigxfoki5p54, 4018mzhxmp3prb, 17o53wuev9wtbr, l521898scta1lj.
Last audit: 681s build, 1246s run, 2070s billed, $0.0345; DELETE 204,
GET 404 and absence from fleet recorded in teardown.txt. Failed 40GB create
request allocated no pod. Do not touch other owners' pods or worktrees.

Resource cap: two cloud vCPUs plus one local single-thread worker, max three
cores total. Local numerical/compiler work uses nice 19 and mac_slot.py;
all numerical thread variables one. Every rental needs watchdog plus external
dead-man and verified teardown; preserve failures. No new unrelated lane.

Next: record the missing current neural parts and independent GPU columns,
then continue vendor/device and optional-property qualification. Keep all
holds until independently measured. Never silently replace references to
make a disagreement disappear. This completed batch is committed and pushed
from the isolated worktree, with main integration as the final workflow step.


## Validation checkpoint

372 focused tests pass, including live base-fixture parity between the source
harness and verifier across all public CPU lanes. Updated evidence reports
176 non-parallel numerical lanes with all-nine historical native training
controls. The strict five-neural admission audit retains missing parts and
single-class references; no new lane is promoted by this audit. Evidence and
counts are being checked in the refreshed installed wheel before main merge.


## Final installed evidence check

The refreshed f41682fb2 wheel passes the five repaired-control lanes: 180
IDENTICAL, 45 N/A, zero DIVERGENT/OWED/REFUSED over nine fixtures twice.
Installed coverage shows 162 available / 17 withheld / 50 excluded and all
176 numerical non-parallel lanes with historical all-nine native training
controls. All release_qualified flags remain false. Bytewise comparison with
the passing 41-lane wheel shows only evidence metadata, COMMIT and wheel
RECORD changed; every numerical code/reference/model/native byte is identical.
Probe native-nine-installed retains full reports and receipts. No local job
or cloud rental remains active from this lane. No PyPI publication occurred.
