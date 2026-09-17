# CPU verification completion — active work

Started 2026-09-17 from main `32acd33d8`, following the user's explicit request
to continue to every exposed algorithm and ship verification in the wheel.
Worktree: `~/mojolearn-wt/cpu-verification-completion`.
Evidence: `~/mojolearn-evidence/cpu-verification-completion`.
This is an active checkpoint, not final release qualification. No PyPI upload.

Baseline: 128 available CPU lanes, 51 withheld, 50 parallel exclusions. The
51 were 27 missing references, five stale references, five held for separate
records, two unwatched, and 12 awaiting additional vendor evidence. Historical
native controls covered every mapped lane/part for 52/246 appendix entries
on at least one fixture, not all-fixture/current-artifact qualification.

At `0dcf5b5a0`, candidate availability is 142, with 37 withheld. Historical
native-control entry coverage is 60/246. All available lanes now have core
references on all nine fixtures (including N/A where actually applicable).
The installed candidate passed the 16 changed/reference-filled lanes: 423
IDENTICAL, 297 N/A, zero DIVERGENT, REFUSED or OWED. The comparator self-test
passed; receipts and reports are in the CPU completion probe directory.

## Implemented and observed

- `tools/verify_cpu_batch.py` runs bounded per-lane clean/native sabotage arms,
  preserves incomplete records, checks source/input/device context, and requires
  all nine training controls. Step/full is recorded by default; `--properties`
  additionally records gradient, batch-size and ragged checks.
- Scoped strict reference generation: `verify --all --lanes ...
  --reference-table BASE --emit-reference CANDIDATE`. Missing/conflicting parts
  and incomplete checkpoints refuse. Unselected cells/indices and the legacy
  table policy stay unchanged. Coverage exposes each updated lane's policy.
- `embedding`, `embedding-sort`, `ivf-euclidean`: installed CPU development-wheel
  replay passed 90 IDENTICAL, 45 N/A, no missing/refused/divergent parts; all-nine
  historical training controls are present. Legacy `identity` remains limited
  to its own fixed record, so promoted lanes cannot compare pre-fix answers.
- `ivf` remains withheld: its ties training control still owes a qualifying
  fresh pair. The native value perturbation already exists; rerun before adding
  a new fault. The first four-lane installed assertion failed here and is retained.
- BPE training and fold construction: their already-public host routes now have
  references from repeated all-nine CPU records. These are Python algorithms;
  no native-control or final-release claim is made for them.
- H/C/V metric: Linux source `f0d53b4ed` moved eight fixtures, but not negative.
  Its constant-label branch ignored an in-range label fault. Sabotage builds now
  return the deliberately wrong zero from that zero-entropy branch. Production
  arithmetic is unchanged. Fresh Mac builds at `b9902bb46` detected all nine.
- Ten classical lanes at `f0d53b4ed`: weighted GBDT and RF scores, GBDT NaN modes,
  parametric losses, Newton-cosine, pair-logit, YetiRank, both HDBSCAN variants,
  and kmeans-sqrt. All 90 native training controls passed; all clean core parts
  are stable or legitimately N/A. Complete source/native provenance and failures
  are committed under `identity_break/2026-09-17_cpu-classical-completion/`.
  Native controls use the combined sabotage set, not one fault at a time.
- No preexisting numerical reference hash was replaced. Kmeans-sqrt's 18 changed
  values were old `n/a:no-save` / `n/a:no-predict` entries replaced by newly
  observed numerical results. Unrelated cells remain unchanged.

278 focused tests passed after the classical additions. The local neural
extended-property preflight hit its explicit 300-second limit during
mamba2-dtlimit batch-size checks. Four cells completed (byte-lm, mamba3,
transformer, samba), but the whole checkpoint is incomplete and NOT admitted.
No fixture floors or repetitions were reduced. No Metal run.

## Cloud and restart

First pod `0krigxfoki5p54` is VERIFIED DELETED. Two vCPUs, $0.06/hr,
376 seconds building, 901 seconds executing, total cost $0.0238. Overall exit
1 correctly retains the original metric control failure; the ten passing lanes
are admitted separately. Results: `cloud-classical/remote/leg_out/`.

In flight: second RunPod CPU pod `4018mzhxmp3prb`, source `dcddb8345`, two vCPUs,
60-minute watchdog and external dead-man, $0.06/hr. SSH was
`-p 39596 root@38.80.152.147`. It builds ten production and eight sabotage
families and runs 25 lanes: IVF, corrected metric, ordered gradient sum,
14 low-bit lanes, and eight neural/reference-stale lanes. Budget 600 seconds
per arm, two repetitions, all nine fixtures, step/full where applicable.
Logs: `cloud-neural-run.log`; results will be in
`cloud-neural/remote/leg_out/`. Root exec session was 5651. Do not kill an owed
run merely to start another pod. Verify deletion even if the command fails.

Only one cloud job (two vCPUs) plus one local single-thread numerical/compiler
worker may run. Use mac_slot.py, numerical thread limits one, compiler at most
two cloud workers or one local worker. Cloud runners use R2-backed caches.

Next: finish installed replay (`check_installed.py ... classical` under the
external evidence directory), retain its receipts, integrate the completed
batch, and inspect/recover the neural cloud run. Still owed: UMAP, the 12
vendor-held candidates, all other missing native/property coverage, hardware
and multi-GPU pairs, and final artifact qualification. The development wheels
reuse Mac bindings except the freshly compiled metrics binding; their receipts
must not be treated as certification of all current native sources.

## Additional completed work at `5e5e0a84c`

UMAP and ordered gradient sum now have all-nine strict references and native
controls, bringing availability to 144 and withholding to 35. UMAP uses the
fresh metrics builds and passes held-out, saved model and batch checks. Ordered
sum needed a native accumulation fault: reversing two addends cannot change an
addition. Sabotage now zeros the first accumulated result; production is
unchanged. The independent oracle catches all nine wrong outputs. The harness
retains repeated measured hashes as DIVERGENT, still exits one, and never admits
those failed results as references. Ordinary exceptions remain REFUSED.

The CPU batch runner now rejects clean property failures even when the training
control moved. The in-flight pod used the earlier runner: its reported training
successes must be reevaluated. Its Mamba/Samba stepfull properties currently
refuse because the build list omitted the neural binding. Keep these failures;
rerun affected lanes with the full native dependency set. The original ordered
sum native control was unchanged and remains a failed control.

307 focused tests pass. The next installed development artifact adds the freshly
built training binding alongside metrics; all other Mac bindings remain reused.

Installed follow-up `b74de8f54` passes UMAP and ordered sum: 45 IDENTICAL,
45 N/A, no DIVERGENT/REFUSED/OWED, current coverage snapshot, all-nine historical
controls, default selection and comparator self-test. Receipts are committed in
the completion probe directory. A bounded single-thread local Mamba2 clean
record is now running with all native dependencies to fill the missing stepfull
property; external `mamba2-complete-clean.json`, root session 28100.

Oracle diff follow-up: equally wrong numerical outputs remain DIVERGENT when
compared to each other. A regression exercises the recorder and diff together;
37 runtime/control tests pass. This small fix was isolated in
`~/mojolearn-wt/cpu-oracle-diff-fix` while the Mamba2 record retained its frozen
source worktree for safe checkpoint resumption.
