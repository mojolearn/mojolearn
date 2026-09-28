# Bounded par-gmm reference and placement qualification

This workflow is prepared for a future exclusive job on the existing two-GPU CUDA host. It has not been run. Do not overlap it with another GPU job. Use a committed source containing the current classic k-means++ initialization and the `verify --par --reference-table` fix, with matching source-built bindings. The mixture binding must expose `gmm_parallel_available() == 1`.

One eligible independent AMD **one-device par-gmm** record already exists at:

```
/tmp/consolidated-existing-records/do-amd/ev-apple-merged-9f20e20ac/s/clean_0of1/par-gmm.gpu.json
```

Its SHA-256 is `5abd39df5941bd63cc0990b4357c787caa8102064e3f279414a35ee3c17160e7`, source commit is `9f20e20ac80a773b4bfc4f5d7ac3c5ffe0faff0a`, enforced backend is HIP, revision is `classic-kmeanspp-init-1`, and all nine fixture/held-out witnesses and eight declared properties match the current protocol. Preserve its original bytes when staging it as `amd.par-gmm.json`. Its CPU counterpart refused; it supplies no CPU evidence. An ordinary `gmm` record cannot substitute for this driver record.

## Commands inside the separately authorized job

Run from the frozen checkout with its Python environment. Set `PAR_GMM_OUT` to a new persistent evidence directory and stage the AMD record there first. The CUDA visibility mask must expose two distinct physical GPUs, not aliases of the same device. Do not set internal family device-count overrides; the driver establishes these inside its cooperative worker.

```sh
export MOJOLEARN_NUMERIC_MODE=identical
export CUDA_VISIBLE_DEVICES=0,1
unset MOJOLEARN_PAR_DEVICES MOJOLEARN_GMM_DEVICE_COUNT MOJOLEARN_KMEANS_DEVICE_COUNT
PAR_GMM_OUT="$HOME/mojolearn-evidence/par-gmm-current"
mkdir -p "$PAR_GMM_OUT"
# Stage the unchanged AMD witness as $PAR_GMM_OUT/amd.par-gmm.json.
# Check its SHA-256 against the value above before proceeding.

MOJOLEARN_PAR_DEVICES=0 python tools/identity_break.py \
  --lanes par-gmm --fixtures base,ties,hashed,wide,denormal,denormal_ftz,dupes,odd,negative \
  --repeats 1 --require-backend cuda --fail-on-refused \
  --json "$PAR_GMM_OUT/cuda.par-gmm.json"

python -m mojolearn verify --all --lanes par-gmm --batch-checks \
  --records "$PAR_GMM_OUT/amd.par-gmm.json" \
  --records "$PAR_GMM_OUT/cuda.par-gmm.json" \
  --reference-table python/mojolearn/verify_reference/table.json \
  --emit-reference "$PAR_GMM_OUT/candidate-table.json"

python -m mojolearn verify --par all --lanes par-gmm --par-devices 0,1 \
  --repeats 1 --reference-table "$PAR_GMM_OUT/candidate-table.json" --json \
  > "$PAR_GMM_OUT/two-device.json"
```

Run these steps with failure propagation (`set -e` in the job script); preserve stdout/stderr, the frozen commit, binding hashes and raw JSON. The candidate step runs no fits, merges only `par-gmm`, and reports incomplete/conflicted evidence. A successful emit exit alone does not
prove complete candidate coverage; validate every required part before physical checks. Each numerical reference requires agreement from the AMD and CUDA classes; two A40s are not two vendor classes. The final step performs the two-device column once for each fixture against that candidate, without rewriting the shipped table.

## Admission conditions and scope

Both one-device records must pass strict full-property admission at the current revision, with matching fixture/held-out hashes and protocols. The two-device result must report agreement with no refused/missing parts and a successful **per-fixture physical placement witness**. GMM uses a cooperative worker seeing both GPUs, so validate two distinct GPU UUIDs and the requested device group, not two worker processes. Every fixture must have a witnessed pool; a one-device fallback or process-only witness is insufficient.

The reference builder deliberately admits the one-device baseline only. `admit(..., par_axis=True)` admits two-device records for separate physical-axis coverage, not as a substitute baseline. The observed two-device result remains separate evidence. Promote the scoped candidate only after this physical check passes. This can qualify current NVIDIA sharding without another multi-GPU host; it does not claim current AMD two-device, Apple multi-device, or CPU cooperative-GMM support.


## Bounded queued runner

`tools/par_gmm_qualification/run.py` implements the reviewed sequence with one
fresh process per fixture and a 120-second limit, including process-group cleanup
on timeout. It pins the numerical source to `c46f776140e997a3db602c97dea53985601e5131`.
Run the script from a separate committed tools checkout against a clean, prebuilt
checkout at that source; its own SHA-256 is recorded in the final summary.

```sh
python3 tools/par_gmm_qualification/run.py --tree /owned/par-gmm-source \
  --out /owned/evidence/par-gmm-new --amd-record /owned/amd.par-gmm.json
```

The command above only prints the plan. Add `--execute` **inside an exclusive
two-GPU queue job**, after source-built bindings and the Python environment are
ready. Preserve the queue's `CUDA_VISIBLE_DEVICES` mask; the driver uses logical
ordinals `0,1` within it. Compiling belongs outside this GPU job, after the
existing consolidated prebuild has completed. Reuse only digest-matching copies
from that completed private native store. Do not mutate the 445-lane run's trees.

The runner preserves nine original CUDA records and the original AMD record;
it passes all ten directly to the existing reference builder. It does **not**
merge raw records with different fixture dictionaries or invent merged provenance.
Per-fixture records are admissible because they are complete for their declared
scope, retain all property fields, and bind their fixture/held-out bytes. The
strict candidate validator requires all nine fixtures and eight declared parts,
agreement from both AMD and NVIDIA classes, the current lane revision, and no
conflicts. The physical checks then run all nine fixtures separately against that
candidate and require each fixture's successful physical placement witness.
Only after every check succeeds is `qualification.json` written. Existing output
directories are refused, preserving failed or interrupted evidence for diagnosis.
This workflow allocates no machine, submits no job itself, and never publishes
or overwrites the shipped reference table.
