# Compact byte-LM continuation handoff

The compact path avoids uploading either vendor's complete 128-step raw
baseline to the other host. Root keeps those raw baselines locally for final
admission. The remote host receives a pinned compact witness, its exact
retained baseline binding, and the actual foreign head checkpoint with its
original root receipt chain.

This is a separate path. `byte_lm_resume_serial.sh` retains its strict full-
baseline preflight, and `byte_lm_state_compare.py` is unchanged. Compact remote
results always say `identity_admitted=false` and `learning_admitted=false`.
A successful remote script exit means its diagnostic checks completed; it is
not cross-vendor admission.

## Root authors the compact artifacts locally

`tools/byte_lm_resume_handoff.py` is file-only and uses the standard library.
Before writing anything, it calls `admit()` on the complete baseline campaign,
which checks all raw training steps, successful guards, the independent oracle,
the retained binding and the learning gate. Root must retain that full baseline
and its original parent-level `leg.txt` and `source_inventory.json`.

```sh
python tools/byte_lm_resume_handoff.py \
  --baseline /local/complete/vendor/byte-lm-validation \
  --output /local/new/vendor-baseline-handoff
```

The output contains:

- `handoff.json`, bounded at 2MiB, with source/runtime/profile/config/schedule,
  initial/final state signatures, every step's complete eleven-array hashes,
  original step-manifest hashes, and terminal checkpoint SHA;
- the exact retained vendor binding under `binding/`;
- unchanged original full-run, first-step and oracle root receipts with their
  command/log/result files under `evidence/`.

The raw FP64 reference archive and raw baseline steps stay local. Their absence
is one reason a compact witness cannot independently admit the baseline on the
remote host. Bundle traversal is bounded to 128 entries, depth 6, 32MiB total,
and 16MiB per file; paths with symlinks or traversal are refused. Actual
transport size is the retained binding plus manifests and small receipt files,
not the full raw trajectories.

After a complete root-run head64 campaign has been fetched locally, root
authors the foreign head handoff against its original complete baseline:

```sh
python tools/byte_lm_resume_handoff.py \
  --baseline /local/complete/nvidia/byte-lm-validation \
  --head /local/complete/nvidia-head \
  --output /local/new/nvidia-head-handoff
```

This first admits the baseline, validates the actual head capture and its
successful root receipt, and compares all 64 raw steps with the baseline.
Its bundle adds `head64.checkpoint.json` containing the actual complete head
checkpoint bytes. The foreign binding is identified by SHA; it is not copied
into the receiving vendor's executable binding location.

Both commands print the `handoff.json` SHA and checkpoint SHA. Root pins those
values before transport. New output directories and exclusive read-only files
are required. The helper's own source SHA is retained, and the remote helper
must match it. Finish source changes before authoring a handoff.

## Remote compact execution

Root's existing transport controller copies only the compact bundles and the
required matching source/helper files. It stages the exact receiving-vendor
binding in `python/mojolearn/identical/_mojolearn_byte_lm.so` and supplies the
existing Python environment. The compact script checks installed binding bytes
against both the handoff SHA and the uploaded retained binding. It does not
build, install, provision, rent, or transport anything.

To create a head on a host with a compact baseline instead of raw baseline:

```sh
export MOJOLEARN_BYTE_LM_EXPECT_VENDOR=cuda
export MOJOLEARN_GPU_ARCHS=sm_ACTUAL_ARCH
export MOJOLEARN_PYTHON=/absolute/existing/environment/bin/python
export MOJOLEARN_BYTE_LM_BASELINE_HANDOFF_SHA256=ROOT_PINNED_BASELINE_HANDOFF_SHA
unset MOJOLEARN_BYTE_LM_FOREIGN_HANDOFF_SHA256 MOJOLEARN_BYTE_LM_FOREIGN_SHA256
bash tools/byte_lm_resume_compact_serial.sh head64 \
  /remote/nvidia-baseline-handoff /remote/new/nvidia-head
```

Fetch that complete new head campaign to root, then use the local authoring
command above to validate it against the full raw baseline and prepare its
foreign head bundle. Do not manufacture a foreign checkpoint from local RNG
initialization or regenerate a checkpoint from only its hashes.

On the receiving AMD host:

```sh
export MOJOLEARN_BYTE_LM_EXPECT_VENDOR=hip
export MOJOLEARN_GPU_ARCHS=gfxACTUAL_ARCH
export MOJOLEARN_PYTHON=/absolute/existing/environment/bin/python
export MOJOLEARN_BYTE_LM_BASELINE_HANDOFF_SHA256=ROOT_PINNED_AMD_BASELINE_HANDOFF_SHA
export MOJOLEARN_BYTE_LM_FOREIGN_HANDOFF_SHA256=ROOT_PINNED_NVIDIA_HEAD_HANDOFF_SHA
export MOJOLEARN_BYTE_LM_FOREIGN_SHA256=ROOT_PINNED_ACTUAL_HEAD_CHECKPOINT_SHA
bash tools/byte_lm_resume_compact_serial.sh resume128 \
  /remote/amd-baseline-handoff /remote/new/amd-resume \
  /remote/nvidia-head-handoff
```

Replace architecture placeholders with the actual explicit rented GPU
architecture. The reverse direction uses the corresponding vendor arguments.

Preflight checks the complete current numerical-source inventory against the
root handoff, exact installed/retained binding bytes, original root receipt
linkage, and foreign checkpoint bytes and state signature. It compares the
foreign first 64 step hashes with the receiver's baseline witnesses before
starting a model. It retains the offered handoff bundles in the new output.

The actual foreign checkpoint is published exclusively read-only in the
output. Each existing capture then loads it into a sealed memfd and retains
the exact incoming bytes. `resume128` must match all corresponding compact
step manifests and the terminal checkpoint SHA before `zero-moments65` runs.
The control compares actual local raw gradients/loss/tokens/flags against
legitimate step 65, proves only incoming moments were zeroed, and requires all
post-step parameters/m/v arrays to differ. It never claims that hash-only
baseline comparison is an all-raw comparison.

The serial guards, cancellation forwarding and bounds follow the strict
helper: one guarded job at a time, two child CPU cores/threads, 12GiB RSS cap,
GPU occupancy/free-memory checks, 2100-second default total with a 120–2400
range, shrinking per-job deadlines, and final cleanup reserve. Root receipts
are written only after actual successful model-guard exits. Do not nest this
whole script under the same guard; its per-job guards own the exclusion lock.

## Final admission still uses all raw bytes locally

Root fetches **all new raw head/resume/control captures**, command logs,
receipts and campaign exit records. It then uses the exact full comparator
command in `BYTE_LM_RESUME_SERIAL.md`, pointing `--cuda` and `--hip` to the
original complete local raw baselines, and `--head`, `--resume`, `--control`
to the fetched complete raw captures. Supply both original independent oracle
reports and all original root receipts. A compact handoff is never accepted
as a substitute for any of those five raw capture directories.

The final file-only comparison belongs to root, with its CPU/resource limits
and retained exit status. Successful local all-raw comparison and the separate
learning requirements are the admission boundary. This source addition was
not executed, tested, built, measured, or used to provision anything by its
authoring subagent.
