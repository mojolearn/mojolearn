# Compact byte LM resume transport (source candidate)

Profile 6 of `tools/gemm_remote_leg.sh` reuses the root-retained vendor binding.
It does not build a model. The historical `MOJOLEARN_NVIDIA_CAMPAIGN` name also
selects this profile on AMD. Profiles 0–5 retain their separate workloads.

Root must commit the transport/helpers, preserve the qualified numerical source
inventory, then author the compact bundles with that committed handoff helper.
Bundle directories are root-created inputs, each at most 32 MiB. Pin the SHA256
of **handoff.json**, not the compressed transport archive. Source is shipped or
fetched first and its inventory compared before data transfer/model launch.
Bundles are validated before rental, packed using the committed helper, and
validated again after bounded extraction outside the source directory.

Root-only example (replace all capitalized placeholders; output must be new):

```sh
MOJOLEARN_NVIDIA_CAMPAIGN=6 \
MOJOLEARN_PUBLIC_GITHUB_SOURCE=mojolearn/mojolearn \
MOJOLEARN_GPU_ARCHS=sm_90 \
MOJOLEARN_BYTE_LM_RESUME_ACTION=head64 \
MOJOLEARN_BYTE_LM_BASELINE_HANDOFF_DIR=/absolute/CUDA_BASELINE_BUNDLE \
MOJOLEARN_BYTE_LM_BASELINE_HANDOFF_SHA256=HANDOFF_JSON_SHA256 \
MOJOLEARN_GEMM_LEG_OUT=/absolute/NEW_OUTPUT \
tools/gemm_remote_leg.sh nvidia --payload mamba --source-ref HELPER_COMMIT \
  --rent --minutes 60 --work-timeout 3000 --gpu MATCHING_GPU
```

For AMD use `amd`, `MOJOLEARN_GPU_ARCHS=gfx942`, the HIP baseline bundle and an
MI300X rental. Use an image with Python and `venv` already installed. Torch is
not required for this profile. For a subsequent foreign resume set action
`resume128` and additionally set `MOJOLEARN_BYTE_LM_FOREIGN_HANDOFF_DIR`,
`MOJOLEARN_BYTE_LM_FOREIGN_HANDOFF_SHA256` and
`MOJOLEARN_BYTE_LM_FOREIGN_SHA256` (the checkpoint SHA). The supplied head must
be from the opposite vendor. No foreign inputs are accepted for `head64`.

The pinned Pixi lock installs the runtime; a fresh environment installs only
binary NumPy 1.26.4. Setup copies the exact retained binding into the public
IDENTICAL package location. Existing destinations are refused. The existing
remote watchdog/deletion flow and global deadline remain active. NVIDIA uses its qualified runtime defaults; AMD retains its successful 1 GiB
pool override. Guards cap GPU memory at 85%, process-group RSS at 12 GiB, and inherited
affinity/thread settings cap CPU work at two cores/threads. Setup has its own
guard; the compact helper then owns each model job's guard without nested locks.

Fetched `byte-lm-resume-setup` and `byte-lm-resume` records support transport
completion diagnostics only. Root must compare **all original baseline raw
arrays and all newly fetched head/resume/control arrays** with the final local
comparator. Compact hashes and a zero exit do not admit cross-vendor identity.

This source has not been executed by its author. Root must review and run the
file-only transport checks/dry run before any rental. Image Python ABI/runtime
compatibility is still checked by the actual guarded capture. Slow SSH data
uploads can exhaust the existing lease; public GitHub transport accelerates
source only and sends no private handoff data to GitHub.
