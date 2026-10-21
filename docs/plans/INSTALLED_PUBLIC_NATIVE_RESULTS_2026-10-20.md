# Selected installed public native-runtime evidence

Internal qualification record, not a release declaration. The actual installed
wheel is `mojolearn-0.8.55-py3-none-macosx_11_0_arm64.whl`, 104,270,735 bytes,
SHA256 `55fe777eb72aa1c104f3e384820d38bbda43e8a90c2175decdc03db0b4ceb3c2`.
Package Python source is frozen commit
`72f11f70dd6774ab25a776cf745fdb6b7b6f0478` (stage58).

The wheel contains tokenizer HOST stage58, sequence HOST/Metal stage57,
105 other extensions retained from 0.8.55, and five runtime libraries: 113
native files altogether. All wheel native hashes match the already staged
inputs, and package Python bytes remained unchanged. Raw, relocated and
installed hashes are distinguished in the retained assembly receipt.
This mixed set does not qualify every native family against stage58 source,
later source changes, or a full release.

The fresh installed environments ran outside the checkout. Every wheel RECORD
entry, observed product Python module and cached native extension was checked
against the wheel. Test code was copied from the frozen commit into a separate
testing namespace. No product source directory was added to the import path.
The existing whole-operation audits and independent numerical/state checks ran
unchanged.

| Selected public family | Metal-default run | CPU run | Scope |
|---|---:|---:|---|
| Samba training step | 12 pass | 10 pass, 2 skip | Standard recipe, microbatches, resident/rebound/external parameter owners, rollback and callback refusal |
| Samba inference/state | 12 pass | 12 pass | Forward/loss, state allocation, dedicated step, recurrent caches and ragged inputs |
| Samba checkpoints | 12 pass | 12 pass | Streamed/legacy formats, resume, corruption, atomic failed save and path refusal |
| Samba admission | 27 pass | 27 pass | Protocol/metaclass/property/custom-buffer refusal before callbacks |
| Sequential execution | 21 pass | 21 pass | Fit/predict/score, continuation, validation/early stopping and consumed state |
| Sequential checkpoints | 23 pass | 23 pass | Import/export, safetensors/shards, checkpoint continuation and corruption |
| Sequential admission | 18 pass | 18 pass | Native descriptors, metadata, mapping factories and path storage |
| Generator draws | 31 pass | 31 pass | Uniform/normal/dropout, stream ordering, integer oracle and failed-call state |
| CausalLM generation | 23 pass **on CPU** | 23 pass | Prefill/greedy/sampling/EOS/stream/text/resume, including Mamba/Mamba2 |

The original Metal-default run passed 179/179 with zero skips, but generation
fixtures explicitly defaulted to CPU. That original receipt remains unchanged.
Controller repair `69b751b0ada678abf2e0d2a9d69e38a03ce02d55` sets the fixture device
explicitly and records the fixture's readback. A separate generation-only
installed retry then passed **23/23 on GPU with Metal/identical selected**,
zero skips or failures. Unrelated cases were not repeated. These are 179 unique
selected cases, not 381 distinct cases or complete API coverage.

The CPU run passed 177/179 with zero failures and two skips. Both are
`test_native_step_preserves_device_parameter_owner` with clipping `None` and
`0.3`. `training/estimator.mojo:715` excludes CPU from
`IDN_OPT_PARAMS_RESIDENT`; `_mojolearn_training.mojo:1836` conditionally registers
the corresponding device entries. These are expected GPU-only exclusions.
The unchanged controller completeness gate returned exit1 and
`failed-or-incomplete`, while pytest returned exit0. The skips were not turned
into passes.

Original failures are retained separately: wheel assembly rejected four
verification-only lazy NumPy imports, repaired by exact-name classification
`e0e8261a9`; the first installed controller stopped before pytest on six tracked
historical benchmark `.mojo` snapshots omitted by committed `export-ignore`,
repaired by `9170650cf`. The repaired inventory verifies 2,940 retained native
sources and records those six omissions with their committed blobs and ignored
ancestor. Missing or modified product sources still refuse, including product
paths marked `export-ignore`. Twelve controller tests and four vendor subtests
pass. No native code was compiled for these controller repairs or reruns.

[The evidence JSON](INSTALLED_PUBLIC_NATIVE_EVIDENCE_2026-10-20.json) records
full wheel/source hashes, per-family results, device scope, original failures,
artifact paths/hashes and explicit incomplete qualification fields. Full logs,
JUnit reports and installed-module receipts are under
`~/mojolearn-evidence/core-completion-20261020/installed-public-stage58-*`.

## Proposed next qualification sets

The following are pending installed coverage, not claims from the nine suites.
Use unchanged stage58 bytes only for stage58 claims. Extend the controller's
explicit test selection in a separate testing commit; do not bypass its
installed-origin or frozen-test checks.

| Set | Existing frozen tests | Remaining public boundary |
|---|---|---|
| Samba construction and public state | `test_samba_construct_native.py`; public snapshot/restore case in `test_samba_state_native.py`; public gradient cases in `test_samba_gradients_native.py` | Seeded/lazy construction, weight replacement and aliases, public `state_dict`/`load_state_dict`, direct `loss_and_grads`. Private bridge cases remain supporting evidence only. |
| CPU SambaInference | `test_samba_cpu_inference_native.py` | Separate CPU inference class, low-bit construction, complete cache state and checkpoint loading. Do not label it GPU coverage. |
| Sequential declaration and buffer coverage | `test_sequential_native_surface.py`, `test_sequential_native_buffers.py` | Public layer/compile configuration, plain/strided/readonly/nested buffers, dtype promotion and label extremes. Internal descriptor checks are supporting evidence only. |
| Generator lifecycle and initializers | `test_generator_state_native.py`, `test_generator_initializer_native.py` | Constructor/state/terminal stream, Kaiming/Xavier, exact scalar math and refusal stream order. |
| Model/tokenizer loading | `test_causal_model_native_loading.py`, `test_tokenizer_native_loading.py` | Model-family mapping, BF16/int8/fixed15 construction and standard native file paths. The frozen tokenizer caller-path test still expects a Python callback; repair that obsolete expectation before claiming a complete native-contract suite. Standard string/bytes/Path support must remain. |

Actual Torch mode guards, newer graph optimization/derivative repairs and
classical owner changes require their own current qualified source/binary
closure. The stage58 wheel cannot certify newer source by running newer tests
against older native code. Capture current package/native provenance first;
then run the actual Torch poison modes and current graph/whole-model suites.
Likewise, installed CUDA/HIP qualification, physical multi-GPU execution,
all supported algorithms, full-model vendor bit identity, performance A/B
campaigns and the upstream PythonObject lifetime blocker remain separate.

No public docs/support claims, optimizer defaults or performance claims are
changed by this record.
