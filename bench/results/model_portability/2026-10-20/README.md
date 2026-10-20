# Custom models and portable inference: October 20 evidence

This record supports the custom-model and imported-inference results in the
MLSys paper. It retains existing correctness results and one recovered
saved-model comparison. It admits **no comparative performance result**.

## Results

- The public `Sequential` example and the custom mixture classifier built from
  `ops.Session` operations have matching trained-parameter and prediction
  digests on Apple, NVIDIA and AMD. A session-training result and exported-model
  labels also match: six selected digests, recorded individually in `review.json`.
- Six pretrained models have identical output digests across all three vendors
  and repeat within each vendor: MNIST-12, MobileNet V3 Small, ResNet-18,
  ResNet-50, SmolLM2-135M and SqueezeNet 1.1. These are eighteen vendor/model
  records, not eighteen different models.
- The same ResNet-18 program file saved on an NVIDIA RTX 3090 reloads on an
  Apple M4 with the same input and output digests. The output digest is
  `b71faa2876f07f24`. Both verification receipts and the successful comparison
  command's complete output and exit status are retained.

`manifest.json` binds each retained input by SHA-256 and byte count;
`review.json` is computed from those inputs. The full test logs, original model
reports, source recipes and catalog snapshot are under `raw/` and `source/`.
The saved model's bytes are retained on the Mac and in R2 at the key in
`manifest.json`; `raw/reload/model-retention.json` records a complete R2
download whose SHA-256 equals the NVIDIA model receipt.

## Scope and provenance

This is a composite source-built record, separate from the older 0.8.10 verifier
census and from installed-wheel release qualification.

NVIDIA and AMD used source revision
`457c727d48fd298f56cfd8fb371da0aadf7dca45`, reporting version 0.8.54. The
0.8.55 catalog carries these model records forward with its explicit
`measured_at` note; it does not claim a new execution of every model on 0.8.55.
The Apple custom-model evidence is the corrected compile-52 native-lowering
worktree run. Its exact worktree HEAD was not retained. The three complete
test logs end with 121, 253 and 214 passing tests respectively; these are
different suites and are not added together as one qualification count.
Only the six named, matching outputs are admitted here.

The original NVIDIA saved-program receipt was retained, but the Apple
comparison output was not. On October 20, the existing test environment and
source checkout at `cdfeb4984f63c82c58825a42f4fed349675e4fca` reloaded that
unchanged file on the M4 and retained the receipt. The loaded binding reports
0.8.55. This is one observed compatible reload, not a general promise of
identity between library versions.

The original missing-pytest failure on NVIDIA, initial compile-52 failures,
and the first local reload command's missing-module error are preserved in
`raw/context/` and `raw/reload/`, beside the successful corrected records.
The separate layer-pool two-GPU check is outside this record and remains
incomplete. No whole-suite or two-GPU claim is inferred from selected digests.

The imported-model checks use batch one and fixed seeded inputs: four
ImageNet-shaped networks use `(1, 3, 224, 224)`, MNIST uses `(1, 1, 28, 28)`,
and SmolLM2 uses 32 token IDs. ONNX model-zoo files were converted from opset
12 to 13; the source and converted hashes are retained. SmolLM2 uses the
pinned Hugging Face revision recorded in its reports and is imported through
`torch.export`. Cross-vendor digest agreement is distinct from source-framework
numerical agreement. These inputs do not establish task accuracy on a held-out
image or language dataset.

## Check the record

From the repository root:

```sh
python3 tools/check_model_portability_evidence.py \
  bench/results/model_portability/2026-10-20
```

This checks every retained file, requires all three vendors for every selected
output, checks model input/source consistency and the catalog, and compares
the saved-program receipts. It runs no model and takes no timings.

## Performance evidence remains separate

The original model JSON files contain timing observations, which remain raw
evidence here. They are not imported into the paper's performance tables.
The forthcoming measurement run is separate.

The current benchmark implementation reads back and checks settings through
`tools/bench_board_params.py`; neural workers reject a mismatch before timed
rounds in `tools/bench_board_neural.py`. It uses persistent workers, an excluded
warm-up, synchronized timed calls and per-lane timing spans. Captured-program
inference has weights resident before timing and includes input upload,
execution and output return. Language-model generation records its own
prefill/generation boundaries. Memory sampling is a separate pass.

For classical opponents, `tools/board_clock_audit.py` derives whole-operation
and kernel-only clocks only from retained copy measurements and labels mixed
boundaries where neither common clock can be recovered. The publication
selection is applied by `tools/race_fairness.py` and the board ingestion tools.
Thus the paper must describe matched work settings, permitted identity-related
arithmetic differences and recorded timing boundaries, rather than saying
that each arm chooses its own algorithm, defaults and clock.
