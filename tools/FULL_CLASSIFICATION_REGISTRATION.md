# Full classification A/B admission

`classification-full-v1` is a distinct input variant. Original capped cells and
all previous measurements remain untouched. The reviewed source contracts cover
44 Apple FAST and 40 NVIDIA IDENTICAL cells of the complete proposed paired
configuration. Apple StandardScaler/MinMaxScaler have no accepted paired
`_mojolearn_preprocessing` binding and remain pending; `x_prep` is not a substitute.
No AMD classification recipes are registered by this change.

Preparation uses `six_lane_prepare_full_classification.py` once on the original
archives: Taxi credit-card population (4,110,786 training and the final 500,000
filtered query rows), Istella full 2,043,304 training and first 500,000 separate
test rows. The six numeric standardized/raw/categorical archives preserve the
saved transforms. Categorical `min_categories` is retained during preparation.

Transfer all archives and small metadata before measurement. Preserve the exact
preparation plan and receipt bytes when relocating them. Preparation may precede
the final measurement commit: registration checks exact helper and preprocessing
source hashes, and records both SHAs. File names alone never establish identity.

Under the device's existing shared measurement lock, run the metadata admission
command with accepted workload-scoped A/B deployment receipts:

```sh
python tools/six_lane_register_full_classification.py \
  --preparation-receipt /external/full-inputs/receipt.json \
  --preparation-plan /external/classification-full-plan.json \
  --data-directory /external/full-inputs \
  --deployments /external/paired-deployments.json \
  --vendor apple --target-track apple \
  --output /external/fresh-classification-admission
```

Use `--vendor nvidia --target-track nvidia-native` for the retained native pair.
`--workload` optionally selects exact original IDs, never inferred subsets.
Registration hashes full files through the existing materializer; do not run
this I/O concurrently with timing. The generated queue is unauthorized until
the owner-controlled durable controller authorizes it. One excluded warmup and
one scored execution per arm remain the queue's policy, with the original
recipe's explicit repeated operation captured separately.

All cells retain `changes_frozen_race=true`. The worker validates population,
shapes, exact parameter records, source hashes, split, original operation and
declared output dtype/shape. Dynamic learned output widths stay explicitly
dynamic. NMF's complete fitted-row W is consumed and captured, not sampled.
SimpleImputer's intentional parameter NaN alone uses a tagged metadata value;
constructors, timings, outputs, metrics and strict finite comparison are unchanged.

Full input/declared output admission is not quality, constituent reach or complete
fitted-state qualification. Their limitations remain visible. Missing artifacts,
refusals and failures are retained, and only affected cells may be repaired.
