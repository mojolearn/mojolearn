# Auditing parallel cross-vendor coverage

`tools/audit_parallel_coverage.py` checks the current harness contract against the reference table and emits a deterministic collection plan. It performs no training, starts no workers, and rents no hardware.

```sh
PYTHONPATH=python python tools/audit_parallel_coverage.py \
  --output /tmp/parallel-coverage.json --fail-on-incomplete
```

Exit 0 means complete coverage; exit 5 means incomplete coverage when `--fail-on-incomplete` is supplied. Without that flag, incomplete reports are still written and the command exits 0 for inspection. The output's `collection_plan` maps vendor → lane → fixture → required parts with reasons. `--vendors`, `--fixtures`, and `--reference-table` can narrow or redirect inspection. `--contract` accepts an independent JSON mapping from lane to part to `numeric`, `recorded`, or a current `n/a:` declaration, for example:

```json
{"par-new": {"train": "numeric", "model": "numeric", "future-property": "numeric"}}
```

The expected lane/fixture/part set comes from the harness, not from the entries that happen to exist in the table. An entirely missing cell or a newly required part therefore remains a gap. Batch and optional-property applicability use the existing harness registries. `REQUIRED_NUMERIC_PARTS` supplies independent numerical requirements, including newly supported saved-model bytes. Core probes with runtime applicability require an actual consistent recorded N/A declaration or numerical answers. A numerical witness on any device makes an old N/A on a required GPU stale; a skipped run never counts as an applicability declaration. Statically declared N/A properties do not require duplicate GPU N/A recordings.

The audit distinguishes missing cells, missing parts, missing vendor values, stale N/A, invalid values, conflicting entries and unequal values. Collection requirements preserve those reasons rather than replacing evidence with a majority value.

On the existing reference table at implementation time, the report identifies **810 numeric part-cells** needing work: 774 with missing vendor evidence and 36 with stale `n/a:no-save` values. These are **1,539 vendor values**, not 810 executions: AMD needs 729 (693 missing plus 36 stale), Apple 720 (684 plus 36), and NVIDIA 90 (54 plus 36). The same contract contains 1,116 already matching numeric part-cells and 2,853 inapplicable part-cells. These counts are an inspection result, not evidence that the missing work has run.

This audits **one-device cross-vendor reference coverage**. It does not establish physical multi-GPU execution. That separate claim uses `verify --par all` and its per-lane, per-fixture physical-device witness. Neither equal one-device hashes nor this collection plan can replace that qualification.
