# Complete nonparallel CPU and three-vendor fixture evidence

The reference table now records **7,452 numeric cell parts matching exactly on CPU, Apple, AMD and NVIDIA**, across 214 nonparallel numeric lanes and all nine fixtures. No applicable nonparallel numeric vendor gaps remain. Eighteen further model-byte parts agree on all three GPUs: categorical and tensor CTR tables each have nine fixtures whose explicit CPU role loads the GPU-saved model rather than writing one. N/A declarations are not numerical comparisons. The complete table still includes all 273 lanes and reports zero conflicts.

This is a reference-data update after the 0.8.9 wheel freeze. It does not modify or rebuild the published wheel bytes, whose source remains `819a47ae48166e91951f54f54e64ee173658e32a`.

## Recorded inputs

- AMD: `bench/results/identity_break/2026-09-20_final-amd-vendor-coverage/amd-mi325x-gfx942.json`, built and measured from frozen source `819a47ae48166e91951f54f54e64ee173658e32a`.
- NVIDIA: `bench/results/identity_break/2026-09-20_final-nvidia-vendor-coverage/nvidia-h100.json`, built and measured from that same source; 648 complete cells and 2,934 numeric CPU comparisons exact.
- Apple: `bench/results/release089/apple-vendor-completion/apple.json`, 342 complete cells and 1,647 numeric CPU comparisons exact. Harness witness `c738f5261`, native build `a97676ba18a09fb577ef9faae45ab19a01eec848`; the record README documents unchanged GPU arithmetic relative to frozen 819 source.

The existing `verify --all --emit-reference --reference-table --batch-checks` path admitted the union of 100 lanes from these three records, reading the full `bench/results/identity_break` tree plus the explicit Apple JSON path. The scoped merge retained earlier CPU, linalg, saved-model and fixture evidence. Final table SHA256: `bbab5d379bdac40c110408c67e57fd8645b0de8343a537571292936e5f382063`.

## Limits and audit

`release089-all-vendor-fixture-audit.json` decodes actual values for each device column and includes all recorded numeric protocol parts, not only training and inference. It explicitly reports parallel APIs separately: 432 CPU numeric parts match all three GPUs; 279 CPU numeric parts still lack an equal value on at least one GPU. Another 684 parallel parts agree across all three GPUs without a numeric CPU role, while 531 lack at least one GPU value. These counts are not included in the nonparallel completeness claim. The artifact retains every parallel gap for inspection instead of excluding or rewriting it.
