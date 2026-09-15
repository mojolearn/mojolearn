# Lane status: lane/cpu-training-embedding-ivf

Agent "cpuembed", fan-out of 2026-09-15. Task: give the non-block lanes with no
declared CPU path one (embedding, embedding-sort, ivf, ivf-euclidean, byte-lm,
byte-lm-resident) and settle the tokenizer's status. No box rented (and, per
Andrew's Sep 15 rule, none will be: every lane here is diffed against GPU
columns already in the repo).

## Done on this branch (not merged yet)

- Embedding: `embedding/host/embedding_host.mojo` restates the device launch
  (gather, seed, PLAN_SCAN and PLAN_SORT run structure, fold, pad row);
  `bindings/_mojolearn_embedding_host.mojo` exports the GPU binding's names.
- IVF: `ivf/host/ivf_host.mojo` restates `ivf_flat_build_and_search_host`
  (k-means quantizer through `cluster/host/kmeans_oracle.mojo`, pinned tile,
  identical top-k); `bindings/_mojolearn_ivf_host.mojo`.
- Byte LM trainer: `python/mojolearn/_byte_lm_trainer_host.py` serves the GPU
  byte LM binding's single-device ABI over the CPU byte LM binding;
  `_backend._cpu_only_binding` returns it (`host_surface.ADAPTED_MODULES`);
  `byte_lm_host_sabotage` now reports gemm_oracle's `MOJOLEARN_HOST_SABOTAGE` arm.
- Manifest: embedding and ivf families; the four lanes diffed against their own
  records through `TRAINING_FIX_COLUMNS` (nine JSONs:
  `2026-09-14_kmeans-sqrt-fix`, `2026-09-15_embedding-sort`,
  `2026-09-14_ivf-euclidean`); byte-lm and byte-lm-resident on the byte_lm family
  against the 166-lane record. Doc spans regenerated; the no-CPU-path sentence no
  longer names the Embedding layer.
- Tokenizer: CPU column IDENTICAL x4 (9 train, 9 infer) against the 166-lane
  record; not a training lane, so stated in SUPPORT_MATRIX's tokenizer row.

## Evidence (M4, one core, shared machine)

- embedding + embedding-sort: 18 train, 18 infer, 18 batch IDENTICAL x4;
  sabotage DIVERGENT 18 of 18 train.
- ivf + ivf-euclidean: 18/18/18 IDENTICAL x4; sabotage DIVERGENT 17 of 18 train
  (ivf/ties exact).
- byte-lm + byte-lm-resident: 18 train, 36 infer and model, 18 batch IDENTICAL
  x4; sabotage DIVERGENT on all 72.
- Scratch outputs: `$SP/cpuembed/` (`final/` is the fresh-build run).

## Waiting on

- lane/cpu-training-gate-budget (gatehyg) merging to main: without its sharded
  covered-lanes steps the seven-runner job exceeds 60 minutes on main already,
  and this branch adds about 5 minutes per IVF run on one M4 core, twice.
- lane/identity-record-next (record2): its 178-lane record carries the four
  lanes; when it lands, drop them from `TRAINING_FIX_LANES`.

## Next commands

    git fetch origin && git merge origin/main   # after gate-budget merges
    python3 tools/docs_facts.py --write && python3 tools/docs_facts.py --check
    bash $SP/cpuembed/final_local.sh            # fresh builds, both sets, all diffs
    git push origin HEAD:lane/cpu-training-embedding-ivf   # a commit without [skip ci] queues the gate
