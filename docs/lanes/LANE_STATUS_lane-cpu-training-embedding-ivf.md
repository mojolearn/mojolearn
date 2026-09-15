# Lane status: lane/cpu-training-embedding-ivf

Agent "cpuembed", fan-out of 2026-09-15. Task: give the non-block lanes with no
declared CPU path one (embedding, embedding-sort, ivf, ivf-euclidean, byte-lm,
byte-lm-resident) and settle the tokenizer's status.

## Done (on this branch, not merged)

- `embedding/host/embedding_host.mojo` restates the device launch of the
  Embedding layer (gather, seed, PLAN_SCAN and PLAN_SORT run structure, fold,
  pad row); `bindings/_mojolearn_embedding_host.mojo` exports the GPU binding's
  names; `bindings/build_embedding_host.sh` is the shim; the manifest declares
  the embedding family with the embedding and embedding-sort lanes.
- M4, one core: all 18 train, 18 infer and 18 batch cells of the two lanes read
  IDENTICAL x4 against `bench/results/identity_break/2026-09-15_embedding-sort/`
  (`--require-columns 4`), and the `-D MOJOLEARN_HOST_SABOTAGE=1` build reads
  DIVERGENT on 18 of 18 train cells (parts differ: dw), every cell STABLE.

## In progress

- IVF host family (`ivf/host/ivf_host.mojo` over `cluster/host/kmeans_oracle.mojo`),
  the extra-records mechanism in `host_surface.py` and the gate workflow, tests,
  the doc spans. No box is rented.

## Next commands

    SP=<scratchpad>; bash $SP/cpuembed/build_set.sh <outdir> "" embedding ivf
    bash $SP/cpuembed/ib.sh <outdir> cpu.json embedding,embedding-sort,ivf,ivf-euclidean
    python3 tools/identity_break.py --diff <three GPU columns> cpu.json --require-columns 4 --lanes <lanes>

(`build_set.sh` and `ib.sh` are one-core wrappers over `bindings/build_<family>_host.sh`
and `tools/identity_break.py` with `MOJOLEARN_HOST_DIR` set.)
