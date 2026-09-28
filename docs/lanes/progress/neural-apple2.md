# neural-apple2: progress (Apple speed round 2, neural family)

Branch `lane/neural-apple2`, forked from `lane/apple-merged` at 037daa353.
Family: transformer, Mamba-1/2/3, Samba, MLP, embedding, byte LM, training.
Brief: ~/mojolearn-evidence/apple2_speed_brief.md. Round 1:
docs/lanes/progress/neural-apple.md.

Rule for every A/B: IDENTICAL bits must not move (GPT-3 route B on AMD
depends on them). Every row below names its digests (bench lane output
digests, byte LM final witness and loss digests).

## Jobs

| steward id | Mac | commit | what |
|---|---|---|---|
| 1790603073363 | m3ultra-b | 037daa353 | BEFORE profile (every bench lane, T3 shard step, census) + mamba1/mamba2/transformer forward A/B b11745d8e vs 30497d57e vs 037daa353 |
