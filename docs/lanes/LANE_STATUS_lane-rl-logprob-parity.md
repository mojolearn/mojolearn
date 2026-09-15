# Lane status: lane/rl-logprob-parity (the `rlpair` harness part)

Goal: prove "the sampler's log-probabilities equal the trainer's" bit for
bit on every column, as a new `rlpair` part of tools/identity_break.py, for
the byte LM, TransformerBlock, Mamba1/2/3Block and SambaStack.

## Done
- 2026-09-15 09:30 ET: worktree created from origin/main 4eac5719a; design read
  (batch part, block state APIs, SambaStack has no decode API, byte LM has no
  KV cache API).

## Running
- nothing

## Next commands
- implement `rlpair` in tools/identity_break.py and a SambaStack stateful
  forward/step; run on the M4 (Metal set copied from wt-apple-R, one core).
