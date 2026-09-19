# Ten `par-*` cells that could not fail, and what each one is held to now

2026-09-19. `tools/lane_applicability.py --oracle-counts` read ten of the 53
`par-*` lanes `recorded`: nothing inside the cell could raise, so the cell
passed by comparing today's bytes to a previous hash of the same code, which
in a column total is indistinguishable from coverage.

Six of the ten already had an oracle the classifier could not see. Four had
none. Both halves are fixed; `--oracle-counts` now reads **0 `recorded` among
the 53 `par-*` lanes**.

## The column everything here was measured on

    MOJOLEARN_NUMERIC_MODE=identical
    MOJOLEARN_HOST_DIR=<prebuilt CPU host bindings>
    PYTHONPATH=<python-only package staged by tools/identity_iterate.cpu_package>
    python3 tools/identity_break.py ... --require-cpu

vendor `cpu-apple-m4`, commit e138146052bb4ab6b6ee6674421ffc31a7a2a128, one
core, `nice -n 19`. No Metal or GPU job was run. `par-byte-lm`,
`par-byte-lm-model-pool` and `par-byte-lm-offload` REFUSE on any CPU column,
before this change and after it, because `bindings/_mojolearn_byte_lm_host.mojo`
registers no `byte_lm_parallel_*` entry.

## THE HASHES DID NOT MOVE

`after.cpu-apple-m4.json` is `par-mlp`, `par-samba` and `par-samba-clip` on all
nine fixtures, `--repeats 2`, with train, infer, model, reload and batch parts.
Held against five committed columns -- the four in
`2026-09-14_166-lanes/` (apple-m4, nvidia-h100, nvidia-2xh100 with
`--par-devices 0,1`, amd-mi325x) and `2026-09-16_cpu-par-samba/` --
**660 hash and part comparisons matched, 0 moved.**

`after-seven-lanes.cpu-apple-m4.json` adds the four lanes whose bodies were NOT
touched (`par-scaler`, `par-scaler-minmax`, `par-reference-knn`,
`par-reference-knn-reg`) over the same nine fixtures: 1008 matched. The only
differences against the 2026-09-14 record are `par-scaler`'s `model` column,
which was `n/a:no-save` then and is a hash now -- a change on main that
predates this one and touches a lane this pass did not edit.

`before.cpu-apple-m4.json` is the same ten lanes before the change, for the
train hashes: `par-mlp` 6e5cde4953362927, `par-samba` 359834bee4b2789a,
`par-samba-clip` 387a63aafbcce06d -- the same values the committed columns
carry, and the same after.

## THE GUARD WAS WATCHED FAILING

Every arm perturbs the SHARDED path only, in a staged copy of the package, and
leaves the reference reconstruction alone. The lane bodies are the ones in this
commit unless the file says `.before.`.

| perturbation | par-mlp | par-samba | par-samba-clip |
|---|---|---|---|
| `sabotage-extra-shard.after.json` shard 0 folded twice into the ordered sum | REFUSED | REFUSED | REFUSED |
| `sabotage-extra-shard.before.json` the SAME sabotage, pre-change lane bodies | STABLE | STABLE | STABLE |
| `sabotage-reversed-fold.after.json` the shard fold reversed | REFUSED | stable (inert) | stable (inert) |
| `sabotage-dropped-clip.after.json` the driver's update drops `max_norm` | n/a | stable (inert) | REFUSED |

The second row is the whole point. Under the same deliberate defect the three
cells read **cells=3 stable=3 moved=0 refused=0** before this change, with
their hashes silently moved (`par-mlp` d74b88302e9b57b8 for 6e5cde4953362927),
and **refused=3** after it. Nothing in the cell could see it; only a record
could, and only if one existed for that column.

The two "inert" rows matter as much: the new arms do not fire on everything.
Reversing a two-shard sum is bitwise identical for these Samba gradients, and
dropping a clip does nothing to a lane whose `max_norm` is `None`. A guard that
raised there would be a guard that cannot pass.

Messages, from the JSONs' `.errors.txt` siblings at capture time:

    par-mlp        ParallelNeuralTrainer weights_[bias1] and ordered single-device
                   replay differ: 37 bytes of 64
    par-samba      ParallelNeuralTrainer parameters()[embed.weight] and ordered
                   single-device replay differ: 21450 bytes of 32768
    par-samba-clip ParallelNeuralTrainer export_gradients[0] and ordered
                   single-device replay gradient differ: 28166 bytes of 32768

## `par-byte-lm`: the guard, not the binding

`par-byte-lm` cannot run on any CPU column, so its new arm has not been
exercised through a binding here and that run is owed on a GPU column. What
WAS watched is the helper it now uses. `replay_guard_probe.py` drives
`_byte_lm_replay` with two stub trainers and no binding at all:

    agreeing pair                : PASSED
    losses perturbed (+1 ulp)    : RAISED: step 0 losses and replica trainer losses differ: 2 bytes of 8
    state perturbed              : RAISED: trainer parameters and replica trainer parameters differ: 11 bytes of 32
    gradient perturbed           : RAISED: export_gradients and replica trainer export_gradients differ: 8 bytes of 32

That the comparison will PASS is not an assumption either. `_byte_lm_seed` is
`par-byte-lm`'s own seed construction verbatim and `_byte_lm_replay`'s loop is
its own loop verbatim, and the committed 166-lane columns record
`par-byte-lm`'s `loss`, `params` and `m` parts BIT-IDENTICAL, on all nine
fixtures on all four vendors, to the two sibling lanes that already hold their
driver to this same replicated reference through this same helper.

## The classifier

`tools/lane_applicability.py --selfcheck` gained five arms, each watched
failing with its own mechanism removed:

    helper-following off     -> kmeans, par-byte-lm-model-pool FAIL
    raised-mismatch off      -> par-scaler, the constructed raised case FAIL
    derivation chains off    -> kmeans, embedding, the constructed view case FAIL
    every _mismatch_bytes on -> the constructed dropped-message case FAILs
    primitives followed      -> five arms FAIL (the collapse)
