# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# THE BATCH SOURCE RESUME (lane batch-source, 2026-11-25; Andrew's deciding test): streamed, variable-length batches
# from a tiny on-disk shard directory (`mojolearn.ops.BatchSource`, autograd/batch_source.mojo), trained by
# `Program.fit(source=)`, stopped MID-EPOCH after the source has READ AHEAD the next batch, checkpointed with ONE call
# (`Program.save_training`), and resumed with ONE call (`Session.load_training`) in a FRESH PROCESS. The resumed run
# must train the same next batch (the same record ids: the checkpoint holds the CONSUMED position, not the
# read-ahead), draw the same dropout masks (the step counts key them: the loss of the step reads the masked forward)
# and make the same parameter updates as the uninterrupted run, bit for bit, for two steps (the second crosses the
# epoch boundary, so the next epoch's shuffled, bucketed plan is drawn on both sides).
#
# A fragment of tools/identity_break.py (loaded by `_load_lane_fragments` into its namespace; np, os, tempfile, _h,
# _fit, _same_bytes, _hw, _lane_side, lane and _batch_decl are that namespace's). Device-agnostic
# (docs/plans/VERIFY_THROUGH_DEVICE.md contract 1): no `device=` in this process; the fresh process is handed the
# lane's side (`_lane_side`), the one place a lane must read it (a saved file the side loads). Helpers are prefixed
# `_bsrc_`. The reference is OWED.
#
# Fixture: 7 records of lengths 1 to 4 in two shards, F = 2, batch 2, two batches a bucket window, shuffled from
# seed 5: 4 batches an epoch. The model: steps (B, L, 2) -> reshape (-1, 2) -> dropout 0.5 -> matmul W (2, 1), plus
# lengths (B, 1) matmul v (1, 1); loss = sum of both (loss='output'); AdamW with W and v in two parameter groups.
# Three steps, stop mid-epoch, save, then two more steps uninterrupted vs resumed.
#
# Lane fable-finalize (2026-11-25), three guarantees more, all checked here:
#   THE REPLACEMENT: the checkpoint the fresh process resumes from is read through a HARD LINK taken on the first
#     checkpoint before a SECOND save to the same name. A save replaces the file (a new file renamed over the name)
#     and never writes into the previous one, so the link still holds the first checkpoint: an interrupted save
#     cannot destroy the last good checkpoint.
#   THE SEAL: the shards are sealed (`BatchSource.seal`: each shard's content digest in index.txt), and a resume on a
#     twin directory with other step values under the same record lengths is refused by name.
#   THE FIELDS: a resume with a source that feeds other input fields is refused by name.
#
# Sabotage arms (tools/identity_lanes/coverage_sabotage.tsv): batch_source_read_ahead (the saved position is one
# batch past the consumed one: the resumed next batch ids REFUSE) and program_training_drop_group (the resume restores group
# 0's state only: v's update moves and the parameter check REFUSES), program_training_in_place (the second save
# overwrites the first checkpoint's file: the link holds the second checkpoint and the resumed batch ids REFUSE) and
# program_training_source_unbound (the twin and the remapped source are accepted: the refusal checks REFUSE), each
# on one side.

_BSRC_LENGTHS = ((1, 3, 2, 4), (2, 1, 3))
_BSRC_F = 2
_BSRC_SETTINGS = dict(inputs=("steps", "lengths"), shuffle=True, seed=5, bucket_batches=2)

_BSRC_CHILD = r'''
import sys
import numpy as np
import mojolearn as ml
ckpt, shards, side, out = sys.argv[1:5]
src = ml.ops.BatchSource(shards, 2, inputs=("steps", "lengths"), shuffle=True, seed=5, bucket_batches=2)
with ml.ops.Session(numeric_mode="identical", device=side) as s:
    state = s.load_training(ckpt, source=src)
    prog, opt, ps = state["program"], state["optimizer"], state["parameters"]
    rows = {}
    for k in range(2):  # two resumed steps
        rows[f"ids{k}"] = np.asarray(src.next_records(), np.int64)
        prog.fit(source=src, optimizer=opt, parameters=ps, loss="output", steps=1)
        rows[f"curve{k}"] = np.asarray(prog.loss_curve_, np.float32)
        for j, h in enumerate(ps):
            rows[f"p{k}_{j}"] = np.asarray(s.numpy(h))
np.savez(out, **rows)
'''


def _bsrc_write(ml, tmp, name="shards", draw="steps"):
    """A sealed shard directory: two shards, float32 steps from the lane's fixed draw `draw`, int64 offsets, then
    `BatchSource.seal`. Another `draw` gives other step values under the same record lengths."""
    root = os.path.join(tmp, name)
    os.makedirs(root)
    lines = ["mojolearn-batch-shards v1"]
    for k, lengths in enumerate(_BSRC_LENGTHS):  # driver-loop: over the two shards (fixture)
        stem = f"part{k}"
        steps = _hw((sum(lengths), _BSRC_F), f"ops-batch-source:{draw}{k}", -1.0, 1.0)
        offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        np.save(os.path.join(root, stem + ".steps.npy"), np.ascontiguousarray(steps, np.float32))
        np.save(os.path.join(root, stem + ".offsets.npy"), offsets)
        lines.append("shard " + stem)
    with open(os.path.join(root, "index.txt"), "w", encoding="ascii") as fh:
        fh.write("\n".join(lines) + "\n")
    if ml.ops.BatchSource.seal(root) != len(_BSRC_LENGTHS):
        raise ValueError("ops-batch-source-resume: BatchSource.seal did not seal every shard")
    return root


def _bsrc_must_refuse(what, needle, call):
    """`call` must raise a ValueError naming `needle`; anything else REFUSES the lane."""
    try:
        call()
    except ValueError as exc:
        if needle in str(exc):
            return
        raise ValueError(f"ops-batch-source-resume: {what} was refused with other words: {exc}") from None
    raise ValueError(f"ops-batch-source-resume: {what} was accepted; it must be refused by name")


def _bsrc_program(ml, c):
    """The captured model on inference session c and its two parameters (W, v)."""
    w = c.tensor(_hw((_BSRC_F, 1), "ops-batch-source:w", -0.5, 0.5))
    v = c.tensor(_hw((1, 1), "ops-batch-source:v", -0.5, 0.5))
    xs = c.tensor(np.zeros((2, 4, _BSRC_F), np.float32))
    xl = c.tensor(np.zeros((2, 1), np.float32))
    d = c.dropout(c.reshape(xs, (-1, _BSRC_F)), 0.5, seed=11, counter=0)
    loss = c.add(c.sum(c.matmul(d, w)), c.sum(c.matmul(xl, v)))
    prog = c.capture(inputs=[xs, xl], outputs=[loss])
    opt = c.param_group_optimizer([{"params": [w], "lr": 0.05}, {"params": [v], "lr": 0.02, "weight_decay": 0.0}],
                                  "adamw", weight_decay=0.01)
    return prog, opt, [w, v]


@lane("ops-batch-source-resume")
def _(ml, X, yc, yr, Xh=None):
    """Streamed batches from a two-shard directory (BatchSource: shuffle, length buckets, per-batch padding),
    Program.fit(source=) three steps (mid-epoch), Program.save_training, then two
    steps uninterrupted against Session.load_training + two steps in a fresh process: the same next batch ids, the
    same step losses (the dropout masks) and the same parameters, bit for bit. Parts: stream (the first three steps'
    curve and the parameters at the stop), resume (the resumed batch ids, losses and parameters)."""
    import subprocess
    import sys
    parts = {}
    with tempfile.TemporaryDirectory(prefix="ib-ops-batch-source-") as tmp:
        root = _bsrc_write(ml, tmp)
        twin = _bsrc_write(ml, tmp, "twin", "other-steps")
        ckpt = os.path.join(tmp, "run.ckpt")
        first_ckpt = os.path.join(tmp, "first.ckpt")
        straight = {}
        with ml.ops.Session(numeric_mode="identical") as c:
            prog, opt, params = _bsrc_program(ml, c)
            order = list(prog.params)  # glue: the program's parameter order (the checkpoint's)
            src = ml.ops.BatchSource(root, 2, **_BSRC_SETTINGS)
            if src.n_batches != 4:
                raise ValueError(f"ops-batch-source-resume: the fixture has {src.n_batches} batches an epoch, not 4")
            prog.fit(source=src, optimizer=opt, parameters=params, loss="output", steps=3)
            first = np.asarray(prog.loss_curve_, np.float32)
            stopped = [np.asarray(c.numpy(h)) for h in order]  # glue: two parameters
            if src.state_dict()["consumed"] != 3:
                raise ValueError("ops-batch-source-resume: after three steps the source's consumed position is "
                                 f"{src.state_dict()['consumed']}, not 3")
            prog.save_training(ckpt, optimizer=opt, source=src)
            os.link(ckpt, first_ckpt)  # the first checkpoint's file, by a second name (module note THE REPLACEMENT)
            for k in range(2):  # driver-loop: over the two uninterrupted steps
                straight[f"ids{k}"] = np.asarray(src.next_records(), np.int64)
                prog.fit(source=src, optimizer=opt, parameters=params, loss="output", steps=1)
                straight[f"curve{k}"] = np.asarray(prog.loss_curve_, np.float32)
                for j, h in enumerate(order):  # driver-loop: over the two parameters
                    straight[f"p{k}_{j}"] = np.asarray(c.numpy(h))
            # the second save to the same name: it must replace the file, never write into the first one
            prog.save_training(ckpt, optimizer=opt, source=src)
        with ml.ops.Session(numeric_mode="identical") as d:
            other = ml.ops.BatchSource(twin, 2, **_BSRC_SETTINGS)
            if (other.n_records, other.n_batches, other.max_length) != (src.n_records, src.n_batches, src.max_length):
                raise ValueError("ops-batch-source-resume: the twin directory must have the trained shards' shape")
            _bsrc_must_refuse("a resume on same-shaped replacement data", "not the data the checkpoint was saved with",
                              lambda: d.load_training(first_ckpt, source=other))
            masked = ml.ops.BatchSource(root, 2, **dict(_BSRC_SETTINGS, inputs=("steps", "mask")))
            _bsrc_must_refuse("a resume with other input fields", "are not the fields the checkpoint was trained on",
                              lambda: d.load_training(first_ckpt, source=masked))
        script = os.path.join(tmp, "resume_child.py")
        with open(script, "w", encoding="utf-8") as fh:
            fh.write(_BSRC_CHILD)
        out = os.path.join(tmp, "resumed.npz")
        done = subprocess.run([sys.executable, script, first_ckpt, root, _lane_side(), out], capture_output=True,
                              text=True, timeout=900, check=False)
        if done.returncode != 0:
            raise ValueError("ops-batch-source-resume: the fresh resume process failed: "
                             + (done.stderr or done.stdout)[-2000:])
        with np.load(out) as z:
            resumed = {k: np.array(z[k]) for k in z.files}  # glue: the child's arrays
    for k in range(2):  # driver-loop: over the two resumed steps
        _same_bytes(f"uninterrupted batch {k} record ids", straight[f"ids{k}"],
                    f"resumed batch {k} record ids", resumed[f"ids{k}"])
        _same_bytes(f"uninterrupted step {k} loss (dropout masks)", straight[f"curve{k}"],
                    f"resumed step {k} loss (dropout masks)", resumed[f"curve{k}"])
        for j in range(2):  # driver-loop: over the two parameters
            _same_bytes(f"uninterrupted step {k} parameter {j}", straight[f"p{k}_{j}"],
                        f"resumed step {k} parameter {j}", resumed[f"p{k}_{j}"])
    parts["stream"] = _h(first, *stopped)
    parts["resume"] = _h(*[resumed[k] for k in sorted(resumed)])
    return _fit(parts)


_batch_decl("n/a:training-step (streamed batches from a shard directory, a mid-epoch one-call checkpoint and a "
            "fresh-process resume; digested whole)", "ops-batch-source-resume")
