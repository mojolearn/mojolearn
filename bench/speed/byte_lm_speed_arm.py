#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""OUR side of the byte-level language-model speed lanes: the IDENTICAL
`SmallByteLanguageModelTrainer`, timed on the same recipe the torch twin in
`tools/speed_torch_byte_lm.py` consumes.

    pixi run python bench/speed/byte_lm_speed_arm.py --lane lm-train
    MOJOLEARN_SPEED_ROUNDS=5 \\
      pixi run python bench/speed/byte_lm_speed_arm.py --lane lm-infer

AUTHORED, NOT EXECUTED. No model, benchmark, build or test was run to
produce this file. The trainer it drives is not yet in this worktree
(`python/mojolearn/_byte_lm_impl.py` and `python/mojolearn/language_model.py`
are merged in before anything runs), so the first run is a BUILD of the
`_mojolearn_byte_lm` binding on the box and its output is read as such.

THE QUESTION THIS FILE ANSWERS
-------------------------------
What the cross-vendor bitwise identity of the byte-LM trainer COSTS against
PyTorch on the same H100, for the same 34,944-parameter model, for training
(128 steps) and for a forward. This is the IDENTICAL arm by construction:
`SmallByteLanguageModelTrainer` refuses any tier but `identical`
(`python/mojolearn/_byte_lm_impl.py::_mode`), no mode is passed here, and
the header prints the tier `mojolearn.numeric_mode()` reads back after
import so a run of the wrong arm cannot be mislabeled. The opponent's file
holds the whole argument for the torch arms and the latency-not-throughput
caveat (DEVIATION 2191); it is not repeated here beyond the note every run
prints.

WHAT LIVES WHERE
-----------------
Everything shared -- the pinned corpus load and SHA check, the 128 training
batches, the eight held-out batches, the `INIT_ID` initializer reproduced
bit for bit, the held-out mean, the hash inputs, the `FSPEED-*` emitters --
lives in `tools/speed_torch_byte_lm.py` and imports NOTHING from
`mojolearn` and nothing from torch at import time. This file adds the
`ours` arm only. Both arms therefore build their initial parameters from
one function and their batches from one function, and the recipe note each
side prints carries `initial_parameters_sha256` so two runs whose recipes
drifted cannot be laid side by side unnoticed.

HOW WE ARE INVOKED, AND WHY THROUGH PYTHON
--------------------------------------------
Through `mojolearn.language_model.SmallByteLanguageModelTrainer`, because
that is the only surface the byte LM has: `bindings/_mojolearn_byte_lm.mojo`
exposes one `byte_lm_run`, and `python/mojolearn/_byte_lm_impl.py::_run`
is the caller every capture, oracle and installed-wheel witness goes
through (`tools/byte_lm_real_text_capture.py`,
`tools/byte_lm_installed_step.py`). Timing anything else would time a
program no user runs.

The cost of that surface is named rather than hidden (DEVIATION 2194).
Every `train_step` validates and COPIES the full state on the way in
(`_validate_state` copies parameters, m and v), hands the binding host
buffers, checks that no input buffer moved, validates the outputs, and
returns the 34,944 pre-update gradients as fresh arrays. That is Python and
memcpy work per step that torch's loop does not do, and it stays inside
the timer for the reason `forest_speed_arm.py` gives for its transpose: a
user of this surface pays it. The host-to-device copy of each step's token
batch is inside the timer on BOTH sides.

WHAT ONE ROUND IS
------------------
`lm-train` (shape `bytelm-2x33-steps128`): a FRESH trainer from the same
initial parameters, 128 `train_step` calls on the pinned schedule, timed
end to end with the host clock. `train_step` returns host arrays that the
wrapper has already inspected, so the call is complete on return and the
return IS the synchronization (the same argument `forest_speed_arm.py::
_our_sync` makes); per-step times are host wall-clock around each call
(DEVIATION 2193). The round hash is SHA256 of `parameters_` after step 128
(first 16 hex on the `FSPEED` line, full digest in a note, DEVIATION 2192),
and every round must repeat it: this arm is IDENTICAL, so a hash that
moves between rounds is a defect and is printed as one. The round also
writes the public checkpoint to a scratch directory and prints its byte
count and SHA256, so the number can be laid directly against the retained
three-vendor checkpoint file. Held-out loss is measured once on the fresh
initial trainer (`heldout_loss_initial`) and after every round
(`heldout_loss_final` is the last round's; a round that disagrees is
printed), exactly as `tools/byte_lm_real_text_capture.py::evaluate_heldout`
computes it: eight FP32 batch means, `math.fsum` over them, divided by 8.
`step_ms_median` is the median over every timed step of every timed round.

`lm-infer` (shape `bytelm-2x33-forward`): one `evaluate` over the first
held-out batch on the INITIAL parameters per round (DEVIATION 2199), the
round hash being SHA256 of the FP32 loss bits.

DEVIATION 2190: the optimizer is `lr=.003, betas=(.9, .999), eps=1e-8,
weight_decay=.01`, the values `tools/byte_lm_real_text_capture.py:348-349`
constructed the pinned run with, NOT the trainer's `lr=1e-3` default. The
constants live in the shared module so both arms move together.

DEVIATION 2202: the registry the shared module spells is cross-checked
against `SmallByteLanguageModelTrainer.parameter_registry()` before any
parameter is built, and a mismatch is a refusal, not a warning.
"""

import os
import statistics
import sys
import tempfile
import time

import numpy as np

# `tools/` is not a package; the recipe is imported by path, the way
# `bench/speed/forest_speed_arm.py` imports `speed_gbdt_arm`. `python/` goes
# on the path too so an in-repo, not-yet-installed `mojolearn` resolves.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (os.path.join(_ROOT, "tools"), os.path.join(_ROOT, "python")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import speed_torch_byte_lm as recipe        # noqa: E402

ARM = "ours"


def check_registry(cls):
    """DEVIATION 2202. `parameter_registry()` spells `size` where the capture
    tool spells `count` (tools/byte_lm_real_text_capture.py:329-330 does
    this same rename before `validate_registry`)."""
    theirs = [dict(name=x["name"], shape=list(x["shape"]), offset=x["offset"], count=x["size"])
              for x in cls.parameter_registry()]
    if theirs != recipe.registry():
        raise RuntimeError("SmallByteLanguageModelTrainer.parameter_registry() differs from the "
                           "20-tensor registry tools/speed_torch_byte_lm.py transcribed")


def check_config(trainer):
    """tools/byte_lm_real_text_capture.py:351-357: the state the trainer
    holds must be the pinned run's, including the FP32-rounded scalars."""
    state = trainer.state_dict()
    scalars = recipe.optimizer_scalars()
    expected = dict(kind=2, lr=scalars["lr"], beta1=scalars["betas"][0], beta2=scalars["betas"][1],
                    eps=scalars["eps"], weight_decay=scalars["weight_decay"],
                    momentum=0., dampening=0., nesterov=False, max_norm=0.)
    if (state["profile"] != recipe.PROFILE or state["config"] != expected
            or state["completed_steps"] != 0 or state["next_batch_index"] != 0):
        raise RuntimeError("trainer profile/optimizer/cursor differ from the pinned run")


def make_trainer(cls, initial, schedule):
    trainer = cls(initial, data_schedule=schedule, lr=recipe.LR, betas=recipe.BETAS,
                  eps=recipe.EPS, weight_decay=recipe.WEIGHT_DECAY)
    check_config(trainer)
    return trainer


def evaluate_heldout(trainer, heldout):
    """`tools/byte_lm_real_text_capture.py::evaluate_heldout` without the
    retention: eight `evaluate` calls, FP32 means, fsum / 8. The trainer
    itself refuses if an evaluation changed any state byte."""
    losses = []
    for ids in heldout:
        value = trainer.evaluate(ids)
        if not np.isfinite(value) or float(np.float32(value)) != value:
            raise RuntimeError("held-out loss is not an exact finite FP32 value")
        losses.append(value)
    return recipe.heldout_mean(losses)


def checkpoint_witness(trainer):
    """Public checkpoint bytes and SHA256, written to scratch and discarded;
    comparable to the retained three-vendor `final.checkpoint.json`."""
    with tempfile.TemporaryDirectory(prefix="bytelm-speed-") as directory:
        path = os.path.join(directory, "final.checkpoint.json")
        trainer.save_checkpoint(path)
        with open(path, "rb") as stream:
            raw = stream.read()
    return len(raw), recipe.sha256_hex(raw)


def run_train_lane(cls, lane, initial, schedule, batches, heldout, n_rounds, warmups):
    shape = recipe.SHAPE_TAGS[lane]

    def one_run(trainer):
        """(end-to-end ms, per-step ms). The call returns host arrays the
        wrapper already validated, so the return is the sync (DEVIATION 2193)."""
        step_ms = []
        t0 = time.perf_counter()
        for step in range(recipe.STEPS):
            s0 = time.perf_counter()
            result = trainer.train_step(batches[step])          # DEVIATION 2194: H2D inside
            step_ms.append((time.perf_counter() - s0) * 1000.0)
            if result["completed_steps"] != step + 1 or not np.isfinite(result["loss"]):
                raise RuntimeError("training cursor/loss mismatch at step %d" % (step + 1))
        total_ms = (time.perf_counter() - t0) * 1000.0
        if trainer.step_ != recipe.STEPS:
            raise RuntimeError("trainer did not complete %d steps" % recipe.STEPS)
        return total_ms, step_ms

    recipe.emit_acc(lane, ARM, "heldout_loss_initial",
                    evaluate_heldout(make_trainer(cls, initial, schedule), heldout))

    for _ in range(warmups):
        ms, _ = one_run(make_trainer(cls, initial, schedule))     # DEVIATION 2200
        recipe.emit_warmup(lane, ARM, shape, ms)

    all_step_ms, finals, hashes = [], [], []
    for index in range(1, n_rounds + 1):
        trainer = make_trainer(cls, initial, schedule)
        ms, step_ms = one_run(trainer)
        flat = trainer.parameters_
        if not np.isfinite(flat).all():
            raise RuntimeError("round %d produced nonfinite parameters" % index)
        digest = recipe.sha256_hex(recipe.flat_bytes(flat))
        hashes.append(digest)
        recipe.emit_round(lane, ARM, shape, index, ms, digest[:16])
        recipe.emit_hash_note(lane, ARM, index, "final_parameters", digest)
        recipe.emit_note(lane, ARM, "round=%d step_ms_median=%.4f step_ms_min=%.4f step_ms_max=%.4f"
                         % (index, statistics.median(step_ms), min(step_ms), max(step_ms)))
        size, checkpoint_sha = checkpoint_witness(trainer)
        recipe.emit_note(lane, ARM, "round=%d checkpoint_bytes=%d checkpoint_sha256=%s"
                         % (index, size, checkpoint_sha))
        all_step_ms.extend(step_ms)
        finals.append(evaluate_heldout(trainer, heldout))
    if len(set(hashes)) > 1:
        # This arm is IDENTICAL. Two rounds with two hashes is a defect in
        # the arm, not in the harness, and it is said in those words.
        recipe.emit_note(lane, ARM, "DEFECT: IDENTICAL arm hash moved across rounds: %s %s"
                         % (hashes[0][:16], next(h for h in hashes if h != hashes[0])[:16]))
    else:
        recipe.emit_note(lane, ARM, "hash repeated across %d rounds" % len(hashes))
    recipe.emit_acc(lane, ARM, "heldout_loss_final", finals[-1])
    if len(set(finals)) > 1:
        recipe.emit_note(lane, ARM, "DEFECT: heldout_loss_final moved across rounds: %r" % (finals,))
    recipe.emit_acc(lane, ARM, "step_ms_median", statistics.median(all_step_ms))


def run_infer_lane(cls, lane, initial, schedule, heldout, n_rounds, warmups):
    shape = recipe.SHAPE_TAGS[lane]
    trainer = make_trainer(cls, initial, schedule)
    ids = heldout[0]                                             # DEVIATION 2199

    def one_forward():
        t0 = time.perf_counter()
        value = trainer.evaluate(ids)                            # DEVIATION 2194: H2D inside
        return (time.perf_counter() - t0) * 1000.0, value

    for _ in range(warmups):
        ms, _ = one_forward()
        recipe.emit_warmup(lane, ARM, shape, ms)
    hashes, last = [], None
    for index in range(1, n_rounds + 1):
        ms, value = one_forward()
        digest = recipe.sha256_hex(recipe.loss_bytes(value))
        hashes.append(digest)
        last = value
        recipe.emit_round(lane, ARM, shape, index, ms, digest[:16])
    if len(set(hashes)) > 1:
        recipe.emit_note(lane, ARM, "DEFECT: IDENTICAL arm hash moved across rounds: %s %s"
                         % (hashes[0][:16], next(h for h in hashes if h != hashes[0])[:16]))
    else:
        recipe.emit_note(lane, ARM, "hash repeated across %d rounds" % len(hashes))
    recipe.emit_acc(lane, ARM, "forward_loss", last)
    recipe.emit_note(lane, ARM, "forward_loss_fp32_bits=0x%08x forward_loss_sha256=%s batch_start=%d"
                     % (int(np.asarray([np.float32(last)]).view(np.uint32)[0]), hashes[-1],
                        recipe.VALIDATION_STARTS[0]))


def main(argv=None):
    args = recipe.build_parser("byte_lm_speed_arm", with_arm=False).parse_args(argv)
    lane = args.lane
    n_rounds = recipe.round_count(args.rounds, lane)
    if args.warmups < 0:
        raise SystemExit("warmups must be nonnegative")

    # MODE FROM IMPORT. `MOJOLEARN_NUMERIC_MODE` is read by `mojolearn` at
    # import (python/mojolearn/_backend.py:714, unset selects `identical`),
    # and the trainer refuses any other tier, so the header can only ever
    # say IDENTICAL or the run refuses; both are printed, never assumed.
    try:
        import mojolearn
        from mojolearn.language_model import SmallByteLanguageModelTrainer as cls
        mode = str(mojolearn.numeric_mode()).upper()
    except Exception as exc:                        # noqa: BLE001
        recipe.emit_refused(lane, ARM, "mojolearn import: %s: %s"
                            % (exc.__class__.__name__, recipe.first_line(exc)))
        return 1

    try:
        check_registry(cls)
        raw, _manifest, manifest_raw = recipe.read_corpus(_ROOT)
        initial = recipe.initialize()
        schedule = recipe.data_schedule(raw, manifest_raw, initial)
        batches = recipe.train_batches(raw)
        heldout = recipe.heldout_batches(raw)
        probe = make_trainer(cls, initial, schedule)
        runtime = probe.run_metadata()          # binding constants only, no model call
    except Exception as exc:                        # noqa: BLE001
        # The EXPECTED failure on a box whose first CUDA build of the byte-LM
        # binding failed: a refusal line, not a traceback.
        recipe.emit_refused(lane, ARM, "%s: %s" % (exc.__class__.__name__, recipe.first_line(exc)))
        return 1
    if runtime["native_numeric_mode"] != 1 or runtime["native_profile"] != recipe.PROFILE:
        recipe.emit_refused(lane, ARM, "native binding witness differs: mode=%r profile=%r"
                            % (runtime["native_numeric_mode"], runtime["native_profile"]))
        return 1

    recipe.emit_header(lane, ARM, mode, recipe.device_string(), n_rounds)
    recipe.emit_note(lane, ARM, "native_vendor=%s binding_sha256=%s surface=mojolearn.language_model."
                                "SmallByteLanguageModelTrainer via _mojolearn_byte_lm.byte_lm_run"
                     % (runtime["native_vendor"], runtime["binding_sha256"]))
    scalars = recipe.optimizer_scalars()
    recipe.emit_note(lane, ARM, "recipe init=%s corpus_sha256=%s initial_parameters_sha256=%s steps=%d "
                                "optimizer=AdamW(lr=%r,betas=%r,eps=%r,weight_decay=%r)"
                     % (recipe.INIT_ID, recipe.CORPUS_SHA, recipe.sha256_hex(recipe.flat_bytes(initial)),
                        recipe.STEPS, scalars["lr"], scalars["betas"], scalars["eps"], scalars["weight_decay"]))
    recipe.emit_latency_note(lane, ARM)

    try:
        if lane == "lm-train":
            run_train_lane(cls, lane, initial, schedule, batches, heldout, n_rounds, args.warmups)
        else:
            run_infer_lane(cls, lane, initial, schedule, heldout, n_rounds, args.warmups)
    except Exception as exc:                        # noqa: BLE001
        recipe.emit_refused(lane, ARM, "%s: %s" % (exc.__class__.__name__, recipe.first_line(exc)))
        return 1
    print("FSPEED-DONE lane=%s arm=%s" % (lane, ARM))
    return 0


if __name__ == "__main__":
    sys.exit(main())
