# Landing numpy-free-0.7, and what each stage still owes

2026-09-10. `numpy-free-0.7` was cut at `fcdcabb3` on 2026-09-07 and then sat
behind `main` while the identical fan-out landed, reaching 91 commits behind
`origin/main` by 2026-09-10. It cannot land as one merge, because `main`
moves through the same Python files faster than the branch can be rebased
onto it. This file records the staged landing instead.

## The split

Of the 61 files the branch touches, measured against `origin/main`:

| set | count | what it is |
|---|---|---|
| new modules | 5 | files `main` does not have at all |
| converted, uncontested | 34 | `main` has them and has not touched them since the cut |
| contested | 20 | both sides changed them |

## Stage 1, landed by this commit

The four new modules (`_array.py`, `_buffer.py`, `_bufcheck.py`,
`_labels.py`) and this contract. They import only the standard library and
each other, and nothing on `main` imports THEM, so landing them changes no
behavior and cannot move a bit of any IDENTICAL result. They exist here so
they stop drifting while the rest is staged.

Gate: exercised standalone against NumPy as the oracle, 15 checks green.
Bit-exact conversions (`as_f32_colmajor` and `as_f32_c` byte-equal to
`numpy.asfortranarray` / `ascontiguousarray` at float32), the zero-copy
borrow for float32 inputs already in the target order, `memcopy`, slicing,
reshape, `all_finite`, and both refusals by name.

The branch's own `test_numpy_free_core.py` and `test_native_helpers.py` are
NOT here. They import `_arrays` and `_serialize` in their branch form, which
are contested files, so on `main` they would fail. They land with stage 3.

## Stage 2, the 34 uncontested conversions

Behavioral. Each one changes a module's returns from `numpy.ndarray` to
`mojolearn.Array`, so each needs its surface test run before it lands.
RUN OWED.

Stage 2 was gated on the converter tax and THAT GATE IS CLEARED
(2026-09-10, e9f40d69, 313ce4a1, 485caa24). The pure-Python converter cost
1,235 ms for a float64 cast NumPy does in 5.7 ms at 2M x 20; the native
host converters of DEVIATION 2470-2472 in `bindings/_mojolearn.mojo`,
selected by `_buffer._convert` whenever the base binding is built, bring
it to NumPy's number on the flat cast and 2.5x to 5.2x FASTER than NumPy
on every layout flip. Numbers in
`bench/results/native_convert_2026-09-10/run5_m4_warmup.txt`; the brief
that produced them is `docs/lanes/BRIEF_native_convert_2026-09-10.md`.
Without the binding the pure-Python path still runs and produces the same
bytes (119 tests, both arms).

## Stage 3, the 20 contested files

These need a per-file merge against a `main` that keeps moving, so they
should land last and in small batches. A trial merge on 2026-09-10 resolved
all of them against `b3279609` in 20 hunks; that resolution is on branch
`numpy-free-onto-main-20260910` and is worth reading, but it is already
stale against `origin/main` and must be redone, not replayed.

Two resolutions from that trial are worth carrying forward.
`_transformer_impl.py` needs `main`'s sliding-window ring cache expressed
without the `ring[:, :, positions % w, :]` gather; the ring layout makes the
held positions at most two contiguous runs per (batch, kv head), so two
`memcopy`s do it. `_training_impl.py` keeps the branch's numpy-free
optimizer, clip and loss while inheriting `main`'s `lr_schedule` and
`accumulation_steps` validation.

## The three that are not a merge problem

These are shipped modules that will still import NumPy after every stage
above, because they are code `main` added or redesigned AFTER the branch was
cut. They need converting on their own terms.

**`_byte_lm_impl.py`** is a DESIGN conflict. The branch hard-codes
`_N = 34944` and a fixed `_OFFSETS` tuple for the fixed
B2/L32/DM32/H4/KV2/FF64/V256 model. `main` generalized the same file to
configurable shapes through `_byte_lm_config.ByteLanguageModelConfig`. The
generalization is the better design; the NumPy removal has to be rewritten
on top of it and the branch's version of this file must not be replayed.
This is the largest single owed item.

**`_training_impl.py`** gained four NumPy-based sections after the cut: the
LR schedules, gradient accumulation, the `Generator` RNG, and the
embedding/rms_norm/linear layer helpers.

**`_samba_impl.py`** did not exist when the branch was cut.

`_verify.py` and `transformer.py` import NumPy lazily behind a guard with an
install hint. That is the intended optional-diagnostic shape and is not an
owed conversion.

## The DEVIATION 1887 measurement, CLOSED

DEVIATION 1887 has a row-tile arm on `main` that the branch does not carry.
`_arrays.as_f32_colmajor` copies large C-order inputs in 256 KB row tiles for
cache locality; the pure-Python `_buffer.as_f32_colmajor` does not tile.
RESOLVED 2026-09-10 by moving the tiling into compiled code: the native
`cast_colmajor_f64_to_f32` and `transpose_f32` walk 128 x 64 tiles (64 KB
in, 32 KB out) and measure 24.5 ms against NumPy's 128 ms for the float64
case and 20.0 vs 53.7 ms for float32, at 2M x 20 on the M4, minimum of 9
interleaved pairs. The untiled Python path remains only as the fallback for
an unbuilt checkout. What is still owed is the estimator-level number: no
estimator on `main` calls `_buffer` yet, so the fit-time effect appears
only once stage 2 lands, and each converted estimator's surface run will
show it.
