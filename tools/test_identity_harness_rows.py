# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""The identity harness's whole-array row form against the per-row form it
replaced (2026-11-06, lane/harness-per-row-python).

The per-row implementation is kept HERE, verbatim, as the oracle
(`_old_*`). Every test runs the old and the new code side by side on the
same synthetic outputs and asserts the same digest bytes, the same verdict
string and the same raised error (type and text). NumPy only: no Mojolearn
import and no library call; seconds on one core.

Run: `python -m pytest -q tools/test_identity_harness_rows.py`, or
`python tools/test_identity_harness_rows.py` where pytest is not installed.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import identity_break as ib  # noqa: E402


# ---------------------------------------------------------------- the oracle
# tools/identity_break.py at a7a445563e (integ/perf-20261105), verbatim but
# for the names.

def _old_as_rows(outputs, n, what):
    rows = [[] for _ in range(n)]
    for k, o in enumerate(outputs):
        if isinstance(o, ib._PerRow):
            if len(o) != n:
                raise ValueError(f"{what}: output {k} has {len(o)} rows, expected {n}")
            items = [np.ascontiguousarray(np.asarray(v)) for v in o]
        else:
            a = np.asarray(o)
            if a.ndim == 0 or a.shape[0] != n:
                raise ValueError(f"{what}: output {k} has shape {a.shape}, expected leading axis {n}")
            items = [np.ascontiguousarray(a[i]) for i in range(n)]
        for i, v in enumerate(items):
            if v.dtype == object or v.dtype.kind in "OUSV":
                raise TypeError(f"{what}: output {k} is dtype={v.dtype}; pass the numeric arrays")
            rows[i].append((v.dtype.str, v.shape, v.tobytes()))
    return rows


def _old_row_mismatch(whole, other, name_other):
    for k, ((dw, sw, bw), (do, so, bo)) in enumerate(zip(whole, other)):
        if dw != do or sw != so:
            return f"output {k}: whole {dw}{list(sw)} vs {name_other} {do}{list(so)}"
        if bw != bo:
            size = np.dtype(dw).itemsize
            first = next(j for j in range(len(bw)) if bw[j] != bo[j])
            e = first // size
            return (f"output {k} element {e}: whole {ib._elem_hex(bw, size, e)} "
                    f"vs {name_other} {ib._elem_hex(bo, size, e)}")
    if len(whole) != len(other):
        return f"whole has {len(whole)} outputs, {name_other} has {len(other)}"
    return None


def _old_sabotage_rows(rows):
    for kinds in ("f", "iub"):
        for r in rows:
            for k, (dt, shape, b) in enumerate(r):
                if np.dtype(dt).kind not in kinds or not b:
                    continue
                flipped = bytearray(b)
                flipped[0] ^= 1
                r[k] = (dt, shape, bytes(flipped))
                return


def _old_eval_batch_rows(call, alone, sabotage, digest):
    n, mb = call.R.shape[0], call.min_batch
    if n < mb:
        raise ValueError(f"{call.label}: {n} rows is below the declared min_batch {mb}")
    if mb > 1:
        try:
            call.fn(np.ascontiguousarray(call.R[:mb - 1]))
        except Exception as exc:
            if call.refusal not in str(exc):
                raise RuntimeError(f"{call.label}: a batch of {mb - 1} raised something other than the declared "
                                   f"refusal {call.refusal!r}: {type(exc).__name__}: {exc}") from None
        else:
            raise RuntimeError(f"{call.label}: a batch of {mb - 1} no longer refuses ({call.refusal!r} was "
                               f"declared); the refusal was lifted, so drop min_batch and ask rows alone")
    whole = _old_as_rows(call.fn(np.ascontiguousarray(call.R)), n, f"{call.label} whole")
    if sabotage:
        _old_sabotage_rows(whole)
    for i in range(min(alone, n)):
        a = min(i, n - mb)
        one = _old_as_rows(call.fn(np.ascontiguousarray(call.R[a:a + mb])), mb,
                           f"{call.label} row {i} alone" if mb == 1 else f"{call.label} rows [{a},{a + mb})")
        where = "alone" if mb == 1 else f"in a batch of {mb}"
        m = _old_row_mismatch(whole[i], one[i - a], where)
        if m:
            return f"BATCH_MOVED:{call.label}:row {i} of {n} {where}:{m}"
    bounds = ib._split_bounds(n, mb)
    got = []
    for a, b in zip(bounds, bounds[1:]):
        got.extend(_old_as_rows(call.fn(np.ascontiguousarray(call.R[a:b])), b - a, f"{call.label} chunk [{a},{b})"))
    split_name = "split " + ",".join(str(b - a) for a, b in zip(bounds, bounds[1:]))
    for i in range(n):
        m = _old_row_mismatch(whole[i], got[i], split_name)
        if m:
            return f"BATCH_MOVED:{call.label}:row {i} of {n} in {split_name}:{m}"
    digest.update((f"rows:{call.label}:{n}" + (f":min_batch={mb}" if mb > 1 else "")).encode())
    for r in whole:
        for dt, shape, b in r:
            digest.update(f"{dt}{shape}".encode())
            digest.update(b)
    return None


def _old_eval_scale_rows(call, sabotage, digest, notes):
    n = call.R.shape[0]
    whole = _old_as_rows(call.fn(call.R), n, f"{call.label} whole B={n}")
    if sabotage:
        _old_sabotage_rows(whole)
    asked = []
    for b in ib.SCALE_BATCHES:
        if b > n:
            notes.append(f"{call.label}: B={b} n/a:the whole call is {n} rows, narrower than the sub-batch")
            continue
        asked.append(b)
        for a in sorted({0, (n - b) // 2, n - b}):
            got = _old_as_rows(call.fn(np.ascontiguousarray(call.R[a:a + b])), b, f"{call.label} B={b} at {a}")
            for i in range(b):
                m = _old_row_mismatch(whole[a + i], got[i], f"B={b} starting at row {a}")
                if m:
                    return f"BATCH_MOVED:{call.label}:row {a + i} of {n} in B={b} starting at row {a}:{m}"
    digest.update(f"scale:{call.label}:{n}:{','.join(map(str, asked))}".encode())
    for r in whole:
        for dt, shape, raw in r:
            digest.update(f"{dt}{shape}".encode())
            digest.update(raw)
    return None


def _old_first_byte(x, y):
    return next(j for j in range(len(x)) if x[j] != y[j])


def _old_first_nonzero(b):
    return next(j for j, c in enumerate(b) if c)


# ---------------------------------------------------------------- the matrix

_RNG = np.random.default_rng(20261106)

_F32_SPECIAL = np.array([0x7f800001, 0x7fc00123, 0xffa00000, 0xffffffff, 0x80000000, 0x7f800000, 0xff800000,
                         0x00000001, 0x3f800000], dtype=np.uint32).view(np.float32)
_F64_SPECIAL = np.array([0x7ff0000000000001, 0x7ff8000000000abc, 0xfff4000000000000, 0x8000000000000000,
                         0x7ff0000000000000, 0xfff0000000000000, 0x0000000000000001, 0x3ff0000000000000],
                        dtype=np.uint64).view(np.float64)
_F16_SPECIAL = np.array([0x7c01, 0x7e12, 0xfd00, 0x8000, 0x7c00, 0xfc00, 0x0001, 0x3c00], dtype=np.uint16).view(np.float16)


def _fill(n, shape, dtype):
    """n rows of `shape` in `dtype`, the special values spread through."""
    size = n * int(np.prod(shape, dtype=np.int64))
    dt = np.dtype(dtype)
    if dt.kind == "f":
        base = {2: _F16_SPECIAL, 4: _F32_SPECIAL, 8: _F64_SPECIAL}.get(dt.itemsize)
        vals = _RNG.standard_normal(size).astype(dt)
        if base is not None and size:
            idx = _RNG.integers(0, size, size=min(size, 3 * base.size))
            vals[idx] = np.resize(base.astype(dt.newbyteorder("=")).view(dt.newbyteorder("=")), idx.size)
    elif dt.kind == "c":
        vals = (_RNG.standard_normal(size) + 1j * _RNG.standard_normal(size)).astype(dt)
        if size:
            vals.view(np.dtype(f"f{dt.itemsize // 2}"))[:2] = [np.nan, -0.0]
    elif dt.kind == "b":
        # bytes 0, 1 and 2 (a uint8 view): a 1-D row boxes 2 as True (1)
        vals = _RNG.integers(0, 3, size=size, dtype=np.uint8).view(np.bool_)
    elif dt.kind in "iu":
        info = np.iinfo(dt)
        vals = _RNG.integers(info.min, info.max, size=size, dtype=dt, endpoint=True)
    elif dt.kind in "mM":
        vals = _RNG.integers(-2**40, 2**40, size=size).astype(np.int64).view(dt)
        if size:
            vals[0] = np.array("NaT", dtype=dt)
    else:
        raise AssertionError(dtype)
    return np.asarray(vals).astype(dtype, copy=False).reshape((n,) + tuple(shape))


NUMERIC = ["<f2", "<f4", "<f8", "<c8", "<c16", "|i1", "<i2", "<i4", "<i8", "|u1", "<u2", "<u4", "<u8", "|b1",
           "<M8[ns]", "<M8[D]", "<m8[s]"]
SHAPES = [(), (3,), (0,), (2, 2), (1,)]


def _arrays(n):
    """(label, array) over every numeric dtype and row shape, plus the
    layouts and byte orders the old row form handled differently."""
    for dt in NUMERIC:
        for sh in SHAPES:
            yield f"{dt}{sh}", _fill(n, sh, dt)
    base = _fill(n, (4,), "<f8")
    yield "fortran", np.asfortranarray(base)
    yield "reversed", base[::-1]
    yield "column-slice", base[:, 1:3]
    yield "strided-1d", _fill(2 * n, (), "<f4")[::2]
    yield ">f8 1-D", _fill(n, (), "<f8").astype(">f8")
    yield ">f8 2-D", _fill(n, (3,), "<f8").astype(">f8")
    yield ">i4 1-D", _fill(n, (), "<i4").astype(">i4")
    yield "longdouble 1-D", _fill(n, (), "<f8").astype(np.longdouble)
    yield "longdouble 2-D", _fill(n, (2,), "<f8").astype(np.longdouble)
    yield "clongdouble 1-D", _fill(n, (), "<c16").astype(np.clongdouble)
    yield "object floats 1-D", np.array([float(v) for v in _fill(n, (), "<f8")] or [], dtype=object)
    yield "object ints 1-D", np.array(list(range(n)), dtype=object)
    yield "matrix", np.asmatrix(_fill(n, (3,), "<f4")) if n else np.zeros((0, 3), dtype=np.float32)
    yield "masked", np.ma.masked_array(_fill(n, (2,), "<f8"), mask=False)


REFUSED = ["<U3", "|S2", "object strings", "V8", "bad leading axis", "0-d", "ragged list", "per-row text"]


def _refused(label, n):
    if label == "<U3":
        return np.array(["abc"] * n)
    if label == "|S2":
        return np.array([b"ab"] * n)
    if label == "object strings":
        return np.array(["x"] * n, dtype=object)
    if label == "V8":
        return np.zeros(n, dtype="V8")
    if label == "bad leading axis":
        return np.zeros((n + 1, 2))
    if label == "0-d":
        return np.float64(1.0)
    if label == "ragged list":
        return [np.zeros(i % 3) for i in range(n)]
    if label == "per-row text":
        return ib._PerRow(np.array(["a"]) for _ in range(n))
    raise AssertionError(label)


def _per_row(n, dtype="<f4"):
    return ib._PerRow(np.ascontiguousarray(_fill(i % 4, (), dtype)) for i in range(n))


def _outcome(fn):
    try:
        return ("ok", fn())
    except Exception as exc:  # the type and text the record would carry
        return ("raise", type(exc).__name__, str(exc))


def _hex(d):
    return d.hexdigest()


# ---------------------------------------------------------------- row form

def _check_rows(outputs, n, sabotage=False):
    def old():
        rows = _old_as_rows(outputs, n, "probe")
        if sabotage:
            _old_sabotage_rows(rows)
        d = hashlib.sha256()
        for r in rows:
            for dt, shape, b in r:
                d.update(f"{dt}{shape}".encode())
                d.update(b)
        return rows, d.hexdigest()

    def new():
        rows = ib._as_rows(ib._host_outputs(outputs), n, "probe")
        if sabotage:
            ib._sabotage_rows(rows)
        d = hashlib.sha256()
        ib._digest_rows(d, rows)
        return list(rows), d.hexdigest()

    o, w = _outcome(old), _outcome(new)
    assert o == w, (o[:1], w[:1], o[1:] if o[0] == "raise" else "", w[1:] if w[0] == "raise" else "")
    return o


def test_row_form_and_digest_every_dtype_and_shape():
    checked = 0
    for n in (0, 1, 2, 7, 33):
        for label, a in _arrays(n):
            for sab in (False, True):
                _check_rows((a,), n, sab)
                checked += 1
    assert checked > 500


def test_row_form_tuples_of_outputs_and_per_row_outputs():
    for n in (0, 1, 5, 17):
        arrays = dict(_arrays(n))
        combos = [
            (arrays["<f8(3,)"], arrays["<i8()"]),
            (arrays["|b1()"], arrays["<f4(2, 2)"], arrays["<u2(0,)"]),
            (arrays["<f4(0,)"], arrays["<i4()"]),  # sabotage skips the empty float rows
            (arrays["<i8(3,)"],),  # no float output: the integer one is flipped
            (_per_row(n), _per_row(n, "<i8")),
            (arrays["<f8()"], _per_row(n)),
            (_per_row(n, "<i4"), arrays["<f2(3,)"]),
            (arrays["object floats 1-D"], arrays["<f4()"]),
            (),
        ]
        for outs in combos:
            for sab in (False, True):
                _check_rows(outs, n, sab)


def test_refusals_raise_the_same_error_in_the_same_order():
    for n in (0, 1, 4):
        for label in REFUSED:
            bad = _refused(label, n)
            good = _fill(n, (2,), "<f8")
            for outs in ((bad,), (good, bad), (bad, good), (_refused("bad leading axis", n), bad)):
                _check_rows(outs, n)


# ---------------------------------------------------------------- the batch protocol

class _Fn:
    """A row-wise call over R: outputs per row from the row's own bytes, with
    an optional fault keyed on the batch asked (start row, length)."""

    def __init__(self, R, make, fault=None):
        self.R, self.make, self.fault = R, make, fault

    def __call__(self, r):
        start = int(np.flatnonzero((self.R == r[:1]).all(axis=1))[0]) if r.shape[0] else 0
        outs = list(self.make(r))
        if self.fault is not None:
            outs = self.fault(start, r.shape[0], outs)
        return tuple(outs)


def _makers():
    def f64(r):
        return (np.ascontiguousarray(r[:, :3], dtype=np.float64) * 1.5,)

    def f32_1d(r):
        return (r[:, 0].astype(np.float32),)

    def two(r):
        return (r[:, :2].astype(np.float32), (r[:, 0] * 10).astype(np.int64))

    def proba_bool(r):
        return ((r[:, 0] > 0), r[:, :4].astype(np.float64))

    def per_row(r):
        return (ib._PerRow(np.arange(int(abs(v) * 3) % 4, dtype=np.int64) for v in r[:, 0]),
                r[:, 1].astype(np.float32))

    def empty_rows(r):
        return (np.zeros((r.shape[0], 0), dtype=np.float32),)

    return dict(f64=f64, f32_1d=f32_1d, two=two, proba_bool=proba_bool, per_row=per_row, empty_rows=empty_rows)


def _faults():
    def flip_byte(row_in_batch, k=0):
        def fault(start, length, outs):
            o = outs[k]
            if isinstance(o, ib._PerRow) or length <= row_in_batch:
                return outs
            o = np.array(o, copy=True)
            o.reshape(length, -1).view(np.uint8)[row_in_batch, -1] ^= 0x10
            outs[k] = o
            return outs
        return fault

    def when(pred, fault):
        return lambda s, L, outs: fault(s, L, outs) if pred(s, L) else outs

    def dtype_change(s, L, outs):
        outs[0] = np.asarray(outs[0]).astype(np.float16) if not isinstance(outs[0], ib._PerRow) else outs[0]
        return outs

    def drop_output(s, L, outs):
        return outs[:-1]

    def extra_output(s, L, outs):
        return outs + [np.zeros(L)]

    def wrong_rows(s, L, outs):
        outs[0] = np.zeros((L + 1, 2))
        return outs

    def raise_it(s, L, outs):
        raise RuntimeError("Parameter n_rows: number of rows cannot be less than two")

    return {
        "none": None,
        "alone row 3": when(lambda s, L: L == 1 and s == 3, flip_byte(0)),
        "rest chunk row 40": when(lambda s, L: s == 8, flip_byte(32)),
        "chunk of 7 last row": when(lambda s, L: s == 1 and L == 7, flip_byte(6)),
        "rest chunk last row output 1": when(lambda s, L: s == 8, flip_byte(-1 % 2, k=-1)),
        "dtype in small batches": when(lambda s, L: L < 8, dtype_change),
        "dtype in rest chunk": when(lambda s, L: s == 8, dtype_change),
        "drops an output alone": when(lambda s, L: L == 1, drop_output),
        "extra output in rest": when(lambda s, L: s == 8, extra_output),
        "wrong rows in chunk": when(lambda s, L: s == 1 and L == 7, wrong_rows),
        "raises alone": when(lambda s, L: L == 1, raise_it),
        "raises below 2": when(lambda s, L: L < 2, raise_it),
    }


def _run_batch(evaluate, R, make, fault, alone, sabotage, mb=1, refusal=None):
    call = ib._BatchRows("predict", R, _Fn(R, make, fault), mb, refusal)
    d = hashlib.sha256()

    def go():
        return evaluate(call, alone, sabotage, d), d.hexdigest()
    return _outcome(go)


def test_batch_rows_same_verdict_and_digest():
    makers, faults = _makers(), _faults()
    checked = moved = 0
    for n in (1, 2, 9, 60):
        R = np.ascontiguousarray(_RNG.standard_normal((n, 6)).astype(np.float32))
        for mname, make in makers.items():
            for fname, fault in faults.items():
                for alone in (0, 16):
                    for sab in (False, True):
                        for mb, refusal in ((1, None), (2, "cannot be less than two")):
                            if mb > n:
                                continue
                            o = _run_batch(_old_eval_batch_rows, R, make, fault, alone, sab, mb, refusal)
                            w = _run_batch(ib._eval_batch_rows, R, make, fault, alone, sab, mb, refusal)
                            assert o == w, (n, mname, fname, alone, sab, mb, o, w)
                            checked += 1
                            moved += o[0] == "ok" and bool(o[1][0])
    assert checked > 1000 and moved > 100, (checked, moved)


def test_scale_rows_same_verdict_and_digest():
    makers = _makers()
    faults = {
        "none": None,
        "B=17 middle": lambda s, L, outs: _flip_at(outs, 5) if (L == 17 and s > 0) else outs,
        "B=256 end": lambda s, L, outs: _flip_at(outs, 255) if (L == 256 and s > 0) else outs,
        "dtype B=1": lambda s, L, outs: [np.asarray(outs[0]).astype(np.float16)] + list(outs[1:]) if L == 1 else outs,
    }
    for n in (20, ib.SCALE_WHOLE):
        R = np.ascontiguousarray(_RNG.standard_normal((n, 6)).astype(np.float32))
        for mname in ("f64", "two", "per_row", "f32_1d"):
            for fname, fault in faults.items():
                for sab in ("", "1"):
                    res = []
                    for ev in (_old_eval_scale_rows, ib._eval_scale_rows):
                        call = ib._ScaleRows("probe", R, _Fn(R, makers[mname], fault))
                        d, notes = hashlib.sha256(), []
                        res.append((_outcome(lambda: ev(call, sab, d, notes)), d.hexdigest(), notes))
                    assert res[0] == res[1], (n, mname, fname, sab, res)


def _flip_at(outs, i):
    o = outs[0]
    if isinstance(o, ib._PerRow) or np.asarray(o).shape[0] <= i:
        return outs
    o = np.array(o, copy=True)
    o.reshape(o.shape[0], -1).view(np.uint8)[i, 0] ^= 1
    return [o] + list(outs[1:])


# ---------------------------------------------------------------- the byte searches

def test_first_differing_byte_matches_the_per_byte_walk():
    for size in (1, 2, 7, 64, 1000):
        for _ in range(50):
            x = _RNG.integers(0, 256, size=size, dtype=np.uint8).tobytes()
            y = bytearray(x)
            for j in _RNG.choice(size, size=min(size, int(_RNG.integers(1, 4))), replace=False):
                y[j] ^= int(_RNG.integers(1, 256))
            y = bytes(y)
            assert ib._first_byte_diff(x, y) == _old_first_byte(x, y)
            z = bytes(size // 2) + b"\x01" + bytes(size)
            assert int(np.flatnonzero(np.frombuffer(z, dtype=np.uint8))[0]) == _old_first_nonzero(z)


def test_first_diff_message_unchanged():
    for dt in ("<f4", "<f8", "<i2", "|u1"):
        a = _fill(6, (3,), dt)
        b = a.copy()
        b.reshape(-1).view(np.uint8)[11] ^= 4
        old_e = _old_first_byte(a.tobytes(), b.tobytes()) // a.dtype.itemsize
        msg = ib._first_diff(a, b)
        assert msg.startswith(f"element {old_e}: whole "), msg


# ---------------------------------------------------------------- the stage clock

def test_harness_work_is_booked_beside_the_stage_not_in_it():
    import time
    timer = ib.CellTimer("probe/base")
    with timer.stage("repeat=1/batch"):
        time.sleep(0.02)  # stands for the library call
        with ib._harness_work():
            time.sleep(0.03)
    with timer.stage("repeat=1/train"):
        pass
    st = timer.result()["stages"]
    assert st["repeat=1/batch"] >= 0.015 and st["repeat=1/batch-harness"] >= 0.029, st
    # the stage is the wall less the harness: never the harness's 30 ms plus the call's 20 ms
    assert st["repeat=1/batch"] < 0.02 + (timer.result()["total_seconds"] - 0.05), st
    assert "repeat=1/train-harness" not in st
    assert timer.result()["total_seconds"] >= st["repeat=1/batch"] + st["repeat=1/batch-harness"]
    assert ib.timing_environment()["harness_timing"] == ib.HARNESS_TIMING_VERSION == 2


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"{len(tests)} passed")
