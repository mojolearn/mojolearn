# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU training for the embedding, embedding-sort, ivf, ivf-euclidean, byte-lm
and byte-lm-resident lanes, checked at runtime where the host bindings are
built and the package took the CPU-only path (skipped, and SAID to be
skipped, when a binding is absent or a GPU set loaded).

Embedding: the host forward is the gather; the host backward is the ascending
sum with the padding row stored +0.0, the same bytes under both plans, and a
carried microbatch equals the unsplit gradient; an out-of-range id refuses by
name. IVF: a search at n_probes == n_lists returns the brute force neighbours
on distinct distances, the euclidean metric returns the root of the squared
one on the same ids, and n_probes > n_lists refuses by name. Byte LM: a
stateless and a resident trainer take the same three steps to the same loss,
parameters and gradient bytes; the trainer's logits are the CPU binding's
reference-path logits; an evaluation changes no state; a checkpoint round
trip restores the bytes; a session rollback restores the last step's shadow
once; a multi-GPU entry is absent by name. Each host binding's sabotage
read-back is off. The bit claim against the GPU columns is the verification's.

    cd python && python3 -m mojolearn.tests.test_cpu_training_embedding_ivf
"""

# Gate-runner scope: host runtime checks require the CPU-only route.
GATE_BACKENDS = ("cpu",)
import sys

import mojolearn
from mojolearn import _backend


def _cpu_only_with(basename):
    if _backend._CPU_ONLY is None:
        print("SKIP: a GPU set loaded; the host route is not taken here")
        return False
    if basename not in _backend.host_families_built():
        print(f"SKIP: {basename} is not built")
        return False
    return True


def test_embedding_runs_on_the_host_when_built():
    if not _cpu_only_with("_mojolearn_embedding_host"):
        return
    import numpy as np
    module = _backend.load_host_module("_mojolearn_embedding_host")
    assert not bool(module.embedding_host_sabotage()), "a sabotage build loaded outside the gate"
    assert str(module.embedding_host_column()) == "cpu"
    rng = np.random.default_rng(4)
    V, D, T = 11, 5, 37
    w = rng.standard_normal((V, D)).astype(np.float32)
    ids = rng.integers(0, V, T).astype(np.int32)
    dy = rng.standard_normal((T, D)).astype(np.float32)
    assert np.asarray(mojolearn.Embedding(V, D, weight=w).forward(ids)).tobytes() == w[ids].tobytes()
    want = np.zeros((V, D), dtype=np.float32)
    for t in range(T):
        want[ids[t]] += dy[t]
    want[2] = 0.0
    got = {}
    for plan in ("scan", "sort"):
        e = mojolearn.Embedding(V, D, padding_idx=2, weight=w, plan=plan)
        full = np.asarray(e.backward(ids, dy))
        carried = np.asarray(e.backward(ids[13:], dy[13:], grad=e.backward(ids[:13], dy[:13])))
        assert carried.tobytes() == full.tobytes(), plan
        assert full.tobytes() == want.tobytes(), plan
        assert np.signbit(full[2]).sum() == 0, "the padding row is not +0.0"
        got[plan] = full.tobytes()
    assert got["scan"] == got["sort"]
    bad = ids.copy()
    bad[5] = V
    try:
        mojolearn.Embedding(V, D, weight=w).forward(bad)
    except Exception as exc:
        assert "outside [0, 11) REFUSED" in str(exc), str(exc)
    else:
        raise AssertionError("an out-of-range id was gathered")


def test_ivf_runs_on_the_host_when_built():
    if not _cpu_only_with("_mojolearn_ivf_host"):
        return
    import numpy as np
    module = _backend.load_host_module("_mojolearn_ivf_host")
    assert not bool(module.ivf_host_sabotage()), "a sabotage build loaded outside the gate"
    rng = np.random.default_rng(5)
    x = rng.standard_normal((300, 6)).astype(np.float32)
    q = rng.standard_normal((9, 6)).astype(np.float32)
    k = 5
    m = mojolearn.IVFIndex(n_lists=4, n_probes=4, n_neighbors=k, random_state=1).fit(x)
    d2, i2 = m.search(q)
    assert np.asarray(m.n_candidates_).tolist() == [300] * 9
    exact = ((q[:, None, :].astype(np.float64) - x[None, :, :]) ** 2).sum(-1)
    assert np.asarray(i2).tolist() == np.argsort(exact, axis=1, kind="stable")[:, :k].tolist()
    e = mojolearn.IVFIndex(n_lists=4, n_probes=4, n_neighbors=k, metric="euclidean", random_state=1).fit(x)
    d1, i1 = e.search(q)
    assert np.asarray(i1).tolist() == np.asarray(i2).tolist()
    assert np.allclose(np.asarray(d1), np.sqrt(np.asarray(d2)), rtol=1e-6)
    try:
        mojolearn.IVFIndex(n_lists=4, n_probes=5, random_state=1).fit(x).search(q)
    except Exception as exc:
        assert "exceeds n_lists (4)" in str(exc), str(exc)
    else:
        raise AssertionError("n_probes > n_lists was clamped")


def test_byte_lm_trainer_runs_on_the_host_when_built():
    if not _cpu_only_with("_mojolearn_byte_lm_host"):
        return
    import os
    import tempfile
    import numpy as np
    from mojolearn import _byte_lm_trainer_host as adapter
    module = _backend.load_host_module("_mojolearn_byte_lm_host")
    assert not bool(module.byte_lm_host_sabotage()), "a sabotage build loaded outside the gate"
    binding = _backend.binding("_mojolearn_byte_lm", "identical")
    assert binding.byte_lm_vendor() == "cpu"
    assert getattr(binding, "byte_lm_parallel_create", None) is None
    try:
        binding.byte_lm_offload_step
    except AttributeError as exc:
        assert "byte_lm_offload_step" in str(exc)
    else:
        raise AssertionError("the adapter served a multi-GPU entry")
    shape = mojolearn.ByteLanguageModelConfig()
    rng = np.random.default_rng(6)
    flat = rng.uniform(-0.125, 0.125, shape.n_total).astype(np.float32)
    ids = rng.integers(0, 256, (6, shape.length + 1)).astype(np.int32)
    schedule = {"dataset": "test"}
    s = mojolearn.SmallByteLanguageModelTrainer(flat, data_schedule=schedule, step_result="full")
    r = mojolearn.SmallByteLanguageModelTrainer(flat, data_schedule=schedule, resident=True)
    for k in range(3):
        a = s.train_step(ids[2 * k:2 * k + 2])
        b = r.train_step(ids[2 * k:2 * k + 2])
        assert a["loss"] == b["loss"]
    assert np.asarray(a["flat_gradients"]).tobytes() == np.asarray(r.export_gradients()["flat_gradients"]).tobytes()
    assert np.asarray(s.parameters_).tobytes() == np.asarray(r.parameters_).tobytes()
    # The reference path's logits, from the same module the adapter loaded
    # (LanguageModelInference reads its own path variable, not
    # MOJOLEARN_HOST_DIR).
    p = np.ascontiguousarray(np.asarray(s.parameters_))
    q = np.ascontiguousarray(ids[:2, :-1])
    want = np.zeros((2, shape.length, shape.vocab_size), dtype=np.float32)
    module.byte_lm_host_logits([p.ctypes.data, q.ctypes.data, want.ctypes.data], [2, shape.length],
                               list(shape.native_shape), 0, 1)
    assert np.asarray(s.logits(ids[:2, :-1])).tobytes() == want.tobytes()
    assert np.asarray(r.logits(ids[:2, :-1])).tobytes() == want.tobytes()
    before = np.asarray(s.parameters_).tobytes()
    loss = s.evaluate(ids[:2])
    assert np.isfinite(loss) and np.asarray(s.parameters_).tobytes() == before and s.step_ == 3
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ck.json")
        s.save_checkpoint(path)
        back = mojolearn.SmallByteLanguageModelTrainer.from_checkpoint(path)
        assert np.asarray(back.parameters_).tobytes() == before and back.step_ == 3
    from mojolearn._buffer import addr, addr_ro, zeros
    session = binding.byte_lm_session_create()
    st = s.state_dict()
    params = [0, 3, 2, st["config"]["lr"], st["config"]["beta1"], st["config"]["beta2"], st["config"]["eps"],
              st["config"]["weight_decay"], 0.0, 0.0, 0, 0.0]
    native = list(shape.native_shape)
    arrays = [st["parameters"], st["m"], st["v"], st["flags"]]
    assert binding.byte_lm_session_open(session, [addr_ro(x, name="s") for x in arrays], params, native) == 3
    assert binding.byte_lm_session_rollback(session) == 3, "a rollback with no step restored something"
    tok = np.ascontiguousarray(ids[:2])
    loss_out = zeros((1,), "<f4")
    flags_out = zeros((shape.n_tensors,), "<i4")
    step = list(params)
    step[0] = 1
    assert binding.byte_lm_session_step(session, [tok.ctypes.data, addr_ro(st["flags"], name="f"),
                                                  addr(loss_out, name="l"), addr(flags_out, name="f")],
                                        step, native) == 4
    assert binding.byte_lm_session_info(session) == [4, 4, 1, 1]
    assert binding.byte_lm_session_rollback(session) == 3
    assert binding.byte_lm_session_info(session) == [3, -1, 1, 1]
    assert binding.byte_lm_session_rollback(session) == 3, "the shadow was restored twice"
    out = [zeros((shape.n_total,), "<f4") for _ in range(3)] + [zeros((shape.n_tensors,), "<i4")]
    assert binding.byte_lm_session_export_state(session, [addr(x, name="o") for x in out], native) == 3
    assert out[0].tobytes() == st["parameters"].tobytes() and out[2].tobytes() == st["v"].tobytes()
    binding.byte_lm_session_close(session)
    assert binding.byte_lm_session_info(session) == [-1, -1, 0, 0]


if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for name in names:
        globals()[name]()
        print("ok", name)
    print(f"{len(names)} passed")
    sys.exit(0)
