# SPDX-License-Identifier: Apache-2.0
"""Write the NumPy inputs (and the NumPy brute-force reference) of the batch-7 Mojo checks that replaced numeric
Python tests (lane py2mojo-D, 2026-11-22; docs/plans/PY2MOJO_CONVERSION.md batch 7).

Every array is the one the original test built (same seed, draw order and dtype); the reference is the NumPy
expression the original test asserted against. Raw little-endian words, row-major (`_put` of
`tools/py2mojo_native_fixtures.py`).

    fixtures/byte_lm_routes/  byte_lm_host_column.mojo, byte_lm_trainer_session.mojo
                              (test_byte_lm_step_routes.py `initial()` seed 41 and `batches()` seed 43)
    fixtures/byte_lm_host6/   byte_lm_trainer_session.mojo
                              (test_cpu_training_embedding_ivf.py::test_byte_lm_trainer_runs_on_the_host_when_built)
    fixtures/emb_host/        embedding_host_fold.mojo
                              (test_cpu_training_embedding_ivf.py::test_embedding_runs_on_the_host_when_built)
    fixtures/ivf_host/        ivf_host_brute_force.mojo
                              (test_cpu_training_embedding_ivf.py::test_ivf_runs_on_the_host_when_built; the stable
                              argsort of the float64 squared distances is the expected neighbour ids)

Generated with numpy 2.5.5 (2026-11-22). Regenerate and repack:

    python tools/py2mojo_native_fixtures_b7.py && python3 tools/fixture_packs.py pack checks/native_tests/fixtures

The sets are independent and are written one per process across the host cores.
"""
from concurrent.futures import ProcessPoolExecutor
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from py2mojo_native_fixtures import _put  # noqa: E402

BYTE_N_TOTAL = 34944  # ByteLanguageModelConfig().n_total, the default shape (training/byte_lm_config.mojo)
BYTE_LENGTH = 32


def byte_lm_routes():
    # test_byte_lm_step_routes.py::initial and ::batches
    initial = (np.random.default_rng(41).standard_normal(34944) * .02).astype(np.float32)
    batches = np.random.default_rng(43).integers(0, 256, (3, 2, 33), dtype=np.int32)
    return [_put("byte_lm_routes", "initial.f32", initial, np.float32),
            _put("byte_lm_routes", "batches.i32", batches, np.int32)]


def byte_lm_host6():
    # test_cpu_training_embedding_ivf.py::test_byte_lm_trainer_runs_on_the_host_when_built
    rng = np.random.default_rng(6)
    flat = rng.uniform(-0.125, 0.125, BYTE_N_TOTAL).astype(np.float32)
    ids = rng.integers(0, 256, (6, BYTE_LENGTH + 1)).astype(np.int32)
    return [_put("byte_lm_host6", "flat.f32", flat, np.float32),
            _put("byte_lm_host6", "ids.i32", ids, np.int32)]


def emb_host():
    # test_cpu_training_embedding_ivf.py::test_embedding_runs_on_the_host_when_built
    rng = np.random.default_rng(4)
    V, D, T = 11, 5, 37
    w = rng.standard_normal((V, D)).astype(np.float32)
    ids = rng.integers(0, V, T).astype(np.int32)
    dy = rng.standard_normal((T, D)).astype(np.float32)
    return [_put("emb_host", "w.f32", w, np.float32), _put("emb_host", "ids.i32", ids, np.int32),
            _put("emb_host", "dy.f32", dy, np.float32)]


def ivf_host():
    # test_cpu_training_embedding_ivf.py::test_ivf_runs_on_the_host_when_built
    rng = np.random.default_rng(5)
    x = rng.standard_normal((300, 6)).astype(np.float32)
    q = rng.standard_normal((9, 6)).astype(np.float32)
    exact = ((q[:, None, :].astype(np.float64) - x[None, :, :]) ** 2).sum(-1)
    want = np.argsort(exact, axis=1, kind="stable")[:, :5]
    return [_put("ivf_host", "x.f32", x, np.float32), _put("ivf_host", "q.f32", q, np.float32),
            _put("ivf_host", "want-ids.i32", want, np.int32)]


SETS = (byte_lm_routes, byte_lm_host6, emb_host, ivf_host)


def _run(index):
    return SETS[index]()


def main():
    with ProcessPoolExecutor(max_workers=min(len(SETS), os.cpu_count() or 1)) as pool:
        for lines in pool.map(_run, range(len(SETS))):
            for line in lines:
                print(line)


if __name__ == "__main__":
    main()
