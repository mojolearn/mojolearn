# SPDX-License-Identifier: Apache-2.0
"""Writes the inputs of the transformer_softmax fixtures of
checks/native_tests/transformer_ops_program.mojo.

Moved from python/mojolearn/tests/test_ops_transformer_program.py
(2026-11-14): the inputs are the same NumPy PCG64 draws the Python test made
(default_rng(718 + columns), x = normal(0, .7, (2, 3, columns)), then
seed = normal(0, .3, x.shape)), stored so the Mojo check needs no NumPy
stream. The check's assertion that the two softmax reduction contracts give
different words depends on these exact inputs. Every oracle is computed in
Mojo from them.

Run from the repository root:
    python checks/native_tests/fixtures/transformer_ops_program/generate.py
Files: softmax<columns>-x.f32 and softmax<columns>-seed.f32 (little-endian
float32, C order).
"""
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    for columns in (5, 33, 65):
        rng = np.random.default_rng(718 + columns)
        x = rng.normal(0, .7, (2, 3, columns)).astype(np.float32)
        seed = rng.normal(0, .3, x.shape).astype(np.float32)
        (HERE / f"softmax{columns}-x.f32").write_bytes(x.astype("<f4").tobytes())
        (HERE / f"softmax{columns}-seed.f32").write_bytes(seed.astype("<f4").tobytes())
        print(f"softmax{columns}", x.shape)


if __name__ == "__main__":
    main()
