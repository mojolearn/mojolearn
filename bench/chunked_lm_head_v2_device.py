# SPDX-License-Identifier: Apache-2.0
"""Local-device timing and bit witness for the opt-in LM-head v2 stage."""
import hashlib
import argparse
import importlib.util
from pathlib import Path
import platform
import struct
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SO = ROOT / "python/mojolearn/identical/_mojolearn_training.so"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--vocab", type=int, default=4096)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--so", type=Path, default=SO)
    cli = parser.parse_args()
    spec = importlib.util.spec_from_file_location("_mojolearn_training", cli.so)
    binding = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(binding)
    rows, vocab, width = cli.rows, cli.vocab, cli.width
    hidden = (((np.arange(rows * width, dtype=np.int64) * 17) % 251) - 125).astype(np.float32) / np.float32(127)
    weight = (((np.arange(vocab * width, dtype=np.int64) * 37) % 509) - 254).astype(np.float32) / np.float32(257)
    targets = ((np.arange(rows, dtype=np.int32) * 127) % vocab).astype(np.int32)
    loss = np.zeros(1, np.float32)
    maxima = np.zeros(rows, np.float32)
    denom = np.zeros(rows, np.float32)
    d_hidden = np.zeros_like(hidden)
    d_weight = np.zeros_like(weight)
    args = [loss.ctypes.data, maxima.ctypes.data, denom.ctypes.data,
            d_hidden.ctypes.data, d_weight.ctypes.data,
            hidden.ctypes.data, weight.ctypes.data, targets.ctypes.data]
    params = [rows, vocab, width]
    binding.chunked_lm_head_v2_train(args, params)
    samples = []
    for _ in range(cli.reps):
        start = time.perf_counter()
        binding.chunked_lm_head_v2_train(args, params)
        samples.append(time.perf_counter() - start)
    witness = hashlib.sha256(loss.tobytes() + maxima.tobytes() + denom.tobytes()
                             + d_hidden.tobytes() + d_weight.tobytes()).hexdigest()
    loss_before = loss.copy()
    weight -= np.float32(0.001) * d_weight
    binding.chunked_lm_head_v2_train(args, params)
    print("host", platform.platform())
    print("vendor", binding.training_vendor())
    print("shape", rows, vocab, width)
    print("seconds", " ".join(f"{x:.6f}" for x in samples))
    print("loss_bits", hex(struct.unpack("I", loss.tobytes())[0]))
    print("loss_before_update", float(loss_before[0]))
    print("loss_after_update", float(loss[0]))
    print("witness_sha256", witness)
    print("scratch_bytes", (3 * rows + 1) * 4)
    print("full_logits_bytes_avoided", rows * vocab * 4)


if __name__ == "__main__":
    main()
