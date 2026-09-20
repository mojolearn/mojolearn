# SPDX-License-Identifier: Apache-2.0
"""Local-device timing and bit witness for the opt-in LM-head v2 stage."""
import hashlib
import importlib.util
from pathlib import Path
import platform
import struct
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SO = ROOT / "python/mojolearn/identical/_mojolearn_training.so"


def main():
    spec = importlib.util.spec_from_file_location("_mojolearn_training", SO)
    binding = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(binding)
    rows, vocab, width = 64, 4096, 128
    hidden = (((np.arange(rows * width, dtype=np.int64) * 17) % 251) - 125).astype(np.float32) / np.float32(127)
    weight = (((np.arange(vocab * width, dtype=np.int64) * 37) % 509) - 254).astype(np.float32) / np.float32(257)
    targets = ((np.arange(rows, dtype=np.int32) * 127) % vocab).astype(np.int32)
    loss = np.zeros(1, np.float32)
    maxima = np.zeros(rows, np.float32)
    denom = np.zeros(rows, np.float32)
    args = [loss.ctypes.data, maxima.ctypes.data, denom.ctypes.data,
            hidden.ctypes.data, weight.ctypes.data, targets.ctypes.data]
    params = [rows, vocab, width]
    binding.chunked_lm_head_v2_loss(args, params)
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        binding.chunked_lm_head_v2_loss(args, params)
        samples.append(time.perf_counter() - start)
    witness = hashlib.sha256(loss.tobytes() + maxima.tobytes() + denom.tobytes()).hexdigest()
    print("host", platform.platform())
    print("vendor", binding.training_vendor())
    print("shape", rows, vocab, width)
    print("seconds", " ".join(f"{x:.6f}" for x in samples))
    print("loss_bits", hex(struct.unpack("I", loss.tobytes())[0]))
    print("witness_sha256", witness)
    print("scratch_bytes", (3 * rows + 1) * 4)
    print("full_logits_bytes_avoided", rows * vocab * 4)


if __name__ == "__main__":
    main()
