#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""GRU and LSTM layers on the scan against the per-step chain: our arms only (lane rnn-scan). Run only when Andrew
asks for this timing (AGENTS.md "Measurement is ad hoc").

The two arms are two wheels of the SAME commit, installed one after the other in the SAME pod job (AGENTS.md
"Times are compared only on the same box"; workers bound to the GPU's NUMA node):
  chain  the default build: GRU and LSTM layers run the per-step chain (the step projection as its own GEMM above
         the step rule, then the cell; T launches and more each way), the tanh RNN the scan;
  scan   the build with -D MOJOLEARN_SEQ_FAST_RNN_SCAN_GATED: every cell runs the scan (one launch forward and one
         backward per layer and direction).
Both arms fold every product the same way (the scan takes the chain's fold at each shape), so quality is "the
outputs are equal": each row prints a digest of the fitted parameters and the predicted probabilities, and
`--compare` requires the two arms' digests to be equal row by row (`same_words`), beside the time ratio.

Rows, per cell (gru, lstm; rnn as the control, the scan in both arms): GRUClassifier / LSTMClassifier /
RNNClassifier, numeric_mode identical, adam, seed 0, two classes, on the full-size shapes
  T 16:  N 65536, F 8, hidden 64 and 128, batch 256 (the recurrent board recipe's sequence length and batch)
  T 512: N 2048, F 8, hidden 64, batch 256
  forward     predict_proba on the N rows (predict_chunk 4096)
  step        fit, max_epochs 1: N / batch training steps; the row prints the fit's ms and ms a step
Timing: one excluded warmup; a call under 10 ms after it is the median of 20 consecutive calls (AGENTS.md "Small
rows"), else one call. `--devices cpu` runs the CPU column for its words only (our CPU is never timed, AGENTS.md
"Measurement"), so the CPU rows carry digests and no times.

    python tools/ops/rnn_scan_timing.py --arm chain > chain.txt      # with the default wheel installed
    python tools/ops/rnn_scan_timing.py --arm scan > scan.txt        # with the -D wheel installed
    python tools/ops/rnn_scan_timing.py --compare chain.txt scan.txt
"""
import argparse
import hashlib
import os
import platform
import statistics
import subprocess
import time

import numpy as np

SHAPES = (  # (T, N, F, hidden, batch)
    (16, 65536, 8, 64, 256),
    (16, 65536, 8, 128, 256),
    (512, 2048, 8, 64, 256),
)


def probe():
    print("PROBE host", platform.node(), "cpu", platform.processor() or platform.machine(), "cores", os.cpu_count())
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,pcie.link.gen.current,pcie.link.width.current",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
        print("PROBE gpu", out or "none")
    except (OSError, subprocess.SubprocessError):
        print("PROBE gpu none")
    import mojolearn
    print("PROBE mojolearn", mojolearn.__version__, "vendor", mojolearn._backend.vendor())


def clock(fn, timed):
    out = fn()  # the excluded warmup
    if not timed:
        return float("nan"), out
    t0 = time.perf_counter()
    out = fn()
    first = time.perf_counter() - t0
    if first >= 0.010:
        return first * 1e3, out
    times = [first]
    for _ in range(19):  # glue: the small-row median (AGENTS.md)
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1e3, out


def data(n, t, f, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, t, f)).astype(np.float32)
    # class: the sign of a decayed sum of channel 0 (depends on the whole sequence)
    w = 0.9 ** np.arange(t)[::-1]
    y = ((x[:, :, 0] * w[None, :]).sum(axis=1) > 0).astype(np.int64)
    return x, y


def digest(model, proba):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):  # glue: digest of the returned arrays
        h.update(name.encode())
        h.update(np.ascontiguousarray(np.asarray(value, dtype=np.float32)).tobytes())
    h.update(np.ascontiguousarray(np.asarray(proba, dtype=np.float32)).tobytes())
    return h.hexdigest()[:16]


def run(args):
    import mojolearn as ml
    probe()
    classes = {"gru": ml.GRUClassifier, "lstm": ml.LSTMClassifier, "rnn": ml.RNNClassifier}
    for device in args.devices.split(","):  # glue: one row group per device
        timed = device == "gpu"
        for t, n, f, h, batch in SHAPES:  # glue: one row group per shape
            x, y = data(n, t, f, 0)
            for cell in args.cells.split(","):  # glue: one row per cell
                def make():
                    return classes[cell](hidden_size=h, num_layers=1, optimizer="adam", learning_rate=1e-3,
                                         batch_size=batch, max_epochs=1, shuffle=True, random_state=0,
                                         numeric_mode="identical", predict_chunk=4096, device=device)
                fit_ms, m = clock(lambda: make().fit(x, y), timed)
                fwd_ms, proba = clock(lambda: m.predict_proba(x), timed)
                steps = (n + batch - 1) // batch
                acc = float((np.asarray(proba).argmax(axis=1) == y).mean())
                print(f"RNN_SCAN arm={args.arm} device={device} cell={cell} T={t} N={n} F={f} H={h} B={batch} "
                      f"forward_ms={fwd_ms:.3f} fit_ms={fit_ms:.3f} step_ms={fit_ms / steps:.4f} steps={steps} "
                      f"train_acc={acc:.4f} digest={digest(m, proba)}", flush=True)


def parse(path):
    rows = {}
    with open(path) as fh:
        for line in fh:  # glue: reading the arm's printed rows
            if not line.startswith("RNN_SCAN "):
                continue
            kv = dict(item.split("=", 1) for item in line.split()[1:])
            rows[(kv["device"], kv["cell"], kv["T"], kv["H"])] = kv
    return rows


def compare(base_path, arm_path):
    base = parse(base_path)
    arm = parse(arm_path)
    for key in sorted(base):  # glue: one line per row
        b = base[key]
        a = arm.get(key)
        if a is None:
            print("COMPARE", key, "missing in", arm_path)
            continue
        same = a["digest"] == b["digest"]
        line = f"COMPARE device={key[0]} cell={key[1]} T={key[2]} H={key[3]} same_words={same}"
        if key[0] == "gpu":
            line += (f" forward {b['forward_ms']} -> {a['forward_ms']} ms"
                     f" ({float(a['forward_ms']) / float(b['forward_ms']):.3f})"
                     f" step {b['step_ms']} -> {a['step_ms']} ms ({float(a['step_ms']) / float(b['step_ms']):.3f})")
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="chain", help="label of the installed wheel: chain or scan")
    ap.add_argument("--cells", default="gru,lstm,rnn")
    ap.add_argument("--devices", default="gpu", help="gpu (timed), cpu (words only), or gpu,cpu")
    ap.add_argument("--compare", nargs=2, metavar=("BASE", "ARM"))
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
    else:
        run(args)


if __name__ == "__main__":
    main()
