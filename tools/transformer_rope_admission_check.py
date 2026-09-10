#!/usr/bin/env python3
"""Compare exported production RoPE with the original NumPy FP32 reference.

No timing; reports original-reference error and error after using the exact
production inverse-frequency bits, isolating constants from trig evaluation.
"""
import argparse
import json
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('probe_log')
    args = ap.parse_args()
    rows = {}
    for line in open(args.probe_log):
        if not line.startswith('ROPE '):
            continue
        hd, p, i, inv, cos, sin = map(int, line.split()[1:])
        rows.setdefault((hd, p), []).append((i, inv, cos, sin))
    if not rows:
        raise SystemExit('No ROPE rows: probe did not complete')
    for (hd, p), entries in sorted(rows.items()):
        entries.sort()
        assert [e[0] for e in entries] == list(range(hd // 2))
        got = np.asarray([e[1:] for e in entries], dtype=np.uint32).view(np.float32)
        inv, cos, sin = got.T
        e = np.arange(0, hd, 2, dtype=np.float32) / np.float32(hd)
        ref_inv = np.float32(1) / (np.float32(10000) ** e)
        original_angle = np.float32(p) * ref_inv
        aligned_angle = np.float32(p) * inv
        out = {'hd': hd, 'position': p,
               'inv_moved': int(np.count_nonzero(inv.view(np.uint32) != ref_inv.view(np.uint32))),
               'inv_max_abs': float(np.max(np.abs(inv.astype(np.float64) - ref_inv))),
               'angle_max_abs': float(np.max(np.abs(aligned_angle.astype(np.float64) - original_angle)))}
        for name, values, fn in [('cos', cos, np.cos), ('sin', sin, np.sin)]:
            out[name + '_original_max_abs'] = float(np.max(np.abs(values.astype(np.float64) - fn(original_angle))))
            out[name + '_aligned_inv_max_abs'] = float(np.max(np.abs(values.astype(np.float64) - fn(aligned_angle))))
        print(json.dumps(out, sort_keys=True))


if __name__ == '__main__':
    main()
