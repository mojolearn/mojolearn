#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare two tools/ag_opbench.py JSONs (NVIDIA and AMD), one table per op.

Columns per (variant, shape): NV us and AMD us (median net of the sync),
launches per call (NV/AMD), us per launch, bytes moved (the op's minimum
traffic from its shapes, float32), achieved GB/s and the share of the
--*-peak-gbs bandwidth. A cell is flagged LAUNCH when its us per launch
exceeds --factor (3) times that vendor's per-launch floor (10k one-element
adds), and BW when it reaches --bw-share (0.5) of the vendor's peak, so the
ideas lane can see which ops are launch-bound and which are bandwidth-bound.

Usage:
  python tools/ag_opbench_report.py opbench-nvidia.json opbench-amd.json [--out report.txt]
"""
import argparse
import json


def _load(path):
    with open(path) as f:
        doc = json.load(f)
    return doc, {r["key"]: r for r in doc["cases"]}


def _fmt(v, spec):
    return "-" if v is None else format(v, spec)


def _cell(r, floor, peak, factor, bw_share):
    """us, launches, us/launch, GB/s, %peak, flags of one vendor's case."""
    if r is None:
        return None, None, None, None, None, ["missing"]
    if "error" in r:
        return None, r.get("launches"), None, None, None, ["error"]
    us = r.get("net_us")
    la = r.get("launches")
    upl = (us / la) if (us is not None and la) else None
    gbs = (r["bytes"] / (us * 1e3)) if (r.get("bytes") and us and us > 0) else None
    share = (gbs / peak) if (gbs is not None and peak) else None
    flags = []
    if upl is not None and floor and upl > factor * floor:
        flags.append("LAUNCH")
    if share is not None and share >= bw_share:
        flags.append("BW")
    return us, la, upl, gbs, share, flags


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("nv", help="opbench JSON from the NVIDIA box")
    ap.add_argument("amd", help="opbench JSON from the AMD box")
    ap.add_argument("--out", default=None, help="write the report here as well as stdout")
    ap.add_argument("--factor", type=float, default=3.0, help="LAUNCH flag: us/launch > factor x floor")
    ap.add_argument("--nv-peak-gbs", type=float, default=864.0, help="NVIDIA DRAM peak GB/s (L40S 864)")
    ap.add_argument("--amd-peak-gbs", type=float, default=6000.0, help="AMD HBM peak GB/s (MI325X 6000)")
    ap.add_argument("--bw-share", type=float, default=0.5, help="BW flag: achieved >= share x peak")
    args = ap.parse_args()

    nv_doc, nv = _load(args.nv)
    amd_doc, amd = _load(args.amd)
    nv_floor = (nv_doc.get("floor") or {}).get("per_launch_us")
    amd_floor = (amd_doc.get("floor") or {}).get("per_launch_us")
    out = []
    out.append(f"NV  {args.nv}: sync_us={_fmt(nv_doc.get('sync_us'), '.1f')} "
               f"launch_floor_us={_fmt(nv_floor, '.2f')} "
               f"synced_add_us={_fmt((nv_doc.get('floor') or {}).get('synced_add_us'), '.1f')} "
               f"gpu={nv_doc.get('device', {}).get('gpu', '?')}")
    out.append(f"AMD {args.amd}: sync_us={_fmt(amd_doc.get('sync_us'), '.1f')} "
               f"launch_floor_us={_fmt(amd_floor, '.2f')} "
               f"synced_add_us={_fmt((amd_doc.get('floor') or {}).get('synced_add_us'), '.1f')} "
               f"gpu={amd_doc.get('device', {}).get('gpu', '?')}")
    for d, name in ((nv_doc, "NV"), (amd_doc, "AMD")):
        for o in d.get("owed", []):
            out.append(f"OWED {name}: {o}")
    out.append(f"us = median net of one sync; LAUNCH = us/launch > {args.factor:g} x floor; "
               f"BW = >= {args.bw_share:g} x peak ({args.nv_peak_gbs:g} / {args.amd_peak_gbs:g} GB/s)")

    keys = list(nv.keys()) + [k for k in amd.keys() if k not in nv]
    ops = []
    for k in keys:
        op = k.split("|", 1)[0]
        if op not in ops:
            ops.append(op)
    flagged = {"LAUNCH": 0, "BW": 0}
    for op in ops:
        out.append("")
        out.append(f"== {op}")
        out.append(f"{'variant':<14} {'shape':<22} {'NV us':>10} {'AMD us':>10} {'launch':>9} "
                   f"{'NV us/l':>8} {'AMD us/l':>8} {'MB':>9} {'NV GB/s':>8} {'AMD GB/s':>8} "
                   f"{'NV%pk':>6} {'AMD%pk':>6}  flags")
        for k in keys:
            if k.split("|", 1)[0] != op:
                continue
            a = nv.get(k)
            b = amd.get(k)
            ref = a or b
            nus, nla, nupl, ngbs, nsh, nfl = _cell(a, nv_floor, args.nv_peak_gbs, args.factor, args.bw_share)
            aus, ala, aupl, agbs, ash, afl = _cell(b, amd_floor, args.amd_peak_gbs, args.factor, args.bw_share)
            flags = [f"NV:{f}" for f in nfl] + [f"AMD:{f}" for f in afl]
            for f in nfl + afl:
                if f in flagged:
                    flagged[f] += 1
            mb = (ref.get("bytes") / 1e6) if ref.get("bytes") else None
            out.append(f"{ref['variant']:<14} {ref['shape']:<22} {_fmt(nus, '.1f'):>10} {_fmt(aus, '.1f'):>10} "
                       f"{(_fmt(nla, 'd') + '/' + _fmt(ala, 'd')):>9} {_fmt(nupl, '.2f'):>8} {_fmt(aupl, '.2f'):>8} "
                       f"{_fmt(mb, '.2f'):>9} {_fmt(ngbs, '.1f'):>8} {_fmt(agbs, '.1f'):>8} "
                       f"{_fmt(None if nsh is None else 100 * nsh, '.0f'):>6} "
                       f"{_fmt(None if ash is None else 100 * ash, '.0f'):>6}  {' '.join(flags)}")
    out.append("")
    out.append(f"AG_OPBENCH_REPORT ops={len(ops)} cells={len(keys)} launch_flags={flagged['LAUNCH']} "
               f"bw_flags={flagged['BW']}")
    text = "\n".join(out) + "\n"
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
