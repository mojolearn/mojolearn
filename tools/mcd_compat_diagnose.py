#!/usr/bin/env python3
"""Bounded diagnosis of existing MCD A/B artifacts; never fits or times a model."""
import argparse
import json
import numpy as np


def summarize(z):
    report = {}
    eps = np.finfo(np.float32).eps
    for key in ('raw_covariance_', 'covariance_'):
        c = np.asarray(z[key], dtype=np.float64)
        w = np.linalg.eigvalsh((c + c.T) / 2)
        cut = np.max(np.abs(w)) * c.shape[0] * eps
        kept = np.abs(w) > cut
        keep_w = np.abs(w[kept])
        report[key] = dict(eigenvalues=[float(x) for x in w], pinvh_cut=float(cut),
                           pinvh_rank=int(kept.sum()),
                           kept_condition=float(keep_w.max()/keep_w.min()) if keep_w.size else None)
    c = np.asarray(z['covariance_'], dtype=np.float64)
    p = np.asarray(z['precision_'], dtype=np.float64)
    report['precision_consistency_rel'] = float(np.linalg.norm(c @ p @ c - c) / max(np.linalg.norm(c), 1e-30))
    report['precision_norm'] = float(np.linalg.norm(p))
    for key in ('distances', 'dist_'):
        report[key + '_quantiles'] = [float(x) for x in np.quantile(z[key], [0, .25, .5, .75, 1])]
    report['flag_fraction'] = float(np.mean(z['flags']))
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('a')
    ap.add_argument('b')
    args = ap.parse_args()
    with np.load(args.a) as a, np.load(args.b) as b:
        for key in ('dataset', 'lane', 'shape', 'data_sha'):
            if not np.array_equal(a[key], b[key]):
                raise ValueError('Different inputs: ' + key)
        # Experiment supports d<=64; refuse unexpectedly unbounded spectra.
        if a['covariance_'].shape[0] > 64:
            raise ValueError('diagnostic limited to candidate d<=64')
        for arm, z in (('A', a), ('B', b)):
            print('MCD-DIAG ' + json.dumps(dict(arm=arm, **summarize(z))), flush=True)
        for key in ('support_', 'raw_support_'):
            aa, bb = a[key].astype(bool), b[key].astype(bool)
            print('MCD-SUPPORT ' + json.dumps(dict(field=key, a=int(aa.sum()), b=int(bb.sum()),
                   common=int((aa & bb).sum()), different=int((aa != bb).sum()))), flush=True)


if __name__ == '__main__':
    main()
