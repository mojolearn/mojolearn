#!/usr/bin/env python3
# F01: qualification pending. Build every independent variant; device quality,
# complete-call speed, peak scratch and opponent admission remain separate gates.
# New experiment mechanisms remain opt-in; existing promoted defaults are retained.
"""Actual PCA projection/Gram geometry; generic neighboring shapes, cold + reuse.

Measurement revision 2026-10-06 enables the previously compiled complete scoped
adapter set: PCA covariance uses split-K, transform/inverse may use NT layouts.
The initial recipe omitted those adapters and failed its positive reach gate.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'apple_fast'))
from support import capture_main, binding_check, consumed


def exercise(args):
    import numpy as np
    from mojolearn import PCA
    from mojolearn import _mojolearn_estimators as binding
    identity = binding_check(binding, 'estimators')
    cases = {}
    # Generic aspect-ratio spread covers tails and neighboring widths, not board rows.
    for rows, columns, rank in ((997, 37, 7), (1009, 41, 9), (4093, 129, 17), (4127, 137, 19)):
        rng = np.random.default_rng(904)
        x = rng.normal(size=(rows, columns)).astype('float32')
        q = rng.normal(size=(113, columns)).astype('float32')
        model = PCA(n_components=rank, numeric_mode='fast')
        before = [int(binding.scoped_gemm_count(route, arm)) for route in range(3) for arm in range(3)]
        import time
        start=time.perf_counter_ns();model.fit(x)
        singular=np.asarray(model.singular_values_,float);singular.tobytes();np.asarray(model.components_).tobytes()
        fit_ms=(time.perf_counter_ns()-start)/1e6
        centered=x.astype(float)-x.astype(float).mean(axis=0)
        expected=np.linalg.svd(centered,compute_uv=False)
        singular_error=float(np.linalg.norm(singular-expected[:rank])/np.linalg.norm(expected[:rank]))
        noise_truth=float(np.sum(expected[rank:]**2)/((rows-1)*(columns-rank)))
        noise_error=abs(float(model.noise_variance_)-noise_truth)
        transformed, cold = consumed(lambda: model.transform(q))
        projected, repeated = consumed(lambda: model.transform(q))
        restored, inverse = consumed(lambda: model.inverse_transform(projected))
        after = [int(binding.scoped_gemm_count(route, arm)) for route in range(3) for arm in range(3)]
        reached = [b - a for a, b in zip(before, after)]
        if args.arm == 'B' and not any(reached[i] for i in (1, 2, 4, 5, 7, 8)):
            raise AssertionError('candidate did not reach an actual fitted caller')
        ref = np.asarray(q, np.float64)
        error = float(np.linalg.norm(np.asarray(restored, np.float64) - ref) / np.linalg.norm(ref))
        cases[f'{rows}x{columns}'] = dict(contract=dict(rows=rows, columns=columns, rank=rank, seed=904),
             metrics=dict(reconstruction=dict(value=error, rtol=1e-3, atol=1e-6),singular_error=dict(value=singular_error,rtol=.1,atol=2e-6),noise_error=dict(value=noise_error,rtol=.1,atol=1e-6)),
             fit_ms=fit_ms,cold_transform_ms=cold, repeated_transform_ms=repeated, inverse_ms=inverse, reach=reached,
             fitted_state_finite=bool(np.isfinite(np.asarray(model.components_)).all()))
    return dict(binding=identity, cases=cases)


if __name__ == '__main__':
    capture_main(exercise)
