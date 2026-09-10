#!/usr/bin/env python3
"""Compare fresh-prefill output/reports with the original explicit-state path."""
from pathlib import Path
import runpy
import numpy as np
from mojolearn import Mamba3Block

root = Path(__file__).resolve().parents[1]
weights_for = runpy.run_path(str(root / 'python/mojolearn/tests/test_mamba_surface.py'), run_name='m3_helpers')['m3_weights']
checked = 0
for batch, length, dm in [(1, 1, 32), (2, 63, 64), (2, 64, 64), (2, 65, 64), (1, 129, 32)]:
    rng = np.random.default_rng(71 + length)
    weights = weights_for(rng, dm)
    fresh = Mamba3Block(weights)
    stateful = Mamba3Block(weights)
    state = stateful.allocate_state(batch)
    x = rng.uniform(-0.2, 0.2, (batch, length, dm)).astype(np.float32)
    has_fresh = hasattr(fresh._extension(), 'mamba3_forward_fresh')
    if has_fresh:
        def forbidden_cache(*args, **kwargs):
            raise AssertionError('fresh path allocated a discarded host state')
        fresh.allocate_state = forbidden_cache
    y = fresh.forward(x)
    expected = stateful.forward(x, state)
    for name, actual, reference in [('y', y, expected)] + [
        (name, getattr(fresh, name), getattr(stateful, name))
        for name in ['h_last_', 'k_last_', 'v_last_', 'theta_last_']
    ]:
        assert np.array_equal(actual.view(np.uint32), reference.view(np.uint32)), (batch, length, dm, name)
        checked += actual.size
    # Both routes must reread caller-backed mutable weights on every call.
    weights['D'][0] = np.nan
    errors = []
    for block, explicit in [(fresh, None), (stateful, stateful.allocate_state(batch))]:
        try:
            block.forward(x, explicit)
        except Exception as exc:
            errors.append(str(exc))
        else:
            raise AssertionError('mutated nonfinite weight escaped refusal')
    assert errors[0] == errors[1], errors
    print('M3_FRESH_CASE_PASS', batch, length, dm, 'dedicated_entry', has_fresh)
print('M3_FRESH_PASS', checked, 'output/report cells and matching mutable-weight refusals')
