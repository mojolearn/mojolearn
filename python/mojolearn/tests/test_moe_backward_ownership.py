"""Backward must retain every native output buffer until its writer finishes."""
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("rows", [7, 256])
def test_backward_output_owner_survives_two_blocks_and_reuse(rows):
    # A child preserves an actionable failure when the original defect corrupts
    # the heap. The second differently shaped block exercises allocator reuse.
    script = r'''
import gc
import sys
import numpy as np
import mojolearn as ml

rows = int(sys.argv[1])
rng = np.random.default_rng(912)
x = rng.standard_normal((rows, 16)).astype(np.float32)
dy = rng.standard_normal(x.shape).astype(np.float32)
blocks = []
for width, experts, top_k, normalize, seed in [(24, 6, 2, True, 1), (12, 5, 3, False, 2)]:
    block = ml.MoEBlock(16, width, num_experts=experts, top_k=top_k,
                        norm_topk_prob=normalize, random_state=seed)
    if not normalize:
        router = block.router.copy()
        router[3] = router[1]
        block.router = router
    dx = block.backward(x, dy)
    first = [v.copy() for v in (dx, block.router_grad_, block.gate_up_proj_grad_, block.down_proj_grad_)]
    assert [v.shape for v in first] == [(rows, 16), (experts, 16),
                                       (experts, 2 * width, 16), (experts, 16, width)]
    assert all(np.isfinite(v).all() for v in first)
    # Allocate and consume unrelated host storage after the native writes.
    scratch = [np.full((rows, 16), i, dtype=np.float32) for i in range(6)]
    assert sum(float(v[0, 0]) for v in scratch) == 15
    again = block.backward(x, dy)
    for saved, current in zip(first, (again, block.router_grad_, block.gate_up_proj_grad_, block.down_proj_grad_)):
        assert saved.tobytes() == current.tobytes()
    blocks.append(block)
del blocks, block
gc.collect()
print("moe-backward-output-owner-complete", flush=True)
'''
    result = subprocess.run([sys.executable, "-c", script, str(rows)],
                            env=os.environ.copy(), capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
    assert "moe-backward-output-owner-complete" in result.stdout
