"""Original grid runner restricted to its two public rows and eager FP32 arm.
Only arm/row selection changes; fixtures, kernels, warmups and timing are
provided unchanged by the Sep7 tools/speed_torch_seq.py snapshot.
"""
import sys
sys.path.insert(0, '/root/attn/tools')
import speed_torch_seq as grid
original_shapes = grid.load_shapes
def public_only():
    _, constants = original_shapes()
    return [], constants
grid.load_shapes = public_only
original_arms = grid._llama_arms
def fp32_only(torch, lane, args):
    return [arm for arm in original_arms(torch, lane, args) if arm[0] == 'torch-gpu-fp32']
grid._llama_arms = fp32_only
sys.argv = ['speed_torch_seq.py', '--lane', 'transformer', '--rounds', '5', '--warmups', '5', '--no-crosscheck', '--no-bf16', '--dump-dir', '/root/jobs/attn-public/base_dump', '--mojo-log', '/root/jobs/attn-public/baseline.log']
raise SystemExit(grid.main())
