#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Host-only planning arithmetic for the generalized LM; never a fit/speed gate.

Counts a disjoint subset of allocations in the current trainer. Workspaces,
linear activations, duplicate weights/gradients, host snapshots and allocator
overhead are excluded. Source hashes let a saved report identify its inventory.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
# Load only the shape definition: do not import mojolearn's GPU backend.
_spec = importlib.util.spec_from_file_location(
    "_lm_capacity_config", ROOT / "python/mojolearn/_byte_lm_config.py")
_config = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _config
_spec.loader.exec_module(_config)

SOURCES = (
    "python/mojolearn/_byte_lm_config.py",
    "training/byte_lm.mojo",
    "transformer/impl/llama/modeling_llama.mojo",
    "transformer/checks/transformer_backward.mojo",
)


def report(shape, *, materialized_attention=False):
    tokens = shape.batch * shape.length
    attention = 4 * shape.batch * shape.n_heads * shape.length ** 2
    vocabulary = 4 * tokens * shape.vocab_size
    # ByteTrainer constructs both stage types with lean=True. Forward:
    # scores, masked, aexp, weights. Backward: d_attn_weights,
    # d_attn_masked, d_attn_scores, d_qk_cell. Fused execution retains one
    # element each; eager/diagnostic fallback may materialize all matrices.
    allocations = {
        "parameters_gradient_adam_m_v": 4 * 4 * shape.n_total,
        "eight_attention_buffers_per_layer": 8 * shape.n_layers * (attention if materialized_attention else 4),
        # ByteBuffers: logits, ce_shift, ce_expo, ce_weights, ce_dlogits.
        "five_token_vocabulary_matrices": 5 * vocabulary,
    }
    return {
        "schema": "mojolearn.lm-capacity.v2",
        "attention_allocation": "materialized fallback" if materialized_attention else "lean fused",
        "qualification": "host arithmetic only; not measured memory or throughput",
        "model_shape": shape.to_dict(),
        "profile": shape.profile,
        "parameters": shape.n_total,
        "tokens_per_microbatch": tokens,
        "allocation_subset_bytes": allocations,
        "allocation_subset_total_gib": sum(allocations.values()) / 2 ** 30,
        "omissions": ["GEMM/optimizer/loss workspaces", "linear activations",
                      "reusable prefill KV workspace",
                      "duplicate weights and gradients", "host state and captures",
                      "allocator/runtime overhead"],
        "fit_admitted": False,
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in SOURCES},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=int, nargs=9, required=True,
                        metavar=("B", "L", "DM", "H", "KV", "HD", "FF", "LAYERS", "VOCAB"))
    parser.add_argument("--materialized-attention", action="store_true",
                        help="Count eager/diagnostic fallback quadratic buffers")
    args = parser.parse_args()
    print(json.dumps(report(_config.ByteLanguageModelConfig(*args.shape),
                            materialized_attention=args.materialized_attention), indent=2))


if __name__ == "__main__":
    main()
