#!/usr/bin/env python3
"""Compare reference_torch.py with MojoLearn's own CPU forward pass.

Both read the same trained weights (checkpoint.blm for MojoLearn, model.safetensors
for PyTorch) and the same held-out token windows. The script reports the largest
absolute logit difference and how often the two pick the same greedy next token.
Agreement here is numerical closeness, not bitwise identity.

    python3 tools/compare_reference.py WINDOWS.i32 [rows] [length]

Requires mojolearn, torch, safetensors and numpy.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import reference_torch  # noqa: E402
from mojolearn import _byte_lm_checkpoint as ck  # noqa: E402
from mojolearn import ByteLanguageModelConfig, LanguageModelInference  # noqa: E402


def main():
    windows = sys.argv[1]
    rows = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    length = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    ids = np.fromfile(windows, '<i4').reshape(-1, 2049)[:rows, :length].copy()
    state = ck.load(ROOT / 'checkpoint.blm')
    shape = ByteLanguageModelConfig(**{**state['model_shape'], 'batch': rows, 'length': length})
    ours = LanguageModelInference(state['parameters'], shape=shape).logits(ids)
    ours = np.asarray(ours, np.float32).reshape(rows, length, -1)
    with torch.no_grad():
        theirs = reference_torch.load()(torch.from_numpy(ids).long()).numpy()
    diff = np.abs(ours - theirs)
    result = {
        'rows': rows, 'length': length,
        'max_abs_logit_diff': float(diff.max()),
        'mean_abs_logit_diff': float(diff.mean()),
        'max_abs_logit': float(np.abs(ours).max()),
        'greedy_token_agreement': float((ours.argmax(-1) == theirs.argmax(-1)).mean()),
        'torch_version': torch.__version__,
    }
    (ROOT / 'results' / 'reference-agreement.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
