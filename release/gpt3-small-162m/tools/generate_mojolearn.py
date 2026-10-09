#!/usr/bin/env python3
"""Greedy generations from checkpoint.blm through MojoLearn's CPU forward pass.

Uses the public mojolearn package (pip install mojolearn): the checkpoint reader
`mojolearn._byte_lm_checkpoint.load` and `LanguageModelInference.next_bytes`, whose
argmax runs in Mojo (ties go to the lowest id). Writes results/generations.json.

    python3 tools/generate_mojolearn.py [new_tokens]

Requires mojolearn, numpy and tokenizers.
"""
import json
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from mojolearn import _byte_lm_checkpoint as ck
from mojolearn import ByteLanguageModelConfig, LanguageModelInference

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = [
    "The water cycle describes how water",
    "Photosynthesis is the process by which",
    "In 1776, the American colonies",
    "To find the area of a circle, you",
    "The most important thing about learning a new language is",
]


def main():
    new_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 48
    tok = Tokenizer.from_file(str(ROOT / 'tokenizer.json'))
    state = ck.load(ROOT / 'checkpoint.blm')
    shape = ByteLanguageModelConfig(**{**state['model_shape'], 'batch': 1, 'length': 128})
    model = LanguageModelInference(state['parameters'], shape=shape)
    out = []
    for prompt in PROMPTS:
        seq = tok.encode(prompt).ids
        for _ in range(new_tokens):
            seq.append(int(model.next_bytes(np.array([seq[-shape.length:]], dtype=np.int32))[0]))
        out.append({'prompt': prompt, 'completion': tok.decode(seq)[len(prompt):], 'token_ids': seq})
        print(json.dumps(out[-1]['prompt'] + '|' + out[-1]['completion']), flush=True)
    result = {'decoding': 'greedy', 'new_tokens': new_tokens, 'runtime': 'mojolearn LanguageModelInference (CPU)',
              'parameters_sha256': model.parameters_sha256(), 'generations': out}
    (ROOT / 'results' / 'generations.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
