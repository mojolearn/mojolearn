#!/usr/bin/env python3
"""Zero-shot HellaSwag (validation, 10,042 items) through reference_torch.py.

Scoring follows EleutherAI lm-evaluation-harness `hellaswag`: the query is
preprocess(activity_label + ": " + ctx_a + " " + ctx_b.capitalize()), each choice is
" " + preprocess(ending), and the model picks the choice with the highest summed
log-probability (acc) or the highest log-probability per UTF-8 byte of the choice
(acc_norm). Writes results/hellaswag.json.

    python3 tools/eval_hellaswag.py [limit]

Requires torch, safetensors, tokenizers and datasets.
"""
import json
import re
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import reference_torch  # noqa: E402


def preprocess(text):
    text = text.strip().replace(' [title]', '. ')
    text = re.sub('\\[.*?\\]', '', text)
    return text.replace('  ', ' ')


@torch.no_grad()
def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    torch.set_num_threads(max(1, torch.get_num_threads()))
    tok = Tokenizer.from_file(str(ROOT / 'tokenizer.json'))
    model = reference_torch.load()
    data = load_dataset('Rowan/hellaswag', split='validation')
    if limit:
        data = data.select(range(limit))
    correct = correct_norm = 0
    t0 = time.time()
    for n, ex in enumerate(data, 1):
        query = preprocess(ex['activity_label'] + ': ' + ex['ctx_a'] + ' ' + ex['ctx_b'].capitalize())
        ctx = tok.encode(query).ids
        choices = [' ' + preprocess(e) for e in ex['endings']]
        conts = [tok.encode(query + c).ids[len(ctx):] for c in choices]
        width = max(len(ctx) + len(c) for c in conts)
        ids = torch.zeros((4, width), dtype=torch.long)
        for i, c in enumerate(conts):
            seq = ctx + c
            ids[i, :len(seq)] = torch.tensor(seq)
        logp = torch.log_softmax(model(ids), dim=-1)
        scores, norms = [], []
        for i, c in enumerate(conts):
            pos = torch.arange(len(ctx) - 1, len(ctx) + len(c) - 1)
            s = float(logp[i, pos, torch.tensor(c)].sum())
            scores.append(s)
            norms.append(s / len(choices[i].encode('utf-8')))
        label = int(ex['label'])
        correct += int(max(range(4), key=scores.__getitem__) == label)
        correct_norm += int(max(range(4), key=norms.__getitem__) == label)
        if n % 500 == 0:
            print(f'{n} acc {correct / n:.4f} acc_norm {correct_norm / n:.4f} {time.time() - t0:.0f}s', flush=True)
    result = {'task': 'hellaswag', 'split': 'validation', 'shots': 0, 'items': len(data),
              'acc': correct / len(data), 'acc_norm': correct_norm / len(data),
              'scoring': 'lm-evaluation-harness hellaswag formulation; acc_norm = per-byte',
              'model': 'reference_torch.py on model.safetensors', 'torch_version': torch.__version__}
    (ROOT / 'results' / 'hellaswag.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
