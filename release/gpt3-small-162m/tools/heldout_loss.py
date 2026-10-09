"""Held-out FineWeb-Edu loss of checkpoint.blm through mojolearn's CPU forward pass.

Each batch is the training layout [4, 2049]; LanguageModelInference.loss returns the
mean next-token cross-entropy (nats) over its 4 x 2,048 targets, computed in Mojo.
"""
import sys, json, math, time
import numpy as np
from mojolearn import _byte_lm_checkpoint as ck
from mojolearn import LanguageModelInference, ByteLanguageModelConfig
hub, windows, out = sys.argv[1], sys.argv[2], sys.argv[3]
limit = int(sys.argv[4]) if len(sys.argv) > 4 else None
state = ck.load(hub + '/checkpoint.blm')
shape = ByteLanguageModelConfig(**state['model_shape'])
model = LanguageModelInference(state['parameters'], shape=shape)
arr = np.fromfile(windows, '<i4').reshape(-1, shape.length + 1)
nbatch = arr.shape[0] // shape.batch if limit is None else limit
token_bytes = np.zeros(shape.vocab_size, np.int64)
for line in open(hub + '/ranks.tsv'):
    i, h = line.rstrip('\n').split('\t'); token_bytes[int(i)] = len(h) // 2
losses = []; t0 = time.time()
for b in range(nbatch):
    ids = np.ascontiguousarray(arr[b * shape.batch:(b + 1) * shape.batch])
    losses.append(model.loss(ids))
    print(f'batch {b} loss {losses[-1]:.5f} elapsed {time.time()-t0:.0f}s', flush=True)
targets = arr[:nbatch * shape.batch, 1:]
n_tok = targets.size; n_bytes = int(token_bytes[targets].sum())
mean = float(np.mean(np.array(losses, np.float64)))
res = {'parameters_sha256': model.parameters_sha256(), 'batches': nbatch, 'target_tokens': n_tok,
       'target_utf8_bytes': n_bytes, 'loss_nats_per_token': mean, 'perplexity': math.exp(mean),
       'bits_per_byte': mean * n_tok / math.log(2) / n_bytes,
       'batch_losses_f32': losses, 'mojolearn_profile': model.profile}
json.dump(res, open(out, 'w'), indent=1)
print({k: v for k, v in res.items() if k != 'batch_losses_f32'})
