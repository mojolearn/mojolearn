"""Requested generation configurations; executed routes are recorded separately."""
import itertools
import re

FLAGS = ('MOJOLEARN_EXPERIMENT_RESIDENT_SAMPLING_FP32',
         'MOJOLEARN_EXPERIMENT_SAMPLING_KV_UPLOAD_REUSE',
         'MOJOLEARN_GENERATION_NATIVE_ENGINE')

def configurations():
    return [dict(case=f's{s}-k{k}-e{e}', sampler=s, kv_reuse=k, engine=e)
            for s, k, e in itertools.product((0, 1), repeat=3)]

def parse_witness(text):
    pattern = r'MOJOLEARN_GENERATION_ROUTE route=(\w+) capacity=(\d+) produced=(\d+) retained_kv=([01])'
    return [dict(route=r, capacity=int(c), produced=int(p), retained_kv=bool(int(k)))
            for r, c, p, k in re.findall(pattern, text)]

def validate_witness(events, tokens, chunk):
    expected = (tokens + chunk - 1) // chunk
    if len(events) != expected:
        raise ValueError(f'expected {expected} completed warmup reads, observed {len(events)}')
    produced = 0
    for event in events:
        capacity = min(chunk, tokens - produced)
        produced += capacity
        if event['capacity'] != capacity or event['produced'] != produced:
            raise ValueError('completed route witness does not match consumed token/read counts')
    return sorted(set(event['route'] for event in events))
