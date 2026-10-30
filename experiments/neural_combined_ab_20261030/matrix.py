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
    events=[]
    for line in text.splitlines():
        matched=re.search(pattern,line)
        if not matched:continue
        r,c,p,k=matched.groups()
        event=dict(route=r,capacity=int(c),produced=int(p),retained_kv=bool(int(k)))
        event['applicability']={key:value for key,value in re.findall(r'(\w+)=([^\s]+)',line[matched.end():])}
        events.append(event)
    return events

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


def route_configurations():
    """Only supported combinations; no manufactured residency off workload."""
    rows=[]
    for route, samplers, forward, engine, actual in (
        ('legacy', (0,1), 1, 0, 'legacy_forward'),
        ('resident', (1,), 1, 0, 'resident_sampling'),
        ('native_engine', (0,1), 0, 1, 'native_engine'),
    ):
        for sampler, reuse in itertools.product(samplers,(0,1)):
            rows.append(dict(case=f'{route}-s{sampler}-k{reuse}-e{engine}-f{forward}',
                sampler=sampler,kv_reuse=reuse,engine=engine,route=route,
                resident_forward=forward,expected_route=actual))
    rows.append(dict(case='complete-auto-s1-k1-e1-f1',sampler=1,kv_reuse=1,
                     engine=1,route='auto',resident_forward=1,expected_route='resident_sampling'))
    return rows
