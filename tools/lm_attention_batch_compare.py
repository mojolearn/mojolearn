#!/usr/bin/env python3
"""B4 exact witnesses and a same-device B1/B4 throughput comparison."""
import copy
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from lm_attention_repair_compare import HASHES, compare, default_capacity


def price_record(r, batch):
    assert r['shape'] == [batch,2048,768,12,12,64,2048,12,50257], 'price shape'
    assert len(r['steps']) == 700, 'price coverage'
    for i, row in enumerate(r['steps']):
        assert row['step'] == i and row['completed_steps'] == i + 1, 'price sequence'
        assert row['attention']['repair_masked_tail'] is True, 'price arm'
        assert math.isfinite(row['seconds']) and row['seconds'] > 0, 'price timing'
    default_capacity(r, batch)


def main():
    root = Path(sys.argv[1])
    a, b, one = [json.loads((root/name/'result.json').read_text()) for name in ('legacy4','default4','default1')]
    for key in HASHES:
        broken = copy.deepcopy(b)
        broken['steps'][-1][key] = 'corrupted'
        try: compare(a, broken, False, batch=4)
        except AssertionError as e:
            assert str(e) == key, str(e)
            print('EXPECTED FAIL B4', key)
        else: raise AssertionError('BLIND B4 ' + key)
    for left, right, reason in ((a,a,'repaired arm'),(b,b,'legacy arm')):
        try: compare(left, right, False, batch=4)
        except AssertionError as e:
            assert str(e) == reason, str(e)
            print('EXPECTED FAIL B4 same arm', reason)
        else: raise AssertionError('BLIND B4 arm')
    compare(a,b,batch=4)
    default_capacity(a,4)
    if not any(x['attention']['released_eager_bytes'] for x in a['steps']): print('INERT legacy B4 release')
    for label, mutate in (
        ('price shape', lambda x: x['shape'].__setitem__(0, 2)),
        ('price coverage', lambda x: x['steps'].pop()),
        ('price sequence', lambda x: x['steps'][0].__setitem__('completed_steps', 9)),
        ('price arm', lambda x: x['steps'][0]['attention'].__setitem__('repair_masked_tail', False)),
        ('price timing', lambda x: x['steps'][0].__setitem__('seconds', 0)),
    ):
        broken=copy.deepcopy(one); mutate(broken)
        try: price_record(broken, 1)
        except AssertionError as e:
            assert str(e) == label, str(e)
            print('EXPECTED FAIL', label)
        else: raise AssertionError('BLIND ' + label)
    price_record(one,1); price_record(b,4)
    for row in one['steps']: print('OBSERVED B1 step',row['step'],'loss',row['loss'],'seconds',row['seconds'])
    times={}
    for name, r, batch in (('legacy4',a,4),('default1',one,1),('default4',b,4)):
        rows=r['steps']; median=statistics.median(x['seconds'] for x in rows[-50:]); times[name]=median
        print(name,'tail_median_seconds',median,'tokens_per_second',batch*2048/median,
              'sampled_peak_mib',max(x['device_used_mb'] or 0 for x in rows),
              'backward_status',dict(Counter(v for x in rows for v in x['attention']['backward_status'])),
              'repair_sites',dict(Counter(v for x in rows for v in x['attention']['backward_repair_sites'])))
    print('SAME_DEVICE_BATCH_MULTIPLIER',4*times['default1']/times['default4'])
    print('B4_REPLAY_SPEEDUP',times['legacy4']/times['default4'])


if __name__ == '__main__': main()
