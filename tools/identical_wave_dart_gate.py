#!/usr/bin/env python3
"""Our source DART quality against immutable stored results; never run opponents."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True,type=Path)
    p.add_argument('--report',required=True,type=Path)
    p.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--reference',type=Path,default=Path(__file__).with_name('identical_wave_dart_reference.json'))
    a=p.parse_args();root=a.source.resolve()
    sys.path.insert(0,str(root/'tools'));sys.path.insert(0,str(root/'python'))
    import numpy as np
    import bench_board_algos as board
    from classical_two_datasets import sha256_array
    from identical_wave_worker import source_provenance
    stored=json.loads(a.reference.read_text());results=[]
    for row in stored['rows']:
        lane=row['lane'];block,record=board._load_block(lane,row['dataset'],str(a.data))
        for name,expected in row['input_arrays'].items():
            value=np.ascontiguousarray(block[name])
            assert list(value.shape)==expected['shape'] and str(value.dtype)==expected['dtype'],(lane,name,'shape/dtype mismatch')
            assert sha256_array(value)==expected['sha256'],(lane,name,'fixture hash mismatch')
        assert board.lane_config(lane)['params']==row['params'],(lane,'parameters changed from stored fixture')
        arrays=board.lane_arrays(lane,block)
        runner=board.build(lane,'ours',arrays)
        provenance=source_provenance(root,os.environ['MOJOLEARN_VENDOR'])
        assert runner.info.get('numeric_mode_used')=='identical','mode readback missing'
        runner.fit();runner.infer()  # Exactly one untimed own fit; zero opponent calls.
        output=runner.outputs();q=board.quality(lane,arrays,{'ours':output})['ours']
        value=q[row['metric']]
        passed=bool(np.isfinite(value) and value>=row['minimum'])
        results.append({'lane':lane,'dataset':row['dataset'],'status':'PASS' if passed else 'FAILED','quality':q,'metric':row['metric'],'minimum':row['minimum'],'tolerance':row['tolerance'],'stored_opponent_quality':row['stored_opponent_quality'],'source_reference_sha256':row['source_sha256'],'provenance':provenance})
        print('DART_STORED_ROW',lane,results[-1]['status'],json.dumps(q),flush=True)
    a.report.parent.mkdir(parents=True,exist_ok=True)
    status='PASS' if all(r['status']=='PASS' for r in results) else 'FAILED'
    a.report.write_text(json.dumps({'status':status,'opponents_executed':0,'checks':results},indent=2)+'\n')
    print('DART_STORED_QUALITY',status,'checks',len(results),'opponents_executed=0')
    return 0 if status=='PASS' else 1
if __name__=='__main__':sys.exit(main())
