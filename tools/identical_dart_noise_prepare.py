#!/usr/bin/env python3
"""Prepare temporal resampling inputs from canonical DATA ONLY, never predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def prepare(data,out,source,protocol_template):
    if out.exists():raise ValueError('refuse overwrite: '+str(out))
    reference=json.loads((source/'tools/identical_wave_dart_reference.json').read_text())
    arrays={};audit={}
    for block,name in [('cls','classification'),('reg','regression')]:
        path=data/(block+'-taxi.npz')
        with np.load(path,allow_pickle=False) as z:
            record=next(r for r in reference['rows'] if r['lane']==('dart' if block=='cls' else 'dart-reg'))
            for key,expected in record['input_arrays'].items():
                value=np.ascontiguousarray(z[key]);h=hashlib.sha256()
                h.update(str(value.dtype).encode());h.update(str(value.shape).encode());h.update(value.data)
                if h.hexdigest()!=expected['sha256']:raise ValueError('canonical fixture mismatch: '+block+'/'+key)
            train=z['X']; query=z['Xq']; arrays[name+'_y']=np.ascontiguousarray(z['yq'])
            if len(query)!=100000 or train.shape!=(1000000,11):
                raise ValueError('exact full canonical DART shape required')
            decoded=[]
            for col,n,offset in [(2,24,0),(3,7,0),(4,31,1)]:
                codes=np.unique(train[:,col]); idx=np.searchsorted(codes,query[:,col])
                if len(codes)!=n or np.any(idx>=n) or not np.array_equal(codes[idx],query[:,col]):
                    raise ValueError('standardized time-code recovery not exact')
                decoded.append(idx+offset)
            hour,weekday,day=decoded
            jan=weekday==(day-1)%7
            feb=(day<=29)&(weekday==(day-1+3)%7)
            # Monthly source is Jan/Feb2024. One observed reg row has day1,
            # weekdayFriday: consistent with the immediately adjacent Mar1.
            # Explicit inference, not a recovered original timestamp/year.
            spill=(day==1)&(weekday==4)
            if not np.all(jan|feb|spill):raise ValueError('unexplained calendar code')
            if np.any(jan):raise ValueError('unexpected January holdout; protocol requires rereview')
            if np.count_nonzero(spill)!=(1 if block=='reg' else 0):
                raise ValueError('spillover profile changed; protocol requires rereview')
            t=(np.where(feb,31,60)+day-1)*24+hour
            order=np.argsort(t,kind='stable').astype(np.int64)
            arrays[name+'_order']=order
            arrays[name+'_calendar_hour']=t.astype(np.int64)
            audit[name]={'dataset_sha256':sha(path),'n':len(query),
                         'observed_hour_groups':int(len(np.unique(t))),
                         'source_order_hour_descents':int(np.count_nonzero(np.diff(t)<0)),
                         'inferred_march1_rows':int(spill.sum()),
                         'order_sha256':hashlib.sha256(order.astype('<i8').tobytes()).hexdigest()}
    out.mkdir(parents=True)
    targets=out/'targets.npz';np.savez(targets,**arrays)
    evidence=[]
    for rel in ['tools/speed_gbdt_arm.py','tools/bench_board_more.py','tools/classical_two_datasets.py']:
        path=source/rel;evidence.append({'path':str(path.resolve()),'sha256':sha(path)})
    metadata={'targets_sha256':sha(targets),'sampling_design':'temporal_tail_stride_hour_order',
              'iid_justified':False,'resampling_unit':'ordered_holdout_rows',
              'order_verified':True,'source_evidence':evidence,
              'dependence_justification':'Stable calendar-hour order recovered exactly from standardized discrete time codes and Jan/Feb2024 source, with one explicitly inferred March1 spillover. Circular contiguous-row blocks retain local dependence; fixed1024/2048/4096 sensitivity addresses block-length uncertainty. All rows retained.',
              'limitations':['No original minutes/seconds; stable original order within each hour.',
                             'Regression one March1 date is inferred from day/weekday and adjacent monthly source; original year/month not retained.',
                             'Classification covers only about six days; no long-horizon drift or training-seed claim.',
                             'Fixed row blocks span varying clock durations; finite-window bootstrap is conditional on observed holdout.'],
              'data_audit':audit}
    (out/'target-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    base=json.loads(protocol_template.read_text());paths=[]
    for length in [1024,2048,4096]:
        p=json.loads(json.dumps(base));p.update(reviewed=True,status='BASELINE_ONLY_POLICY_REVIEWED',
          policy_review='Data/provenance-only selection before candidate comparison; root owns final decision.')
        p['resampling']={'kind':'moving_block','unit':'ordered_holdout_rows','minimum_units':20,
                         'block_length':length,'justification':metadata['dependence_justification']}
        for name in ['classification','regression']:p['cases'][name]['order_key']=name+'_order'
        p['sensitivity_policy']={'primary_block_length':4096,'required_block_lengths':[1024,2048,4096],
                                'acceptance':'Both metrics must PASS all three frozen policies; never select favorable block size.'}
        path=out/('protocol-block'+str(length)+'.json');path.write_text(json.dumps(p,indent=2)+'\n');paths.append(path)
    manifest={'status':'DATA_ONLY_READY_NOT_CALIBRATED','targets_sha256':sha(targets),
              'metadata_sha256':sha(out/'target-metadata.json'),
              'protocol_sha256':{p.name:sha(p) for p in paths},'predictions_read':False,
              'baseline_margins':'NOT_CALIBRATED; OFF artifacts and source receipts required',
              'data_audit':audit}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--protocol-template',type=Path,default=Path(__file__).with_name('identical_dart_noise_protocol.json'))
    a=p.parse_args()
    if sys.platform!='linux':p.error('authorized Linux statistics server only')
    r=prepare(a.data.resolve(),a.out.resolve(),a.source.resolve(),a.protocol_template.resolve())
    print('DART_TEMPORAL_PREP',r['status'],'protocols',len(r['protocol_sha256']))
if __name__=='__main__':main()
