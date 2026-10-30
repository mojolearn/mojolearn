"""Recover quality only from an identical retained model/input/output evaluation.

Never rerun a timed call or overwrite the original failure. A matching existing
quality artifact is required; otherwise the cell stays pending.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from matrix import parse_witness, validate_witness


def fingerprint(row, directory):
    return (row['model_key'], row['smoke_only'], row['options'],
            hashlib.sha256((directory/'prompt.npy').read_bytes()).hexdigest(),
            row['rounds'][1]['output_sha256'])


def recover(root):
    manifest=json.loads((root/'manifest.json').read_text())
    rows=json.loads((root/'summary.json').read_text())
    candidates=[]
    for row in rows:
        path=root/row['model_key']/('read'+str(row['read_chunk']))/row['case']
        if row.get('status')=='measured' and row.get('quality',{}).get('finite_logits'):
            candidates.append((row,path,fingerprint(row,path)))
    recovered=[]
    for row in rows:
        if row.get('status')!='failed-quality':continue
        path=root/row['model_key']/('read'+str(row['read_chunk']))/row['case']
        error=row.get('quality',{}).get('error','')
        if 'native generation engine does not replace the existing resident route' not in error:
            recovered.append(dict(cell=str(path),status='pending',reason='unrelated quality failure requires separate repair'));continue
        wanted=fingerprint(row,path)
        matches=[(good,d) for good,d,f in candidates if f==wanted]
        if not matches:
            recovered.append(dict(cell=str(path),status='pending',reason='no exact retained quality evaluation'));continue
        good,d=matches[0]
        with np.load(d/'quality-logits.npz') as quality:
            if not np.array_equal(quality['target_tokens'],np.load(path/'scored.npy')):
                raise ValueError('retained target bytes differ')
            if not np.isfinite(quality['logits']).all() or not np.isfinite(quality['nll']).all():
                raise ValueError('retained quality is nonfinite')
        actual=validate_witness(parse_witness((path/'execution.log').read_text()),row['generated_tokens'],row['read_chunk'])
        target=path/'quality-recovery';target.mkdir(exist_ok=True)
        shutil.copy2(d/'quality-logits.npz',target/'quality-logits.npz')
        receipt=dict(status='recovered-from-identical-retained-evaluation',source_commit=manifest['source_commit'],
                     recovery_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     original_status=row['status'],original_error=error,cell=str(path),quality_source_cell=str(d),
                     exact_match='model key, same frozen source/binary, options, exact prompt file hash and scored token bytes',
                     quality=good['quality'],actual_generation_routes=actual,scored_output_sha256=row['rounds'][1]['output_sha256'],
                     quality_artifact_sha256=hashlib.sha256((target/'quality-logits.npz').read_bytes()).hexdigest(),
                     interpretation='Same-model finite/NLL check only; no independent statistical quality claim; no timing rerun or original failure rewrite')
        (target/'receipt.json').write_text(json.dumps(receipt,indent=2))
        recovered.append(receipt)
    (root/'quality-recovery-summary.json').write_text(json.dumps(recovered,indent=2))
    return recovered


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
    rows=recover(a.root)
    print(json.dumps(dict(cells=len(rows),recovered=sum(x['status']=='recovered-from-identical-retained-evaluation' for x in rows),pending=sum(x['status']=='pending' for x in rows))))
