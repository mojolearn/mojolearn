"""Offline preparation: reuse uncapped regression blocks and derive class labels."""
import argparse
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'tools'))
    import numpy as np
    import classical_two_datasets as ctd
    args.output.mkdir(parents=True,exist_ok=True)
    for dataset in ['taxi','istella']:
        source=args.input/('reg-'+dataset)
        meta=json.loads(source.with_suffix('.json').read_text())
        assert meta['full_dataset_coverage'] is True and not meta['intrinsic_caps']
        assert meta['fit_rows']==[0,meta['fit_rows_available']]
        for suffix in ['.npz','.json']:
            (args.output/('reg-'+dataset+suffix)).symlink_to(source.with_suffix(suffix))
    # cpu-route: offline benchmark input preparation, identical to the full
    # classification recipe's relevance>0 labels; all feature values retained.
    source=args.input/'reg-istella.npz'
    meta=json.loads(source.with_suffix('.json').read_text())
    with np.load(source,allow_pickle=False) as z:
        arrays={k:np.ascontiguousarray(z[k]) for k in ['X','Xq','y','yq']}
    arrays['y']=(arrays['y']>0).astype(np.float32)
    arrays['yq']=(arrays['yq']>0).astype(np.float32)
    meta.update(block='cls',target='binary relevance>0',source_block=str(source))
    ctd._write_block(str(args.output),'cls-istella',arrays,meta)
    print(json.dumps({'full_dataset_coverage':True,'classification_train':list(arrays['X'].shape),
                      'classification_eval':list(arrays['Xq'].shape)}))


if __name__=='__main__':
    main()
