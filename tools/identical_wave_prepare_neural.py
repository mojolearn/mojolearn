#!/usr/bin/env python3
"""Generate forward-only canonical fixtures once, then transfer unchanged via R2.

Uses board shape/range helpers and seeded NumPy arrays. Does not construct or run
mojolearn, Torch reference models, or opponent training. Run on NVIDIA Linux only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if sys.platform!='linux':p.error('canonical preparation runs on the rented Linux box')
    a.out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(a.source.resolve()/'tools'))
    import bench_board_neural as neural
    sha=subprocess.check_output(['git','-C',str(a.source),'rev-parse','HEAD'],text=True).strip()
    manifest={}
    for lane in ('mamba3-forward','transformer-forward'):
        file=a.out/(lane+'-small.npz')
        record=neural.make_inputs(lane,'small',2,str(file))
        record.update(sha256=hashlib.sha256(file.read_bytes()).hexdigest(),generator_source_sha=sha,
                      generation_only=True,models_or_opponents_executed=0)
        file.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n')
        manifest[file.name]=record['sha256']
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('WAVE_NEURAL_FIXTURES',len(manifest),'models_or_opponents_executed=0')
if __name__=='__main__':main()
