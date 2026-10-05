#!/usr/bin/env python3
"""Generate forward-only canonical fixtures once, then transfer unchanged via R2.

Uses board shape/range helpers and seeded NumPy arrays. Does not construct or run
mojolearn, Torch reference models, or opponent training. Run on NVIDIA Linux only.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def mamba_metadata(source, names):
    """Read only authoritative shape/range definitions, without importing Torch.

    The corpus module imports and configures Torch at module scope. Its four
    pure metadata functions require only math and these five literal constants.
    Keep their exact source bodies instead of maintaining a second shape table.
    """
    path=source/'mamba/corpus/gen_corpus.py'
    wanted={'fan_in_scale','m3_dims','m3_default_ranges','m3_shapes_for',
            'M3_EXPAND','M3_HEADDIM','M3_NGROUPS','M3_D_STATE','M3_ROPE'}
    selected=[]; found=set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node,ast.FunctionDef) and node.name in wanted:
            selected.append(node);found.add(node.name)
        elif isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            name=node.targets[0].id
            if name in wanted:
                ast.literal_eval(node.value)  # constants must remain literal
                selected.append(node);found.add(name)
    if found!=wanted:raise RuntimeError('corpus metadata definitions changed: '+str(wanted-found))
    import math
    namespace={'math':math}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)
    def spec(model,d):
        if model!='mamba3':raise ValueError('this fixture prepares Mamba3 only')
        shapes=namespace['m3_shapes_for'](d['d_model'],d['batch'],d['length'])
        ranges=namespace['m3_default_ranges'](d['d_model'])
        return [(n,tuple(shapes[n]),ranges[n]) for n in names[model]],(tuple(shapes['x']),ranges['x'])
    return spec,hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if sys.platform!='linux':p.error('canonical preparation runs on the rented Linux box')
    a.out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(a.source.resolve()/'tools'))
    import bench_board_neural as neural
    spec,metadata_hash=mamba_metadata(a.source.resolve(),neural.MAMBA_NAMES)
    sha=subprocess.check_output(['git','-C',str(a.source),'rev-parse','HEAD'],text=True).strip()
    manifest={}
    for lane in ('mamba3-forward','transformer-forward'):
        file=a.out/(lane+'-small.npz')
        record=neural.make_inputs(lane,'small',2,str(file),mamba_spec=spec)
        record.update(sha256=hashlib.sha256(file.read_bytes()).hexdigest(),generator_source_sha=sha,
                      generation_only=True,models_or_opponents_executed=0,
                      corpus_metadata_source_sha256=metadata_hash)
        file.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n')
        manifest[file.name]=record['sha256']
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    assert 'torch' not in sys.modules,'fixture generation imported an opponent'
    print('WAVE_NEURAL_FIXTURES',len(manifest),'models_or_opponents_executed=0')
if __name__=='__main__':main()
