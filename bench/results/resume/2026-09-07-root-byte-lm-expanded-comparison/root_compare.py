from pathlib import Path
import json,hashlib
from byte_lm_validation_admit import admit
from byte_lm_state_compare import load_capture,compare_continuous,sha,COUNTS
r=Path('/Users/andrewhendel/CascadeProjects/mojolearn');base=r/'bench/results/resume'
paths={'cuda':base/'2026-09-07-root-byte-lm-nvidia-common/run2/remote/byte-lm-validation','hip':base/'2026-09-07-root-byte-lm-do-amd/run6/remote/byte-lm-do-output/byte-lm-validation'}
admissions={k:admit(p) for k,p in paths.items()};captures={k:load_capture(p/'full128','continuous',k) for k,p in paths.items()}
expected=json.loads((base/'2026-09-07-root-byte-lm-do-amd/preparation-r6/expanded-source-inventory.json').read_text())['files']
assert all(a['source_commit']=='eac39c367beeddb8ba4792551d154673e654ce21' for a in admissions.values())
assert all(c['source']==expected for c in captures.values())
assert compare_continuous(captures['cuda'],captures['hip'])==128
out=base/'2026-09-07-root-byte-lm-expanded-comparison';out.mkdir(exist_ok=True)
result={'schema':'mojolearn.byte-lm.continuous-admission.v1','status':'QUALIFIED_BOUNDED_CONTINUOUS','identity_admitted':True,'learning_admitted':True,'resume_admitted':False,'metal_admitted':False,'commit':'eac39c367beeddb8ba4792551d154673e654ce21','source_files':len(expected),'compared_steps':128,'raw_32bit_cells_per_step':sum(COUNTS.values()),'checkpoint_sha256':sha(captures['cuda']['checkpoint']),'initial_heldout_loss':captures['cuda']['summary']['initial_heldout']['mean_loss'],'final_heldout_loss':captures['cuda']['summary']['final_heldout']['mean_loss'],'devices':{'cuda':'RunPod RTX4090 sm_89','hip':'DigitalOcean MI325X VF gfx942'},'campaigns':{k:str(p.relative_to(r)) for k,p in paths.items()},'summary_sha256':{k:c['summary_sha256'] for k,c in captures.items()},'admissions':admissions,'scope':'Complete retained raw state at all128 steps, identical pinned inputs/configuration/source, heldout tokens/losses and final checkpoint; both independent first-step gradient/AdamW oracles. Expanded-source continuous result only; no Metal, new resume, speed or larger-model claim.'}
raw=json.dumps(result,sort_keys=True,indent=2)+'\n';(out/'comparison.json').write_text(raw)
print({k:result[k] for k in ('status','source_files','compared_steps','initial_heldout_loss','final_heldout_loss','checkpoint_sha256')});print('comparison_sha256',hashlib.sha256(raw.encode()).hexdigest())
