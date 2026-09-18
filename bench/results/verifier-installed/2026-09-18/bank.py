from pathlib import Path
import gzip, hashlib, json, shutil

base=Path(__file__).resolve().parent
root=Path('/Users/andrewhendel/mojolearn-wt/verification-coverage-continuation')
out=root/'bench/results/verifier-installed/2026-09-18'
files=['build.py','replay.py','followup.py','validate.py','review_gp.py','bank.py',
       'wheel-receipt.json','api-audit.json','models.json','models.log','self-test.json','self-test.log',
       'replay-receipt.json','replay.log','mamba3.log','slot.json','ordinary-slot.json','classical-slot.json',
       'loaded-lm-capture.json','loaded-lm-compare-cpu.json','loaded-lm-compare-apple.json','loaded-lm-compare-amd.json',
       'models-count-fixed.json','focused-tests.log','post-admission-tests.log','post-admission-slot.json',
       'gp-admission-review.json','gp-admission-applied.json']
files += [str(p.relative_to(base)) for folder in ('ordinary','classical-extended') for p in sorted((base/folder).glob('*')) if p.suffix in ('.json','.log') and p.stat().st_size]
inventory={}
for name in files:
    p=base/name
    data=p.read_bytes()
    target=out/name
    target.parent.mkdir(parents=True,exist_ok=True)
    if len(data)>100_000:
        target=target.with_suffix(target.suffix+'.gz')
        target.write_bytes(gzip.compress(data,mtime=0))
        if (out/name).exists():
            (out/name).unlink()
    else:
        target.write_bytes(data)
    inventory[str(target.relative_to(out))]=dict(original_bytes=len(data),original_sha256=hashlib.sha256(data).hexdigest(),stored_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
(out/'artifact-index.json').write_text(json.dumps(inventory,indent=2)+'\n')
print(len(inventory),'artifacts banked')
