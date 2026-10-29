from pathlib import Path
import hashlib,json,sys
r=Path(__file__).parent
root=Path('/Users/andrewhendel/mojolearn-wt/host-attribute-error-repair-20261029')
sys.path.insert(0,str(root/'tools'))
import macos_cpu_compile_contract as mc
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
stage=json.loads((r/'stage-receipt.json').read_text())
ctx={'selected_binding_build_flags':{},'selected_binding_compile_jobs':{},'selected_macos_cpu_toolchains':{}}
producers=[]
for own in stage['owners']:
 rp=Path(own['compile_receipt']);d=json.loads(rp.read_text());raw=Path(own['raw']);assert sha(raw)==own['raw_sha256']==d['sha256']
 files=d['fields']['source']['files'];assert all(sha(root/rel)==value for rel,value in files.items())
 tool=d['fields']['toolchain'];key='osx-arm64/host/-/host/'+own['stem']
 ctx['selected_macos_cpu_toolchains'][key]=mc.contract(tool['compiler_sha256'],tool['locked']['pixi_lock_sha256'],mc.read_image(raw))
 ctx['selected_binding_build_flags'][key]={}
 ctx['selected_binding_compile_jobs'][key]='10'
 producers.append({'binding_key':key,'producer':'development-build','receipt':str(rp),'receipt_sha256':sha(rp),'source_commit':stage['source_commit'],'raw_binary':{'path':str(raw),'sha256':sha(raw)}})
(r/'configuration-replacement-fragment.json').write_text(json.dumps({'dependency_context':ctx},indent=2)+'\n')
(r/'replacement-producers.json').write_text(json.dumps(producers,indent=2)+'\n')
print('two exact default-optimized CPU contracts; ten workers encoded as decimal strings; no ledger writes')
