#!/usr/bin/env python3
"""Append a missing linalg pair to an immutable, verified medium-wave revision.

Copies existing artifacts byte for byte; never edits the donor or reruns quality.
Numerical source, compiler environment, arm and architecture remain identical.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

SHA='a4d01a130c8d05ff0208a39989d36b86b69aa0ed'

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def require(value,message):
    if not value: raise ValueError(message)
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')
def inventory(source):
    result={}
    for path in (source/'python/mojolearn').rglob('*.so'):
        require(not path.is_symlink() and path.resolve().is_relative_to(source.resolve()),'unsafe artifact path')
        result[str(path.relative_to(source))]=digest(path)
    return result
def clean(source):
    require(subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==SHA,'wrong numerical source')
    require(subprocess.run(['git','-C',str(source),'diff','--quiet','HEAD','--']).returncode==0,'dirty numerical source')
def copy_checked(source,destination,records):
    for relative,wanted in records.items():
        src=source/relative;dst=destination/relative
        require(src.resolve().is_relative_to(source.resolve()) and not src.is_symlink(),'source path escape')
        require(dst.resolve().is_relative_to(destination.resolve()),'destination path escape')
        require(digest(src)==wanted,'donor changed during copy')
        require(not dst.exists(),'destination artifact already exists')
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst)
        require(digest(dst)==wanted,'copied artifact differs')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--plan',required=True,type=Path);p.add_argument('--old-plan',required=True,type=Path)
    p.add_argument('--repo',type=Path,default=Path('/root/mojolearn'))
    p.add_argument('--supplement-prefix',type=Path,default=Path('/root/lq/linalg-a4d'))
    a=p.parse_args();old=read(a.old/'prepare.json');plan=read(a.plan);oldplan=read(a.old_plan)
    require(old['status']=='PASS' and old['identity']['sha']==SHA,'old preparation incomplete')
    require(digest(a.old_plan)==old['identity']['plan_sha256'],'original plan differs')
    require(plan['builders']==oldplan['builders']+['build_linalg.sh','build_linalg_host.sh'],'unexpected dependency delta')
    require(plan['cases']==oldplan['cases'],'case inventory changed')
    require(plan['required_quality_gates']==oldplan['required_quality_gates'],'quality gate inventory changed')
    # The only reviewed quality-plan change fixes the missing IF output argument.
    normalized=json.loads(json.dumps(plan));normalized['builders']=oldplan['builders']
    for gate in normalized['quality_gates']:
        if gate['id']=='iforest_lifetime':
            require(gate['args']==['--out','{report}'],'unexpected IF repair')
            gate.pop('args')
    require(normalized==oldplan,'unreviewed plan changes')
    require(set(old['arms'])=={'on','off'},'old arms incomplete')
    identity=dict(old['identity'],plan_sha256=digest(a.plan))
    prepared={};origins={}
    for arm in ('on','off'):
        source=a.old/arm/'source';clean(source)
        recorded=read(a.old/arm/'build-products.json')
        require(recorded and inventory(source)==recorded,'old product inventory changed')
        require(len(recorded)==51,'expected fifty bindings plus helper')
        folder=Path(str(a.supplement_prefix)+'-'+arm);supp=read(folder/'native-build.json');clean(folder/'source')
        for key,value in dict(sha=SHA,arm=arm,vendor=identity['vendor'],arch=identity['gpu_arch'],mode='identical',status='PASS').items():
            require(supp.get(key)==value,'supplement context mismatch: '+key)
        require(supp['expected_builders']==['linalg','linalg_host'] and set(supp['modules'])=={'linalg','linalg_host'},'supplement module inventory differs')
        require(supp['effective_mojo_build_flags']==('-D MOJOLEARN_IDN_ALL_OFF=1' if arm=='off' else ''),'supplement flags differ')
        require(not supp['explicit_defines'],'unexpected numerical define')
        require(supp['bootstrap'] and all(x['rc']==0 for x in supp['bootstrap']),'supplement bootstrap failed')
        extra={}
        for name,row in supp['modules'].items():
            require(row['status']=='PASS' and row['rc']==0 and row['import_smoke']['rc']==0,'supplement build/import failed')
            relative='python/mojolearn/'+('host' if name.endswith('_host') else 'identical')+'/_mojolearn_'+name+'.so'
            require(Path(row['artifact'])==folder/'source'/relative,'supplement artifact path differs')
            require(digest(row['artifact'])==row['sha256'],'supplement artifact changed')
            extra[relative]=row['sha256']
        expected=dict(extra);helper='python/mojolearn/.libs/libMojolearnMath.so'
        expected[helper]=recorded[helper]
        require(inventory(folder/'source')==expected,'supplement helper or complete inventory differs')
        require(digest(source/'pixi.lock')==digest(folder/'source/pixi.lock')==digest(a.repo/'pixi.lock'),'locked toolchain differs')
        prepared[arm]=(source,recorded,folder,extra)
        origins[arm]={'old_inventory_sha256':digest(a.old/arm/'build-products.json'),'supplement_receipt':str(folder/'native-build.json'),'supplement_receipt_sha256':digest(folder/'native-build.json'),'reused_artifacts':recorded,'added_artifacts':extra}
    a.out.mkdir(parents=True,exist_ok=False)
    provenance={'schema':1,'numerical_source':SHA,'identity':identity,'old_revision':str(a.old),'old_prepare_sha256':digest(a.old/'prepare.json'),'old_quality_sha256':digest(a.old/'quality.json'),'old_plan_sha256':digest(a.old_plan),'new_plan_sha256':digest(a.plan),'arms':origins,'status':'COPYING','compiler_sha256':digest(a.repo/'.pixi/envs/default/bin/mojo'),'pixi_lock_sha256':digest(a.repo/'pixi.lock'),'composer_sha256':digest(__file__)}
    save(a.out/'revision-provenance.json',provenance)
    report={'identity':identity,'phase':'prepare','status':'INCOMPLETE','arms':{},'method':'verified unchanged artifacts plus missing linalg dependency pair; no repeated builds'}
    save(a.out/'wave.json',identity)
    for arm,(source,recorded,folder,extra) in prepared.items():
        destination=a.out/arm/'source';destination.parent.mkdir(parents=True)
        with (a.out/(arm+'-worktree.log')).open('w') as log:
            subprocess.run(['git','-C',str(a.repo),'worktree','add','--detach',str(destination),SHA],stdout=log,stderr=subprocess.STDOUT,check=True)
        (destination/'.pixi').symlink_to(a.repo/'.pixi',target_is_directory=True)
        copy_checked(source,destination,recorded);copy_checked(folder/'source',destination,extra)
        expected=dict(recorded,**extra);require(inventory(destination)==expected,'new complete inventory differs')
        clean(destination);save(a.out/arm/'build-products.json',expected)
        origins[arm]['new_inventory_sha256']=digest(a.out/arm/'build-products.json')
        report['arms'][arm]=[{'id':'verified-original-artifacts','rc':0,'count':len(recorded),'inventory_sha256':origins[arm]['old_inventory_sha256']},{'id':'verified-linalg-supplement','rc':0,'count':len(extra),'receipt_sha256':origins[arm]['supplement_receipt_sha256']}]
    report['status']='PASS';save(a.out/'prepare.json',report)
    provenance['status']='PRODUCTS_FROZEN_QUALITY_RECONCILIATION_REQUIRED';save(a.out/'revision-provenance.json',provenance)
    print('REVISION_PRODUCTS_FROZEN',a.out,identity['vendor'],flush=True)

if __name__=='__main__':main()
