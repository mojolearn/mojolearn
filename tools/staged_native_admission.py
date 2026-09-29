"""Strict, explicit provenance for staging previously compiled native outputs.

This is not a build proof. Raw original compiler receipts/stamps are embedded
unchanged, and every admitted output is bound to its original successful build,
source closure, tier, compiler and shipped bytes. No inferred build success.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

SCHEMA = 'mojolearn.linux.staged-admission.v1'
BUILD_SCHEMA = 'mojolearn.linux.build-provenance.v1'


def require(ok, why):
    if not ok:
        raise ValueError(why)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def witness(path):
    raw = Path(path).read_bytes()
    return {'sha256': sha(raw), 'utf8': raw.decode('utf-8')}


def decoded(record):
    require(isinstance(record, dict) and isinstance(record.get('utf8'), str), 'Missing raw witness')
    raw = record['utf8'].encode('utf-8')
    require(sha(raw) == record.get('sha256'), 'Raw witness SHA mismatch')
    return json.loads(raw)


def valid_sha(value, length=64):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{'+str(length)+'}', value) is not None


def compiler_identity(raw):
    require(isinstance(raw,str), 'Missing compiler identity')
    # Preserve full raw stdout in the witness; tcmalloc diagnostics contain PIDs.
    versions=[line for line in raw.splitlines() if re.fullmatch(r'Mojo \S+ \([^)]+\)',line)]
    require(len(versions)==1, 'Missing or ambiguous compiler version')
    return versions[0]


def expected_outputs():
    import verify_linux_surface_qualification as surface
    rows = {('' if mode == 'fast' else mode+'/')+name+'.so': mode
            for mode in surface.MODES for name in surface.expected_bindings(mode, True)}
    rows.update({'host/'+name+'.so': 'host' for name in surface.wheel_host_bindings()})
    return rows


def validate_staged(proof, source_root=None):
    require(proof.get('schema') == SCHEMA and proof.get('action') == 'stage-validated-native'
            and proof.get('complete') is True, 'Incomplete staged admission')
    require('build_exit' not in proof, 'Staged admission must not impersonate a build')
    source = proof.get('source_commit')
    require(valid_sha(source, 40), 'Invalid admitted source')
    admission = decoded(proof.get('stage_witness'))
    require(admission.get('schema') == 'mojolearn.native-staging-admission.v1'
            and admission.get('status') == 'STAGED', 'Missing successful staging receipt')
    require(admission.get('admitted_package_source_commit') == source, 'Wrong admitted source')
    require(valid_sha(admission.get('qualified_source_commit'), 40), 'Invalid qualified source')
    compiler = compiler_identity(admission.get('compiler'))
    vendor, arch = admission.get('vendor'), admission.get('arch')
    require(vendor in ('cuda','hip') and isinstance(arch,str)
            and re.fullmatch('sm_[0-9]+[a-z]*' if vendor=='cuda' else 'gfx[0-9a-f]+',arch), 'Invalid vendor/architecture')
    inventory = proof.get('source_inventory')
    require(isinstance(inventory,list) and inventory and inventory == admission.get('source_inventory'), 'Wrong source inventory')
    require(len(inventory)==len({r[0] for r in inventory}), 'Duplicate source inventory')
    for rel, digest in inventory:
        require(not PurePosixPath(rel).is_absolute() and '..' not in PurePosixPath(rel).parts and valid_sha(digest), 'Unsafe source inventory')
    require(sha(json.dumps(inventory,separators=(',',':')).encode()) == proof.get('source_sha256') == admission.get('source_sha256'), 'Wrong source inventory SHA')
    if source_root is not None:
        from check_linux_release_qualification import tracked_native_inventory
        require(tracked_native_inventory(source_root)==inventory, 'Current source differs from admitted source inventory')
    expected = expected_outputs()
    outputs = admission.get('outputs')
    require(isinstance(outputs,list) and len(outputs)==len(expected)
            and {r.get('relative') for r in outputs}==set(expected), 'Missing, duplicate or unexpected admitted output')
    primary_rows = {r['relative']:r for r in outputs}
    manifest = decoded(proof.get('stage_manifest_witness'))
    require(len(manifest.get('extensions',[]))==len(expected) and {r['path']:r['sha256'] for r in manifest.get('extensions',[])}
            == {r['relative']:r['staged_sha256'] for r in outputs}, 'Original stage manifest SHA mismatch')
    libraries = {r['name']:r['sha256'] for r in manifest.get('staged_libs',[])}
    require(libraries and len(libraries)==len(manifest.get('staged_libs',[])) and all(valid_sha(v) for v in libraries.values()), 'Missing runtime manifest')
    if 'canonical_host_stage_witness' in proof:
        canonical = decoded(proof['canonical_host_stage_witness'])
        require(canonical.get('schema')==admission['schema'] and canonical.get('status')=='STAGED'
                and canonical.get('admitted_package_source_commit')==source
                and canonical.get('source_inventory')==inventory and canonical.get('source_sha256')==proof['source_sha256']
                and compiler_identity(canonical.get('compiler'))==compiler and canonical.get('vendor')=='cuda'
                and isinstance(canonical.get('arch'),str) and re.fullmatch('sm_[0-9]+[a-z]*',canonical['arch']), 'Canonical host source/compiler differs')
        crows = canonical.get('outputs',[])
        require(len(crows)==len(expected) and {r.get('relative') for r in crows}==set(expected), 'Incomplete canonical host stage')
        crows = {r['relative']:r for r in crows}
        cm = decoded(proof.get('canonical_host_manifest_witness'))
        require(len(cm.get('extensions',[]))==len(expected) and {r['path']:r['sha256'] for r in cm.get('extensions',[])}
                == {r['relative']:r['staged_sha256'] for r in crows.values()}, 'Canonical stage manifest SHA mismatch')
        composition = decoded(proof.get('composition_witness'))
        require(composition.get('schema')=='mojolearn.canonical-host-assembly.v1'
                and composition.get('package_source')==source, 'Missing explicit canonical composition')
        source_links = composition.get('sources',{})
        require(source_links.get('device_manifest_sha256')==proof['stage_manifest_witness']['sha256']
                and source_links.get('canonical_manifest_sha256')==proof['canonical_host_manifest_witness']['sha256'], 'Composition manifest linkage differs')
        overrides = composition.get('overrides',[])
        host_names = {rel for rel,tier in expected.items() if tier=='host'}
        require(len(overrides)==len(host_names)+1
                and {r.get('relative') for r in overrides}==host_names|{'.libs/libMojolearnMath.so'}, 'Incomplete or device-changing composition')
        for override in overrides:
            rel=override['relative']
            if rel.startswith('host/'):
                old=primary_rows[rel]['staged_sha256'];new=crows[rel]['staged_sha256']
            else:
                old=libraries['libMojolearnMath.so']
                new=next(r['sha256'] for r in cm['staged_libs'] if r['name']=='libMojolearnMath.so')
                libraries['libMojolearnMath.so']=new
            original_key='original_hip_sha256' if vendor=='hip' else 'original_hopper_sha256'
            require(override.get(original_key)==old and override.get('canonical_cuda_sha256')==new, 'Wrong composition override SHA')
        outputs = [crows[r['relative']] if expected[r['relative']]=='host' else r for r in outputs]
    require(proof.get('runtime_libraries')==libraries, 'Runtime composition SHA mismatch')
    builds = {}
    for build in proof.get('native_builds',[]):
        meta, results = decoded(build.get('manifest')), decoded(build.get('results'))
        require(compiler_identity(meta.get('compiler'))==compiler and valid_sha(meta.get('source_commit'),40), 'Wrong original compiler/source')
        require(isinstance(results,list), 'Missing original build results')
        key = build['results']['sha256']
        require(key not in builds, 'Duplicate original build witness')
        builds[key]=(meta, results)
    require(builds, 'Missing original builds')
    links = proof.get('output_witnesses',{})
    require(set(links)==set(expected), 'Missing or unexpected output witnesses')
    extensions, hosts = {}, {}
    checked_closures = {}
    for row in outputs:
        rel=row['relative']; tier=expected[rel]; link=links[rel]
        raw_sha, staged_sha = row.get('original_sha256'), row.get('staged_sha256')
        require(valid_sha(raw_sha) and valid_sha(staged_sha), 'Invalid raw/staged SHA')
        original=decoded(link.get('stamp'))
        require(original==row.get('original_stamp') and original.get('binding')==rel
                and original.get('scope')=='closure' and valid_sha(original.get('commit'),40), 'Wrong original source stamp')
        closure=row.get('admitted_closure')
        require(isinstance(closure,dict) and closure.get('digest')==original.get('digest')
                and valid_sha(closure.get('digest')) and closure.get('sources')==original.get('sources')
                and closure.get('files')==original.get('files'), 'Changed native source closure')
        if source_root is not None:
            import binding_stamps
            script = original['script']
            if script not in checked_closures:
                checked_closures[script] = binding_stamps.digest(script, repo=Path(source_root))
            require(checked_closures[script] == closure, 'Native closure does not match admitted checkout')
        meta, results = builds.get(link.get('build_results_sha256'), ({},[]))
        matches=[r for r in results if r.get('relative')==rel and r.get('sha256')==raw_sha
                 and r.get('tier')==tier and r.get('script')==original.get('script') and type(r.get('exit')) is int and r['exit']==0
                 and r.get('exists') is True and r.get('timeout') is False]
        require(len(matches)==1 and meta.get('source_commit')==original.get('commit')
                and (tier=='host' or meta.get('arch')==arch), 'Missing, failed or wrong-tier original build receipt')
        receipt=row.get('original_tier_receipt',{})
        require(receipt.get('relative')==rel and receipt.get('tier')==tier and receipt.get('sha256')==raw_sha
                and ('exit' not in receipt or (type(receipt['exit']) is int and receipt['exit']==0)), 'Wrong original tier receipt')
        mode=row.get('readback',{}).get('mode')
        require(row.get('readback',{}).get('vendor')==('cpu' if tier=='host' else vendor), 'Wrong vendor readback')
        no_mode=Path(rel).stem in ('_mojolearn_solver','_mojolearn_tsa') and tier in ('fast','identical')
        require((type(mode) is int and mode=={'fast':0,'deterministic':2,'identical':1,'host':1}[tier])
                or (mode is None and no_mode), 'Wrong mode readback')
        member=f'mojolearn/{vendor}/{arch}/{rel}'
        (hosts if tier=='host' else extensions)[member]=staged_sha
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'packaging/linux'))
    from device_glue import DEVICE_GLUE
    by_rel={r['relative']:r for r in outputs}
    for rel,tier in expected.items():
        images=by_rel[rel].get('embedded_architectures')
        if tier=='host':
            require(images==[], 'Host artifact carries a device image')
        elif images!=[arch]:
            delegates=DEVICE_GLUE.get((tier,Path(rel).stem))
            require(images==[] and delegates, 'Missing or wrong embedded architecture')
            for delegate in delegates:
                key=('' if tier=='fast' else tier+'/')+delegate+'.so'
                require(by_rel[key].get('embedded_architectures')==[arch], 'Missing device delegate image')
    require(proof.get('extensions')==extensions and proof.get('host_extension')==hosts, 'Shipped SHA/inventory mismatch')
    return True


def complete_native_proof(proof, source_root=None):
    """Keep traditional complete-build admission unchanged; add explicit staging."""
    try:
        if proof.get('schema')==SCHEMA:
            return validate_staged(proof,source_root)
        return (proof.get('schema')==BUILD_SCHEMA and proof.get('complete') is True
                and type(proof.get('build_exit')) is int and proof['build_exit']==0
                and proof.get('action')=='build')
    except (ValueError,TypeError,KeyError,IndexError,OSError,StopIteration,AttributeError):
        return False


def adapt(admission_path, stamps, set_root, source_root, native_builds, raw_root,
          stage_manifest, canonical_admission=None, canonical_stamps=None,
          canonical_raw_root=None, canonical_manifest=None, composition=None):
    stage = witness(admission_path); admission = decoded(stage)
    builds = [{'manifest':witness(m),'results':witness(r)} for m,r in native_builds]
    require(admission['status']=='STAGED', 'Staging incomplete')
    extra={};outputs=admission['outputs'];host_rows={}
    primary_manifest=witness(stage_manifest)
    libraries={r['name']:r['sha256'] for r in decoded(primary_manifest)['staged_libs']}
    if canonical_admission is not None:
        require(all((canonical_stamps,canonical_raw_root,canonical_manifest,composition)), 'Incomplete canonical source paths')
        extra=dict(canonical_host_stage_witness=witness(canonical_admission),
                   canonical_host_manifest_witness=witness(canonical_manifest),
                   composition_witness=witness(composition))
        host_rows={r['relative']:r for r in decoded(extra['canonical_host_stage_witness'])['outputs']
                   if r['relative'].startswith('host/')}
        outputs=[host_rows[r['relative']] if r['relative'].startswith('host/') else r for r in outputs]
        libraries['libMojolearnMath.so']=next(r['sha256'] for r in decoded(extra['canonical_host_manifest_witness'])['staged_libs'] if r['name']=='libMojolearnMath.so')
    links={};ext={};hosts={}
    for row in outputs:
        rel=row['relative']; tier=row['original_tier_receipt']['tier']
        use_host=rel in host_rows
        stamp=witness(Path(canonical_stamps if use_host else stamps)/(rel.replace('/','__')+'.json'))
        require(decoded(stamp)==row['original_stamp'], 'Original raw stamp differs')
        raw=Path(canonical_raw_root if use_host else raw_root)/rel;shipped=Path(set_root)/rel
        require(sha(raw.read_bytes())==row['original_sha256'], 'Raw artifact SHA mismatch')
        require(sha(shipped.read_bytes())==row['staged_sha256'], 'Staged artifact SHA mismatch')
        choices=[]
        for build in builds:
            meta=decoded(build['manifest']);results=decoded(build['results'])
            if meta.get('source_commit')!=row['original_stamp']['commit']:
                continue
            if any(r.get('relative')==rel and r.get('tier')==tier and r.get('sha256')==row['original_sha256']
                   and type(r.get('exit')) is int and r['exit']==0 and r.get('exists') is True
                   and r.get('timeout') is False for r in results):
                choices.append(build['results']['sha256'])
        require(choices, 'No original successful compiler receipt for '+rel)
        links[rel]={'stamp':stamp,'build_results_sha256':sorted(choices)[0]}
        key=f"mojolearn/{admission['vendor']}/{admission['arch']}/{rel}"
        (hosts if tier=='host' else ext)[key]=row['staged_sha256']
    for name,digest in libraries.items():
        require(sha((Path(set_root)/'.libs'/name).read_bytes())==digest, 'Runtime artifact SHA mismatch')
    proof=dict(schema=SCHEMA,action='stage-validated-native',complete=True,
               source_commit=admission['admitted_package_source_commit'],
               source_inventory=admission['source_inventory'],source_sha256=admission['source_sha256'],
               stage_witness=stage,stage_manifest_witness=primary_manifest,
               native_builds=builds,output_witnesses=links,runtime_libraries=libraries,
               extensions=ext,host_extension=hosts,**extra)
    validate_staged(proof,source_root)
    return proof


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--admission',required=True);p.add_argument('--stamps',required=True)
    p.add_argument('--set',dest='set_root',required=True);p.add_argument('--source',required=True)
    p.add_argument('--native-build',nargs=2,action='append',required=True,metavar=('MANIFEST','RESULTS'))
    p.add_argument('--raw-root',required=True);p.add_argument('--stage-manifest',required=True)
    p.add_argument('--canonical-admission');p.add_argument('--canonical-stamps')
    p.add_argument('--canonical-raw-root');p.add_argument('--canonical-manifest')
    p.add_argument('--composition');p.add_argument('--out',required=True)
    args=p.parse_args()
    result=adapt(args.admission,args.stamps,args.set_root,args.source,args.native_build,
                 args.raw_root,args.stage_manifest,args.canonical_admission,args.canonical_stamps,
                 args.canonical_raw_root,args.canonical_manifest,args.composition)
    output=Path(args.out);require(not output.exists(),'Refuse overwriting a provenance proof')
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(f"Staged admission: {len(result['extensions'])} device-tier outputs, {len(result['host_extension'])} host outputs")
