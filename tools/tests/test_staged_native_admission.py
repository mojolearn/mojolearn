"""File-only adversarial provenance tests; no native loading or fitting."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import staged_native_admission as a


def w(value):
    text=json.dumps(value,indent=1)+'\n'
    return {'utf8':text,'sha256':a.sha(text.encode())}


def fixture():
    inventory=[['source.py',a.sha(b'source')]]; outputs=[];results=[];stamps={}
    for rel,tier in a.expected_outputs().items():
        raw=a.sha(('raw:'+rel).encode());staged=a.sha(('staged:'+rel).encode())
        script='build_'+Path(rel).stem+'.sh'
        stamp=dict(binding=rel,commit='a'*40,digest='b'*64,files=1,sources=['source.mojo'],scope='closure',script=script)
        receipt=dict(relative=rel,tier=tier,sha256=raw,script=script,exit=0,timeout=False,exists=True)
        results.append(receipt);stamps[rel]=w(stamp)
        outputs.append(dict(relative=rel,original_sha256=raw,staged_sha256=staged,original_stamp=stamp,
            original_tier_receipt=receipt,admitted_closure={k:stamp[k] for k in ('digest','files','sources','scope')},
            readback=dict(mode={'fast':0,'deterministic':2,'identical':1,'host':1}[tier],vendor='cpu' if tier=='host' else 'hip'),
            embedded_architectures=[] if tier=='host' or Path(rel).stem=='_mojolearn_x_trees' else ['gfx942']))
    stage=dict(schema='mojolearn.native-staging-admission.v1',status='STAGED',admitted_package_source_commit='c'*40,
               qualified_source_commit='d'*40,compiler='Mojo 1.0.0 (test)',vendor='hip',arch='gfx942',outputs=outputs,
               source_inventory=inventory,source_sha256=a.sha(json.dumps(inventory,separators=(',',':')).encode()))
    build=dict(manifest=w(dict(source_commit='a'*40,compiler=stage['compiler'],arch='gfx942')),results=w(results))
    manifest=dict(extensions=[dict(path=r['relative'],sha256=r['staged_sha256']) for r in outputs],
                  staged_libs=[dict(name='libMojolearnMath.so',sha256='e'*64)])
    proof=dict(schema=a.SCHEMA,action='stage-validated-native',complete=True,source_commit='c'*40,
               source_inventory=inventory,source_sha256=stage['source_sha256'],stage_witness=w(stage),
               stage_manifest_witness=w(manifest),native_builds=[build],runtime_libraries={'libMojolearnMath.so':'e'*64},
               output_witnesses={rel:dict(stamp=stamp,build_results_sha256=build['results']['sha256']) for rel,stamp in stamps.items()},
               extensions={},host_extension={})
    for r in outputs:
        member='mojolearn/hip/gfx942/'+r['relative']
        proof['host_extension' if r['relative'].startswith('host/') else 'extensions'][member]=r['staged_sha256']
    return proof


def canonicalize(proof):
    primary=a.decoded(proof['stage_witness']);canonical=copy.deepcopy(primary)
    canonical.update(vendor='cuda',arch='sm_89')
    new_results=[];overrides=[]
    for row in canonical['outputs']:
        rel=row['relative']
        if not rel.startswith('host/'):continue
        old=row['staged_sha256'];row['original_sha256']=a.sha(('canonicalraw:'+rel).encode())
        row['staged_sha256']=a.sha(('canonicalstaged:'+rel).encode());row['original_stamp']['commit']='f'*40
        row['original_tier_receipt']['sha256']=row['original_sha256'];new_results.append(row['original_tier_receipt'])
        overrides.append(dict(relative=rel,original_hip_sha256=old,canonical_cuda_sha256=row['staged_sha256']))
        proof['host_extension']['mojolearn/hip/gfx942/'+rel]=row['staged_sha256']
    native=dict(manifest=w(dict(source_commit='f'*40,compiler=primary['compiler'],arch='sm_89')),results=w(new_results))
    proof['native_builds'].append(native)
    for row in canonical['outputs']:
        if row['relative'].startswith('host/'):
            proof['output_witnesses'][row['relative']]=dict(stamp=w(row['original_stamp']),build_results_sha256=native['results']['sha256'])
    proof['canonical_host_stage_witness']=w(canonical)
    proof['canonical_host_manifest_witness']=w(dict(extensions=[dict(path=r['relative'],sha256=r['staged_sha256']) for r in canonical['outputs']],staged_libs=[dict(name='libMojolearnMath.so',sha256='f'*64)]))
    overrides.append(dict(relative='.libs/libMojolearnMath.so',original_hip_sha256='e'*64,canonical_cuda_sha256='f'*64))
    proof['composition_witness']=w(dict(schema='mojolearn.canonical-host-assembly.v1',package_source=proof['source_commit'],overrides=overrides,sources=dict(device_manifest_sha256=proof['stage_manifest_witness']['sha256'],canonical_manifest_sha256=proof['canonical_host_manifest_witness']['sha256'])))
    proof['runtime_libraries']['libMojolearnMath.so']='f'*64
    return proof


class Admission(unittest.TestCase):
    def setUp(self):self.proof=fixture()

    def change_stage(self,fn):
        value=a.decoded(self.proof['stage_witness']);fn(value);self.proof['stage_witness']=w(value)

    def change_results(self,fn):
        old=self.proof['native_builds'][0]['results']['sha256'];value=a.decoded(self.proof['native_builds'][0]['results']);fn(value)
        self.proof['native_builds'][0]['results']=w(value);new=self.proof['native_builds'][0]['results']['sha256']
        for r in self.proof['output_witnesses'].values():
            if r['build_results_sha256']==old:r['build_results_sha256']=new

    def refused(self):self.assertFalse(a.complete_native_proof(self.proof))

    def test_full_inventory_and_legacy_build_are_distinct(self):
        self.assertTrue(a.validate_staged(self.proof));self.assertEqual(len(self.proof['extensions']),68);self.assertEqual(len(self.proof['host_extension']),42)
        legacy=dict(schema=a.BUILD_SCHEMA,action='build',complete=True,build_exit=0)
        self.assertTrue(a.complete_native_proof(legacy))
        self.proof['build_exit']=0;self.refused()

    def test_missing_duplicate_and_unexpected_output(self):
        for mutation in (lambda r:r.pop(),lambda r:r.append(r[0]),lambda r:r[0].update(relative='unregistered.so')):
            self.proof=fixture();self.change_stage(lambda s:mutation(s['outputs']));self.refused()

    def test_duplicate_manifest_output_is_refused(self):
        value=a.decoded(self.proof['stage_manifest_witness'])
        value['extensions'].append(value['extensions'][0])
        self.proof['stage_manifest_witness']=w(value);self.refused()

    def test_original_build_failure_missing_receipt_and_wrong_tier(self):
        for mutation in (lambda r:r.pop(),lambda r:r[0].update(exit=1),lambda r:r[0].update(tier='host'),lambda r:r[0].update(timeout=True)):
            self.proof=fixture();self.change_results(mutation);self.refused()

    def test_raw_and_shipped_sha_and_closure_mismatches(self):
        for field in ('original_sha256','staged_sha256'):
            self.proof=fixture();self.change_stage(lambda s:s['outputs'][0].update({field:'f'*64}));self.refused()
        self.proof=fixture();self.change_stage(lambda s:s['outputs'][0]['admitted_closure'].update(digest='f'*64));self.refused()

    def test_wrong_compiler_source_and_witness_tampering(self):
        for field,value in [('compiler','Mojo another'),('source_commit','f'*40)]:
            self.proof=fixture();m=a.decoded(self.proof['native_builds'][0]['manifest']);m[field]=value;self.proof['native_builds'][0]['manifest']=w(m);self.refused()
        self.proof=fixture();self.proof['source_commit']='f'*40;self.refused()
        self.proof=fixture();self.proof['stage_witness']['utf8']+=' ';self.refused()

    def test_wrong_mode_and_unregistered_architecture_waiver(self):
        self.change_stage(lambda s:s['outputs'][0]['readback'].update(mode=2));self.refused()
        self.proof=fixture();self.change_stage(lambda s:s['outputs'][0].update(embedded_architectures=[]));self.refused()

    def test_canonical_hosts_are_explicit_and_cannot_replace_device_outputs(self):
        self.proof=canonicalize(self.proof);self.assertTrue(a.validate_staged(self.proof))
        for mutation in (lambda c:c['overrides'].pop(),lambda c:c['overrides'][0].update(relative='_mojolearn_gbdt.so'),lambda c:c['overrides'][0].update(canonical_cuda_sha256='0'*64)):
            self.proof=canonicalize(fixture());value=a.decoded(self.proof['composition_witness']);mutation(value);self.proof['composition_witness']=w(value);self.refused()
        self.proof=canonicalize(fixture());del self.proof['canonical_host_manifest_witness'];self.refused()

    def test_current_source_inventory_must_match(self):
        with patch('check_linux_release_qualification.tracked_native_inventory',return_value=[['changed.py','f'*64]]):
            self.assertFalse(a.complete_native_proof(self.proof,Path('/irrelevant')))

    def test_adapter_refuses_actual_raw_and_staged_bytes_mismatch(self):
        # Failure occurs before source admission: real files disagree with genuine receipt hashes.
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);stage=a.decoded(self.proof['stage_witness']);row=stage['outputs'][0]
            (p/'admission').write_text(self.proof['stage_witness']['utf8']);(p/'manifest').write_text(self.proof['stage_manifest_witness']['utf8'])
            (p/'stamps').mkdir();(p/'raw').mkdir();(p/'staged').mkdir()
            (p/'stamps'/(row['relative'].replace('/','__')+'.json')).write_text(self.proof['output_witnesses'][row['relative']]['stamp']['utf8'])
            (p/'raw'/row['relative']).write_bytes(b'wrong')
            with self.assertRaisesRegex(ValueError,'Raw artifact SHA'):
                a.adapt(p/'admission',p/'stamps',p/'staged',p,[],p/'raw',p/'manifest')
            (p/'raw'/row['relative']).write_bytes(('raw:'+row['relative']).encode());(p/'staged'/row['relative']).write_bytes(b'wrong')
            with self.assertRaisesRegex(ValueError,'Staged artifact SHA'):
                a.adapt(p/'admission',p/'stamps',p/'staged',p,[],p/'raw',p/'manifest')


if __name__=='__main__':unittest.main()
