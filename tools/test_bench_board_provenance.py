#!/usr/bin/env python3
"""CPU-only regression tests for board path evidence boundaries."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent

def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

P = load('bench_board_provenance')
B = load('bench_board')

class Provenance(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.native = dict(schema=P.SCHEMA, source_commit='a'*40, code_path='native',
                           numeric_mode='identical', files={'cuda_native/sm_89/identical/a.so':'b'*64})
        self.ptx = dict(self.native, code_path='ptx-baseline')

    def tearDown(self):
        self.temp.cleanup()

    def test_nested_and_aliased_extensions_are_pinned_by_actual_file(self):
        import types
        gpu = self.root/'_mojolearn_linalg.so'; gpu.write_bytes(b'gpu')
        host = self.root/'_mojolearn_training_host.so'; host.write_bytes(b'host')
        core = self.root/'_mojolearn.so'; core.write_bytes(b'core')
        modules = {
            'mojolearn._mojolearn': types.SimpleNamespace(__file__=str(core)),
            'mojolearn._sets.identical._mojolearn_linalg': types.SimpleNamespace(__file__=str(gpu)),
            'mojolearn._host.training_alias': types.SimpleNamespace(__file__=str(host)),
            'mojolearn.missing': types.SimpleNamespace(),
            'other._mojolearn_fake': types.SimpleNamespace(__file__=str(gpu)),
        }
        with patch.dict(P.sys.modules, modules, clear=True):
            rows = P.loaded_bindings()
        self.assertEqual(len(rows), 3)
        self.assertEqual({r['role'] for r in rows}, {'host', 'gpu'})
        self.assertEqual(next(r for r in rows if r['role']=='host')['sha256'], P.sha(host))
        self.assertTrue(all(r['module'].startswith('mojolearn.') for r in rows))

    def test_forced_path_requires_manifest_and_opt_in(self):
        f = self.root/'manifest.json'; f.write_text(json.dumps(self.native))
        with patch.dict(os.environ, {'MOJOLEARN_CUDA_PATH':'ptx-baseline'}, clear=True):
            with self.assertRaisesRegex(ValueError, 'require'): P.identity(None)
            with self.assertRaisesRegex(ValueError, 'differs'): P.identity(f)
            f.write_text(json.dumps(self.ptx))
            with self.assertRaisesRegex(ValueError, 'opt-in'): P.identity(f)
        with patch.dict(os.environ, {}, clear=True): self.assertIsNone(P.identity(None))

    def test_installed_artifact_changed(self):
        p=self.root/'identity_columns'; p.mkdir(); (p/'COMMIT').write_text('a'*40)
        binary=self.root/'a.so'; binary.write_bytes(b'original')
        doc=dict(self.native,files={'a.so':P.sha(binary)})
        P.installed_check(doc,self.root)
        binary.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'bytes differ'):P.installed_check(doc,self.root)

    def test_wrong_path_and_legacy_smoke_cannot_qualify_guarded_run(self):
        r=dict(family='algos',lane='pca',dataset='taxi')
        doc=dict(schema=B.SMOKE_SCHEMA,vendor='nvidia',files_sha256=B.smoke_files_sha256(),
                 races={B.smoke_key(r):{'pass':True}},artifact_identity=self.native)
        f=self.root/'smoke.json';f.write_text(json.dumps(doc))
        self.assertIsNone(B.smoke_gate(str(self.root/'out'),'nvidia',[r],[str(f)],self.native))
        self.assertIsNotNone(B.smoke_gate(str(self.root/'out'),'nvidia',[r],[str(f)],self.ptx))
        changed=dict(self.native,files={'x.so':'c'*64})
        self.assertIsNotNone(B.smoke_gate(str(self.root/'out'),'nvidia',[r],[str(f)],changed))
        del doc['artifact_identity'];f.write_text(json.dumps(doc))
        self.assertIsNotNone(B.smoke_gate(str(self.root/'out'),'nvidia',[r],[str(f)],self.native))
        self.assertIsNone(B.smoke_gate(str(self.root/'out'),'nvidia',[r],[str(f)]))

    def test_resume_identity_includes_path_and_bytes(self):
        old={'gpu':{'vendor':'nvidia'},'artifact_identity':self.native}
        self.assertNotEqual(B.box_key(old),B.box_key(dict(old,artifact_identity=self.ptx)))
        self.assertNotEqual(B.box_key(old),B.box_key({'gpu':{'vendor':'nvidia'}}))

    def test_missing_invalid_and_wrong_arm_receipts_fail(self):
        with self.assertRaisesRegex(ValueError,'No same-process'):P.read_receipts(self.root,self.native,['ours'])
        f=self.root/'1.json';row=dict(status='verified',final=True,artifact_identity=self.native,
                                    argv=['driver.py','--arm','ours'])
        f.write_text(json.dumps(row));self.assertEqual(len(P.read_receipts(self.root,self.native,['ours'])),1)
        with self.assertRaisesRegex(ValueError,'Missing worker'):P.read_receipts(self.root,self.native,['ours-base'])
        row['artifact_identity']=self.ptx;f.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError,'Invalid final'):P.read_receipts(self.root,self.native,['ours'])

    def test_actual_worker_native_and_ptx_selection_are_checked(self):
        import types
        package = self.root / 'mojolearn'; package.mkdir()
        (package/'identity_columns').mkdir(); (package/'identity_columns/COMMIT').write_text('a'*40)
        (package/'_mojolearn_x.so').write_bytes(b'kernel')
        digest=P.sha(package/'_mojolearn_x.so')
        manifest=dict(self.native,files={'_mojolearn_x.so':digest})
        ml=types.ModuleType('mojolearn');ml.__file__=str(package/'__init__.py')
        ml.vendor=lambda:'cuda';ml.numeric_mode=lambda:'identical'
        backend=types.ModuleType('mojolearn._backend')
        backend.gpu_plugin=lambda:{'code_format':'native'}
        backend.baseline_selection_receipt=lambda:None
        extension=types.ModuleType('mojolearn._sets.identical._mojolearn_x')
        extension.__file__=str(package/'_mojolearn_x.so')
        with patch.dict(P.sys.modules,{'mojolearn':ml,'mojolearn._backend':backend,'mojolearn._sets.identical._mojolearn_x':extension}), patch.object(P.subprocess,'check_output',return_value='GPU-123, RTX 4090, 8.9, 580.1'):
            receipt=P.collect(manifest)
            self.assertEqual(receipt['loaded_files'][0]['sha256'],digest)
            self.assertFalse(receipt['identical_qualified'])
            backend.gpu_plugin=lambda:{'code_format':'ptx-baseline'}
            with self.assertRaisesRegex(ValueError,'wrong CUDA'):P.collect(manifest)
            manifest=dict(manifest,code_path='ptx-baseline')
            with self.assertRaisesRegex(ValueError,'Missing forced'):P.collect(manifest)
            backend._BASELINE_ROOT=str(package)
            backend.baseline_selection_receipt=lambda:dict(requested='ptx-baseline',selected='ptx-baseline',native_fallback=False,source_commit='a'*40,loaded_files=[{'file':'_mojolearn_x.so','sha256':digest}])
            self.assertEqual(P.collect(manifest)['status'],'verified')
            changed=dict(manifest,files={'_mojolearn_x.so':'d'*64})
            with self.assertRaisesRegex(ValueError,'outside pinned'):P.collect(changed)

    def test_no_opponent_smoke_only_omits_explicitly_skipped_opponents(self):
        race=dict(family='algos',lane='pca',dataset='taxi',arms=['ours','sklearn'],opponents=['sklearn'])
        rec=dict(status='done',params_check='MATCHED',skipped_opponents=['sklearn'],cells=[dict(arm='ours',status='ok',median_ms=1.0,quality={'x':1})])
        self.assertEqual(B.smoke_verdict(race,rec,'nvidia'),[])
        rec['skipped_opponents']=['ours','sklearn'];rec['cells']=[]
        self.assertTrue(B.smoke_verdict(race,rec,'nvidia'))

    def test_source_native_requires_clean_exact_checkout_and_rejects_ptx(self):
        import subprocess
        repo=self.root/'repo';package=repo/'python/mojolearn';package.mkdir(parents=True)
        (package/'__init__.py').write_text('# frozen source\n')
        subprocess.run(['git','init','-q',str(repo)],check=True)
        subprocess.run(['git','-C',str(repo),'add','.'],check=True)
        subprocess.run(['git','-C',str(repo),'-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','frozen'],check=True)
        head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
        manifest=dict(self.native,source_commit=head,installation='source',source_root=str(repo))
        P.source_check(manifest,package)
        (package/'__init__.py').write_text('# changed source\n')
        with self.assertRaisesRegex(ValueError,'clean frozen'):P.source_check(manifest,package)
        f=self.root/'source-manifest.json';f.write_text(json.dumps(dict(manifest,code_path='ptx-baseline')))
        with patch.dict(os.environ,{'MOJOLEARN_CUDA_PATH':'ptx-baseline','MOJOLEARN_EXPERIMENTAL_PTX':'1'}):
            with self.assertRaisesRegex(ValueError,'native-only'):P.identity(f)

    def test_comparison_refuses_mismatched_source_settings_data_hardware(self):
        import copy
        def board(identity):
            cell=dict(arm='ours',library='mojolearn',status='ok',median_ms=2,hash='abc',settings={'rounds':1})
            return dict(plan=['r'],config=dict(artifact_identity=identity,rounds=1,data_sha256={'r':'data'}),
                        box=dict(artifact_hardware='GPU-1'),races={'r':dict(status='done',params_check='MATCHED',cells=[cell],worker_provenance=[dict(status='verified',artifact_identity=identity,hardware='GPU-1')])})
        runtime={'/common/libKGEN.so':'f'*64}
        native,ptx=board(dict(self.native,runtime_files=runtime)),board(dict(self.ptx,runtime_files=runtime))
        for doc in (native,ptx):
            doc['races']['r']['worker_provenance'][0]['loaded_runtime_files']=[dict(file='/common/libKGEN.so',sha256='f'*64)]
        self.assertEqual(P.compare_boards(native,ptx)['races'],1)
        for category,key,value in [('config','rounds',2),('config','data_sha256',{'r':'changed'}),('box','artifact_hardware','GPU-2')]:
            changed=copy.deepcopy(ptx);changed[category][key]=value
            with self.assertRaises(ValueError):P.compare_boards(native,changed)
        changed=copy.deepcopy(ptx);changed['config']['artifact_identity']['source_commit']='c'*40
        with self.assertRaisesRegex(ValueError,'source_commit'):P.compare_boards(native,changed)
        changed=copy.deepcopy(ptx);changed['races']['r']['cells'][0]['hash']='different'
        with self.assertRaisesRegex(ValueError,'digest mismatch'):P.compare_boards(native,changed)

    def test_ptx_requires_common_runtime_inventory(self):
        f=self.root/'ptx.json';f.write_text(json.dumps(self.ptx))
        with patch.dict(os.environ,{'MOJOLEARN_CUDA_PATH':'ptx-baseline','MOJOLEARN_EXPERIMENTAL_PTX':'1'}):
            with self.assertRaisesRegex(ValueError,'runtime_files'):P.identity(f)
            f.write_text(json.dumps(dict(self.ptx,runtime_files={'relative.so':'f'*64})))
            with self.assertRaisesRegex(ValueError,'absolute'):P.identity(f)
            f.write_text(json.dumps(dict(self.ptx,runtime_files={'/common/libKGEN.so':'f'*64})))
            self.assertEqual(P.identity(f)['runtime_files'],{'/common/libKGEN.so':'f'*64})

    def test_actual_runtime_mapping_rejects_bundled_override_and_tamper(self):
        common=self.root/'common';common.mkdir();lib=common/'libKGEN.so';lib.write_bytes(b'common')
        bundled=self.root/'bundled';bundled.mkdir();other=bundled/'libKGEN.so';other.write_bytes(b'common')
        maps=self.root/'maps';manifest=dict(self.native,runtime_files={str(lib):P.sha(lib)})
        maps.write_text('0-1 r-xp 0 00:00 1 '+str(lib)+'\n')
        rows=P.loaded_runtime(manifest,maps);self.assertEqual(rows,[dict(file=str(lib.resolve()),sha256=P.sha(lib))])
        maps.write_text('0-1 r-xp 0 00:00 1 '+str(other)+'\n')
        with self.assertRaisesRegex(ValueError,'outside pinned'):P.loaded_runtime(manifest,maps)
        maps.write_text('0-1 r-xp 0 00:00 1 '+str(lib)+'\n');lib.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'outside pinned'):P.loaded_runtime(manifest,maps)
        maps.write_text('')
        with self.assertRaisesRegex(ValueError,'No actual'):P.loaded_runtime(manifest,maps)

    def test_registration_performed_before_clock_collection_only_at_exit(self):
        P._registered=False
        with patch.dict(os.environ,{P.ENV:'manifest',P.RECEIPTS:str(self.root)}), patch.object(P.atexit,'register') as reg, patch.object(P,'collect') as collect:
            P.register_worker('torch');reg.assert_not_called()
            P.register_worker('mojolearn');reg.assert_called_once();collect.assert_not_called()
            P.register_worker('mojolearn');reg.assert_called_once()
        P._registered=False

if __name__=='__main__':unittest.main()
