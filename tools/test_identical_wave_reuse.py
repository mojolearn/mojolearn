"""Metadata refusal tests; no GPU, rental, compiler or package installation."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import identical_wave_reuse as reuse


class ReuseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.out = self.root / 'donor'
        self.source = self.out / 'source'
        self.environment = self.root / 'environment'
        def put(path, content=b'fixture'):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            return path
        self.put = put
        self.receipt = self.out / 'native-build.json'
        self.producer = put(self.out / 'native-builder-producer.py', b'audited producer')
        put(self.source / 'tools/identical_wave_native_build.py', self.producer.read_bytes())
        self.patch = patch.object(reuse, 'AUDITED_PRODUCER', reuse.digest(self.producer))
        self.patch.start(); self.addCleanup(self.patch.stop)
        self.clean = patch.object(reuse, 'clean_source')
        self.clean.start(); self.addCleanup(self.clean.stop)
        self.lock = put(self.source / 'pixi.lock', b'locked')
        put(self.environment / 'pixi.lock', b'locked')
        self.compiler = put(self.environment / '.pixi/envs/default/bin/mojo', b'compiler')
        self.python = put(self.environment / 'python', b'python')
        self.pixi = put(self.environment / 'pixi', b'pixi')
        self.helper = put(self.source / 'python/mojolearn/.libs/libMojolearnMath.so', b'portable')
        self.artifact = put(self.source / 'python/mojolearn/identical/_mojolearn.so', b'base-native')
        def record(path):
            return dict(path=str(path), resolved_path=str(path.resolve()), sha256=reuse.digest(path))
        self.att = dict(schema=1, source_path=str(self.source), source_sha='a'*40, source_clean=True,
            pixi_lock=record(self.lock), compiler=dict(record(self.compiler),version='Mojo pinned'),
            pixi=record(self.pixi), python=record(self.python), producer=record(self.producer),
            vendor='nvidia', gpu_arch='sm_120', mode='identical', arm='on', jobs=1, mojo_build_flags='',
            compile_environment=dict(MOJOLEARN_NUMERIC_MODE='identical',MOJOLEARN_COMPILE_JOBS='1',
                MOJOLEARN_GPU_ARCHS='sm_120',MOJOLEARN_TARGET_COLUMN='nvidia'),
            host_override=dict(MOJOLEARN_GPU_ARCHS=None,MOJOLEARN_TARGET_COLUMN='cpu'))
        self.doc = dict(sha='a'*40,vendor='nvidia',arch='sm_120',arm='on',mode='identical',status='PASS',
            expected_builders=['base'],bootstrap=[dict(rc=0)],modules={'base':dict(status='PASS',rc=0,
                import_smoke=dict(rc=0),artifact=str(self.artifact),sha256=reuse.digest(self.artifact))})
        self.flush()
        self.kwargs = dict(sha='a'*40,vendor='nvidia',arch='sm_120',arm='on',builders=['build.sh'],
            environment=self.environment,python=self.python,pixi=self.pixi)

    def flush(self):
        self.receipt.write_text(json.dumps(self.doc))
        (self.out/'producer-attestation.json').write_text(json.dumps(self.att))
        (self.out/'artifact-inventory.json').write_text(json.dumps({str(p.relative_to(self.source)):reuse.digest(p)
            for p in (self.source/'python/mojolearn').rglob('*.so')}))

    def validate(self):
        return reuse.validate_receipt(self.receipt,**self.kwargs)

    def test_valid_copies_to_fresh_source(self):
        result=self.validate(); dest=self.root/'fresh'
        reuse.import_artifacts(result,dest)
        self.assertEqual((dest/'python/mojolearn/identical/_mojolearn.so').read_bytes(),b'base-native')
        self.assertEqual(self.artifact.read_bytes(),b'base-native')
        self.assertIn('artifact_inventory_sha256',result)

    def test_context_mismatch_refused(self):
        for key,value in [('sha','b'*40),('vendor','amd'),('arch','sm_89'),('arm','off'),('mode','fast'),('status','RUNNING')]:
            with self.subTest(key=key):
                old=self.doc[key];self.doc[key]=value;self.flush()
                with self.assertRaises(ValueError):self.validate()
                self.doc[key]=old

    def test_partial_or_failed_import_refused(self):
        self.doc['expected_builders'].append('x_cnn');self.flush()
        with self.assertRaisesRegex(ValueError,'partial'):self.validate()
        self.doc['expected_builders']=['base'];self.doc['modules']['base']['import_smoke']['rc']=1;self.flush()
        with self.assertRaisesRegex(ValueError,'did not pass'):self.validate()

    def test_flags_and_toolchain_refused(self):
        self.att['mojo_build_flags']='-D MOJOLEARN_IDN_ALL_OFF=1';self.flush()
        with self.assertRaisesRegex(ValueError,'flags'):self.validate()
        self.att['mojo_build_flags']='';self.flush();self.compiler.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'hash'):self.validate()

    def test_lock_and_producer_refused(self):
        (self.environment/'pixi.lock').write_bytes(b'newlock')
        with self.assertRaisesRegex(ValueError,'lock'):self.validate()
        (self.environment/'pixi.lock').write_bytes(b'locked')
        self.producer.write_bytes(b'unreviewed');self.att['producer']['sha256']=reuse.digest(self.producer);self.flush()
        with self.assertRaisesRegex(ValueError,'audited'):self.validate()

    def test_artifact_and_helper_tampering_refused(self):
        self.artifact.write_bytes(b'newmodule')
        with self.assertRaisesRegex(ValueError,'inventory'):self.validate()
        self.artifact.write_bytes(b'base-native');self.helper.write_bytes(b'newhelper')
        with self.assertRaisesRegex(ValueError,'inventory'):self.validate()

    def test_added_binary_or_forged_path_refused(self):
        extra=self.put(self.source/'python/mojolearn/identical/extra.so')
        self.flush()
        with self.assertRaisesRegex(ValueError,'unexpected'):self.validate()
        extra.unlink();self.doc['modules']['base']['artifact']=str(self.root/'elsewhere.so');self.flush()
        with self.assertRaisesRegex(ValueError,'path'):self.validate()

    def test_symlink_and_late_mutation_refused(self):
        result=self.validate();self.artifact.write_bytes(b'late')
        with self.assertRaisesRegex(ValueError,'changed after'):reuse.import_artifacts(result,self.root/'fresh')
        self.artifact.unlink();outside=self.put(self.root/'outside.so',b'base-native');self.artifact.symlink_to(outside);self.flush()
        with self.assertRaisesRegex(ValueError,'escapes'):self.validate()

    def test_dirty_source_refused(self):
        with patch.object(reuse,'clean_source',side_effect=ValueError('dirty')):
            with self.assertRaisesRegex(ValueError,'dirty'):self.validate()

    def test_donor_proof_changed_refused(self):
        self.doc['modules']['base']['reused_from']=dict(module='base',receipt=str(self.receipt),receipt_sha256='0'*64)
        self.flush()
        with self.assertRaisesRegex(ValueError,'donor receipt'):self.validate()


class ClosureTests(unittest.TestCase):
    def test_initializer_change_invalidates_only_own_module(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def git(*args):
                return subprocess.check_output(['git','-C',str(root),*args],stderr=subprocess.DEVNULL)
            git('init','-q');(root/'bindings').mkdir();(root/'tools').mkdir()
            (root/'bindings/_mojolearn.mojo').write_text('def main(): pass\n')
            linear=root/'bindings/_mojolearn_x_linear.mojo';linear.write_text('def main(): pass\n')
            diag=root/'tools/identical_dart_bit_diagnostic.py';diag.write_text('external=1\n')
            def commit():
                git('add','.');git('-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','fixture')
            commit();base=reuse.dependency_fingerprint(root,'base');old=reuse.dependency_fingerprint(root,'x_linear')
            linear.write_text('@export\ndef main(): pass\n');diag.write_text('external=2\n');commit()
            self.assertEqual(base,reuse.dependency_fingerprint(root,'base'))
            self.assertNotEqual(old,reuse.dependency_fingerprint(root,'x_linear'))
            (root/'bindings/_mojolearn.mojo').write_text('from bindings._mojolearn_x_linear import main\n');commit()
            with self.assertRaisesRegex(ValueError,'imported'):reuse.dependency_fingerprint(root,'base')


if __name__=='__main__':unittest.main()
