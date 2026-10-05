"""Reject unauthorized diagnostic overlays without a GPU or provider call."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import zipfile

import amd_diagnostic_patch as subject
import diagnose_amd_oob as diagnostic


class PatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.fixed = 'a' * 40, 'b' * 40
        self.original = self.root / subject.MEMBER
        self.original.parent.mkdir(parents=True)
        self.original.write_bytes(b'original binary')
        self.binding = self.root / 'patched.so'
        self.binding.write_bytes(b'fixed binary')
        self.wheel = self.root / 'original.whl'
        with zipfile.ZipFile(self.wheel, 'w') as z:
            z.write(self.original, subject.MEMBER)
        self.rows = [dict(distribution='mojolearn-amd-gfx942', path=str(self.wheel), sha256=subject.sha(self.wheel))]
        self.proof = self.root / 'proof.json'
        self.proof.write_text(json.dumps(dict(schema='mojolearn.amd-diagnostic-patch-build.v1',
            source_commit=self.fixed, architecture='gfx942', numeric_mode='identical',
            archive_path=subject.MEMBER, sha256=subject.sha(self.binding), complete=True)))
        self.doc = dict(schema='mojolearn.amd-oob-patch.v1', base_source_commit=self.source,
            patch_source_commit=self.fixed, archive_path=subject.MEMBER,
            original_sha256=subject.sha(self.original), patched_sha256=subject.sha(self.binding),
            original_wheel_sha256=subject.sha(self.wheel), build_proof_sha256=subject.sha(self.proof),
            experimental=True, release_qualified=False)
        self.manifest = self.root / 'manifest.json'
        self.save()

    def save(self):
        self.manifest.write_text(json.dumps(self.doc))

    def check(self):
        return subject.validate(self.manifest, self.proof, self.source,
                                binding=self.binding, wheels=self.rows)

    def test_apply_preserves_original_wheel(self):
        before = self.wheel.read_bytes()
        witness = subject.apply(self.manifest, self.proof, self.binding, self.root, self.source, self.rows)
        self.assertEqual(self.original.read_bytes(), b'fixed binary')
        self.assertEqual(self.wheel.read_bytes(), before)
        self.assertFalse(witness['qualification'])

    def test_wrong_member_or_qualification_refused(self):
        for key, value in [('archive_path', '../outside.so'), ('archive_path', subject.MEMBER.replace('identical','fast')),
                           ('release_qualified', True), ('base_source_commit', 'c'*40), ('original_wheel_sha256','d'*64)]:
            with self.subTest(key=key,value=value):
                saved=self.doc[key]; self.doc[key]=value; self.save()
                with self.assertRaises(ValueError): self.check()
                self.doc[key]=saved

    def test_tampered_binary_and_proof_refused(self):
        self.binding.write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'binding hash'): self.check()
        self.binding.write_bytes(b'fixed binary')
        self.proof.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'proof hash'): self.check()

    def test_matching_proof_hash_does_not_allow_wrong_source(self):
        d=json.loads(self.proof.read_text()); d['source_commit']='c'*40
        self.proof.write_text(json.dumps(d)); self.doc['build_proof_sha256']=subject.sha(self.proof); self.save()
        with self.assertRaisesRegex(ValueError, 'source/completeness'): self.check()

    def test_changed_installed_original_refused(self):
        self.original.write_bytes(b'different original')
        with self.assertRaisesRegex(ValueError, 'Installed original'):
            subject.apply(self.manifest,self.proof,self.binding,self.root,self.source,self.rows)

    def test_original_provenance_stays_strict_fixed_provenance_explicit(self):
        package=self.root/'mojolearn'; (package/'identity_columns').mkdir()
        (package/'identity_columns/COMMIT').write_text(self.source)
        ml=SimpleNamespace(__file__=str(package/'__init__.py'))
        def dist(name):
            native=[subject.MEMBER] if name=='mojolearn-amd-gfx942' else []
            d=dict(schema='mojolearn.linux-payload.v1', source_commit=self.source,
                split=dict(distribution=name,native_members=native),extensions={subject.MEMBER:self.doc['original_sha256']})
            return SimpleNamespace(read_text=lambda _:json.dumps(d))
        self.original.write_bytes(self.binding.read_bytes())
        def loaded():return [dict(file=str(self.original),sha256=subject.sha(self.original))]
        with patch('diagnose_amd_oob.importlib.metadata.distribution',side_effect=dist):
            with self.assertRaisesRegex(ValueError,'Loaded binding differs'):
                diagnostic.provenance(ml,loaded())
            report=diagnostic.provenance(ml,loaded(),self.doc)
            self.assertEqual(report['diagnostic_patch'],self.doc)
            self.assertFalse(report['qualification'])
            with self.assertRaisesRegex(ValueError,'Exactly one patched'):
                diagnostic.provenance(ml,[],self.doc)


if __name__=='__main__':unittest.main()
