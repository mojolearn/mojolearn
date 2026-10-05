import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import nvidia_extra_capture as capture


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name); self.source='a'*40
        self.binary=self.root/'identical/x.so';self.binary.parent.mkdir();self.binary.write_bytes(b'PTX bytes')
        self.manifest=self.root/'PTX_BASELINE.json';self.manifest.write_text(json.dumps(dict(source_commit=self.source)))
        self.rows=[dict(module='_mojolearn_x',file=str(self.binary),sha256=capture.sha(self.binary))]
        self.ml=SimpleNamespace(__file__=str(self.root/'__init__.py'),vendor=lambda:'cuda',numeric_mode=lambda:'identical')
        self.runtime=dict(schema='mojolearn.ptx-baseline-selection.v1',requested='ptx-baseline',selected='ptx-baseline',native_fallback=False,
                          manifest_sha256=capture.sha(self.manifest),source_commit=self.source,
                          loaded_files=[dict(file='identical/x.so',sha256=capture.sha(self.binary))])
        self.backend=SimpleNamespace(gpu_plugin=lambda:dict(code_format='ptx-baseline'),gpu_arch=lambda:'sm_89',
                                     _BASELINE_ROOT=str(self.root),baseline_selection_receipt=lambda:self.runtime)
        self.qualifier=SimpleNamespace(manifest_files=lambda _: {'identical/x.so':{'sha256':capture.sha(self.binary)}},
                                       installed_source_evidence=Mock(return_value={'source_commit':self.source}))

    def witness(self):
        return capture.witness(self.ml,self.backend,self.rows,self.manifest,'baseline',self.qualifier)

    def test_forced_baseline_witness_uses_actual_loaded_hash(self):
        d=self.witness();self.assertEqual(d['loaded_bindings'][0]['file'],'identical/x.so')
        self.assertFalse(d['runtime']['native_fallback'])
        self.qualifier.installed_source_evidence.assert_called_once()

    def test_fallback_or_wrong_loaded_receipt_refused(self):
        self.runtime['native_fallback']=True
        with self.assertRaisesRegex(ValueError,'forced PTX'):self.witness()
        self.runtime['native_fallback']=False;self.runtime['loaded_files']=[]
        with self.assertRaisesRegex(ValueError,'actual loaded'):self.witness()

    def test_changed_loaded_bytes_refused(self):
        self.binary.write_bytes(b'different')
        with self.assertRaisesRegex(ValueError,'Loaded binding hash'):self.witness()

    def test_native_cannot_claim_baseline_path(self):
        self.backend.gpu_plugin=lambda:dict(code_format='native')
        with self.assertRaisesRegex(ValueError,'selection differs'):self.witness()

    def test_script_failure_retains_data_and_explicit_incomplete_receipt(self):
        script=self.root/'capture.py';script.write_text('import pathlib,sys\npathlib.Path(sys.argv[1]).write_text("{}")\nraise RuntimeError("capture failed")\n')
        self.ml._backend=self.backend
        args=SimpleNamespace(source_tools=self.root,role='baseline',script=script,manifest=self.manifest,out=self.root/'receipt.json')
        modules={'mojolearn':self.ml,'mojolearn._verify':SimpleNamespace(binding_artifacts=lambda:self.rows),
                 'nvidia_baseline_qualification':self.qualifier}
        old_path=sys.path[:]
        try:
            with patch.dict(sys.modules,modules):self.assertEqual(capture.run(args),1)
        finally:sys.path[:]=old_path
        d=json.loads(args.out.read_text());self.assertFalse(d['complete']);self.assertFalse(d['qualification'])
        self.assertIn('capture failed',d['error']);self.assertTrue(args.out.with_suffix('.capture.json').is_file())
        self.assertEqual(d['capture_script_sha256'],capture.sha(script));self.assertIn('witness',d)


if __name__=='__main__':unittest.main()
