"""Offline API fixtures for verified teardown and durable deadman failures."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CURL = r'''#!/usr/bin/env python3
import json, os, pathlib, sys
args=sys.argv[1:]; url=args[-1]; method=args[args.index('-X')+1] if '-X' in args else 'GET'
out=pathlib.Path(args[args.index('-o')+1]); state=pathlib.Path(os.environ['STATE_FILE'])
case=os.environ['CASE']; deleted=state.exists()
code='200'; body=[]
if method=='DELETE':
    if case=='delete-failed': code='503'
    else: state.write_text('deleted'); code='204'
elif url.endswith('/pods'):
    if case=='listing-denied': code='401'; body={'error':'denied'}
    elif case=='listing-invalid': body={'error':'unexpected shape'}
    elif case=='transient' and not state.exists():
        state.write_text('retried'); code='503'; body={}
    else:
        body=[dict(id='pod123',name='fixture',desiredStatus='RUNNING')] if case in ('delete-ok','delete-failed') and not deleted else []
        if case=='wrapped': body={'items':body}
else:
    if case in ('delete-ok','delete-failed') and not deleted: body={'desiredStatus':'RUNNING'}
    elif case=='bad-status': code='500'; body={'desiredStatus':'TERMINATED'}
    else: code='404'; body={}
out.write_text(json.dumps(body));print(code,end='')
'''


class Cleanup(unittest.TestCase):
    def fixture(self, case):
        temporary=tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        root=Path(temporary.name); binaries=root/'bin'; binaries.mkdir()
        for name, source in [('curl',CURL),('sleep','#!/bin/sh\nexit 0\n')]:
            path=binaries/name;path.write_text(source);path.chmod(0o755)
        config=root/'curlrc';config.write_text('credential-sentinel-do-not-print')
        env={**os.environ,'CASE':case,'STATE_FILE':str(root/'deleted'),
             'PATH':str(binaries)+os.pathsep+os.environ['PATH'],
             'TMPD':str(root),'CURLRC':str(config),'POD_NAME':'fixture','LIB':str(ROOT/'tools/runpod_pod_lib.sh')}
        return root,env

    def test_verify_requires_successful_well_formed_listing(self):
        for case, success in [('ok',True),('wrapped',True),('transient',True),
                              ('listing-denied',False),('listing-invalid',False),('bad-status',False)]:
            with self.subTest(case=case):
                root,env=self.fixture(case)
                result=subprocess.run(['bash','-c','source "$LIB"; verify_gone pod123'],env=env,capture_output=True,text=True)
                self.assertEqual(result.returncode==0,success,result.stdout+result.stderr)
                self.assertEqual('VERIFIED:' in result.stdout,success)
                self.assertNotIn('credential-sentinel',result.stdout+result.stderr)

    def deadman(self, case):
        root,env=self.fixture(case); directory=root/'guard';env['GUARD']=str(directory)
        result=subprocess.run(['bash','-c','source "$LIB"; write_deadman "$GUARD" 0'],env=env,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        (directory/'pod_id.txt').write_text('pod123\n')
        result=subprocess.run(['sh',str(directory/'deadman.sh')],env=env,capture_output=True,text=True,timeout=15)
        return directory,result

    def test_deadman_verifies_after_delete_before_discarding_credentials(self):
        directory,result=self.deadman('delete-ok')
        self.assertEqual(result.returncode,0,result.stderr)
        log=(directory/'deadman.log').read_text()
        self.assertIn('DELETE',log)
        self.assertIn('termination VERIFIED',log)
        self.assertFalse((directory/'curlrc').exists())
        self.assertNotIn('credential-sentinel',log)

    def test_deadman_preserves_recovery_credentials_when_api_remains_uncertain(self):
        for case in ('listing-denied','listing-invalid','delete-failed'):
            with self.subTest(case=case):
                directory,result=self.deadman(case)
                self.assertNotEqual(result.returncode,0)
                self.assertTrue((directory/'curlrc').exists())
                log=(directory/'deadman.log').read_text()
                self.assertIn('TEARDOWN UNCONFIRMED',log)
                self.assertNotIn('termination VERIFIED',log)
                self.assertNotIn('credential-sentinel',log)
                self.assertIn('--max-time 30', (directory/'deadman.sh').read_text())


if __name__=='__main__': unittest.main()
