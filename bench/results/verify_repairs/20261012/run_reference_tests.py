import sys,importlib.util,pathlib
import mojolearn
import pytest
root=pathlib.Path('/Users/andrewhendel/mojolearn-wt/verify-neural-20261012')
# Load from the source path without pytest replacing the installed package.
# This preserves each test's repository-relative evidence paths.
for name in ['test_verify_reference_admit','test_verify_ctr_model_roles']:
 p=root/'python/mojolearn/tests'/(name+'.py')
 spec=importlib.util.spec_from_file_location(name,p);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
class Collection:
 def pytest_collect_file(self,file_path,parent):
  if file_path.name=='reference_registry.py':
   return pytest.Module.from_parent(parent,path=file_path)
p=pathlib.Path(__file__).with_name('reference_registry.py')
p.write_text('from test_verify_reference_admit import *\nfrom test_verify_ctr_model_roles import *\n')
raise SystemExit(pytest.main([str(p),'-q']))
