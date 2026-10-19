import importlib.util,sys,pathlib
import pytest,mojolearn
old=pathlib.Path('/tmp/mojolearn-pip-verify-sWrlqV/venv/lib/python3.14/site-packages/mojolearn/_expansion_trees.py')
s=importlib.util.spec_from_file_location('mojolearn._trees_original_for_test',old);module=importlib.util.module_from_spec(s);sys.modules[s.name]=module;s.loader.exec_module(module)
s=importlib.util.spec_from_file_location('dart_lifetime_negative',pathlib.Path('/tmp/mojolearn-pip-verify-sWrlqV/repair-tests/test_dart_buffer_lifetime.py'));t=importlib.util.module_from_spec(s);s.loader.exec_module(t);t.trees=module
with pytest.MonkeyPatch.context() as m:
 try:t.test_binary_probability_zero_column_survives_native_call(m)
 except AssertionError as e:
  assert 'zero column was freed' in str(e)
  print('EXPECTED_FAILURE: original published implementation frees its borrowed zero column')
 else:raise AssertionError('negative control unexpectedly passed')
