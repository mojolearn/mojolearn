"""Compile native_family_raw_attr_probe.mojo as raw_attr_probe.so and put it on PYTHONPATH.
Test-only descriptors exercise CPython generic lookup and exception preservation;
this probe does not expand the product's admission of caller protocols.
"""
import mojolearn
import raw_attr_probe as native
import unittest

class Probe(unittest.TestCase):
 def test_missing(self):
  sentinel=object()
  self.assertIs(native.read(object(),'absent',sentinel),sentinel)
 def test_value(self):
  class Model:pass
  m=Model();m.value=object()
  self.assertIs(native.read(m,'value',None),m.value)
 def test_descriptor_precedence(self):
  expected=object()
  class Model:
   @property
   def value(self):return expected
  m=Model();m.__dict__['value']='shadow'
  self.assertIs(native.read(m,'value',None),expected)
 def test_attribute_error(self):
  class Model:
   @property
   def value(self):raise AttributeError('missing')
  sentinel=object()
  self.assertIs(native.read(Model(),'value',sentinel),sentinel)
 def test_other_exception(self):
  failure=ValueError('preserve original')
  class Model:
   @property
   def value(self):raise failure
  with self.assertRaises(ValueError) as got:native.read(Model(),'value',None)
  self.assertIs(got.exception,failure)
 def test_no_getattr(self):
  class Model:
   def __getattr__(self,name):raise AssertionError('hook called')
  sentinel=object()
  self.assertIs(native.read(Model(),'absent',sentinel),sentinel)

if __name__ == "__main__":
 unittest.main()
