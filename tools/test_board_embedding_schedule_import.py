"""Import/metadata regressions only; no device or timing calls."""
import importlib.util
import pathlib
import sys
import types
import unittest
from unittest.mock import patch
import bench_board_algos as b

class SmokeImportMetadata(unittest.TestCase):
    def test_sequence_schedules_import_without_removed_unused_epsilon(self):
        pkg=types.ModuleType("schedule_import_test");pkg.__path__=[]
        train=types.ModuleType(pkg.__name__+"._training_impl")
        for name in ("_LrTable", "_lr_buffers", "_lr_native", "_lr_status"):
            setattr(train,name,type("Table",(),{}) if name=="_LrTable" else object())
        math=types.ModuleType(pkg.__name__+"._portable_math")
        name=pkg.__name__+"._x_sequence_sched"
        path=pathlib.Path(__file__).resolve().parents[1]/"python/mojolearn/_x_sequence_sched.py"
        spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{pkg.__name__:pkg,train.__name__:train,math.__name__:math}):
            spec.loader.exec_module(module)
        for cls in ("ExponentialLR", "StepLR", "OneCycleLR"):
            self.assertTrue(hasattr(module,cls))
    def test_embedding_reads_actual_embedding_binding(self):
        ml=types.ModuleType("mojolearn");ml.vendor=lambda:"metal"
        seen=[]
        def readback(ml,est,binding):
            seen.append(binding)
            if binding!="_mojolearn_embedding":raise ImportError(binding)
            return "fast","binding constant"
        more=types.SimpleNamespace(_mode_readback=readback)
        with patch.dict(sys.modules,{"mojolearn":ml}),patch.object(b,"_tool",return_value=more),patch.dict(b.os.environ,{"MOJOLEARN_NUMERIC_MODE":"fast"}):
            info=b._ours_info("embedding")
        self.assertEqual(seen,["_mojolearn_embedding"])
        self.assertEqual(info["numeric_mode_used"],"fast")

if __name__=="__main__":unittest.main()
