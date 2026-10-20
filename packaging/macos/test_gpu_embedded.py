"""Static Metal image proof must reject placeholders and missing delegates."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest

_spec = importlib.util.spec_from_file_location('embedded', Path(__file__).with_name('check_gpu_embedded.py'))
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)


def library():
    def field(tag, value):
        return tag + struct.pack('<H', len(value)) + value
    fields = field(b'NAME', b'tiny_kernel\0') + field(b'TYPE', b'\x02') + b'ENDT'
    record = struct.pack('<I', 4 + len(fields)) + fields
    functions = struct.pack('<I', 1) + record
    payload = b'BC\xc0\xde' + b'fixture-bitcode'
    code = b'\xde\xc0\x17\x0b' + struct.pack('<IIII', 0, 20, len(payload), 0xffffffff) + payload
    head = bytearray(88)
    head[:4] = b'MTLB'
    struct.pack_into('<Q', head, 16, 88 + len(functions) + len(code))
    struct.pack_into('<QQ', head, 24, 88, len(record))
    struct.pack_into('<QQ', head, 72, 88 + len(functions), len(code))
    return bytes(head) + functions + code


class EmbeddedTests(unittest.TestCase):
    def test_small_library_does_not_need_ten_strings(self):
        self.assertEqual(gate.metallib_kernels(library()), ['tiny_kernel'])

    def test_strings_and_placeholder_are_not_code(self):
        self.assertEqual(gate.metallib_kernels(b'air.main metallib AIR' * 100), [])
        self.assertEqual(gate.metallib_kernels(b'MTLB' + b'\0' * 130), [])

    def test_truncated_and_out_of_range_records_fail(self):
        for index in (16, 24, 32, 72, 80):
            data = bytearray(library())
            struct.pack_into('<Q', data, index, 2**63)
            self.assertEqual(gate.metallib_kernels(data), [], index)
        self.assertEqual(gate.metallib_kernels(library()[:-1]), [])

    def test_named_noncompute_or_empty_bitcode_rejected(self):
        self.assertEqual(gate.metallib_kernels(library().replace(b'TYPE\x01\0\x02', b'TYPE\x01\0\x00')), [])
        self.assertEqual(gate.metallib_kernels(library().replace(b'BC\xc0\xde', b'NONE')), [])

    def test_glue_requires_both_same_tier_compiled_delegates(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            def put(rel, data):
                p = root / rel;p.parent.mkdir(parents=True, exist_ok=True);p.write_bytes(data);return p
            glue = put('identical/_mojolearn_x_trees.so', b'host glue')
            rf = put('identical/_mojolearn_rf.so', library())
            gbdt = put('identical/_mojolearn_gbdt.so', library())
            wrong_tier = put('_mojolearn_gbdt.so', library())
            unknown = put('identical/_mojolearn_unknown.so', b'host glue')
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(gate.main([]), 1)
                self.assertEqual(gate.main([glue, rf, gbdt]), 0)
                self.assertEqual(gate.main([glue, rf]), 1)
                self.assertEqual(gate.main([glue, rf, wrong_tier]), 1)
                self.assertEqual(gate.main([unknown, rf, gbdt]), 1)
                self.assertEqual(gate.main([rf, rf]), 1)
                gbdt.write_bytes(b'no GPU code')
                self.assertEqual(gate.main([glue, rf, gbdt]), 1)

    def test_merged_files_are_judged_by_their_index(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            def put(rel, data):
                p = root / rel;p.parent.mkdir(parents=True, exist_ok=True);p.write_bytes(data);return p
            def index(rel, files):
                (root / rel).mkdir(parents=True, exist_ok=True)
                (root / rel / '_mojolearn_merged.json').write_text(json.dumps({'format': 1, 'files': files}))
            native = put('identical/_mojolearn_native.so', library())
            glue = put('identical/_mojolearn_native_m1.so', b'host glue')
            fast = put('_mojolearn_native.so', library())
            index('identical', {'_mojolearn_native.so': ['_mojolearn_rf', '_mojolearn_gbdt'],
                                '_mojolearn_native_m1.so': ['_mojolearn_x_trees']})
            index('.', {'_mojolearn_native.so': ['_mojolearn_rf', '_mojolearn_gbdt']})
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(gate.main([native, fast]), 0)
                # a glue-only merged file passes only beside its same-tier delegates
                self.assertEqual(gate.main([native, glue]), 0)
                self.assertEqual(gate.main([fast, glue]), 1)
                # a member served twice in one tier is refused
                own = put('identical/_mojolearn_rf.so', library())
                self.assertEqual(gate.main([native, own]), 1)
                own.unlink()
                # a merged file that is not all glue needs its own kernels
                index('identical', {'_mojolearn_native.so': ['_mojolearn_rf', '_mojolearn_gbdt'],
                                    '_mojolearn_native_m1.so': ['_mojolearn_x_trees', '_mojolearn_estimators']})
                self.assertEqual(gate.main([native, glue]), 1)
                native.write_bytes(b'no GPU code')
                self.assertEqual(gate.main([native, fast]), 1)


if __name__ == '__main__':
    unittest.main()
