"""Boundary test: a safetensors read admits its mmap slice without lm_corpus.

The checkpoint reader hands memoryview(mmap)[slice] to the shared native
exporter admission. The mmap exporter used to be connected only by the lazily
imported lm_corpus, so CausalLM.load in a fresh process failed with "native
buffers require an exact native exporter". Runs in a fresh interpreter so no
earlier test has imported lm_corpus.
"""
import json
import struct
import subprocess
import sys
import textwrap


def _write_safetensors(path):
    values = struct.pack("<6f", 0.5, -1.0, 2.0, 0.0, 3.25, -0.125)
    ids = struct.pack("<3i", 7, -2, 9)
    header = {"w": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]},
              "ids": {"dtype": "I32", "shape": [3], "data_offsets": [24, 36]}}
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(raw)) + raw + values + ids)


def test_checkpoint_read_in_a_process_without_lm_corpus(tmp_path):
    path = tmp_path / "model.safetensors"
    _write_safetensors(path)
    script = textwrap.dedent("""
        import sys
        from mojolearn.models.safetensors import Checkpoint
        assert "mojolearn.lm_corpus" not in sys.modules
        with Checkpoint.open(sys.argv[1]) as ckpt:
            w = ckpt.read("w")
            ids = ckpt.read("ids")
        assert type(w).__name__ == "Array" and w.shape == (2, 3) and w.dtype == "<f4"
        assert w.tolist() == [[0.5, -1.0, 2.0], [0.0, 3.25, -0.125]]
        assert ids.dtype == "<i4" and ids.tolist() == [7, -2, 9]
        assert "mojolearn.lm_corpus" not in sys.modules
        print("ok")
    """)
    done = subprocess.run([sys.executable, "-c", script, str(path)], capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip().endswith("ok")
