"""LatentDirichletAllocation's fit E-step runs one GPU block per document.

The resident fit kit (`DKit.lda_rows` in x_decomp/kit_device.mojo) launched
`lda_rows_kernel`, one GPU thread per document, each scanning every vocabulary
word for every topic on every inner iteration: the measurement smoke's text set
(2000 x 4096 counts, 16 topics, max_iter 20) ran past its 300 s step limit on
gfx942. The fit now takes `launch_lda_rows`, the same launch transform already
used (`lda_block_kernel`: the document's nonzero words compacted once, the
words and topics spread over the block's threads; same bits as `lda_doc_row`).

These checks read the kernel sources, so they bound the per-iteration launch
shape without a GPU: no kit entry may launch the one-thread-per-document kernel
directly; only `launch_lda_rows` may, and only past the block kernel's topic cap.
Written without a run (lane fix-lda-text-time, 2026-11-08).
"""
import pathlib
import re

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[3]
_X_DECOMP = _ROOT / "x_decomp"

pytestmark = pytest.mark.skipif(not (_X_DECOMP / "kit_device.mojo").is_file(),
                                reason="needs the source checkout (x_decomp/*.mojo)")


def _def_body(text, name):
    """The text of `def <name>(` up to the next def at the same or a lower indent."""
    m = re.search(r"^([ \t]*)def " + re.escape(name) + r"\(", text, re.M)
    assert m, name
    indent = len(m.group(1))
    lines = text[m.start():].splitlines()
    body = [lines[0]]
    for line in lines[1:]:
        stripped = line.lstrip()
        if stripped and (len(line) - len(stripped)) <= indent and (
                stripped.startswith("def ") or stripped.startswith("@") or stripped.startswith("struct ")):
            break
        body.append(line)
    return "\n".join(body)


def test_resident_fit_kit_takes_the_block_launch():
    body = _def_body((_X_DECOMP / "kit_device.mojo").read_text(), "lda_rows")
    body = "\n".join(line.split("#", 1)[0] for line in body.splitlines())  # code, not comments
    assert "launch_lda_rows(" in body
    assert "lda_rows_kernel" not in body


def test_only_the_launcher_starts_the_per_thread_kernel():
    sites = []
    for path in sorted(_X_DECOMP.rglob("*.mojo")):
        text = path.read_text()
        for m in re.finditer(r"enqueue_function\[\s*lda_rows_kernel\s*\]", text):
            sites.append((path.relative_to(_ROOT).as_posix(), text[:m.start()].count("\n") + 1))
    assert sites, "lda_rows_kernel is no longer launched anywhere; update this check"
    device = (_X_DECOMP / "device.mojo").read_text()
    launcher = _def_body(device, "launch_lda_rows")
    assert "k <= LDA_K_CAP" in launcher and "lda_block_kernel" in launcher
    assert [s for s in sites if s[0] != "x_decomp/device.mojo"] == []
    assert len(sites) == 1 and "enqueue_function[lda_rows_kernel]" in launcher
