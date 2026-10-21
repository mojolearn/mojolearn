"""Regression check: models.Tokenizer built from a BpeVocabularyTrainer result.

A tokenizer host shell read token bytes from a freed array (Mojo dropped the
`array('B')` before the memcpy), so small tokens read allocator bytes and
`Tokenizer(tokens, merges, pattern=...)` refused with "token 1 repeats
token 0 (80)". Run with MOJOLEARN_NUMERIC_MODE=identical; exits 1 on a failure.
"""
import sys

import mojolearn as ml

raw = (b"the quick brown fox jumps over the lazy dog. " * 40) + bytes(range(256))
v = ml.tokenizer.BpeVocabularyTrainer(vocab_size=320, min_frequency=2).train([raw])
tokens = list(v.tokens); merges = [(tokens[a], tokens[b]) for a, b, _ in v.merges]
T = ml.models.Tokenizer
failed = 0
for p in ("gpt2", "llama3", "qwen2"):
    try:
        t = T(tokens, merges, pattern=p)
        ids = t.encode_bytes(b"the lazy dog")
        back = t.decode_bytes(ids)
        if back != b"the lazy dog":
            raise AssertionError(f"decode_bytes round trip gave {back!r}")
        print(p, "ok", ids[:6])
    except Exception as e:
        failed += 1
        print(p, "FAIL", type(e).__name__, str(e)[:120])
sys.exit(1 if failed else 0)
