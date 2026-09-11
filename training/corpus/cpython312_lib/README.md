# Pinned source-code corpus for the byte language model (CPython 3.12.0 Lib)

The second data kind of ENGINEERING_RULES section 9 for neural timing
claims (English text is `training/corpus/tinyshakespeare`; this is source
code). Andrew, 2026-09-11, on what the two kinds are: "2 different but
relatively normal things to train on, not edge cases; we build our
software to handle GENERAL NORMAL CASES."

`input.txt` is the 163 top-level `Lib/*.py` files of the CPython 3.12.0
release tarball (`Python-3.12.0.tgz` from python.org, tarball sha256
`51412956d24a1ef7c97f1cb5f70e185c13e3de1f50d131c0aac6338080687afb`),
sorted bytewise by file name (`LC_ALL=C`) and concatenated with no
separators, 4,522,096 bytes, sha256
`f08d783cac53829be0da6def7ac74947f3e339915dfee7ca8d8db5ee344fc956`. No
Unicode normalization, no tokenizer, no sampling. The bytes are NOT
committed (the repository fences blobs and evidence, and 4.5 MB of
someone else's source belongs behind its own URL); `manifest.json` pins
the source, the selection rule, the hash and the length, and
`tools/fetch_corpus_cpython312_lib.sh` rebuilds `input.txt` from the
tarball and refuses any byte that does not match. The sha256 above was
computed on the Mac on 2026-09-11 from that tarball with that rule.

Consumers: `tools/lm_step_memory_probe.py --corpus training/corpus/cpython312_lib/input.txt`
(the LM target step on real bytes, `tools/attention_step_leg.sh`). The
schedule field of the manifest is the probe's; `validation_range` is
empty because no learning gate is declared for this corpus.

License: the Python Software Foundation License Version 2 covers the
source files; this repository redistributes none of them, only their
hash.
