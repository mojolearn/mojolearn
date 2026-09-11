# Pinned source-code corpus for the byte language model (the Pile, GitHub)

The code kind of ENGINEERING_RULES section 9 for every neural timing and
quality claim, from 2026-09-11 night (the English kind is
`training/corpus/enwik8`). It replaces `training/corpus/cpython312_lib`
(4.5 MB of one project's Python, no published numbers), because Andrew
asked for two corpora that are the norm, generalize and carry published
benchmarks.

The Pile's GitHub component is ordinary public GitHub code in many
languages, with the READMEs and config files that come with it. It has
published bits per byte: GPT-2 small 1.7912 and GPT-3 davinci 0.5635 (Gao
et al., arXiv 2101.00027), and for byte-level models a plain Transformer
0.655, MegaByte 0.570 and SpaceByte 0.500 (arXiv 2404.14408). Those come
from full training runs; a timing leg matches the metric, not the budget.
The Stack and the-stack-smol need an accepted terms gate and a login, and
codeparrot-clean and CodeSearchNet have no published byte-level numbers.

Source: `val.jsonl.zst` of the Hugging Face dataset
`monology/pile-uncopyrighted` at revision
`3be90335b66f24456a5d6659d9c8d208c0357119` (338,045,152 bytes, sha256
`db5e5d1532bf8dc33a6589b50ecba1a8c96f7b4b9cb343d168e603c393007c26`, no
login). Selection: every record whose `meta.pile_set_name` is `Github`, in
file order, its `text` as UTF-8 followed by one `\n` byte, concatenated and
truncated at 100,000,000 bytes. The file holds 18,337 such records and
97,124,565 bytes, so nothing is truncated; sha256
`52a5b4c36ab9119c15505331c10e3b23690377d40fbac3e598c7fafe13a324df`
(computed on the Mac 2026-09-11). No normalization, no tokenizer.

A quality claim against the published numbers evaluates on the same
selection of `test.jsonl.zst` at the same revision (sha256
`a6cafe820e9c350af95b57ec75d35d7c8b479de2eb87caabe6fb6b567e639c3a`); no
such evaluation file is pinned yet.

The bytes are NOT committed. `manifest.json` pins the source, the selection
rule, the hash and the length, and `tools/fetch_corpus_pile_github.sh`
rebuilds `input.txt` (it installs the `zstd` tool with apt when root and the
tool is missing) and refuses any byte that does not match.

Consumers: `tools/lm_step_memory_probe.py --corpus training/corpus/pile_github/input.txt`,
`tools/torch_lm_step_opponent.py --corpus pile_github`, `tools/attention_step_leg.sh`,
`tools/torch_lm_step_opponent_leg.sh`.

License: the code keeps its authors' licenses; this repository
redistributes none of it, only its hash.
