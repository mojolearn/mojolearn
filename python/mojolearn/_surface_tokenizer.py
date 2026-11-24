"""THE TOKENIZER LANE'S HOST SURFACE FRAGMENT (lane vdf-parallel, 2026-11-24).

Owned by the `tokenizer` expansion lane and merged by host_surface.py at import
(`EXPANSION_LANES`). Literal data only: no imports, no calls but dict(...).
The tokenizer's EXISTING host binding (`_mojolearn_tokenizer_host`) stays
declared where it is; this fragment adds the GPU binding, `_mojolearn_x_tokenizer`
(bindings/build_x_tokenizer.sh), the device="gpu" side of BpeTokenizer and
BpeVocabularyTrainer (tokenizer/gpu/), so every packaging list carries it
(`host_surface.py --expansion-gpu-bindings`). No host family: the host column
is `_mojolearn_tokenizer_host`.
"""
GPU_BINDINGS = ("_mojolearn_x_tokenizer",)
FAMILIES = ()
TRAINING_LANE_NAMES = {}
PUBLIC_PENDING_LANES = {}
