# Pinned real-text byte-language-model fixture

This is the Tiny Shakespeare text distributed in Andrej Karpathy's
[char-rnn repository](https://github.com/karpathy/char-rnn/tree/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare),
containing text by William Shakespeare. `input.txt` is preserved byte for byte;
`manifest.json` records its immutable source URL, SHA256 and byte length.
No Unicode normalization, tokenizer training or random corpus sampling occurs.

The proposed experiment uses a two-block, 34,944-parameter FP32 byte decoder,
batch 2 and context 32. Each row has 33 bytes: 32 inputs and their shifted
next-byte targets. The first 65,536 corpus bytes are the training pool; the
next 8,192 are held out. Training runs 128 fixed steps. For zero-based step
`s` and row `b`, the starting offset is `(64*s + 32*b) % 65504`.
The manifest's eight validation starts each identify a two-row batch; row 1
starts 32 bytes after row 0. Validation is evaluation only and never updates
parameters, optimizer state or the training cursor.

Before any model execution, the learning gate is fixed at final mean held-out
loss at most 90% of initial mean held-out loss, on the same eight batches.
This demonstrates limited next-byte learning. It does not demonstrate useful
generation, contextual reasoning, beating a unigram baseline, or large-model
scaling. A failed gate remains failed; any revised experiment gets a separate
manifest and retained history.

Learning, independent gradient/update correctness, and bitwise cross-vendor
state equality are separate gates. Matching loss alone is insufficient.
The new model and this dataset experiment have **not been executed or
qualified**. Root alone runs bounded remote NVIDIA/AMD jobs; fresh Apple
execution remains deferred. Initialization and optimizer bytes must be frozen
in the run descriptor before those jobs begin.
