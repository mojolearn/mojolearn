# Frozen-source H100 to RTX 5090 replay

RunPod `cvisjmryfcmz6g`, two GeForce RTX 5090 32 GB cards, driver
580.126.09. IDENTICAL, NVIDIA column, `sm_120`; H100 reference builds used
`sm_90a`. The source archive is the exact completed H100 working tree in
`../pooled-source-freeze/`, identified here by `source.sha256`. R2 enwik8
matches the same 100,000,000-byte corpus SHA256. No local tests or builds.

All 16 complete structured receipt groups match their H100 references, with
only prose `scope` excluded. Shape/configuration, checks, flags and every
recorded raw-byte hash remain in the comparison. The reference JSON files,
comparison script, logs and verdict are retained. Some inherited harness scope
strings name H100s; `devices.csv` records the hardware actually used here.

The eight neural groups cover all 60 SGD/Adam/AdamW configurations (three steps
each, including global clipping and split momentum flags), existing byte-LM
replay, byte-LM optimizer pooling, MLP, and Samba with/without clipping and
attention/dropout. The eight classical groups cover 95 fitted cases across
pointwise/greedy boosting, OrderedRMSE, two-level feature frequency, classifier
and regressor adapters, wider Gram/OLS, covariance PCA/SVD and tall full PCA.

The additional native checks pass one-vs-two-device histogram, Gram and QR
comparisons on the 5090s. Their PASS lines alone are not cross-architecture
array receipts. Separately, `trace-comparison.json` directly compares every
saved pointwise histogram dump byte and every non-header pointwise/OrderedRMSE
trace record against H100 artifacts, rather than inferring intermediate
identity from matching fitted models.

`replay-neural`, `replay-classical` and `replay-compare` all exit zero.
Build scripts, binary hashes, compiler/gate logs and source/corpus identities
are retained. This is a two-NVIDIA-architecture result for these fixtures,
not AMD/Apple qualification, a speedup result, or full pooled-model capacity.
The pod remains leased for the separate gradient-scratch pooling batch.
