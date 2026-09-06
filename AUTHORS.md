# Authors and attribution

Andrew Hendel ([@ajhendel](https://github.com/ajhendel)) created and maintains mojolearn.
Every contributor retains copyright in their contribution; Git history is the authoritative record.
The project does not require copyright assignment.

All Mojo source in this repository was written for mojolearn. Much of it deliberately follows
upstream algorithm and kernel designs. That relationship can create derivative-work obligations
even though the source was independently expressed in Mojo.

The principal design sources include CatBoost, cuML, cuVS, RAFT, Hugging Face Transformers,
state-spaces/mamba, Modular MAX kernels, scikit-learn, FAISS-derived historical work, and published
random-number algorithms. Their applicable copyright, license, provenance, and modification notices
are recorded in `NOTICE`, per-file headers, and the lane `DERIVATION_MAP.tsv` files. Those sources
are authoritative; historical attribution analysis remains available in Git and `archive/`.

The cuRAND/XORWOW provenance question remains explicitly unresolved and must be settled before that
implementation is distributed as legally cleared work.

## AI-assisted development

AI tools have assisted drafting, inspection, and integration. They are not authors, maintainers, or
accountable decision makers. Andrew Hendel remains responsible for accepted changes, evidence claims,
licensing decisions, releases, and failure response.
