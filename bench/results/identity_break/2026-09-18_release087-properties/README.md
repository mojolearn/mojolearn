# Current release 0.8.7 property references

Twenty-eight lanes, nine fixtures, two repetitions per cell, measured on
Apple M4, NVIDIA H100 and AMD MI325X. Original capture files are preserved
unchanged. Apple captures use the release diagnostic Metal bindings;
NVIDIA and AMD captures run the installed final Linux wheel. Their package
binding hashes are checked against the qualified wheel before admission.

`admission.json` pins every capture and the Linux wheel. All 252 training
cells per vendor are stable and agree; applicable inference, saved-model,
batch-invariance and RL-pair parts agree across vendors. These captures do
not enable the separate step/full flag and do not qualify physical
multi-GPU execution or automatically promote the public verifier's holds.
