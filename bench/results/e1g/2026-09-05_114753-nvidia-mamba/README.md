# Retained layout-driver compile failure

Frozen source 323b7301a832c831cabfaaf71e014fdffb3ba26f, RTX 4090.
The first baseline check build failed because the new wrapper imported a
module containing main(), which Mojo refuses inside packages. No GPU
correctness or timing result was produced; layout_exit=1.

The controller collected the failure and deleted pod thj6afti3ifqg2; DELETE
returned 204 and GET confirmed 404. Later source fixes and qualifications
must be recorded separately. This is a failed compile, not an identity result.
