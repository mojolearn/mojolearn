# THREE vendors wrote the same checkpoint file, 2026-09-03

    Apple M4      161008 bytes  md5 fa0e2c151f189b4990f3d6c48ae3132d
    AMD MI325X    161008 bytes  md5 fa0e2c151f189b4990f3d6c48ae3132d
    NVIDIA H100   161008 bytes  md5 fa0e2c151f189b4990f3d6c48ae3132d
    cmp           no differences, any pair

Eight training steps of `mojolearn.identical.train.fp32.v1` -- embed,
transformer block forward, backward, cross-entropy, AdamW with gradient
clipping -- serialized through `mojolearn.identical.train.ckpt.file.v1`.
Both files written at commit `5e5bd5e0`, both under IDENTICAL, neither
machine having seen the other's.

## Why this is a different claim from the card

The cards already showed both boxes reaching `h_all = 463245ce6c97e68d`.
That is two machines agreeing on a NUMBER THEY EACH COMPUTED. This is two
machines producing THE SAME BYTES ON DISK, which is the thing a user
actually moves between machines, and it additionally pins the descriptor,
the layout table, the padding and the byte order -- none of which a digest
covers.

`training/checkpoint.mojo` imports no `max.gpu.host`. No `DeviceContext`
and no `DeviceBuffer` appears in any of its signatures, so it cannot name
the device that wrote it. Vendor neutrality is structural here rather than
a convention that happened to hold.

## What it does not cover

One shape, `d_model 32, L 8, B 2, head_dim 8, V 64`, and one model family.
The mamba blocks have no backward, so nothing about them can be trained
and nothing about them is in this file.

And the resume this file makes possible was tested WITHIN one process on
each box -- `whole(8) == file-split(4+4)`, all three boxes reaching
`463245ce6c97e68d`. WRITING ON ONE VENDOR AND RESUMING ON ANOTHER has not
been run. With three byte-identical files it is a weaker gap than it was --
loading any of them is loading the same bytes -- but "the same bytes, so the
same computation" is an argument and not a measurement, and this directory
does not contain one.
