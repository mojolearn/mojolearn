# Shared x_neighbors provider continuation

The existing packed CPU/GPU backend exports77 operations used by neighbor,
clustering, graph, semi-supervised and kernel-feature algorithms. Their scalar
admission and calls to existing numerical owners now have typed native tables,
using the same common descriptor provider as neural/SVM/kernel-method work.

`MOJOLEARN_CLASSICAL_NATIVE_DISPATCH=1` admits one provider at the selected public
native operation. The public method bodies for34 local/semi/graph/OneClassSVM
operations now hand a None binding slot to the same native binding admission.
Packed operation calls within those methods use the retained provider table;
factory lookup and compile-time mode readback occur at admission. CPU and GPU
use their original numerical owners. Optional nan_cells flags and conditional
softmax64 availability retain their existing contracts.

`tools/gen_x_neighbors_provider.py --check` verifies the tables against retained
packed wrapper source. It does not regenerate numerical drivers or overwrite
manual additions to the older x_neighbors/gen.py output. Both binding exporters
and that older generator register the same new private native table factory.

Source checks and the authored public probe are separate. The probe compares
selected public outputs bitwise against the retained packed route and poisons
legacy xn_* entry names during candidate calls. It has not been executed by the
lane. No compilation, timing or device qualification is claimed.

This tranche remains partial: custom resident/capability entry dispatch, input
and output metadata reuse, remaining Python-owned orchestration and other public
frontdoors need subsequent migration. The candidate is not the default yet.

The subsequent auxiliary tranche adds native OneClassSVM alpha initialization
and resident execution plus PageRank vector normalization for their existing
CPU/GPU owners. Compile-time capability words are read once into an operation-local
`CompiledBindingSnapshot`, which retains the existing common provider. Nested
helpers read native words from that snapshot; no input metadata is stored in a
persistent provider cache. The host retains its existing non-resident solver.
The public probe also poisons the corresponding legacy auxiliary entry names.
